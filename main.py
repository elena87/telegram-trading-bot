import os
import math
import asyncio
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telegram import Bot

from supabase_db import (
    db_get_state, db_update_state, db_get_params,
    db_log_command, db_insert_trade, db_upsert_snapshot,
    db_update_param, db_get_last_trade, db_get_last_snapshot
)


ROME = ZoneInfo("Europe/Rome")
PRODUCT_ID = "BTC-EUR"
EXCHANGE_TICKER_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/ticker"
EXCHANGE_CANDLES_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"


def now_rome() -> datetime:
    return datetime.now(tz=ROME)


def hour_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H")


def day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def fetch_bid_ask() -> tuple[float, float, float]:
    r = requests.get(EXCHANGE_TICKER_URL, timeout=10)
    r.raise_for_status()
    j = r.json()
    bid = float(j["bid"])
    ask = float(j["ask"])
    mid = (bid + ask) / 2.0
    return bid, ask, mid


def fetch_last_60_candles_1m() -> list[tuple[float, float, float, float]]:
    # Exchange API candles: [time, low, high, open, close, volume]
    # granularity=60 -> 1 minute
    params = {"granularity": 60}
    r = requests.get(EXCHANGE_CANDLES_URL, params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    arr.sort(key=lambda x: x[0])  # time asc
    last = arr[-60:]
    # open, close, high, low
    candles = [(float(x[3]), float(x[4]), float(x[2]), float(x[1])) for x in last]
    return candles


def compute_features(candles: list[tuple[float, float, float, float]]) -> dict:
    closes = [c[1] for c in candles]

    def ret(a, b):
        return math.log(b / a) if a > 0 else 0.0

    r1 = ret(closes[-2], closes[-1]) if len(closes) >= 2 else 0.0
    r15 = ret(closes[-16], closes[-1]) if len(closes) >= 16 else 0.0
    r60 = ret(closes[0], closes[-1]) if len(closes) >= 60 else 0.0

    rets = [ret(closes[i - 1], closes[i]) for i in range(1, len(closes))]
    if len(rets) >= 2:
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        vol = math.sqrt(var)
    else:
        vol = 0.0

    return {"r1": r1, "r15": r15, "r60": r60, "vol_1m": vol}


def forecast_r_hat(feat: dict) -> float:
    # modello base semplice (poi lo miglioreremo)
    return 0.7 * feat["r60"] + 0.3 * feat["r15"]


def map_rhat_to_wtarget(r_hat: float, w_current: float, w_max: float) -> float:
    w_min = 0.0
    r_dead = 0.0010   # 0.10%
    r_full = 0.0040   # 0.40%

    if abs(r_hat) < r_dead:
        return w_current

    if r_hat <= -r_full:
        return w_min
    if r_hat >= r_full:
        return w_max

    t = (r_hat + r_full) / (2 * r_full)  # 0..1
    return w_min + t * (w_max - w_min)


def paper_rebalance(state: dict, params: dict, bid: float, ask: float, mid: float, r_hat: float, feat: dict):
    eur = float(state["paper_eur"])
    btc = float(state["paper_btc"])
    equity = eur + btc * mid
    w_current = (btc * mid / equity) if equity > 0 else 0.0

    spread_pct = (ask - bid) / mid if mid > 0 else 0.0
    slippage_base = float(params["slippage_base"])
    slippage_pct = max(slippage_base, 0.8 * float(feat["vol_1m"]))
    fee_pct = float(params["fee_taker"])
    buffer_pct = float(params["buffer"])

    cost_pct = fee_pct + spread_pct + slippage_pct + buffer_pct
    edge_mult = float(params["edge_mult"])
    band = float(params["band"])
    w_max = float(params["w_max"])
    max_trade_eur = float(params["max_trade_eur"])

    w_target = map_rhat_to_wtarget(r_hat, w_current, w_max)
    delta_w = w_target - w_current

    action = "HOLD"
    reason = ""
    trade_value = 0.0

    if abs(delta_w) < band:
        reason = f"band gate: |Δw|={abs(delta_w):.4f} < {band:.4f}"
    elif abs(r_hat) <= cost_pct * edge_mult:
        reason = f"cost gate: |r_hat|={abs(r_hat):.4f} <= cost_pct*{edge_mult:.2f} ({cost_pct*edge_mult:.4f})"
    else:
        trade_value = abs(delta_w) * equity
        trade_value = min(trade_value, equity / 24.0, max_trade_eur)
        if trade_value < 2.0:
            reason = f"min trade gate: trade_value={trade_value:.2f}€"
            trade_value = 0.0

    exec_price = None
    fee_eur = None
    notional_eur = None

    if trade_value > 0:
        if delta_w > 0:
            notional = min(trade_value, eur)
            if notional <= 0:
                reason = "insufficient EUR"
            else:
                action = "BUY"
                exec_p = ask * (1.0 + slippage_pct)
                fee = notional * fee_pct
                btc_bought = (notional - fee) / exec_p
                eur -= notional
                btc += btc_bought
                exec_price, fee_eur, notional_eur = exec_p, fee, notional
                if not reason:
                    reason = "rebalance up to w_target"
        else:
            exec_p = bid * (1.0 - slippage_pct)
            btc_to_sell = min(btc, trade_value / mid) if mid > 0 else 0.0
            if btc_to_sell <= 0:
                action = "HOLD"
                reason = "insufficient BTC"
            else:
                action = "SELL"
                eur_gross = btc_to_sell * exec_p
                fee = eur_gross * fee_pct
                eur_net = eur_gross - fee
                btc -= btc_to_sell
                eur += eur_net
                exec_price, fee_eur, notional_eur = exec_p, fee, eur_gross
                if not reason:
                    reason = "rebalance down to w_target"

    equity2 = eur + btc * mid
    w_after = (btc * mid / equity2) if equity2 > 0 else 0.0

    trade_row = {
        "action": action,
        "w_current": w_current,
        "w_target": w_target,
        "notional_eur": notional_eur,
        "exec_price": exec_price,
        "fee_eur": fee_eur,
        "slippage_pct": slippage_pct,
        "spread_pct": spread_pct,
        "cost_pct": cost_pct,
        "r_hat": r_hat,
        "reason": reason,
        "w_after": w_after,
        "eur_after": eur,
        "btc_after": btc,
        "equity_after": equity2,
    }

    new_state = dict(state)
    new_state["paper_eur"] = eur
    new_state["paper_btc"] = btc

    return new_state, trade_row


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_eur(x: float) -> str:
    return f"€{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

def fmt_btc(x: float) -> str:
    return f"{x:.8f}"

def parse_number(s: str) -> float:
    # accetta "0.05" o "0,05"
    return float(s.replace(",", "."))


async def process_telegram_commands(bot: Bot, chat_id: str, state: dict) -> dict:
    offset = int(state.get("telegram_offset") or 0)
    updates = await bot.get_updates(offset=offset, timeout=0)
    max_update_id = offset - 1

    # whitelist parametri modificabili + tipo atteso
    # (numeric: float; int: int)
    allowed_params = {
        "w_max": "float",
        "band": "float",
        "slippage_base": "float",
        "fee_taker": "float",
        "buffer": "float",
        "edge_mult": "float",
        "max_trade_eur": "float",
        "daily_stop": "float",
        "max_trades_day": "int",
    }

    for u in updates:
        if u.update_id > max_update_id:
            max_update_id = u.update_id

        msg = u.message
        if not msg or not msg.text:
            continue

        if str(msg.chat_id) != str(chat_id):
            continue

        text = msg.text.strip()
        db_log_command(msg.from_user.id if msg.from_user else None, text)

        if text == "/pause":
            state["status"] = "PAUSED"
            await bot.send_message(chat_id=chat_id, text="⏸️ Ok, bot in PAUSED.")
            continue

        if text == "/resume":
            state["status"] = "RUNNING"
            await bot.send_message(chat_id=chat_id, text="▶️ Ok, bot in RUNNING.")
            continue

        if text == "/kill":
            state["kill_switch"] = True
            await bot.send_message(chat_id=chat_id, text="🛑 Kill switch attivo. Il bot si fermerà.")
            continue

        if text == "/paper":
            state["mode"] = "PAPER"
            state["armed"] = False
            await bot.send_message(chat_id=chat_id, text="✅ Mode impostato su PAPER.")
            continue

        if text == "/live":
            state["mode"] = "LIVE"
            state["armed"] = True
            await bot.send_message(chat_id=chat_id, text="⚠️ LIVE ARMED. (Trading live non implementato ancora) Usa /confirm_live quando sarà pronto.")
            continue

        if text == "/confirm_live":
            await bot.send_message(chat_id=chat_id, text="ℹ️ LIVE non ancora implementato in questo progetto (paper only).")
            continue

        if text == "/params":
            params = db_get_params()
            msg_out = (
                "⚙️ Parametri attuali\n"
                f"- w_max: {params['w_max']}\n"
                f"- band: {params['band']}\n"
                f"- fee_taker: {params['fee_taker']}\n"
                f"- slippage_base: {params['slippage_base']}\n"
                f"- buffer: {params['buffer']}\n"
                f"- edge_mult: {params['edge_mult']}\n"
                f"- max_trade_eur: {params['max_trade_eur']}\n"
                f"- daily_stop: {params['daily_stop']}\n"
                f"- max_trades_day: {params['max_trades_day']}\n"
                "\nModifica: /set <chiave> <valore>  (es: /set band 0.04)"
            )
            await bot.send_message(chat_id=chat_id, text=msg_out)
            continue

        if text == "/status":
            snap = db_get_last_snapshot()
            last_trade = db_get_last_trade()

            eur = float(state["paper_eur"])
            btc = float(state["paper_btc"])
            base = (
                f"📊 Status\n"
                f"- Mode: {state['mode']} | Status: {state['status']} | Kill: {state['kill_switch']}\n"
                f"- PAPER: EUR {fmt_eur(eur)} | BTC {fmt_btc(btc)}\n"
            )

            if snap:
                equity = float(snap["equity_total"])
                w_btc = float(snap["w_btc"])
                pnl_day = float(snap["pnl_day"] or 0.0)
                base += (
                    f"- Equity: {fmt_eur(equity)} | BTC%: {fmt_pct(w_btc)} | P&L day: {fmt_eur(pnl_day)}\n"
                    f"- Bid/Ask: {float(snap['bid']):.2f} / {float(snap['ask']):.2f}\n"
                )

            if last_trade:
                base += (
                    "\n🧾 Ultima decisione\n"
                    f"- {last_trade['action']} | w {float(last_trade['w_current']):.2%} → {float(last_trade['w_target']):.2%}\n"
                    f"- r_hat: {fmt_pct(float(last_trade['r_hat'] or 0.0))} | cost: {fmt_pct(float(last_trade['cost_pct'] or 0.0))}\n"
                    f"- reason: {last_trade.get('reason')}"
                )

            await bot.send_message(chat_id=chat_id, text=base)
            continue

        if text.startswith("/set "):
            parts = text.split()
            if len(parts) != 3:
                await bot.send_message(chat_id=chat_id, text="Uso: /set <chiave> <valore>  (es: /set band 0.04)")
                continue

            key = parts[1].strip()
            val_raw = parts[2].strip()

            if key not in allowed_params:
                await bot.send_message(
                    chat_id=chat_id,
                    text="Chiave non valida. Usa /params per vedere le chiavi disponibili.",
                )
                continue

            try:
                if allowed_params[key] == "int":
                    val = int(parse_number(val_raw))
                else:
                    val = float(parse_number(val_raw))
            except Exception:
                await bot.send_message(chat_id=chat_id, text="Valore non valido. Esempi: 0.05, 1.5, 20")
                continue

            # controlli di sicurezza base (evita valori “assurdi”)
            if key in ("band", "w_max") and not (0 <= val <= 1):
                await bot.send_message(chat_id=chat_id, text="Valore fuori range (0..1).")
                continue
            if key in ("fee_taker", "slippage_base", "buffer") and not (0 <= val <= 0.05):
                await bot.send_message(chat_id=chat_id, text="Valore fuori range (0..0.05).")
                continue
            if key == "edge_mult" and not (1.0 <= val <= 5.0):
                await bot.send_message(chat_id=chat_id, text="edge_mult fuori range (1..5).")
                continue
            if key == "max_trade_eur" and val <= 0:
                await bot.send_message(chat_id=chat_id, text="max_trade_eur deve essere > 0.")
                continue

            db_update_param(key, val)
            await bot.send_message(chat_id=chat_id, text=f"✅ Aggiornato: {key} = {val}")
            continue

    if updates:
        state["telegram_offset"] = max_update_id + 1

    return state



async def main_async():
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    bot = Bot(token=tg_token)

    state = db_get_state()
    params = db_get_params()

    # Leggi comandi (sempre)
    state = await process_telegram_commands(bot, tg_chat, state)
    db_update_state(state)

    if state.get("kill_switch"):
        await bot.send_message(chat_id=tg_chat, text="🛑 Kill switch attivo. Bot fermo.")
        return

    # Finestra: primi 5 minuti dell'ora
    t = now_rome()

    force = os.environ.get("FORCE_RUN", "").strip() == "1"
    if not force:
        # Finestra normale: primi 5 minuti dell'ora
        if t.minute >= 5:
            return


    hk = hour_key(t)
    ignore_hourkey = os.environ.get("FORCE_IGNORE_HOURKEY", "").strip() == "1"

    if (not ignore_hourkey) and state.get("last_processed_hour") == hk:
        return


    if state.get("status") != "RUNNING":
        await bot.send_message(chat_id=tg_chat, text=f"⏸️ Bot in PAUSED. Nessuna operazione per l'ora {hk}.")
        return

    bid, ask, mid = fetch_bid_ask()
    candles = fetch_last_60_candles_1m()
    feat = compute_features(candles)
    r_hat = forecast_r_hat(feat)

    dk = day_key(t)
    eur = float(state["paper_eur"])
    btc = float(state["paper_btc"])
    equity = eur + btc * mid

    if state.get("day_key") != dk or state.get("day_start_equity") is None:
        state["day_key"] = dk
        state["day_start_equity"] = equity

    new_state, trade_row = paper_rebalance(state, params, bid, ask, mid, r_hat, feat)

    new_state["last_processed_hour"] = hk
    db_update_state(new_state)

    equity2 = float(trade_row["equity_after"])
    pnl_day = equity2 - float(new_state["day_start_equity"])

    db_upsert_snapshot(
        ts=t.astimezone(timezone.utc),
        mode=new_state["mode"],
        eur=float(new_state["paper_eur"]),
        btc=float(new_state["paper_btc"]),
        bid=bid, ask=ask, mid=mid,
        equity=equity2,
        w_btc=float(trade_row["w_after"]),
        pnl_day=pnl_day,
        drawdown=None
    )

    db_insert_trade(
        mode=new_state["mode"],
        action=trade_row["action"],
        w_current=float(trade_row["w_current"]),
        w_target=float(trade_row["w_target"]),
        notional=trade_row["notional_eur"],
        exec_price=trade_row["exec_price"],
        fee_eur=trade_row["fee_eur"],
        slippage_pct=float(trade_row["slippage_pct"]),
        spread_pct=float(trade_row["spread_pct"]),
        cost_pct=float(trade_row["cost_pct"]),
        r_hat=float(trade_row["r_hat"]),
        reason=trade_row["reason"]
    )

    # calcoli utili per spiegazione
    w_cur = float(trade_row["w_current"])
    w_tgt = float(trade_row["w_target"])
    delta_w = w_tgt - w_cur

    cost_pct = float(trade_row["cost_pct"])
    edge_mult = float(params["edge_mult"])
    required = cost_pct * edge_mult

    msg = (
        f"🕒 {hk} (Rome) – BTC/EUR – PAPER\n"
        f"Decisione: {trade_row['action']}\n"
        f"Allocazione BTC: {w_cur:.2%} → {w_tgt:.2%}  (Δ {delta_w:+.2%}, after {float(trade_row['w_after']):.2%})\n"
        f"Prezzo: bid {bid:.2f} | ask {ask:.2f} | spread {float(trade_row['spread_pct']):.2%}\n"
        f"Costi worst: slip {float(trade_row['slippage_pct']):.2%} + fee {float(params['fee_taker']):.2%} + buffer {float(params['buffer']):.2%}\n"
        f"Cost totale: {cost_pct:.2%}  | soglia richiesta (×{edge_mult:.2f}): {required:.2%}\n"
        f"Forecast 1h (r_hat): {r_hat:.2%}\n"
        f"Wallet: EUR {float(new_state['paper_eur']):.2f} | BTC {float(new_state['paper_btc']):.8f}\n"
        f"Equity: {equity2:.2f}€ | P&L oggi: {pnl_day:+.2f}€\n"
        f"Motivo: {trade_row['reason']}\n"
        f"Comandi: /status  /params  /set band 0.04  /pause  /resume  /kill"
    )
    await bot.send_message(chat_id=tg_chat, text=msg)



def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
