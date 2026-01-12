import os
import math
import html
import asyncio
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from telegram import Bot
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.error import BadRequest



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

def h(x) -> str:
    # Escape per Telegram HTML (non serve quote=True qui)
    return html.escape(str(x), quote=False)



def build_main_menu() -> InlineKeyboardMarkup:
    kb = [
        [
            InlineKeyboardButton("📊 Status", callback_data="STATUS"),
            InlineKeyboardButton("⚙️ Params", callback_data="PARAMS"),
        ],
        [
            InlineKeyboardButton("⏸️ Pause", callback_data="PAUSE"),
            InlineKeyboardButton("▶️ Resume", callback_data="RESUME"),
        ],
        [
            InlineKeyboardButton("🧪 Preset", callback_data="PRESETS"),
            InlineKeyboardButton("🛑 Kill", callback_data="KILL"),
        ],
        [
          InlineKeyboardButton("🔄 Refresh", callback_data="REFRESH"),
          InlineKeyboardButton("🧹 Clear queue", callback_data="CLEARQ"),
        ],

    ]
    return InlineKeyboardMarkup(kb)

def build_presets_menu() -> InlineKeyboardMarkup:
    kb = [
        [
            InlineKeyboardButton("🧊 Conservativo", callback_data="PRESET_CONS"),
        ],
        [
            InlineKeyboardButton("⚖️ Bilanciato", callback_data="PRESET_BAL"),
        ],
        [
            InlineKeyboardButton("🔥 Aggressivo", callback_data="PRESET_AGGR"),
        ],
        [
            InlineKeyboardButton("⬅️ Indietro", callback_data="MENU"),
        ],
    ]
    return InlineKeyboardMarkup(kb)

def format_params(params: dict) -> str:
    return (
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
        "\nModifica: /set <chiave> <valore> (es: /set band 0.04)"
    )


async def process_telegram_commands(bot: Bot, chat_id: str, state: dict) -> dict:
    offset = int(state.get("telegram_offset") or 0)
    updates = await bot.get_updates(offset=offset, timeout=0)
    max_update_id = offset - 1

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

    async def send_menu(text: str | None = None):
        await bot.send_message(
            chat_id=chat_id,
            text=text or "📌 Menu",
            reply_markup=build_main_menu(),
            parse_mode="HTML"
        )

    async def reply_status():
        snap = db_get_last_snapshot()
        last_trade = db_get_last_trade()

        eur = float(state["paper_eur"])
        btc = float(state["paper_btc"])
        now = now_rome()
        last_hk = state.get("last_processed_hour")


        out = (
            "<b>📊 STATUS</b>\n"
            f"<b>Mode</b> <code>{state['mode']}</code>  | "
            f"<b>Status</b> <code>{state['status']}</code>  | "
            f"<b>Kill</b> <code>{state['kill_switch']}</code>\n"
            f"<b>Now</b> <code>{now.strftime('%Y-%m-%d %H:%M')}</code> (Rome)\n"
            f"<b>last_processed_hour</b> <code>{last_hk}</code>\n\n"
            "<b>💼 PAPER WALLET</b>\n"
            f"EUR <code>{fmt_eur(eur)}</code>  •  BTC <code>{fmt_btc(btc)}</code>\n"
        )

        if snap:
            equity = float(snap["equity_total"])
            w_btc = float(snap["w_btc"])
            pnl_day = float(snap["pnl_day"] or 0.0)
            out += (
                "\n<b>📈 MARKET / EQUITY</b>\n"
                f"Bid/Ask <code>{float(snap['bid']):.2f}</code> / <code>{float(snap['ask']):.2f}</code>\n"
                f"Equity <code>{fmt_eur(equity)}</code>  •  BTC% <code>{fmt_pct(w_btc)}</code>\n"
                f"P&L day <code>{fmt_eur(pnl_day)}</code>\n"
            )

        if last_trade:
            # opzionale: delta edge nello status
            r_hat_lt = float(last_trade.get("r_hat") or 0.0)
            cost_lt = float(last_trade.get("cost_pct") or 0.0)
            params_now = db_get_params()
            required_lt = cost_lt * float(params_now["edge_mult"])
            delta_edge_lt = abs(r_hat_lt) - required_lt

            out += (
                "\n<b>🧾 LAST DECISION</b>\n"
                f"<b>{last_trade['action']}</b>  "
                f"w <code>{float(last_trade['w_current'])*100:.2f}% → {float(last_trade['w_target'])*100:.2f}%</code>\n"
                f"r_hat <code>{r_hat_lt*100:+.2f}%</code>  •  "
                f"req <code>{required_lt*100:.2f}%</code>  •  "
                f"Δedge <code>{delta_edge_lt*100:+.2f}%</code>\n"
                f"<i>{h(last_trade.get('reason',''))}</i>\n"

            )

        out += "\n<i>Tip: /params • /set band 0.04 • /pause • /resume</i>"


        await bot.send_message(chat_id=chat_id, text=out, reply_markup=build_main_menu(), parse_mode="HTML")


    async def reply_params():
        params = db_get_params()

        out = (
            "<b>⚙️ PARAMS</b>\n"
            "<code>"
            f"w_max          {params['w_max']}\n"
            f"band           {params['band']}\n"
            f"fee_taker      {params['fee_taker']}\n"
            f"slippage_base  {params['slippage_base']}\n"
            f"buffer         {params['buffer']}\n"
            f"edge_mult      {params['edge_mult']}\n"
            f"max_trade_eur  {params['max_trade_eur']}\n"
            f"daily_stop     {params['daily_stop']}\n"
            f"max_trades_day {params['max_trades_day']}\n"
            "</code>\n"
            "Modifica: <code>/set band 0.04</code>\n"
            "Preset: usa il menu 🧪 Preset"
        )

        await bot.send_message(chat_id=chat_id, text=out, reply_markup=build_main_menu(), parse_mode="HTML")



    async def apply_preset(kind: str):
        # Preset = modifica di parametri “chiave” per vedere comportamenti diversi
        # (puoi cambiare numeri quando vuoi)
        if kind == "CONS":
            db_update_param("edge_mult", 2.0)
            db_update_param("band", 0.06)
            db_update_param("w_max", 0.30)
        elif kind == "BAL":
            db_update_param("edge_mult", 1.5)
            db_update_param("band", 0.05)
            db_update_param("w_max", 0.40)
        elif kind == "AGGR":
            db_update_param("edge_mult", 1.2)
            db_update_param("band", 0.03)
            db_update_param("w_max", 0.50)

        p = db_get_params()
        txt = f"✅ Preset applicato.\n\n<pre>{h(format_params(p))}</pre>"
        await bot.send_message(chat_id=chat_id, text=txt, reply_markup=build_main_menu(), parse_mode="HTML")


    for u in updates:
        if u.update_id > max_update_id:
            max_update_id = u.update_id

        # 1) Callback dei bottoni
        if u.callback_query:
            cq = u.callback_query
            if str(cq.message.chat_id) != str(chat_id):
                continue

            data = (cq.data or "").strip()
            try:
                await bot.answer_callback_query(cq.id)
            except BadRequest as e:
                # normale con bot "batch" (Actions): il callback può essere scaduto
                msg = str(e).lower()
                if "query is too old" in msg or "response timeout" in msg or "query id is invalid" in msg:
                    pass
                else:
                    raise


            if data in ("MENU", "REFRESH"):
                await send_menu("📌 Menu aggiornato")
            elif data == "STATUS":
                await reply_status()
            elif data == "PARAMS":
                await reply_params()
            elif data == "PAUSE":
                state["status"] = "PAUSED"
                await bot.send_message(chat_id=chat_id, text="⏸️ Ok, bot in PAUSED.", reply_markup=build_main_menu(), parse_mode="HTML")
            elif data == "RESUME":
                state["status"] = "RUNNING"
                await bot.send_message(chat_id=chat_id, text="▶️ Ok, bot in RUNNING.", reply_markup=build_main_menu(), parse_mode="HTML")
            elif data == "PRESETS":
                await bot.send_message(chat_id=chat_id, text="🧪 Scegli un preset:", reply_markup=build_presets_menu(), parse_mode="HTML")
            elif data == "PRESET_CONS":
                await apply_preset("CONS")
            elif data == "PRESET_BAL":
                await apply_preset("BAL")
            elif data == "PRESET_AGGR":
                await apply_preset("AGGR")
            elif data == "KILL":
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ Conferma KILL", callback_data="CONFIRM_KILL"),
                    InlineKeyboardButton("❌ Annulla", callback_data="MENU"),
                ]])
                await bot.send_message(chat_id=chat_id, text="🛑 Vuoi davvero fermare il bot?", reply_markup=kb, parse_mode="HTML")
            elif data == "CONFIRM_KILL":
                state["kill_switch"] = True
                await bot.send_message(chat_id=chat_id, text="🛑 Kill switch attivo. Il bot si fermerà.", reply_markup=build_main_menu(), parse_mode="HTML")
            elif data == "CLEARQ":
                # setta offset all'ultimo update_id + 1 (li stiamo già iterando, quindi basta aggiornare max_update_id)
                state["telegram_offset"] = max_update_id + 1
                await bot.send_message(chat_id=chat_id, text="🧹 Queue ripulita ✅", reply_markup=build_main_menu(), parse_mode="HTML")

            continue

        # 2) Messaggi testo
        msg = u.message
        if not msg or not msg.text:
            continue
        if str(msg.chat_id) != str(chat_id):
            continue

        text = msg.text.strip()
        db_log_command(msg.from_user.id if msg.from_user else None, text)

        if text in ("/start", "/menu"):
            await send_menu("👋 Ciao! Usa i bottoni qui sotto:")
            continue

        if text == "/pause":
            state["status"] = "PAUSED"
            await bot.send_message(chat_id=chat_id, text="⏸️ Ok, bot in PAUSED.", reply_markup=build_main_menu(), parse_mode="HTML")
            continue

        if text == "/resume":
            state["status"] = "RUNNING"
            await bot.send_message(chat_id=chat_id, text="▶️ Ok, bot in RUNNING.", reply_markup=build_main_menu(), parse_mode="HTML")
            continue

        if text == "/kill":
            state["kill_switch"] = True
            await bot.send_message(chat_id=chat_id, text="🛑 Kill switch attivo. Il bot si fermerà.", reply_markup=build_main_menu(), parse_mode="HTML")
            continue

        if text == "/status":
            await reply_status()
            continue

        if text == "/params":
            await reply_params()
            continue

        if text == "/clear":
            state["telegram_offset"] = max_update_id + 1
            await bot.send_message(chat_id=chat_id, text="🧹 Queue ripulita ✅", reply_markup=build_main_menu(), parse_mode="HTML")
            continue


        if text.startswith("/set "):
            parts = text.split()
            if len(parts) != 3:
                await bot.send_message(
                chat_id=chat_id,
                text="Uso: /set &lt;chiave&gt; &lt;valore&gt;  (es: /set band 0.04)",
                reply_markup=build_main_menu(),
                parse_mode="HTML"
            )
                continue

            key = parts[1].strip()
            val_raw = parts[2].strip()

            if key not in allowed_params:
                await bot.send_message(chat_id=chat_id, text="Chiave non valida. Usa /params.", reply_markup=build_main_menu(), parse_mode="HTML")
                continue

            try:
                if allowed_params[key] == "int":
                    val = int(parse_number(val_raw))
                else:
                    val = float(parse_number(val_raw))
            except Exception:
                await bot.send_message(chat_id=chat_id, text="Valore non valido. Esempi: 0.05, 1.5, 20", reply_markup=build_main_menu(), parse_mode="HTML")
                continue

            if key in ("band", "w_max") and not (0 <= val <= 1):
                await bot.send_message(chat_id=chat_id, text="Valore fuori range (0..1).", reply_markup=build_main_menu(), parse_mode="HTML")
                continue
            if key in ("fee_taker", "slippage_base", "buffer") and not (0 <= val <= 0.05):
                await bot.send_message(chat_id=chat_id, text="Valore fuori range (0..0.05).", reply_markup=build_main_menu(), parse_mode="HTML")
                continue
            if key == "edge_mult" and not (1.0 <= val <= 5.0):
                await bot.send_message(chat_id=chat_id, text="edge_mult fuori range (1..5).", reply_markup=build_main_menu(), parse_mode="HTML")
                continue
            if key == "max_trade_eur" and val <= 0:
                await bot.send_message(chat_id=chat_id, text="max_trade_eur deve essere > 0.", reply_markup=build_main_menu(), parse_mode="HTML")
                continue

            db_update_param(key, val)
            await bot.send_message(chat_id=chat_id, text=f"✅ Aggiornato: {key} = {val}", reply_markup=build_main_menu(), parse_mode="HTML")
            continue


    if updates:
        state["telegram_offset"] = max_update_id + 1

    return state




async def main_async():
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]
    run_mode = os.environ.get("RUN_MODE", "trade").strip().lower()

    bot = Bot(token=tg_token)

    state = db_get_state()

    # 1) commands mode: consuma update Telegram e risponde
    if run_mode == "commands":
        state2 = await process_telegram_commands(bot, tg_chat, state)
        if state2 is None:
            state2 = state
        state = state2
        db_update_state(state)


        if state.get("kill_switch"):
            # opzionale: conferma visiva
            await bot.send_message(chat_id=tg_chat, text="🛑 Kill switch attivo. Bot fermo.", reply_markup=build_main_menu(), parse_mode="HTML")
        return

    # 2) trade mode: NON consuma update Telegram (li lascia al workflow commands)
    params = db_get_params()

    if state.get("kill_switch"):
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
        await bot.send_message(chat_id=tg_chat, text=f"⏸️ Bot in PAUSED. Nessuna operazione per l'ora {hk}.", parse_mode="HTML")
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

    run_time = t.strftime("%H:%M")

    action = trade_row["action"]
    action_emoji = "🟨" if action == "HOLD" else ("🟩" if action == "BUY" else "🟥")

    eur_after = float(new_state["paper_eur"])
    btc_after = float(new_state["paper_btc"])

    # ---- Trade line (label diversa BUY/SELL) ----
    notional = trade_row.get("notional_eur")  # None se HOLD
    exec_price = trade_row.get("exec_price")
    fee_eur = trade_row.get("fee_eur")

    if action == "HOLD" or notional is None or exec_price is None:
        trade_line = f"<b>Trade</b>  <code>—</code>"
    else:
        label = "speso" if action == "BUY" else "incasso lordo"
        trade_line = (
            f"<b>Trade</b>  <b>{action_emoji} {action}</b>  "
            f"{label} <code>{float(notional):.2f}€</code>  "
            f"@ <code>{float(exec_price):.2f}</code>"
        )
        if fee_eur is not None:
            trade_line += f"  fee <code>{float(fee_eur):.2f}€</code>"

    # ---- Gate diagnostics ----
    band = float(params["band"])
    min_trade_ok = True
    # nel tuo paper_rebalance: se trade_value < 2.0 scatta min trade gate
    # qui lo deduciamo: se HOLD e reason min trade gate => non ok
    reason_raw = (trade_row.get("reason") or "").strip().lower()
    if "min trade gate" in reason_raw:
        min_trade_ok = False

    band_ok = abs(delta_w) >= band
    cost_ok = abs(r_hat) > required

    band_mark = "✅" if band_ok else "❌"
    cost_mark = "✅" if cost_ok else "❌"
    min_mark = "✅" if min_trade_ok else "❌"

    # Edge info
    delta_edge = abs(r_hat) - required  # positivo = edge sopra soglia

    # ---- Reason “umana” ----
    if "cost gate" in reason_raw:
        reason = f"Edge insufficiente: <code>{abs(r_hat)*100:.2f}%</code> &lt; <code>{required*100:.2f}%</code>"
    elif "band gate" in reason_raw:
        reason = f"Δw troppo piccolo: <code>{abs(delta_w)*100:.2f}%</code> &lt; band <code>{band*100:.2f}%</code>"
    elif "min trade gate" in reason_raw:
        reason = "Importo trade troppo piccolo (min trade)"
    elif "insufficient" in reason_raw:
        reason = "Fondi insufficienti"
    else:
        reason = h(trade_row.get("reason", ""))


    msg = (
        f"<b>{action_emoji} BTC/EUR • PAPER</b>\n"
        f"<b>Ora (Rome):</b> <code>{hk}</code>  •  <b>run</b> <code>{run_time}</code>\n\n"

        f"<b>Decisione</b>  <b>{action}</b>\n"
        f"{trade_line}\n\n"

        f"<b>Gates</b>  band {band_mark}  cost {cost_mark}  min {min_mark}\n"
        f"<b>Edge</b>  richiesto <code>{required*100:.2f}%</code> • attuale <code>{abs(r_hat)*100:.2f}%</code> • Δedge <code>{delta_edge*100:+.2f}%</code>\n\n"

        f"<b>Allocazione BTC</b>  <code>{w_cur*100:.2f}% → {w_tgt*100:.2f}%</code>"
        f"  (Δ <code>{delta_w*100:+.2f}%</code>, after <code>{float(trade_row['w_after'])*100:.2f}%</code>)\n"

        f"<b>Prezzo</b>  bid <code>{bid:.2f}</code> • ask <code>{ask:.2f}</code> • spread <code>{float(trade_row['spread_pct'])*100:.2f}%</code>\n"
        f"<b>Forecast 1h</b>  r_hat <code>{r_hat*100:+.2f}%</code>\n\n"

        f"<b>Costi (worst)</b>  "
        f"fee <code>{float(params['fee_taker'])*100:.2f}%</code> + "
        f"slip <code>{float(trade_row['slippage_pct'])*100:.2f}%</code> + "
        f"buffer <code>{float(params['buffer'])*100:.2f}%</code>\n"
        f"<b>Cost totale</b> <code>{cost_pct*100:.2f}%</code>  | (×{edge_mult:.2f})\n\n"

        f"<b>Wallet</b>  EUR <code>{eur_after:.2f}</code> • BTC <code>{btc_after:.8f}</code>\n"
        f"<b>Equity</b> <code>{equity2:.2f}€</code>  | <b>P&L day</b> <code>{pnl_day:+.2f}€</code>\n\n"

        f"<b>Motivo</b>  {reason}\n"
        f"<i>Menu: /menu • Status: /status • Params: /params</i>"
    )

    await bot.send_message(chat_id=tg_chat, text=msg, parse_mode="HTML")






def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
