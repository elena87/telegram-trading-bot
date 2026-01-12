import os
from supabase import create_client

_supabase = None

def sb():
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _supabase

def db_get_state() -> dict:
    res = sb().table("bot_state").select("*").eq("id", 1).single().execute()
    return res.data

def db_update_state(state: dict) -> None:
    state2 = dict(state)
    state2.pop("created_at", None)
    state2.pop("updated_at", None)
    sb().table("bot_state").update(state2).eq("id", 1).execute()

def db_get_params() -> dict:
    res = sb().table("bot_params").select("*").eq("id", 1).single().execute()
    return res.data

def db_log_command(user_id, command: str) -> None:
    sb().table("telegram_commands").insert({
        "user_id": user_id,
        "command": command,
        "payload": None
    }).execute()

def db_insert_trade(mode, action, w_current, w_target, notional, exec_price, fee_eur,
                    slippage_pct, spread_pct, cost_pct, r_hat, reason) -> None:
    sb().table("trades").insert({
        "mode": mode,
        "action": action,
        "w_current": w_current,
        "w_target": w_target,
        "notional_eur": notional,
        "exec_price": exec_price,
        "fee_eur": fee_eur,
        "slippage_pct": slippage_pct,
        "spread_pct": spread_pct,
        "cost_pct": cost_pct,
        "r_hat": r_hat,
        "reason": reason
    }).execute()

def db_upsert_snapshot(ts, mode, eur, btc, bid, ask, mid, equity, w_btc, pnl_day, drawdown) -> None:
    # ts deve essere unico (primary key)
    sb().table("equity_snapshots").upsert({
        "ts": ts.isoformat(),
        "mode": mode,
        "eur_balance": eur,
        "btc_balance": btc,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "equity_total": equity,
        "w_btc": w_btc,
        "pnl_day": pnl_day,
        "drawdown": drawdown
    }).execute()

def db_update_param(key: str, value):
    # aggiorna una sola colonna su bot_params (id=1)
    sb().table("bot_params").update({key: value}).eq("id", 1).execute()

def db_get_last_trade() -> dict | None:
    res = (
        sb()
        .table("trades")
        .select("*")
        .order("ts", desc=True)
        .limit(1)
        .execute()
    )
    data = res.data or []
    return data[0] if data else None

def db_get_last_snapshot() -> dict | None:
    res = (
        sb()
        .table("equity_snapshots")
        .select("*")
        .order("ts", desc=True)
        .limit(1)
        .execute()
    )
    data = res.data or []
    return data[0] if data else None


