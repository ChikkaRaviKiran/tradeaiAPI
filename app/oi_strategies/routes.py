"""OI-level defined-risk strategy builder and execution routes."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException

from app.api.routes import _find_broker_for_account_id, _list_all_active_brokers, _state
from app.core.instruments import get_instrument
from app.core.models import OptionsChainRow
from app.execution.broker_base import OrderRequest, OrderSide, OrderStatus, OrderType, ProductType

logger = logging.getLogger(__name__)

STRATEGIES = {
    "BULL_CALL_SPREAD": {
        "label": "Bull Call Spread", "belief": "Move UP toward resistance", "level": "resistance",
        "legs": [("BUY", "CE", "atm"), ("SELL", "CE", "resistance")], "kind": "debit",
        "explanation": "Use when NIFTY is expected to rise toward the OI resistance level. Loss is limited to the debit paid.",
    },
    "BEAR_PUT_SPREAD": {
        "label": "Bear Put Spread", "belief": "Move DOWN toward support", "level": "support",
        "legs": [("BUY", "PE", "atm"), ("SELL", "PE", "support")], "kind": "debit",
        "explanation": "Use when NIFTY is expected to fall toward the OI support level. Loss is limited to the debit paid.",
    },
    "BULL_PUT_SPREAD": {
        "label": "Bull Put Spread", "belief": "Support will HOLD", "level": "support",
        "legs": [("SELL", "PE", "support"), ("BUY", "PE", "protective_low")], "kind": "credit",
        "explanation": "Use when support is expected to hold. Profit is limited to the credit; the long put protects the downside.",
    },
    "BEAR_CALL_SPREAD": {
        "label": "Bear Call Spread", "belief": "Resistance will HOLD", "level": "resistance",
        "legs": [("SELL", "CE", "resistance"), ("BUY", "CE", "protective_high")], "kind": "credit",
        "explanation": "Use when resistance is expected to hold. Profit is limited to the credit; the long call protects the upside.",
    },
}


def _round_strike(value: float, interval: float, direction: str = "nearest") -> float:
    import math
    scaled = value / interval
    if direction == "up":
        return math.ceil(scaled) * interval
    if direction == "down":
        return math.floor(scaled) * interval
    return round(scaled) * interval


def _metrics(strategy: str, legs: list[dict], spot: float, lot_size: int, lots: int) -> dict:
    spec = STRATEGIES[strategy]
    width = abs(float(legs[1]["strike"]) - float(legs[0]["strike"]))
    net = sum((1 if l["side"] == "BUY" else -1) * float(l.get("premium") or 0) for l in legs)
    debit = max(net, 0.0)
    credit = max(-net, 0.0)
    if spec["kind"] == "debit":
        max_loss_unit, max_profit_unit = debit, max(0.0, width - debit)
    else:
        max_profit_unit, max_loss_unit = credit, max(0.0, width - credit)
    breakevens = []
    if strategy == "BULL_CALL_SPREAD": breakevens = [legs[0]["strike"] + debit]
    if strategy == "BEAR_PUT_SPREAD": breakevens = [legs[0]["strike"] - debit]
    if strategy == "BULL_PUT_SPREAD": breakevens = [legs[0]["strike"] - credit]
    if strategy == "BEAR_CALL_SPREAD": breakevens = [legs[0]["strike"] + credit]
    return {
        "width": width, "net_premium_per_unit": round(net, 2),
        "max_profit_per_lot": round(max_profit_unit * lot_size, 2),
        "max_loss_per_lot": round(max_loss_unit * lot_size, 2),
        "margin_required_per_lot": round(max_loss_unit * lot_size, 2),
        "max_profit": round(max_profit_unit * lot_size * lots, 2),
        "max_loss": round(max_loss_unit * lot_size * lots, 2),
        "margin_required": round(max_loss_unit * lot_size * lots, 2),
        "breakevens": [round(x, 2) for x in breakevens],
        "net_premium_total": round(net * lot_size * lots, 2),
        "pricing_note": "Estimated from live option premiums; broker margin is authoritative before execution.",
    }


def _payoff(legs: list[dict], price: float, lot_size: int, lots: int) -> float:
    total = 0.0
    for leg in legs:
        intrinsic = max(price - leg["strike"], 0) if leg["option_type"] == "CE" else max(leg["strike"] - price, 0)
        sign = 1 if leg["side"] == "BUY" else -1
        total += sign * (intrinsic - float(leg.get("premium") or 0))
    return round(total * lot_size * lots, 2)


def _build_preview(body: dict, chain: list[OptionsChainRow], spot: float, support: float, resistance: float, expiry: str, max_pain: float | None = None) -> dict:
    strategy = str(body.get("strategy") or "").upper()
    if strategy not in STRATEGIES: raise HTTPException(400, "Unknown OI strategy")
    instrument = get_instrument(str(body.get("symbol") or "NIFTY").upper())
    interval = float(instrument.strike_interval)
    lots = max(1, int(body.get("lots") or 1))
    width = max(interval, float(body.get("wing_width") or interval * 4))
    atm = _round_strike(spot, interval)
    strikes = {"atm": atm, "support": _round_strike(support, interval), "resistance": _round_strike(resistance, interval),
               "protective_low": _round_strike(support - width, interval, "down"), "protective_high": _round_strike(resistance + width, interval, "up")}
    rows = {round(float(r.strike_price), 2): r for r in chain}
    symbol_expiry = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d%b%y").upper() if "-" in expiry else expiry
    legs = []
    for side, option_type, key in STRATEGIES[strategy]["legs"]:
        strike = strikes[key]
        row = rows.get(round(strike, 2))
        get_value = (lambda key: row.get(key) if isinstance(row, dict) else getattr(row, key, None)) if row else (lambda key: None)
        premium = get_value("call_ltp" if option_type == "CE" else "put_ltp")
        if premium is None: raise HTTPException(503, f"No live premium for {strike} {option_type}")
        legs.append({"side": side, "option_type": option_type, "strike": strike, "premium": round(float(premium), 2), "security_id": get_value("call_security_id" if option_type == "CE" else "put_security_id"),
                     "quantity": lots * int(instrument.lot_size), "symbol": instrument.build_option_symbol(symbol_expiry, strike, option_type)})
    result = _metrics(strategy, legs, spot, int(instrument.lot_size), lots)
    scenario_prices = [support, float(max_pain or spot), resistance]
    result["scenario_payoffs"] = [{"label": label, "price": round(price, 2), "pnl": _payoff(legs, price, int(instrument.lot_size), lots)} for label, price in zip(["Support", "Max Pain", "Resistance"], scenario_prices)]
    return {"strategy": strategy, "strategy_info": STRATEGIES[strategy], "symbol": instrument.symbol, "spot": spot, "support": support, "resistance": resistance,
            "max_pain": scenario_prices[1], "expiry": expiry, "lot_size": instrument.lot_size, "lots": lots, "legs": legs, "metrics": result,
            "generated_at": datetime.now().isoformat()}


async def _levels(symbol: str) -> tuple[float, float, float, str, list[OptionsChainRow], float]:
    brokers = await _list_all_active_brokers()
    data_entry = next((x for x in brokers if x.get("is_data_feed") and x.get("broker_type") == "dhan"), None)
    if data_entry:
        broker = data_entry["broker"]
        dhan_client = getattr(broker, "client", None) or getattr(broker, "_client", None)
        if dhan_client is not None and hasattr(dhan_client, "get_dhan_option_chain"):
            try:
                response = await asyncio.to_thread(dhan_client.get_dhan_option_chain, symbol)
                data = response.get("data") or {}
                oc = data.get("oc") or {}
                chain = []
                for strike_text, contracts in oc.items():
                    strike = float(strike_text)
                    ce, pe = contracts.get("ce") or {}, contracts.get("pe") or {}
                    chain.append({"strike_price": strike, "call_ltp": ce.get("last_price"), "put_ltp": pe.get("last_price"),
                                  "call_oi": int(ce.get("oi") or 0), "put_oi": int(pe.get("oi") or 0),
                                  "call_security_id": ce.get("security_id"), "put_security_id": pe.get("security_id")})
                if chain:
                    support = max(chain, key=lambda x: x["put_oi"])["strike_price"]
                    resistance = max(chain, key=lambda x: x["call_oi"])["strike_price"]
                    pain_by_strike = {strike: sum(max(strike - row["strike_price"], 0) * row["put_oi"] + max(row["strike_price"] - strike, 0) * row["call_oi"] for row in chain) for strike in [x["strike_price"] for x in chain]}
                    max_pain = min(pain_by_strike, key=pain_by_strike.get)
                    return float(data.get("last_price") or 0), float(support), float(resistance), str(response.get("expiry") or ""), chain, float(max_pain)
            except Exception as exc:
                logger.exception("Dhan OI chain fetch failed: %s", exc)
    orch = _state.get("orchestrator")
    snap = _state.get("snapshots", {}).get(symbol) or _state.get("snapshot")
    if not orch or not snap: raise HTTPException(503, "Live market data is not available")
    metrics = snap.options_metrics
    support, resistance = metrics.put_oi_cluster, metrics.call_oi_cluster
    if not support or not resistance: raise HTTPException(503, "OI support/resistance is not available yet")
    expiry = getattr(orch, "_expiries", {}).get(symbol, "")
    chain = getattr(orch, "_last_option_chain", {}).get(symbol, [])
    if not expiry or not chain: raise HTTPException(503, "Option chain is not available yet")
    return float(snap.price or snap.nifty_price), float(support), float(resistance), expiry, chain, float(metrics.max_pain or spot)


def register_routes(app: FastAPI) -> None:
    @app.get("/api/oi-strategies/market")
    async def market(symbol: str = "NIFTY"):
        symbol = symbol.upper()
        spot, support, resistance, expiry, chain, max_pain = await _levels(symbol)
        snap = _state["snapshots"].get(symbol) or _state.get("snapshot")
        return {"symbol": symbol, "spot": spot, "support": support, "resistance": resistance,
                "max_pain": max_pain, "expiry": expiry,
                "lot_size": get_instrument(symbol).lot_size, "strike_interval": get_instrument(symbol).strike_interval,
                "chain_strikes": [r.strike_price for r in chain]}

    @app.post("/api/oi-strategies/preview")
    async def preview(body: dict):
        spot, support, resistance, expiry, chain, max_pain = await _levels(str(body.get("symbol") or "NIFTY").upper())
        return _build_preview(body, chain, spot, support, resistance, expiry, max_pain)

    @app.post("/api/oi-strategies/place")
    async def place(body: dict):
        account_id = int(body.get("account_id") or 0)
        entry = await _find_broker_for_account_id(account_id)
        if not entry: raise HTTPException(404, "Selected trade account is not active")
        spot, support, resistance, expiry, chain, max_pain = await _levels(str(body.get("symbol") or "NIFTY").upper())
        preview_data = _build_preview(body, chain, spot, support, resistance, expiry, max_pain)
        if body.get("confirm") is not True: raise HTTPException(400, "Explicit order confirmation is required")
        instrument = get_instrument(preview_data["symbol"])
        broker = entry["broker"]
        results = []
        for leg in preview_data["legs"]:
            token = ""
            client = getattr(broker, "client", None) or getattr(broker, "_client", None)
            if leg.get("security_id") and hasattr(client, "resolve_option_security_id"):
                token = str(leg["security_id"])
            elif hasattr(client, "_search_symbol"):
                info = await asyncio.to_thread(client._search_symbol, leg["symbol"])
                token = str((info or {}).get("symboltoken") or "")
                if not token: raise HTTPException(503, f"Broker token not found for {leg['symbol']}")
            req = OrderRequest(instrument=instrument, trading_symbol=leg["symbol"], symbol_token=token, exchange=instrument.option_exchange.value,
                side=OrderSide(leg["side"]), order_type=OrderType.MARKET, product_type=ProductType.CARRYFORWARD, quantity=leg["quantity"],
                underlying=instrument.symbol, expiry_date=datetime.strptime(expiry, "%Y-%m-%d") if "-" in expiry else datetime.strptime(expiry, "%d%b%y"), strike=leg["strike"], option_type=leg["option_type"], broker_security_id=token if entry["broker_type"] == "dhan" else None, broker_exchange_segment="NSE_FNO" if entry["broker_type"] == "dhan" else None, wait_for_terminal=True)
            response = await asyncio.to_thread(broker.place_order, req)
            results.append({"symbol": leg["symbol"], "side": leg["side"], "order_id": response.order_id, "status": response.status.value, "message": response.message})
            if response.status == OrderStatus.REJECTED: break
        return {"account_id": account_id, "account_name": entry["account_name"], "strategy": preview_data["strategy"], "results": results,
                "complete": len(results) == len(preview_data["legs"]) and all(r["status"] != "REJECTED" for r in results)}
