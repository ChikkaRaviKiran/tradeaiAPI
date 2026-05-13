"""Point-in-time feature snapshot computation.

CRITICAL: every feature uses ONLY data with timestamp <= ts. No look-ahead.
Reads from existing index_candles + option_candles tables (does not modify them).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
from sqlalchemy import select

from app.db.models import IndexCandle, OptionCandle

_IST = pytz.timezone("Asia/Kolkata")


@dataclass
class FeatureSnapshot:
    ts: datetime
    symbol: str
    spot: float
    vwap: Optional[float] = None
    vwap_dist_pct: Optional[float] = None
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema20_slope_5: Optional[float] = None
    atr_14: Optional[float] = None
    range_pct_15m: Optional[float] = None
    candle_body_pct: Optional[float] = None
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    pdc: Optional[float] = None
    dist_to_pdh_pct: Optional[float] = None
    dist_to_pdl_pct: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_open: Optional[float] = None
    gap_pct: Optional[float] = None
    gap_class: Optional[str] = None
    time_bucket: Optional[str] = None
    minute_of_day: Optional[int] = None
    dow: Optional[int] = None
    dte: Optional[int] = None
    regime: Optional[str] = None
    pcr: Optional[float] = None
    pcr_delta_30m: Optional[float] = None
    atm_ce_oi_change_pct: Optional[float] = None
    atm_pe_oi_change_pct: Optional[float] = None
    iv_atm: Optional[float] = None
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_broken: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Make JSON-serializable: convert datetimes to ISO strings
        if isinstance(d.get("ts"), datetime):
            d["ts"] = d["ts"].isoformat()
        return d


def _time_bucket(t: datetime) -> str:
    h, m = t.hour, t.minute
    if (h, m) < (10, 0):
        return "0915-1000"
    if (h, m) < (11, 0):
        return "1000-1100"
    if (h, m) < (12, 0):
        return "1100-1200"
    if (h, m) < (13, 0):
        return "1200-1300"
    if (h, m) < (14, 0):
        return "1300-1400"
    if (h, m) < (14, 30):
        return "1400-1430"
    return "1430-1530"


def _gap_class(gap_pct: float) -> str:
    if gap_pct is None:
        return "unknown"
    a = abs(gap_pct)
    if a < 0.1:
        return "flat"
    if gap_pct > 0:
        return "up_small" if a < 0.4 else "up_large"
    return "down_small" if a < 0.4 else "down_large"


def _regime_from_emas(ema9: float, ema20: float, ema50: float, atr_pct: float) -> str:
    if any(v is None for v in (ema9, ema20, ema50)):
        return "unknown"
    if atr_pct is not None and atr_pct > 0.6:
        return "volatile"
    if ema9 > ema20 > ema50:
        return "trend_up"
    if ema9 < ema20 < ema50:
        return "trend_down"
    return "range"


def _dte_to_weekly_expiry(ts: datetime) -> int:
    """Days to nearest Tue (NIFTY weekly). Crude approximation."""
    # NIFTY weekly expiry currently Tuesday. Adjust if SEBI changes.
    dow = ts.weekday()  # Mon=0, Tue=1
    days = (1 - dow) % 7
    return days


def _ema(series: pd.Series, span: int) -> Optional[float]:
    if len(series) < span:
        return None
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _vwap(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, 1)  # spot vol can be 0
    cum_vp = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    return float(cum_vp.iloc[-1] / cum_v.iloc[-1])


async def load_intraday_candles(
    session, symbol: str, target_date: str, until_ts: datetime
) -> pd.DataFrame:
    """Load 1-min candles for `target_date` up to and INCLUDING `until_ts`."""
    stmt = (
        select(IndexCandle)
        .where(
            IndexCandle.instrument == symbol,
            IndexCandle.date == target_date,
            IndexCandle.timestamp <= until_ts,
        )
        .order_by(IndexCandle.timestamp.asc())
    )
    with session.no_autoflush:
        rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume or 0,
            }
            for r in rows
        ]
    )
    df.set_index("timestamp", inplace=True)
    return df


async def load_prior_day_levels(
    session, symbol: str, target_date: str
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (PDH, PDL, PDC) from the most recent date strictly before target_date."""
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    cutoff = datetime.combine(target, datetime.min.time())
    stmt = (
        select(IndexCandle)
        .where(IndexCandle.instrument == symbol, IndexCandle.timestamp < cutoff)
        .order_by(IndexCandle.timestamp.desc())
        .limit(2000)  # plenty for last day
    )
    with session.no_autoflush:
        rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return None, None, None
    # Group by date, pick most recent
    last_date = rows[0].date
    same_day = [r for r in rows if r.date == last_date]
    if not same_day:
        return None, None, None
    pdh = max(r.high for r in same_day)
    pdl = min(r.low for r in same_day)
    pdc = sorted(same_day, key=lambda r: r.timestamp)[-1].close
    return pdh, pdl, pdc


async def compute_snapshot(
    session, symbol: str, ts: datetime
) -> Optional[FeatureSnapshot]:
    """Compute a point-in-time feature snapshot at `ts` for `symbol`.

    Returns None if there isn't enough data yet (e.g. before market open).
    """
    target_date = ts.strftime("%Y-%m-%d")
    df = await load_intraday_candles(session, symbol, target_date, ts)
    if df.empty:
        return None

    last = df.iloc[-1]
    spot = float(last["close"])

    # PDH/PDL/PDC
    pdh, pdl, pdc = await load_prior_day_levels(session, symbol, target_date)

    # Day open / gap
    day_open = float(df.iloc[0]["open"])
    gap_pct = ((day_open - pdc) / pdc * 100) if pdc else None

    # VWAP & EMAs (5-min resample)
    df5 = df.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    vwap = _vwap(df5)
    ema9 = _ema(df5["close"], 9)
    ema20 = _ema(df5["close"], 20)
    ema50 = _ema(df5["close"], 50)

    ema20_slope = None
    if len(df5) >= 6 and ema20 is not None:
        ema20_series = df5["close"].ewm(span=20, adjust=False).mean()
        ema20_slope = float(ema20_series.iloc[-1] - ema20_series.iloc[-6])

    atr_14 = _atr(df5)
    atr_pct = (atr_14 / spot * 100) if (atr_14 and spot) else None

    # Range last 15 min
    last_15 = df.last("15min") if len(df) > 0 else df
    range_pct_15m = None
    if len(last_15) > 0:
        rng = last_15["high"].max() - last_15["low"].min()
        range_pct_15m = float(rng / spot * 100) if spot else None

    candle_body_pct = None
    last5 = df5.iloc[-1] if len(df5) > 0 else None
    if last5 is not None:
        rng = float(last5["high"] - last5["low"])
        body = abs(float(last5["close"] - last5["open"]))
        candle_body_pct = (body / rng * 100) if rng > 0 else None

    # ORB (09:15 - 09:30)
    orb_df = df.between_time("09:15", "09:30")
    orb_high = float(orb_df["high"].max()) if not orb_df.empty else None
    orb_low = float(orb_df["low"].min()) if not orb_df.empty else None
    orb_broken = "none"
    if orb_high and orb_low and ts.time() > pd.Timestamp("09:30").time():
        if spot > orb_high:
            orb_broken = "up"
        elif spot < orb_low:
            orb_broken = "down"

    snap = FeatureSnapshot(
        ts=ts,
        symbol=symbol,
        spot=spot,
        vwap=vwap,
        vwap_dist_pct=((spot - vwap) / spot * 100) if vwap else None,
        ema9=ema9,
        ema20=ema20,
        ema50=ema50,
        ema20_slope_5=ema20_slope,
        atr_14=atr_14,
        range_pct_15m=range_pct_15m,
        candle_body_pct=candle_body_pct,
        pdh=pdh,
        pdl=pdl,
        pdc=pdc,
        dist_to_pdh_pct=((spot - pdh) / spot * 100) if pdh else None,
        dist_to_pdl_pct=((spot - pdl) / spot * 100) if pdl else None,
        day_high=float(df["high"].max()),
        day_low=float(df["low"].min()),
        day_open=day_open,
        gap_pct=gap_pct,
        gap_class=_gap_class(gap_pct) if gap_pct is not None else None,
        time_bucket=_time_bucket(ts),
        minute_of_day=ts.hour * 60 + ts.minute,
        dow=ts.weekday(),
        dte=_dte_to_weekly_expiry(ts),
        regime=_regime_from_emas(ema9, ema20, ema50, atr_pct or 0.0),
        orb_high=orb_high,
        orb_low=orb_low,
        orb_broken=orb_broken,
    )
    return snap
