# TradeAI — System Overview

**Platform:** NSE F&O (NIFTY, BANKNIFTY, FINNIFTY) | **Mode:** Index Options Buy-side  
**Stack:** Python / FastAPI / PostgreSQL / Docker (AWS EC2) | **AI:** GPT-4o-mini | **Broker:** AngelOne SmartAPI

---

## How It Works

```
Market Data → Indicators → Regime Detection → 8 Strategies → Score (0-100) → AI Validation → Risk Check → Trade
```

Every minute during market hours (09:15–14:30), the system fetches 1-min candles, computes technical indicators, detects the market regime, runs 8 book-proven strategies, scores any signals on 9 factors, validates via AI, and executes if risk rules pass.

---

## Data Sources

- **1-min spot candles** + **futures volume merge** (spot indices have vol=0)
- **5-min candles** for higher-timeframe trend bias (Elder Triple Screen)
- **Daily candles** for 20-day Donchian levels
- **Options chain** every 3 min (PCR, OI, max pain)
- **Pre-market:** Global markets, FII/DII flows, breadth, news sentiment

---

## Indicators

EMA (9/20/50/200), VWAP (volume-weighted via futures merge), RSI-14, MACD (12/26/9), ATR-14, Bollinger Bands (20/2), ADX-14, Trend Strength (0-3 EMA alignment score)

**Key authors:** Wilder (RSI, ATR, ADX), Elder (EMAs), Shannon (VWAP), Appel (MACD), Bollinger, Weinstein (EMA200)

---

## Market Regime Detection

| Regime | Condition | Favors |
|--------|-----------|--------|
| Trending | ADX > 22 | Momentum, Breakout, ORB |
| Range-Bound | ADX < 20, tight range | Range Breakout, VWAP Reclaim, Liquidity Sweep |
| High Volatility | ATR rising, VIX up | Breakout, ORB, Liquidity Sweep |
| Low Volatility | ATR falling | Pullback, Range |

Strategies only fire in compatible regimes.

---

## 8 Trading Strategies

| # | Strategy | Core Concept | Key Book |
|---|----------|-------------|----------|
| 1 | **ORB** | Breakout above/below 09:15-09:30 range | Crabel, Fisher |
| 2 | **VWAP Reclaim** | Price reclaims VWAP after 5+ candles below | Shannon |
| 3 | **Trend Pullback** | ADX > 20 + pullback to EMA in trend | Raschke & Connors (Holy Grail) |
| 4 | **Liquidity Sweep** | False break of swing high/low + reversal | ICT, Wyckoff (Spring/Upthrust) |
| 5 | **Range Breakout** | Breakout from tight range (ADX < 20) | Bollinger (squeeze → expansion) |
| 6 | **Momentum Breakout** | 20-candle high/low breakout + volume | Minervini (SEPA), O'Neil |
| 7 | **EMA Breakout** | Price crosses EMA50 with trend aligned | Elder, Cooper |
| 8 | **20-Day Donchian** | Daily 20-day high/low channel breakout | Donchian, Curtis Faith (Turtle) |

**Common filters across all strategies:** Volume confirmation (O'Neil: ≥1.5×), RSI sweet spot (Wilder), candle body strength (Nison), EMA alignment (Elder), EMA200 stage filter (Weinstein)

---

## 9-Factor Signal Scoring (Max 100)

| Factor | Max Pts | Source |
|--------|---------|--------|
| Strategy Trigger Quality | 22 | RSI, body strength, breakout margin, MACD |
| Volume Confirmation | 18 | O'Neil (1.5×) / Minervini (2×) |
| Historical Pattern | 16 | EMA stack, Dow Theory, ADX, Bollinger |
| VWAP Alignment | 13 | Shannon — price vs institutional fair value |
| Options OI Signal | 8 | McMillan — PCR, OI change |
| Global Market Bias | 8 | S&P, NASDAQ, VIX, DXY direction |
| FII/DII Flows | 6 | Institutional buying/selling |
| Market Breadth | 5 | Zweig — advance/decline ratio |
| News Sentiment | 4 | Tetlock — GPT-classified headlines |

**MIN_SCORE = 48** (adaptive — lowers to 35 floor when data sources are unavailable)

---

## 3-Path Trade Approval

| Path | Score Range | AI Required? | AI Confidence Needed |
|------|-----------|-------------|---------------------|
| **High Conviction** | ≥ 65 | No (bypass) | — |
| **Normal** | 48–64 | Yes | 60–65% |
| **AI Rescue** | 40–47 | Yes | 80% (high bar) |
| **Rejected** | < 40 | — | Discarded |

This OR logic prevents over-filtering while maintaining quality gates.

---

## Stop Loss & Targets

All based on **ATR** (Wilder), converted to option premium via ATM delta ≈ 0.5:

| Level | Formula | R:R | Book |
|-------|---------|-----|------|
| **Stop Loss** | Entry − 2×ATR (floor: 70% of premium) | — | Wilder |
| **Target 1** | Entry + 2×ATR | 1:1 | — |
| **Target 2** | Entry + 3.5×ATR | 1:1.75 | Van Tharp |
| **Trailing SL** | Activates at 50% toward T1, trails at 40% of profit | — | Elder |

---

## Risk Management (Van Tharp)

| Rule | Value |
|------|-------|
| Risk per trade | **1%** of capital |
| Max daily loss | **3%** of capital |
| Max concurrent positions | **3** |
| Max trades/day | **5** |
| Consecutive loss limit | **3** (stop trading) |

**Position sizing:** `lots = (capital × 1%) / (|entry − SL| × lot_size)`

---

## Daily Schedule

| Time | Action |
|------|--------|
| 08:45 | Startup, authenticate |
| 09:00 | Fetch global markets, FII/DII, breadth, news, Donchian levels |
| 09:15 | Market open — begin 1-min analysis loop |
| 09:15–09:30 | ORB window captured |
| Every 1 min | Full cycle: data → indicators → regime → strategies → score → AI → trade |
| 14:30 | Stop new entries |
| 15:20 | Close all open positions |
| 15:30 | Daily report + strategy evaluation |

---

## Book References

| Author | Book | Used For |
|--------|------|----------|
| Wilder | *New Concepts in Technical Trading* | RSI, ATR, ADX, 2×ATR stop |
| O'Neil | *How to Make Money in Stocks* | Volume ≥1.5× breakout rule |
| Minervini | *Trade Like a Stock Market Wizard* | SEPA, volume ≥2× ideal |
| Elder | *Trading for a Living* | Triple Screen, EMA cross, trailing SL |
| Van Tharp | *Trade Your Way to Financial Freedom* | 1% risk, position sizing |
| Donchian / Curtis Faith | Donchian Channel / *Way of the Turtle* | 20-day breakout |
| Raschke & Connors | *Street Smarts* | Holy Grail pullback pattern |
| Shannon | *Technical Analysis Using Multiple Timeframes* | VWAP, multi-timeframe |
| Nison | *Japanese Candlestick Charting* | Body strength, wick patterns |
| Bollinger | *Bollinger on Bollinger Bands* | Squeeze → breakout |
| Weinstein | *Secrets for Profiting* | EMA200 stage filter |
| Crabel / Fisher | ORB / ACD Method | Opening range breakout |
| McMillan | *Options as a Strategic Investment* | Put/Call ratio |
| Zweig | *Winning on Wall Street* | Market breadth |
| ICT / Wyckoff | Smart Money / Wyckoff Method | Liquidity sweeps, springs |
