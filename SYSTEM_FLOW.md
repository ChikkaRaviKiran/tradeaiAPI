# TradeAI — Complete System Flow

> **Version:** LOCKED v1.0  
> **Last Updated:** April 9, 2026  
> **Architecture:** Python FastAPI + PostgreSQL + AngelOne SmartAPI + React UI  
> **Server:** AWS Lightsail (35.154.9.116) | Docker at /opt/tradeai/

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Daily Lifecycle](#daily-lifecycle)
3. [Pre-Market Phase (8:00–9:15)](#pre-market-phase-800915)
4. [Market Loop (9:15–14:30)](#market-loop-915-1430)
5. [Signal Generation](#signal-generation)
6. [Scoring & Ranking](#scoring--ranking)
7. [Execution](#execution)
8. [Exit Management](#exit-management)
9. [Post-Market Phase (15:30+)](#post-market-phase-1530)
10. [5 Locked Strategies](#5-locked-strategies)
11. [3 Market Regimes](#3-market-regimes)
12. [Micro-Trigger System](#micro-trigger-system)
13. [AI Role (Log-Only)](#ai-role-log-only)
14. [Risk Controls](#risk-controls)
15. [Configuration Reference](#configuration-reference)
16. [Key Design Principles](#key-design-principles)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                      │
│  Dashboard │ Trades │ Signals │ Strategy Eval │ Activity Log  │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST API (15s polling)
┌──────────────────────────────▼──────────────────────────────┐
│                     BACKEND (FastAPI)                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Orchestrator  │──│ Signal Scorer│──│ Strategy Selector │  │
│  │ (60s loop)    │  │ (5 factors)  │  │ (multi-window)    │  │
│  └──────┬───────┘  └──────────────┘  └───────────────────┘  │
│         │                                                    │
│  ┌──────▼───────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 5 Strategies  │  │ Risk Manager │  │ Smart Exit Engine │  │
│  │ (regime-gated)│  │ (3 trades/d) │  │ (120 min max)     │  │
│  └──────┬───────┘  └──────────────┘  └───────────────────┘  │
│         │                                                    │
│  ┌──────▼───────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ AngelOne API  │  │ PostgreSQL   │  │ AI (GPT-4o-mini)│  │
│  │ (broker)      │  │ (trades, DB) │  │ (log-only)        │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Daily Lifecycle

```
08:00  StrategyEvaluator runs (90-day lookback, multi-window blend)
08:30  StrategySelector picks best strategies for today's conditions
08:45  AngelOne auth + instrument setup + WebSocket start
09:00  Pre-market AI analysis (FII/DII, news, breadth → log only)
09:15  ─── MARKET OPEN ───
09:15  Main loop starts (60s cycles)
09:30  ORB window closes → breakout signals eligible
10:00  Day classification (trend/range/volatile)
10:00  VWAP Reclaim eligible
11:30  ORB signals expire | No Trade Day check
14:30  ─── NO NEW ENTRIES ───
15:10  Close all remaining positions
15:30  ─── MARKET CLOSE ───
15:30  Post-market evaluation (updates DB for tomorrow)
15:45  Daily report → Telegram
16:00  Sleep until next pre-market
```

---

## Pre-Market Phase (8:00–9:15)

### Strategy Evaluation (8:00)

The `StrategyEvaluator` runs all 5 strategies against historical data:

- **Data source:** `index_candles` table (1-min candles from PostgreSQL)
- **Lookback:** 90 trading days
- **Method:** Walk-forward simulation — for each day, scan candles from bar 30, take first signal, simulate SL/T1/T2/EOD exit

**Multi-Window Blended Score:**

```
Final Score = 50% × (90-day composite)
            + 30% × (30-day composite)
            + 20% × (7-day composite)
```

This prevents overfitting while keeping adaptability.

**Composite Score Formula (per window):**

| Factor | Weight | Scale |
|--------|--------|-------|
| Win Rate | 30% | 0–100 (direct %) |
| Profit Factor | 25% | PF × 25, capped 100 |
| Sharpe Ratio | 20% | Sharpe × 25, capped 100 |
| Avg PnL | 15% | (avg_pnl / 50) + 50 |
| Signal Frequency | 10% | signals_per_day × 100 |

Penalty: `< 3 trades → ×0.5` | `< 5 trades → ×0.75`

### Strategy Selection (8:30)

The `StrategySelector` queries the `strategy_condition_performance` table:

1. Builds today's conditions: VIX bucket, gap bucket, day type
2. Cascading key match (most specific → least specific):
   - `gap_flat_vix_low_trend` → exact match
   - `gap_flat_vix_low_any` → relaxed day type
   - `gap_flat_any_any` → gap only
   - `any_any_any` → global fallback
3. Requires ≥ 3 historical trades and composite score ≥ 25
4. Selects up to 5 strategies, ranked by score
5. Falls back to all 5 locked strategies if no data

### Pre-Market AI Analysis (9:00)

- FII/DII data, market breadth, news sentiment
- Produces bias + risk advice
- **Used for logging and UI display only** — never affects trade decisions

---

## Market Loop (9:15–14:30)

Every **60 seconds**, per instrument:

```
┌─────────────────────────────────────────────┐
│              _run_analysis_cycle()            │
│                                              │
│  1. Fetch latest candles (WebSocket or API)  │
│  2. Compute indicators (RSI, ADX, EMA, etc.) │
│  3. Update options chain (dynamic refresh)    │
│     • Normal: every 3 min                     │
│     • Active signal formation: every 60s       │
│  4. Detect market regime                     │
│  5. Build market snapshot                    │
│  6. Classify day type (at 10:00)             │
│  7. Check exits on open trades               │
│  8. Gate checks (time, risk, SL, no-trade)   │
│  9. Run strategies → generate signals        │
│  10. Filter → Score → Rank                   │
│  11. Position sizing                         │
│  12. Pre-execution gates                     │
│  13. AI evaluation (log only)                │
│  14. Enter trade                             │
└─────────────────────────────────────────────┘
```

### Gate Checks (step 8)

| Gate | Rule | Action |
|------|------|--------|
| Time gate | After 14:30 | No new entries |
| Risk check | `can_trade()` fails | Skip cycle |
| SL circuit breaker | Any trade hit full SL today | No more trades for instrument |
| No Trade Day | Best score ≤ 55 by 11:30 + no trades placed | Skip rest of day |

---

## Signal Generation

### Market Structure

Computed per candle count change:
- Swing points (HH/HL for bullish, LH/LL for bearish)
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Thesis-break level stored per trade (breakout level ± 0.15% buffer)

### Micro-Trigger Detection

Fires when the **current forming candle** shows conviction:

| Trigger | Condition | Effect |
|---------|-----------|--------|
| Volume spike | Volume ≥ 1.3× average | Strategies lower volume requirements |
| Strong momentum | Candle body ≥ 70% of range | ORB/Range use high/low instead of close |
| VWAP reclaim | Price crosses VWAP intra-candle | VWAP strategy accepts intra-candle cross |

When active, strategies use relaxed thresholds:

| Strategy | Normal | With Micro-Trigger |
|----------|--------|--------------------|
| ORB | close > ORH, vol 1.5× | high > ORH, vol 1.3×, skip next-candle confirm |
| VWAP Reclaim | close > VWAP, vol 1.3× | high > VWAP, vol 1.1× |
| Range Breakout | close > range_high, vol 1.5× | high > range_high, vol 1.3× |
| Trend Pullback | pullback ≤ 0.40%, vol 1.2× | pullback ≤ 0.55%, vol 1.0× |
| Liquidity Sweep | wick ≥ 40%/50%, vol 1.4× | wick ≥ 35%/45%, vol 1.2× |

### Strategy Execution

Each strategy is filtered by regime compatibility, then `evaluate()` is called:

```python
for strategy in self.strategies:
    if not _strategy_compatible_with_regime(strategy, regime):
        continue
    signal = strategy.evaluate(df_today, options_metrics, spot_price,
                                daily_levels=daily_levels,
                                structure_data=structure)  # includes micro_trigger
```

### HTF (Higher Timeframe) Filter

Applied after signal generation:

1. **RSI extreme (≤30 or ≥70):** Hard direction filter — only CALL if RSI ≤ 30, only PUT if RSI ≥ 70
2. **Strong 5-min opposing trend (EMA gap > 0.5%):** Block opposing signals entirely
3. **Weak opposing trend:** Apply −5 score penalty

---

## Scoring & Ranking

### 5-Factor Scoring Model (max 100)

| Factor | Max Points | What It Measures |
|--------|-----------|------------------|
| Strategy Strength | 30 | Trigger quality, RSI position, candle body, MACD |
| Market Alignment | 25 | VWAP alignment, EMA trend direction, regime fit |
| Volume Confirmation | 20 | Futures volume, ATM option volume, candle range |
| Options OI Signal | 15 | Put-Call Ratio, OI change, OI buildup direction |
| Volatility Context | 10 | ATR ratio, VIX alignment with strategy |

### Ranking Rules

```
1. All signals scored → sorted descending
2. Top 2 signals taken (max)
3. Score < 45 → NO TRADE (logged as missed signal)
4. Score 45–50 → allowed, lowest priority (0.25% risk)
5. Best score ≤ 55 by 11:30 + no trades today → NO TRADE DAY
```

### Position Sizing (by score)

| Score Range | Risk % | Example (₹1,00,000 capital) |
|-------------|--------|----------------------------|
| ≥ 75 | 1.5% | ₹1,500 max loss per trade |
| 60–74 | 1.0% | ₹1,000 max loss per trade |
| 55–59 | 0.5% | ₹500 max loss per trade |
| 50–54 | 0.3% | ₹300 max loss per trade |
| 45–49 | 0.25% | ₹250 max loss per trade |
| < 45 | 0% | No trade |

---

## Execution

### Pre-Execution Gates

| Check | Rule |
|-------|------|
| Re-entry cooldown | Same instrument + direction blocked 15 min after SL hit |
| Duplicate strike | Same strike + type never traded twice in one day |
| Option quote | Must have valid LTP with acceptable bid-ask spread |
| Liquidity gate | Bid-ask spread must be < `max_spread_pct` |
| ATR availability | ATR must be > 0 for SL/target computation |

### SL/Target Computation (ATR-based)

```
option_ATR = spot_ATR × 0.5  (ATM delta ~0.5)

Stoploss  = entry − 2.0 × option_ATR  (floor: entry × 0.85, i.e. max 15% loss)
Target 1  = entry + 2.5 × option_ATR
Target 2  = entry + 4.0 × option_ATR
```

### AI Evaluation (Log-Only)

```
Signal → Score → Rank → EXECUTE
                          ↓
                    AI evaluates (async)
                          ↓
                    Logs "AGREES (78%)" or "CAUTION (42%)"
                          ↓
                    Can tighten SL/target (only if safer)
                          ↓
                    TRADE PROCEEDS REGARDLESS
```

AI receives enriched context:
- Market structure (HH/HL, BOS, CHoCH)
- Session state (trades today, PnL, consecutive losses)
- Candle character analysis
- Key price levels
- VIX, global indices
- DataFrame indicators (EMA slopes, ROC)

**AI never blocks trades.** If the API call fails, trade proceeds with "AI unavailable" logged.

### Trade Entry

```python
# Paper trading (default)
paper_trader.enter_trade(signal, entry_price, sl, t1, t2)

# Live trading
broker.place_order(nfo_symbol, qty, "BUY", "MARKET")
```

---

## Exit Management

### Smart Exit Priority (evaluated every cycle)

| Priority | Exit Type | Condition |
|----------|-----------|-----------|
| 1 | Hard stoploss | Price ≤ SL (never violated) |
| 2 | Catastrophic SL | LTP ≤ entry − 3× ATR (ignores candle close) |
| 3 | Thesis-break | Spot reverses through breakout level ± 0.15% buffer |
| 4 | Momentum fade | RSI > 78 (CALL) or < 22 (PUT) with partial profit |
| 5 | Time exit | Position held > 120 minutes (adjusted by day type) |
| 6 | Trailing SL | Triggers at 8% profit, trails at strategy-specific factor |
| 7 | T1 partial | +25% → book 50%, trail rest |

### Trail Factors (per strategy)

| Strategy | Trail Factor | Trail Trigger |
|----------|-------------|---------------|
| ORB | 0.65 | 8% profit (7.2% in TREND) |
| VWAP Reclaim | 0.45 | 8% profit |
| Trend Pullback | 0.30 | 8% profit (7.2% in TREND) |
| Range Breakout | 0.50 | 8% profit (5.6% in RANGE) |
| Liquidity Sweep | 0.40 | 8% profit |

### Day-Type Exit Adjustments

| Day Type | SL | Trail Factor | Hold Time | T1 | Breakeven |
|----------|----|--------------|-----------|----|----------|
| TREND | normal | capped ≤ 0.40 | × 1.3 (156 min) | normal | 7.2% |
| RANGE | normal | floored ≥ 0.60 | × 0.7 (84 min) | × 0.8 | 5.6% |
| VOLATILE | × 1.2 (wider) | capped ≤ 0.45 | normal | normal | 5.6% |

### Additional Exit Triggers

| Exit | Condition |
|------|-----------|
| Momentum fade | RSI > 78 (CALL) or < 22 (PUT) while profit ≥ 50% of breakeven trigger |
| Catastrophic SL | LTP ≤ entry − 3× ATR → immediate exit, ignores candle close |

---

## Post-Market Phase (15:30+)

1. **Close all positions** at 15:10 (pre-close)
2. **Strategy evaluation** re-runs → updates `strategy_condition_performance` DB
3. **Daily report** → trade summary, PnL, strategy performance → Telegram
4. **Missed signals** logged with spot price for review
5. **Option data collection** → saves 1-min option candles for backtesting
6. **Sleep** until next pre-market (8:00 next trading day)

---

## 5 Locked Strategies

### 1. ORB (Opening Range Breakout)

- **Window:** 09:15–09:30 (opening range), signals until 11:30
- **CALL:** Close > ORH + 0.05%, Volume > 1.5× avg, Price > VWAP, EMA9 > EMA20, RSI ≥ 55
- **PUT:** Close < ORL − 0.05%, Volume > 1.5× avg, Price < VWAP, EMA9 < EMA20, RSI ≤ 45
- **Invalidation:** Wick > 60% of candle range, next candle closes inside range
- **Regime:** TREND, VOLATILE

### 2. VWAP Reclaim

- **Window:** 10:00–15:00
- **CALL:** Below VWAP for ≥ 5 candles → closes above, RSI > 50, EMA9 crosses above EMA20
- **PUT:** Above VWAP for ≥ 5 candles → closes below, RSI < 50, EMA9 crosses below EMA20
- **Volume:** > 1.3× average
- **Regime:** TREND, RANGE

### 3. Trend Pullback

- **Condition:** ADX > 20 (trending market)
- **CALL:** EMA20 > EMA50, pullback within 0.40% of EMA, RSI 38–60, bullish candle, Price > EMA200
- **PUT:** EMA20 < EMA50, pullback within 0.40% of EMA, RSI 40–62, bearish candle, Price < EMA200
- **Volume:** > 1.2× average
- **Regime:** TREND only

### 4. Range Breakout

- **Condition:** ADX < 20 (ranging), price range < 0.80% for 30 candles
- **CALL:** Close > range high, Volume ≥ 1.5× avg, RSI ≥ 55, body ≥ 40% of range
- **PUT:** Close < range low, Volume ≥ 1.5× avg, RSI ≤ 45, body ≥ 40% of range
- **Regime:** RANGE only

### 5. Liquidity Sweep

- **CALL (Wyckoff Spring):** Low breaks swing low by ≥ 0.03%, bullish close, lower wick ≥ 40%, Volume ≥ 1.4× avg
- **PUT (Wyckoff Upthrust):** High breaks swing high by ≥ 0.03%, bearish close, upper wick ≥ 50%, Volume ≥ 1.4× avg
- **Lookback:** 15 candles for swing detection
- **Regime:** TREND, RANGE, VOLATILE

---

## 3 Market Regimes

| Regime | Detection | Strategies Allowed |
|--------|-----------|-------------------|
| **TREND** | ADX > 20 | ORB, VWAP Reclaim, Trend Pullback, Liquidity Sweep |
| **RANGE** | ADX < 18 | VWAP Reclaim, Range Breakout, Liquidity Sweep |
| **VOLATILE** | ADX 18–20 + VIX rising or large candle (range/ATR > 1.5) | ORB, Liquidity Sweep |

### Strategy-Regime Compatibility Matrix

| Strategy | TREND | RANGE | VOLATILE | INSUFFICIENT_DATA |
|----------|-------|-------|----------|-------------------|
| ORB | ✅ | ❌ | ✅ | ✅ |
| VWAP Reclaim | ✅ | ✅ | ❌ | ✅ |
| Trend Pullback | ✅ | ❌ | ❌ | ❌ |
| Range Breakout | ❌ | ✅ | ❌ | ❌ |
| Liquidity Sweep | ✅ | ✅ | ✅ | ✅ |

---

## Micro-Trigger System

Detects intra-candle momentum for faster entry **without waiting for candle close**.

### Detection (runs every cycle)

```python
def _detect_micro_trigger(df):
    # 1. Volume spike: current candle volume ≥ 1.3× avg
    # 2. Strong momentum: candle body ≥ 70% of range
    # 3. VWAP reclaim: price crosses VWAP this candle
    # Returns: {"active": True, "type": "volume_spike+momentum", ...}
```

### Effect on Strategies

When micro-trigger is active, it's injected into `structure_data["micro_trigger"]` and each strategy reads it to relax entry conditions — using `high`/`low` instead of `close` for breakout checks, lowering volume thresholds, skipping next-candle confirmation.

---

## AI Role (Log-Only)

### What AI Does

| Function | Description |
|----------|-------------|
| Pre-market summary | Market bias, news interpretation |
| Trade explanation | "VWAP reclaim with strong volume and OI support" |
| Confidence logging | "AGREES (78%)" or "CAUTION (42%)" in audit trail |
| SL/target tightening | Only if AI suggests tighter (safer) values |

### What AI Does NOT Do

| ❌ Forbidden | Reason |
|-------------|--------|
| Block trades | Non-deterministic, can't be backtested |
| Override signals | No statistical grounding |
| Set thresholds | Market doesn't respect arbitrary cutoffs |
| Approve/reject | Adds latency, misses entries |

### AI in Code

```python
# AI called for explanation — trade proceeds regardless
decision = await self.ai_engine.evaluate(signal, snap, score, ...)
# Log result (never blocks)
ai_label = "AGREES" if decision.confidence_score >= 60 else "CAUTION"
self._log_event("ai", f"AI {ai_label} ({ai_confidence:.0f}%): {reason}")
# Trade ALWAYS proceeds to entry
```

---

## Risk Controls

### Position Limits

| Rule | Value |
|------|-------|
| Max trades per day | 3 |
| Max concurrent positions | 1 |
| Max daily loss | 3% of capital |
| Consecutive loss stop | 3 losses → stop |
| Max hold time | 120 minutes |

### Entry Guards

| Guard | Rule |
|-------|------|
| Re-entry cooldown | 15 min after SL hit (same instrument + direction) |
| Duplicate strike | Never trade same strike + type twice in one day |
| Liquidity gate | Skip if bid-ask spread too wide |
| SL circuit breaker | If any trade hits full SL → no more trades today |
| No Trade Day | No score > 55 by 11:30 → skip rest of day |

### Exit Rules

| Type | Trigger |
|------|---------|
| Stoploss | Entry − 2.0× option ATR (floor 15% loss) |
| Target 1 | Entry + 2.5× option ATR → book 50% |
| Target 2 | Entry + 4.0× option ATR → exit all |
| Trailing SL | 8% profit → trail at 50% of gains |
| Time exit | 120 minutes max hold |
| Thesis break | Spot reverses through breakout level |

---

## Configuration Reference

### Core Settings (`config.py` / `.env`)

| Setting | Value | Description |
|---------|-------|-------------|
| `initial_capital` | ₹1,00,000 | Starting capital |
| `max_trades_per_day` | 3 | Hard limit |
| `max_concurrent_positions` | 1 | Sequential trades only |
| `risk_per_trade_pct` | 1.0% | Default (overridden by score) |
| `max_daily_loss_pct` | 3.0% | Daily loss circuit breaker |
| `consecutive_loss_limit` | 3 | Consecutive SL → stop |
| `paper_trading` | True | Paper mode (set False for live) |
| `auto_select_instruments` | True | Auto-pick instruments |
| `max_active_instruments` | 3 | Max simultaneous instruments |
| `v2_max_hold_minutes` | 120 | Max position hold time |
| `v2_openai_model` | gpt-4o-mini | AI model (log-only) |
| `v2_stoploss_pct` | 15 | SL as % of entry |
| `v2_quick_target_pct` | 25 | T1 as % of entry |

### Timing Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MARKET_OPEN` | 09:15 | NSE opening |
| `ORB_END` | 09:30 | Opening range window closes |
| `ORB_EXPIRY` | 11:30 | No more ORB signals |
| `NO_NEW_ENTRY_AFTER` | 14:30 | Stop new entries |
| `PRE_CLOSE` | 15:10 | Begin position closing |
| `MARKET_CLOSE` | 15:30 | NSE close |
| `REPORT_TIME` | 15:30 | Daily report trigger |
| `LOOP_INTERVAL` | 60s | Main cycle interval |

---

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Deterministic execution** | Score → Rank → Execute (no random elements) |
| **Probabilistic logic** | Multi-window blending, regime detection, scoring model |
| **AI as co-pilot** | Explains and logs, never decides |
| **Anti-overfitting** | 50% × 90d + 30% × 30d + 20% × 7d weighting |
| **Fast entry** | Micro-triggers bypass candle-close confirmation |
| **Capital protection** | 3 trades/day, 1 concurrent, 3% daily loss, 3 SL stop |
| **Adaptability** | Daily evaluation updates strategy selection per market conditions |
| **Auditability** | Every signal, score, AI opinion, and missed trade is logged |

### Mental Model

```
Your System = Pilot        (makes all decisions)
AI (GPT)    = Co-pilot     (talks, explains, but doesn't fly)
Scoring     = Flight plan  (deterministic route)
Risk Mgmt   = Autopilot    (enforces limits automatically)
```

---

## File Reference

| File | Purpose |
|------|---------|
| `app/engine/orchestrator.py` | Central controller — 60s loop, all phases |
| `app/engine/signal_scorer.py` | 5-factor scoring model (max 100) |
| `app/engine/regime_detector.py` | 3-regime detection (TREND/RANGE/VOLATILE) |
| `app/engine/strategy_selector.py` | Pre-market condition-based strategy selection |
| `app/engine/ai_decision.py` | GPT-4o-mini evaluation (log-only) |
| `app/backtest/strategy_evaluator.py` | 90-day walk-forward evaluation with multi-window blend |
| `app/backtest/scheduler.py` | Auto-runs evaluator daily |
| `app/trading/risk_manager.py` | Position limits and daily loss control |
| `app/trading/smart_exit.py` | Context-aware exit engine |
| `app/strategies/orb.py` | Opening Range Breakout |
| `app/strategies/vwap_reclaim.py` | VWAP Reclaim |
| `app/strategies/trend_pullback.py` | Trend Pullback |
| `app/strategies/range_breakout.py` | Range Breakout |
| `app/strategies/liquidity_sweep.py` | Liquidity Sweep |
| `app/core/config.py` | All settings and defaults |
| `app/core/models.py` | Data models (Signal, Trade, Score, etc.) |
| `app/api/routes.py` | REST API endpoints for frontend |
| `app/data/index_candle_collector.py` | Downloads 1-min index OHLCV candles → PostgreSQL |
| `app/data/option_data_collector.py` | Downloads 1-min option OHLCV candles → PostgreSQL |
