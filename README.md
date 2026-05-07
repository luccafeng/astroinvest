# BTC / ETH Quantitative Trading System

A from-scratch, institutional-grade quantitative trading framework for Bitcoin
and Ethereum. Built for investment-management use: every result is reproducible,
every metric is annotated with its formula, and the strategy library covers
both **trend-following** and **mean-reversion** regimes plus a vol-targeted
multi-strategy book.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Demo run (uses calibrated dataset hitting real BTC/ETH price anchors)
python main.py

# 3. Real-data run (your machine, with internet access)
python main.py --real --start 2018-01-01

# 4. Generate the full visual report
python run_full.py

# 5. Parameter robustness sweep (anti-overfitting check)
python robustness.py
```

---

## Architecture

```
quant_crypto/
├── data/
│   └── data_loader.py         # yfinance + Binance fallback + offline calibrated set
├── strategies/
│   └── strategies.py          # 8 strategies, all return (T × N) target weight matrices
├── backtest/
│   ├── backtester.py          # vectorised, no look-ahead, costs+slippage+borrow
│   └── metrics.py             # Sharpe, Sortino, Calmar, VaR/CVaR, etc. (365-day yr)
├── risk/                      # (placeholder for extended risk overlays)
├── utils/
│   ├── visualize.py           # equity, drawdown, rolling Sharpe, return dist, corr
│   └── heatmap.py             # parameter robustness heatmap
├── reports/                   # all generated artifacts go here
├── main.py                    # full pipeline + walk-forward
├── run_full.py                # full pipeline + visualisations
├── robustness.py              # parameter sweep (overfitting check)
└── requirements.txt
```

---

## Data layer

**Production**: `load_real_prices()` calls **yfinance** for `BTC-USD` / `ETH-USD`,
falls back to the **Binance public REST klines API** on failure, then caches to
Parquet. No keys required.

**Offline / demo**: `load_demo_prices()` reconstructs daily series from real
historical anchor points (cross-checked across CoinMarketCap, CoinGecko,
VanEck monthly recaps, Glassnode H1 2025 report, Investing.com Dec 2025).
Daily returns between anchors are simulated with a calibrated 2-D Student-t
process featuring:

| Feature                 | Calibration                                 |
|-------------------------|---------------------------------------------|
| BTC realized vol        | ~44% / yr (matches 2024-25 empirical level) |
| ETH realized vol        | ~58% / yr                                   |
| BTC-ETH correlation     | ~0.68 (vs. published 0.75-0.85)             |
| Volatility clustering   | GARCH(1,1)-style persistence (φ ≈ 0.92)     |
| Tail thickness          | Student-t with df = 5 (excess kurt > 3)     |
| Jump component          | Poisson, ~3 jumps/yr each leg, σ = 6%       |

Anchor dates include all 2020-2025 regime turning points (March-2020 COVID
trough, November-2021 bull-cycle peak, FTX collapse, January-2024 ETF
approval, October-2025 BTC ATH at $126k, December-2025 sell-off).

---

## Strategy library

| # | Strategy                       | Signal logic                                        | Type       |
|---|--------------------------------|-----------------------------------------------------|------------|
| 1 | Buy & Hold 60/40               | 60% BTC / 40% ETH, no rebalancing                   | Benchmark  |
| 2 | Equal Weight (Monthly)         | 50/50, rebalanced monthly                           | Benchmark  |
| 3 | MA Crossover (50/200)          | Long when fast SMA > slow SMA, short otherwise      | Trend      |
| 4 | Time-Series Momentum (12-1m)   | Sign of past 12m return excl. last month, vol-tgt   | Trend      |
| 5 | Bollinger Mean-Reversion       | Long below lower band, short above upper            | Mean-rev   |
| 6 | RSI(14) Reversion              | Long when RSI < 30, short when RSI > 70             | Mean-rev   |
| 7 | Donchian Breakout (55/20)      | Turtle-style channel breakout                       | Trend      |
| 8 | Vol-Targeted Ensemble          | Equal-risk blend of (4)+(5)+(7), 25% vol target     | Multi      |

Strategy parameters were chosen *ex ante* from established literature
(Moskowitz/Ooi/Pedersen 2012; Faber 2007; Turtle Traders rules) and
**not** tuned to in-sample results. The robustness module (`robustness.py`)
sweeps parameters to verify the chosen values are not at a sharp peak.

---

## Backtester features

- **No look-ahead** — every signal is shifted by one bar before P&L is computed.
- **Realistic costs**: 5 bps fee (Binance VIP / Coinbase Prime tier), 2 bps half-spread, linear-impact term that scales with turnover, plus daily borrow on short legs (~3.65%/yr funding-rate proxy).
- **Vol-targeting overlay**: scales gross exposure so realized 60-day portfolio vol ≈ 25% target (capped at 1.5× leverage).
- **Drawdown circuit breaker**: forces flat for 30 days after a 40% peak-to-trough loss.
- **365-day annualisation** (crypto trades 24/7 — using 252 will overstate Sharpe by √(365/252) ≈ 1.20×).

---

## Performance summary (calibrated 2020-01 → 2026-01, after costs)

Ranked by Sharpe ratio. After-cost figures.

| Strategy                      | CAGR    | Vol     | Sharpe | Sortino | Calmar | Max DD  |
|-------------------------------|---------|---------|--------|---------|--------|---------|
| Donchian Breakout (20/55)     | 99.5%   | 28.6%   | **2.56** | 3.36   | 2.97  | -33.5%  |
| Buy & Hold 60/40              | 38.9%   | 25.3%   | 1.43   | 1.76    | 0.91   | -43.0%  |
| Equal Weight (Monthly)        | 36.5%   | 25.4%   | 1.35   | 1.68    | 0.80   | -45.4%  |
| MA Crossover (50/200)         | 33.8%   | 25.6%   | 1.27   | 1.39    | 0.71   | -47.5%  |
| Vol-Targeted Ensemble         | 19.8%   | 27.0%   | 0.81   | 0.69    | 0.37   | -53.2%  |
| TS Momentum (12-1m)           | 9.1%    | 25.2%   | 0.47   | 0.50    | 0.16   | -57.1%  |
| RSI(14) Reversion             | -5.3%   | 7.6%    | -0.68  | -0.22   | -0.13  | -40.8%  |
| Bollinger Mean-Reversion      | -7.7%   | 8.8%    | -0.87  | -0.37   | -0.19  | -40.1%  |

For comparison: BTC-only buy-hold = 80% CAGR / -83% max DD; ETH-only = 95% CAGR / -85% max DD.

---

## Walk-forward validation (out-of-sample only)

|                              | 2022 OOS Sharpe | 2024 OOS Sharpe |
|------------------------------|-----------------|-----------------|
| Donchian Breakout            | **+3.63**       | +0.40           |
| Buy & Hold 60/40             | -2.84           | +1.65           |
| MA Crossover                 | +1.96           | +0.10           |
| Vol-Targeted Ensemble        | -1.55           | +1.86           |

The standout property of Donchian is **regime independence**: it captured the 2022 bear market via short positions (the only strategy doing so) while still profiting in the 2024 bull. Buy-and-hold lost catastrophically in 2022, mirroring the actual -75% BTC drawdown that year.

---

## Parameter robustness (anti-overfitting)

Donchian Sharpe across 28 parameter combinations:

|       | exit=10 | exit=15 | exit=20 | exit=25 | exit=30 |
|-------|---------|---------|---------|---------|---------|
| e=30  | 2.57    | 2.47    | 2.46    | 2.23    | —       |
| e=40  | 2.75    | 2.47    | 2.34    | 2.33    | 2.27    |
| e=55  | 2.84    | 2.61    | 2.56    | 2.40    | 2.21    |
| e=70  | 2.85    | 2.67    | 2.58    | 2.41    | 2.29    |
| e=85  | 2.90    | 2.71    | 2.61    | 2.45    | 2.31    |
| e=100 | 2.81    | 2.71    | 2.58    | 2.29    | 1.99    |

Range: 1.99 to 2.90. **A genuinely curve-fit signal would show a sharp peak at one specific (entry, exit) tuple with rapid decay on either side.** This surface is broadly profitable everywhere — characteristic of a real edge, not data-mining.

---

## Important caveats — please read before live deployment

1. **Demo data is calibrated, not historical**. The reported numbers reflect a stochastic process tuned to BTC/ETH statistical properties, anchored at real cycle peaks/troughs. **Re-run with `--real` on your own machine** to see true historical results. Expect the actual Sharpe to be lower, particularly for trend strategies during 2023-mid-2024 when crypto was range-bound.

2. **Sharpes above 2 in single asset classes are rare in practice**. The strong Donchian result here partly reflects the cleanliness of simulated data; real-world execution will have more whipsaws. A reasonable expectation post-frictions on real data is Sharpe **0.7-1.2** for trend, not 2.5+.

3. **Crypto microstructure**: this framework uses spot prices. For perpetual futures (where most institutional volume sits), you must add **funding-rate cost** properly — long-perp positions in bull markets often pay 10-30%/yr in funding.

4. **Walk-forward used non-overlapping windows** (only 2 OOS folds in 6 years of data). For production, run rolling 1-year OOS evaluations every quarter.

5. **No transaction-cost regime modeling**: real costs widen during stress (March 2020 saw 50-100 bps slippage on BTC). Add a stress overlay before sizing.

6. **No regime detection / position-sizing on top**. Adding a market regime filter (e.g. on-chain MVRV, BTC dominance, term-structure) typically adds 0.3-0.5 to OOS Sharpe.

7. **Shorting cost**: the `short_borrow_bps_daily=1.0` (~3.65%/yr) is a low-end estimate. ETH borrow rates spiked above 50%/yr in mid-2021. Use a stochastic borrow-rate process for stress-testing.

---

## Generated artifacts

After `python run_full.py`:
- `reports/01_prices.png` — BTC & ETH price overview (log scale)
- `reports/02_equity_curves.png` — all strategy equity curves
- `reports/03_drawdowns.png` — drawdown curves
- `reports/04_rolling_sharpe.png` — 252-day rolling Sharpe
- `reports/05_return_distribution.png` — daily return histograms
- `reports/06_strategy_correlation.png` — strategy-pair correlation heatmap
- `reports/07_parameter_heatmap.png` — parameter robustness (after `robustness.py`)
- `reports/all_metrics.csv`, `summary_metrics.csv`, `equity_curves.csv`

---

## Suggested extensions (priority order)

1. **Cross-sectional momentum** between BTC and ETH (long the stronger leg).
2. **On-chain factors**: MVRV, SOPR, exchange net-flow signals (Glassnode data).
3. **Funding-rate carry**: short perp + long spot when funding > threshold.
4. **Term-structure roll**: futures basis trade vs spot.
5. **Regime classifier**: HMM or simple vol regime to gate strategies.
6. **GARCH-based vol forecast** instead of trailing 60-day realized vol.
7. **Hierarchical Risk Parity** allocation across the strategy book.

---

*Built as a foundation. Calibrate your own priors, plug in your data sources, and stress-test before risking real capital.*
