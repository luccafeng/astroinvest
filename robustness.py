"""
robustness.py
=============
Parameter-sweep robustness check on the top-performing strategy (Donchian).
A genuinely robust signal should produce reasonably stable Sharpe across a
*neighbourhood* of parameter values — not a sharp peak at one specific tuple
(which would be the canonical signature of curve-fitting).

We sweep entry ∈ {30, 40, 55, 70, 85, 100} and exit ∈ {10, 15, 20, 25, 30}
and report the Sharpe surface.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import get_prices
from strategies.strategies import DonchianBreakout, MovingAvgCrossover
from backtest.backtester import run_backtest, CostModel, RiskOverlay
from backtest.metrics import compute_metrics

prices = get_prices(use_real=False)
cost = CostModel()
risk = RiskOverlay(target_vol=0.25, max_leverage=1.5, drawdown_kill_pct=0.40)


def sweep_donchian():
    entries = [30, 40, 55, 70, 85, 100]
    exits = [10, 15, 20, 25, 30]
    out = pd.DataFrame(index=entries, columns=exits, dtype=float)
    out_dd = pd.DataFrame(index=entries, columns=exits, dtype=float)
    out_cagr = pd.DataFrame(index=entries, columns=exits, dtype=float)
    for e in entries:
        for x in exits:
            if x >= e:
                continue
            strat = DonchianBreakout(entry=e, exit_w=x)
            w = strat.generate_weights(prices)
            res = run_backtest(prices, w, cost=cost, risk=risk)
            m = compute_metrics(res.daily_ret, res.equity,
                                turnover=res.turnover, costs=res.costs)
            out.loc[e, x] = round(m.sharpe, 2)
            out_dd.loc[e, x] = round(m.max_drawdown, 2)
            out_cagr.loc[e, x] = round(m.cagr, 2)
    return out, out_cagr, out_dd


def sweep_ma():
    fasts = [20, 30, 50, 75, 100]
    slows = [100, 150, 200, 250, 300]
    out = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            strat = MovingAvgCrossover(fast=f, slow=s)
            w = strat.generate_weights(prices)
            res = run_backtest(prices, w, cost=cost, risk=risk)
            m = compute_metrics(res.daily_ret, res.equity,
                                turnover=res.turnover, costs=res.costs)
            out.loc[f, s] = round(m.sharpe, 2)
    return out


if __name__ == "__main__":
    print("\n=== DONCHIAN BREAKOUT — Sharpe surface ===")
    print("(rows = entry lookback, cols = exit lookback)\n")
    sharpe, cagr, dd = sweep_donchian()
    print("Sharpe:")
    print(sharpe.fillna("--").to_string())
    print("\nCAGR:")
    print(cagr.fillna("--").to_string())
    print("\nMax DD:")
    print(dd.fillna("--").to_string())

    print("\n\n=== MA CROSSOVER — Sharpe surface ===")
    print("(rows = fast MA, cols = slow MA)\n")
    out = sweep_ma()
    print(out.fillna("--").to_string())
