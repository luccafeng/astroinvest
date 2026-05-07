"""
main.py
=======
End-to-end run: load data -> run every strategy -> compute metrics
-> walk-forward validation -> save report.

Usage:
    python main.py                  # demo data
    python main.py --real           # real BTC/ETH data via yfinance/Binance
    python main.py --start 2021-01-01 --end 2025-12-31
"""
from __future__ import annotations
import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd

# Make package imports work regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import get_prices
from strategies.strategies import ALL_STRATEGIES
from backtest.backtester import run_backtest, CostModel, RiskOverlay
from backtest.metrics import compute_metrics

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def run_all(prices: pd.DataFrame, label: str = "Full sample") -> pd.DataFrame:
    print(f"\n{'='*72}\n  {label}: {prices.index[0].date()} -> {prices.index[-1].date()}  "
          f"({len(prices)} days)\n{'='*72}")

    rows = []
    results = {}
    cost = CostModel()
    risk = RiskOverlay(target_vol=0.25, max_leverage=1.5, drawdown_kill_pct=0.40)

    for strat in ALL_STRATEGIES:
        try:
            w = strat.generate_weights(prices)
            res = run_backtest(prices, w, cost=cost, risk=risk)
            m = compute_metrics(res.daily_ret, res.equity,
                                turnover=res.turnover, costs=res.costs)
            d = m.as_dict()
            d["Strategy"] = strat.name
            rows.append(d)
            results[strat.name] = res
        except Exception as e:
            print(f"[ERROR] {strat.name}: {e}")

    df = pd.DataFrame(rows).set_index("Strategy")
    cols_order = [
        "cagr", "ann_vol", "sharpe", "sortino", "calmar",
        "max_drawdown", "dd_duration_days", "var_95", "cvar_95",
        "hit_ratio", "skew", "kurtosis", "turnover_ann",
        "total_costs_bps", "total_return",
    ]
    return df[cols_order], results


def walk_forward(prices: pd.DataFrame, train_years: int = 2, test_years: int = 1) -> pd.DataFrame:
    """
    Out-of-sample walk-forward: split into rolling (train, test) windows.
    Demonstrates that we are not curve-fitting parameters.
    Strategy params are *fixed* — this just shows OOS stability.
    """
    print(f"\n{'='*72}\n  WALK-FORWARD VALIDATION  ({train_years}y train -> {test_years}y test, rolling)\n{'='*72}")
    cost = CostModel()
    risk = RiskOverlay(target_vol=0.25, max_leverage=1.5, drawdown_kill_pct=0.40)

    start = prices.index[0]
    end = prices.index[-1]
    folds = []
    cur = start
    while True:
        train_end = cur + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append((cur, train_end, min(test_end, end)))
        cur = train_end  # non-overlapping test sets

    rows = []
    for fold_idx, (s, te, tee) in enumerate(folds, 1):
        sub = prices.loc[s:tee]
        oos = prices.loc[te:tee]
        for strat in ALL_STRATEGIES:
            w = strat.generate_weights(sub)
            res = run_backtest(sub, w, cost=cost, risk=risk)
            # Slice OOS portion
            oos_ret = res.daily_ret.loc[te:tee]
            oos_eq = (1 + oos_ret).cumprod()
            m = compute_metrics(oos_ret, oos_eq,
                                turnover=res.turnover.loc[te:tee],
                                costs=res.costs.loc[te:tee])
            rows.append({
                "fold": fold_idx,
                "test_window": f"{te.date()}->{tee.date()}",
                "strategy": strat.name,
                "oos_cagr": m.cagr,
                "oos_sharpe": m.sharpe,
                "oos_max_dd": m.max_drawdown,
            })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="Use real Yahoo/Binance data")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    prices = get_prices(use_real=args.real, start=args.start, end=args.end)

    print(f"\nLoaded {len(prices)} days  ({prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"BTC ann.vol: {prices['BTC'].pct_change().std()*np.sqrt(365):.1%}  "
          f"ETH ann.vol: {prices['ETH'].pct_change().std()*np.sqrt(365):.1%}  "
          f"BTC-ETH corr: {prices.pct_change().corr().iloc[0,1]:.3f}")

    full_metrics, results = run_all(prices, "FULL SAMPLE")
    print("\n--- FULL-SAMPLE PERFORMANCE ---\n")
    print(full_metrics.round(3).to_string())

    wf = walk_forward(prices, train_years=2, test_years=1)
    print("\n--- WALK-FORWARD (OOS only) ---\n")
    pivot = wf.pivot_table(index="strategy",
                           columns="test_window",
                           values="oos_sharpe").round(2)
    print(pivot.to_string())

    print("\n--- OOS Sharpe summary across folds ---\n")
    summary = wf.groupby("strategy").agg(
        mean_sharpe=("oos_sharpe", "mean"),
        median_sharpe=("oos_sharpe", "median"),
        worst_sharpe=("oos_sharpe", "min"),
        best_sharpe=("oos_sharpe", "max"),
        mean_oos_cagr=("oos_cagr", "mean"),
        worst_oos_dd=("oos_max_dd", "min"),
    ).round(3)
    print(summary.to_string())

    # Save outputs
    out_dir = "./reports"
    os.makedirs(out_dir, exist_ok=True)
    full_metrics.to_csv(f"{out_dir}/full_sample_metrics.csv")
    wf.to_csv(f"{out_dir}/walk_forward.csv", index=False)
    eq = pd.DataFrame({k: v.equity for k, v in results.items()})
    eq.to_csv(f"{out_dir}/equity_curves.csv")
    print(f"\nResults saved to {out_dir}/")
    return full_metrics, wf, results


if __name__ == "__main__":
    main()
