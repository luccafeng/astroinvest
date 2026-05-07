"""
Run the full pipeline and produce all output artifacts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import get_prices
from strategies.strategies import ALL_STRATEGIES
from backtest.backtester import run_backtest, CostModel, RiskOverlay
from backtest.metrics import compute_metrics
from utils.visualize import make_all


def main():
    prices = get_prices(use_real=False)
    print(f"Loaded {len(prices)} days  ({prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"BTC ann.vol: {prices['BTC'].pct_change().std()*np.sqrt(365):.1%}  "
          f"ETH ann.vol: {prices['ETH'].pct_change().std()*np.sqrt(365):.1%}  "
          f"corr: {prices.pct_change().corr().iloc[0,1]:.3f}")

    cost = CostModel()
    risk = RiskOverlay(target_vol=0.25, max_leverage=1.5, drawdown_kill_pct=0.40)

    metric_rows = []
    equity_curves = {}
    daily_returns = {}

    for strat in ALL_STRATEGIES:
        w = strat.generate_weights(prices)
        res = run_backtest(prices, w, cost=cost, risk=risk)
        m = compute_metrics(res.daily_ret, res.equity,
                            turnover=res.turnover, costs=res.costs)
        d = m.as_dict()
        d["Strategy"] = strat.name
        metric_rows.append(d)
        equity_curves[strat.name] = res.equity
        daily_returns[strat.name] = res.daily_ret

    metrics = pd.DataFrame(metric_rows).set_index("Strategy")
    equity_df = pd.DataFrame(equity_curves)
    daily_ret_df = pd.DataFrame(daily_returns)

    # Add BTC and ETH-only buy-hold for context
    btc_eq = (1 + prices["BTC"].pct_change().fillna(0)).cumprod() * 1_000_000
    eth_eq = (1 + prices["ETH"].pct_change().fillna(0)).cumprod() * 1_000_000
    equity_df["BTC only"] = btc_eq
    equity_df["ETH only"] = eth_eq
    daily_ret_df["BTC only"] = prices["BTC"].pct_change().fillna(0)
    daily_ret_df["ETH only"] = prices["ETH"].pct_change().fillna(0)

    print("\n=== FULL-SAMPLE METRICS ===\n")
    show_cols = ["cagr", "ann_vol", "sharpe", "sortino", "calmar",
                 "max_drawdown", "var_95", "hit_ratio", "turnover_ann"]
    print(metrics[show_cols].round(3).to_string())

    # Sortable summary
    summary = metrics[show_cols].sort_values("sharpe", ascending=False).round(3)
    summary.to_csv("./reports/summary_metrics.csv")
    print("\n=== RANKED BY SHARPE ===\n")
    print(summary.to_string())

    # Visualisations
    make_all(equity_df, daily_ret_df, prices, out_dir="./reports")

    # Save tables
    metrics.to_csv("./reports/all_metrics.csv")
    equity_df.to_csv("./reports/equity_curves.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
