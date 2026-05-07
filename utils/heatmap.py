"""Generate parameter robustness heatmap."""
import sys, os
# Add project root (parent of this file's parent dir) to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    out_sh = pd.DataFrame(index=entries, columns=exits, dtype=float)
    out_dd = pd.DataFrame(index=entries, columns=exits, dtype=float)
    for e in entries:
        for x in exits:
            if x >= e: continue
            w = DonchianBreakout(entry=e, exit_w=x).generate_weights(prices)
            res = run_backtest(prices, w, cost=cost, risk=risk)
            m = compute_metrics(res.daily_ret, res.equity, turnover=res.turnover, costs=res.costs)
            out_sh.loc[e, x] = m.sharpe
            out_dd.loc[e, x] = m.max_drawdown
    return out_sh, out_dd


def sweep_ma():
    fasts = [20, 30, 50, 75, 100]
    slows = [100, 150, 200, 250, 300]
    out = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    for f in fasts:
        for s in slows:
            if f >= s: continue
            w = MovingAvgCrossover(fast=f, slow=s).generate_weights(prices)
            res = run_backtest(prices, w, cost=cost, risk=risk)
            m = compute_metrics(res.daily_ret, res.equity, turnover=res.turnover, costs=res.costs)
            out.loc[f, s] = m.sharpe
    return out


def heatmap(ax, df, title, fmt=".2f", cmap="RdYlGn", vmin=None, vmax=None):
    arr = df.values.astype(float)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="grey")
            else:
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    return im


sh, dd = sweep_donchian()
ma_sh = sweep_ma()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
heatmap(axes[0], sh, "Donchian Sharpe (entry × exit)", vmin=1.5, vmax=3.0)
axes[0].set_xlabel("Exit lookback (days)"); axes[0].set_ylabel("Entry lookback (days)")
heatmap(axes[1], dd, "Donchian Max DD", fmt=".0%", cmap="RdYlGn", vmin=-0.6, vmax=-0.2)
axes[1].set_xlabel("Exit lookback (days)"); axes[1].set_ylabel("Entry lookback (days)")
heatmap(axes[2], ma_sh, "MA Crossover Sharpe (fast × slow)", vmin=0.5, vmax=2.5)
axes[2].set_xlabel("Slow MA"); axes[2].set_ylabel("Fast MA")

plt.suptitle("Parameter Robustness — broad neighbourhood stability is evidence against overfitting",
             fontweight="bold", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("./reports/07_parameter_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved 07_parameter_heatmap.png")
