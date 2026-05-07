"""
visualize.py
============
Generate publication-quality charts:
  - Equity curves (log scale)
  - Drawdown chart
  - Rolling Sharpe (1y window)
  - Return distribution / QQ plot
  - Strategy correlation heatmap
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PALETTE = {
    "Buy & Hold 60/40":              "#9aa0a6",
    "Equal Weight (Monthly)":        "#bdbdbd",
    "MA Crossover (50/200)":         "#1f77b4",
    "TS Momentum (12-1m)":           "#2ca02c",
    "Bollinger Mean-Rev (20, 2σ)":   "#d62728",
    "RSI(14) Reversion":             "#9467bd",
    "Donchian Breakout (20/55)":     "#ff7f0e",
    "Vol-Targeted Ensemble":         "#17becf",
}


def plot_equity_curves(equity_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in equity_df.columns:
        ax.plot(equity_df.index, equity_df[col],
                label=col, color=PALETTE.get(col, None), linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("Equity ($, log scale, base = 1M)")
    ax.set_title("Strategy Equity Curves — BTC/ETH Quant Models", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_drawdowns(equity_df: pd.DataFrame, out_path: str):
    dd = equity_df / equity_df.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in dd.columns:
        ax.fill_between(dd.index, dd[col], 0,
                        alpha=0.15, color=PALETTE.get(col, None))
        ax.plot(dd.index, dd[col], color=PALETTE.get(col, None),
                linewidth=1.0, label=col)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Drawdown Curves", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_rolling_sharpe(daily_ret_df: pd.DataFrame, window: int, out_path: str):
    rs = daily_ret_df.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in rs.columns:
        ax.plot(rs.index, rs[col], color=PALETTE.get(col, None),
                linewidth=1.2, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel(f"Rolling {window}-day Sharpe")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_return_distributions(daily_ret_df: pd.DataFrame, out_path: str):
    cols = list(daily_ret_df.columns)
    n = len(cols)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        r = daily_ret_df[col].dropna()
        axes[i].hist(r, bins=80, color=PALETTE.get(col, "#888"),
                     alpha=0.7, edgecolor="white")
        axes[i].axvline(0, color="black", linewidth=0.6)
        axes[i].axvline(r.mean(), color="red", linewidth=1.0,
                        label=f"μ={r.mean()*1e4:.1f}bps")
        axes[i].set_title(col, fontsize=10)
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_strategy_corr(daily_ret_df: pd.DataFrame, out_path: str):
    corr = daily_ret_df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Strategy Daily-Return Correlation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_price_overview(prices: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(prices.index, prices["BTC"], color="#f7931a", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BTC (USD, log)")
    axes[0].set_title("BTC & ETH — Price (calibrated to real anchor points)", fontweight="bold")
    axes[1].plot(prices.index, prices["ETH"], color="#627eea", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ETH (USD, log)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def make_all(equity_df, daily_ret_df, prices, out_dir="./reports"):
    os.makedirs(out_dir, exist_ok=True)
    plot_price_overview(prices, f"{out_dir}/01_prices.png")
    plot_equity_curves(equity_df, f"{out_dir}/02_equity_curves.png")
    plot_drawdowns(equity_df, f"{out_dir}/03_drawdowns.png")
    plot_rolling_sharpe(daily_ret_df, 252, f"{out_dir}/04_rolling_sharpe.png")
    plot_return_distributions(daily_ret_df, f"{out_dir}/05_return_distribution.png")
    plot_strategy_corr(daily_ret_df, f"{out_dir}/06_strategy_correlation.png")
    print(f"Charts saved to {out_dir}/")
