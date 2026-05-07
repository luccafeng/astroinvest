"""
Test funding rate carry with REALISTIC simulated funding data.

Real OKX historical funding rate stats (BTC-USDT-SWAP, 2020-2025):
  - Mean per 8h interval: +0.011% (median +0.010%)
  - That's ~+12% annualised average funding paid by longs to shorts
  - Std per interval: ~0.020%  (annualised ~22% vol of rolling mean)
  - Right-skewed: episodic spikes to +0.10%/8h (~+109%/yr) in bull blow-offs
  - Negative episodes: -0.05%/8h (~-55%/yr) during forced unwinds (May 2021,
    June 2022, August 2024)

We simulate funding using an OU process with these moments, plus jump components
keyed to BTC return regimes (high funding follows strong price runs).
"""
import sys, os
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import load_demo_prices
from strategies.perp_strategies import FundingRateCarry, backtest_perp_carry
from backtest.metrics import compute_metrics


def simulate_okx_funding(prices: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Simulate realistic 3x-daily aggregated funding rates for BTC and ETH perp,
    correlated with momentum in spot price (high recent return -> elevated funding).
    """
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=prices.index, columns=["BTC_funding", "ETH_funding"], dtype=float)

    # Empirical means per 8h interval (annualised):
    #   BTC: +0.011%/interval = ~+12%/yr
    #   ETH: +0.013%/interval = ~+14%/yr
    # We sum 3 intervals to get daily funding.
    base_per_interval = {"BTC": 0.00011, "ETH": 0.00013}
    base_std = {"BTC": 0.00020, "ETH": 0.00025}

    for asset in ["BTC", "ETH"]:
        rets = prices[asset].pct_change()
        # 30-day momentum scaled to z-score
        mom = rets.rolling(30).mean() / rets.rolling(30).std()
        mom = mom.fillna(0).clip(-3, 3)
        # OU mean-reverting noise
        noise = np.zeros(len(prices))
        kappa = 0.15
        for i in range(1, len(prices)):
            noise[i] = (1 - kappa) * noise[i-1] + rng.normal(0, base_std[asset])
        # Jump component (rare episodic spikes)
        jumps = rng.binomial(1, 0.005, size=len(prices)) * \
                rng.normal(0, base_std[asset] * 6, size=len(prices))
        # Daily total = 3 intervals * (base + momentum_premium + noise + jumps)
        per_interval = base_per_interval[asset] + 0.0001 * mom.values + noise + jumps
        out[f"{asset}_funding"] = per_interval * 3   # daily sum

    return out


def main():
    prices = load_demo_prices()
    funding = simulate_okx_funding(prices)

    print("=" * 70)
    print("  OKX FUNDING RATE — calibrated simulation summary")
    print("=" * 70)
    for col in funding.columns:
        ann = funding[col].mean() * 365
        print(f"  {col}: avg daily funding = {funding[col].mean()*1e4:.2f} bp/day  "
              f"=> annualised {ann:.2%}")

    # Run carry strategy
    # Trigger at 0.00005 per-8h = 5bp/8h = ~5.5%/yr - profitable threshold after costs
    strat = FundingRateCarry(entry_trigger=0.00005,
                             exit_trigger=0.00001,
                             leg_weight=0.45)
    res = backtest_perp_carry(prices, funding, strategy=strat)

    m = compute_metrics(res["daily_ret"], res["equity"],
                        turnover=res["turnover"], costs=res["costs"])
    print("\n" + "=" * 70)
    print("  FUNDING RATE CARRY BACKTEST  (OKX-calibrated)")
    print("=" * 70)
    print(m.pretty())

    # Time in market
    in_pos = (res["signals"].abs().sum(axis=1) > 0)
    print(f"\n  Time in carry trade : {in_pos.sum() / len(in_pos):.1%}")
    print(f"  Net position: market-neutral (long spot + short perp)")
    print(f"  Final capital: ${res['equity'].iloc[-1]:,.0f} on $1,000,000 start")

    # Save artifacts
    funding.to_csv("./reports/okx_funding_simulated.csv")
    res["equity"].to_csv("./reports/carry_equity.csv")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(prices.index, prices["BTC"], color="#f7931a", label="BTC", linewidth=1)
    axes[0].plot(prices.index, prices["ETH"], color="#627eea", label="ETH (right)", linewidth=1)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Price (log)")
    axes[0].legend(); axes[0].set_title("BTC & ETH spot — 2020-2026", fontweight="bold")

    axes[1].plot(funding.index, funding["BTC_funding"]*100, color="#f7931a",
                 alpha=0.7, label="BTC daily funding", linewidth=0.7)
    axes[1].plot(funding.index, funding["ETH_funding"]*100, color="#627eea",
                 alpha=0.7, label="ETH daily funding", linewidth=0.7)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Funding (%/day)")
    axes[1].legend(); axes[1].set_title("Simulated OKX 8h funding rate, daily-aggregated",
                                        fontweight="bold")

    axes[2].plot(res["equity"].index, res["equity"], color="#16a085", linewidth=1.5)
    axes[2].axhline(1_000_000, color="grey", linewidth=0.6, linestyle="--")
    axes[2].set_ylabel("Equity ($)")
    axes[2].set_title("Funding Rate Carry — equity curve (market-neutral)",
                      fontweight="bold")
    axes[2].fill_between(res["equity"].index, 1_000_000, res["equity"],
                         where=(res["equity"] >= 1_000_000),
                         alpha=0.25, color="#16a085")

    plt.tight_layout()
    plt.savefig("./reports/08_carry_strategy.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("\n  Saved chart -> ./reports/08_carry_strategy.png")


if __name__ == "__main__":
    main()
