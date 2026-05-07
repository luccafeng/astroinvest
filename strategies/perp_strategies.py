"""
perp_strategies.py
==================
OKX-specific strategies that exploit perpetual swap features unavailable on
spot exchanges:

  1. FundingRateCarry      -- short perp + long spot when funding > threshold
                              (cash-and-carry; market-neutral basis trade)
  2. FundingMomentumOverlay -- bias spot trend signals by funding regime
                              (avoid longs when funding extremely positive,
                               avoid shorts when funding extremely negative)

These require funding rate history alongside price data. Use:
    from data.okx_loader import load_perp_with_funding
    bundle = load_perp_with_funding(start='2020-01-01')
    prices, funding = bundle['prices'], bundle['funding']

Funding rate intuition:
  * Funding rate > 0  ==> longs pay shorts every 8h
  * Funding rate < 0  ==> shorts pay longs every 8h
  * Annualised: rate * 3 * 365  (since paid 3x per day)
  * Historical median on BTC perp ~+10% annualised, with episodes of +50-100%
    in bull blow-offs (April 2021, October 2021, March 2024) and -20-30%
    during forced unwinds (May 2021, June 2022).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# A daily funding rate of 0.0003 (0.03%) per 8h period ≈ 32%/yr annualised
DEFAULT_HIGH_FUNDING_TRIGGER = 0.0003   # per 8h interval
DEFAULT_LOW_FUNDING_TRIGGER  = -0.0001


class FundingRateCarry:
    """
    Cash-and-carry style market-neutral strategy.

    Long spot + short perp pays the funding rate (perp shorts COLLECT funding
    when rate is positive). The spread between perp and spot is small for
    BTC-USDT/ETH-USDT (typically <5 bps) so the carry is mostly funding.

    Signal: when 24h trailing funding > trigger, enter the carry. Exit when
    funding falls below half of trigger. We size each leg at 50% of capital
    (so net delta ~ 0).

    Output: weights DataFrame with columns
        'BTC_spot', 'BTC_perp_short', 'ETH_spot', 'ETH_perp_short'
    The backtester for this strategy must consume both prices AND funding;
    see `backtest_perp_carry()` below.
    """
    name = "Funding Rate Carry (BTC+ETH perp)"

    def __init__(self,
                 entry_trigger: float = DEFAULT_HIGH_FUNDING_TRIGGER,
                 exit_trigger: float = DEFAULT_HIGH_FUNDING_TRIGGER / 2,
                 leg_weight: float = 0.45):
        self.entry, self.exit = entry_trigger, exit_trigger
        self.leg = leg_weight   # 45% spot long + 45% perp short = 90% gross, 0% net

    def generate_signals(self, funding: pd.DataFrame) -> pd.DataFrame:
        """funding columns: ['BTC_funding', 'ETH_funding'] (daily total of 3 intervals)."""
        # Smooth funding with 3-day EMA to avoid whipsaw on single funding spikes,
        # then convert daily-total -> per-8h-interval (divide by 3).
        smooth_per_interval = funding.ewm(span=3, adjust=False).mean() / 3.0

        positions = pd.DataFrame(0.0, index=funding.index,
                                 columns=["BTC_spot", "BTC_perp", "ETH_spot", "ETH_perp"])

        for asset, fund_col in [("BTC", "BTC_funding"), ("ETH", "ETH_funding")]:
            f = smooth_per_interval[fund_col].fillna(0).values
            held = np.zeros(len(f))
            cur = 0
            for i in range(len(f)):
                if cur == 0 and f[i] > self.entry:
                    cur = 1
                elif cur == 1 and f[i] < self.exit:
                    cur = 0
                held[i] = cur
            positions[f"{asset}_spot"] = +self.leg * held
            positions[f"{asset}_perp"] = -self.leg * held

        return positions


def backtest_perp_carry(
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    strategy: FundingRateCarry = None,
    fee_bps: float = 6.0,             # OKX VIP1 perp+spot blended round-trip
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Specialised backtester for the funding-carry strategy.

    PnL components per leg per day:
        spot_long_pnl  = w_spot * spot_return
        perp_short_pnl = w_perp_short * (-perp_return + funding_received)

    Assumes spot price ≈ perp price (true for liquid USDT pairs, basis < 5 bps).
    """
    if strategy is None:
        strategy = FundingRateCarry()
    # Align indexes
    common = prices.index.intersection(funding.index)
    prices = prices.loc[common]
    funding = funding.loc[common]

    sigs = strategy.generate_signals(funding)
    spot_ret = prices.pct_change().fillna(0)

    # PnL: spot leg gets spot return; perp short leg gets -spot_return + funding
    daily_pnl = (
        sigs["BTC_spot"].shift(1).fillna(0) * spot_ret["BTC"] +
        sigs["BTC_perp"].shift(1).fillna(0) * spot_ret["BTC"]    # short = neg weight
        - sigs["BTC_perp"].shift(1).fillna(0) * funding["BTC_funding"]    # short receives positive funding
        +
        sigs["ETH_spot"].shift(1).fillna(0) * spot_ret["ETH"] +
        sigs["ETH_perp"].shift(1).fillna(0) * spot_ret["ETH"]
        - sigs["ETH_perp"].shift(1).fillna(0) * funding["ETH_funding"]
    )

    # Costs on turnover (each entry/exit toggles 4 positions × leg_weight)
    turnover = sigs.diff().abs().sum(axis=1)
    txn_cost = turnover * fee_bps / 1e4
    daily_pnl = daily_pnl - txn_cost

    equity = (1 + daily_pnl).cumprod() * initial_capital
    return {
        "equity": equity,
        "daily_ret": daily_pnl,
        "turnover": turnover,
        "costs": txn_cost,
        "signals": sigs,
    }
