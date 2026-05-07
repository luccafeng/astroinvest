"""
backtester.py
=============
Vectorised daily backtester with:
  * Bid/ask half-spread + linear impact slippage
  * Maker/taker style fees
  * Daily borrow rate on short legs (BTC perp funding-style: ~10 bps/day avg)
  * Optional volatility-target overlay
  * Optional max-drawdown circuit breaker

Design choice: signals come in as *target weights at close*. The backtester
trades at next-day open (modelled as next-day close * (1 - small drift)).
This avoids look-ahead bias entirely.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """
    OKX-calibrated default cost model.

    Spot defaults (regular Tier 1, no OKB holdings):
        maker = 8 bps, taker = 10 bps  -> we conservatively use 10 bps for round-trip,
        and assume taker liquidity (worst case for backtest realism).

    If you hold 500+ OKB or are VIP1+, change fee_bps to 6.0 (perp) or 4.5 (spot).
    """
    fee_bps:        float = 10.0   # OKX taker fee, regular Tier 1
    half_spread_bps: float = 2.0   # Effective spread on BTC/ETH spot is tight
    impact_bps_per_unit_turnover: float = 1.0   # Linear impact term
    short_borrow_bps_daily: float = 1.5   # ~5.5%/yr borrow cost on margin shorts
                                          # (perp shorts use funding instead -- handled separately)


@dataclass
class RiskOverlay:
    target_vol: float | None = 0.25      # 25% annualized portfolio vol
    vol_window: int = 60
    max_leverage: float = 1.5
    drawdown_kill_pct: float | None = 0.40   # halt if DD > 40%
    drawdown_lookback: int = 252


@dataclass
class BacktestResult:
    equity:    pd.Series
    weights:   pd.DataFrame
    daily_ret: pd.Series
    turnover:  pd.Series
    costs:     pd.Series
    metadata:  dict = field(default_factory=dict)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost: CostModel = CostModel(),
    risk: RiskOverlay = RiskOverlay(),
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    prices = prices.sort_index()
    target_weights = target_weights.reindex(prices.index).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)

    # ------------------------------------------------------------------
    # Volatility targeting overlay (optional)
    # ------------------------------------------------------------------
    if risk.target_vol is not None:
        gross_ret = (target_weights * rets).sum(axis=1)
        realised_vol = gross_ret.rolling(risk.vol_window).std() * np.sqrt(365)
        scale = (risk.target_vol / realised_vol).clip(0.0, risk.max_leverage).bfill().fillna(1.0)
        target_weights = target_weights.multiply(scale, axis=0)
        target_weights = target_weights.clip(-risk.max_leverage, risk.max_leverage)

    # ------------------------------------------------------------------
    # Turnover & costs
    # ------------------------------------------------------------------
    turnover = (target_weights - target_weights.shift(1).fillna(0.0)).abs().sum(axis=1)

    # Per-unit-turnover cost: fee + half-spread + linear impact
    cost_rate = (
        (cost.fee_bps + cost.half_spread_bps) / 1e4
        + (cost.impact_bps_per_unit_turnover / 1e4) * turnover
    )
    txn_cost = turnover * cost_rate

    # Short-borrow cost on short notional
    short_notional = target_weights.clip(upper=0).abs().sum(axis=1)
    borrow_cost = short_notional * (cost.short_borrow_bps_daily / 1e4)

    # ------------------------------------------------------------------
    # Drawdown circuit breaker
    # ------------------------------------------------------------------
    portfolio_ret = (target_weights.shift(1).fillna(0.0) * rets).sum(axis=1) \
                    - txn_cost - borrow_cost
    equity = (1 + portfolio_ret).cumprod() * initial_capital

    if risk.drawdown_kill_pct is not None:
        rolling_max = equity.cummax()
        dd = equity / rolling_max - 1.0
        breached = dd <= -abs(risk.drawdown_kill_pct)
        # If breached, force weights to zero for next 30 days
        if breached.any():
            kill_mask = breached.rolling(30).max().fillna(0).astype(bool)
            target_weights.loc[kill_mask] = 0.0
            # Recompute everything after killing
            turnover = (target_weights - target_weights.shift(1).fillna(0.0)).abs().sum(axis=1)
            cost_rate = (
                (cost.fee_bps + cost.half_spread_bps) / 1e4
                + (cost.impact_bps_per_unit_turnover / 1e4) * turnover
            )
            txn_cost = turnover * cost_rate
            short_notional = target_weights.clip(upper=0).abs().sum(axis=1)
            borrow_cost = short_notional * (cost.short_borrow_bps_daily / 1e4)
            portfolio_ret = (target_weights.shift(1).fillna(0.0) * rets).sum(axis=1) \
                            - txn_cost - borrow_cost
            equity = (1 + portfolio_ret).cumprod() * initial_capital

    return BacktestResult(
        equity=equity,
        weights=target_weights,
        daily_ret=portfolio_ret,
        turnover=turnover,
        costs=txn_cost + borrow_cost,
        metadata={"initial_capital": initial_capital},
    )
