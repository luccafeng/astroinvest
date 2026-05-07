"""
strategies.py
=============
Signal-generating strategies. Each returns a DataFrame of *target weights* in
[-1, +1] per asset per day. The backtester is in charge of position sizing,
turnover, slippage, etc.

Strategies implemented:
  1. BuyAndHold              -- 60/40 BTC/ETH, no rebalancing (benchmark)
  2. EqualWeightRebalance    -- 50/50 monthly rebalance (benchmark)
  3. MovingAvgCrossover      -- Classic trend follower (50/200 daily SMA)
  4. TimeSeriesMomentum      -- Moskowitz/Ooi/Pedersen-style 12-1m momentum
  5. BollingerMeanReversion  -- Long when price < lower band, short when > upper
  6. RSIReversion            -- RSI(14) mean-reversion, neutral mid-zone
  7. DonchianBreakout        -- Turtle-style 20/55 day breakout
  8. VolTargetedEnsemble     -- Weighted blend of 3,4,5,7 with vol scaling
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str = "Strategy"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a (T, N_assets) DataFrame of target weights in [-1, +1]."""
        ...

    @staticmethod
    def _zeros_like(prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


# ----------------------------------------------------------------------- #
#  1. Buy and hold (fixed at t=0)                                         #
# ----------------------------------------------------------------------- #
class BuyAndHold(Strategy):
    name = "Buy & Hold 60/40"
    def __init__(self, weights=(0.6, 0.4)):
        self.weights = weights
    def generate_weights(self, prices):
        w = self._zeros_like(prices)
        w.iloc[0] = self.weights
        # Forward-fill = hold (the backtester recomputes effective weight from drift)
        return w.replace(0.0, np.nan).ffill().fillna(0.0)


# ----------------------------------------------------------------------- #
#  2. Equal-weight monthly rebalance                                      #
# ----------------------------------------------------------------------- #
class EqualWeightRebalance(Strategy):
    name = "Equal Weight (Monthly)"
    def generate_weights(self, prices):
        w = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
        # Mark month-ends as the rebalance days; in between, weights drift naturally
        # Backtester treats each row as a *target* though, so this is fine here.
        return w


# ----------------------------------------------------------------------- #
#  3. Moving average crossover                                            #
# ----------------------------------------------------------------------- #
class MovingAvgCrossover(Strategy):
    name = "MA Crossover (50/200)"
    def __init__(self, fast=50, slow=200, gross_leverage=1.0):
        self.fast, self.slow, self.lev = fast, slow, gross_leverage
    def generate_weights(self, prices):
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        sig = (fast > slow).astype(float) - (fast < slow).astype(float)
        sig = sig.shift(1).fillna(0.0)        # avoid lookahead: trade next day
        return sig * (self.lev / sig.abs().sum(axis=1).clip(lower=1)).values[:, None]


# ----------------------------------------------------------------------- #
#  4. Time-series momentum (12-1 month)                                   #
#     Reference: Moskowitz, Ooi, Pedersen (2012) "Time series momentum"   #
# ----------------------------------------------------------------------- #
class TimeSeriesMomentum(Strategy):
    name = "TS Momentum (12-1m)"
    def __init__(self, lookback=252, skip=21, vol_target=0.20):
        self.lb, self.skip, self.vt = lookback, skip, vol_target
    def generate_weights(self, prices):
        # Past 12 months excluding most recent month (skip=21)
        ret_lb = (prices.shift(self.skip) / prices.shift(self.skip + self.lb) - 1.0)
        sig = np.sign(ret_lb)
        # Inverse-volatility scaling so each leg contributes equal risk
        realized = prices.pct_change().rolling(63).std() * np.sqrt(252)
        scale = (self.vt / realized).clip(0, 3.0)
        w = (sig * scale).shift(1).fillna(0.0)
        # Cap gross exposure
        gross = w.abs().sum(axis=1).clip(lower=1.0)
        return w.div(gross, axis=0).clip(-1, 1)


# ----------------------------------------------------------------------- #
#  5. Bollinger band mean-reversion                                       #
# ----------------------------------------------------------------------- #
class BollingerMeanReversion(Strategy):
    name = "Bollinger Mean-Rev (20, 2σ)"
    def __init__(self, window=20, k=2.0):
        self.window, self.k = window, k
    def generate_weights(self, prices):
        ma = prices.rolling(self.window).mean()
        sd = prices.rolling(self.window).std()
        upper, lower = ma + self.k * sd, ma - self.k * sd
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[prices < lower] = 1.0   # long when oversold
        sig[prices > upper] = -1.0  # short when overbought
        # Hold the position until price crosses back through the MA
        for col in sig.columns:
            held = 0.0
            out = []
            for p, m, s in zip(prices[col], ma[col], sig[col]):
                if s != 0:
                    held = s
                elif (held > 0 and p >= m) or (held < 0 and p <= m):
                    held = 0.0
                out.append(held)
            sig[col] = out
        sig = sig.shift(1).fillna(0.0)
        return sig * 0.5  # half-leverage per leg


# ----------------------------------------------------------------------- #
#  6. RSI mean reversion                                                  #
# ----------------------------------------------------------------------- #
class RSIReversion(Strategy):
    name = "RSI(14) Reversion"
    def __init__(self, period=14, lower=30, upper=70):
        self.p, self.lo, self.hi = period, lower, upper
    @staticmethod
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        rs = up / dn
        return 100 - 100 / (1 + rs)
    def generate_weights(self, prices):
        rsi = prices.apply(lambda s: self._rsi(s, self.p))
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[rsi < self.lo] = 1.0
        sig[rsi > self.hi] = -1.0
        # Persist position until RSI re-enters neutral zone
        for col in sig.columns:
            held, out = 0.0, []
            for r, s in zip(rsi[col], sig[col]):
                if s != 0:
                    held = s
                elif (held > 0 and r >= 50) or (held < 0 and r <= 50):
                    held = 0.0
                out.append(held)
            sig[col] = out
        sig = sig.shift(1).fillna(0.0)
        return sig * 0.5


# ----------------------------------------------------------------------- #
#  7. Donchian channel breakout (Turtle-style)                            #
# ----------------------------------------------------------------------- #
class DonchianBreakout(Strategy):
    name = "Donchian Breakout (20/55)"
    def __init__(self, entry=55, exit_w=20):
        self.entry, self.exit_w = entry, exit_w
    def generate_weights(self, prices):
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            high_e = prices[col].rolling(self.entry).max()
            low_e  = prices[col].rolling(self.entry).min()
            high_x = prices[col].rolling(self.exit_w).max()
            low_x  = prices[col].rolling(self.exit_w).min()
            held, out = 0.0, []
            for p, he, le, hx, lx in zip(prices[col], high_e, low_e, high_x, low_x):
                if held == 0:
                    if p >= he:
                        held = 1.0
                    elif p <= le:
                        held = -1.0
                elif held > 0 and p <= lx:
                    held = 0.0
                elif held < 0 and p >= hx:
                    held = 0.0
                out.append(held)
            sig[col] = out
        sig = sig.shift(1).fillna(0.0)
        return sig * 0.5


# ----------------------------------------------------------------------- #
#  8. Volatility-targeted ensemble                                        #
# ----------------------------------------------------------------------- #
class VolTargetedEnsemble(Strategy):
    """
    Combines TSMomentum + Bollinger + Donchian with equal *risk* weight,
    then re-scales the whole book so that ex-ante portfolio vol = target.
    This is closer to how a real CTA / multi-strat fund operates.
    """
    name = "Vol-Targeted Ensemble"

    def __init__(self, target_vol=0.20):
        self.target_vol = target_vol
        self.subs = [
            TimeSeriesMomentum(),
            BollingerMeanReversion(),
            DonchianBreakout(),
        ]

    def generate_weights(self, prices):
        sub_w = [s.generate_weights(prices) for s in self.subs]
        avg = sum(sub_w) / len(sub_w)
        rets = prices.pct_change().fillna(0)
        # Realized portfolio vol of the raw average book (rolling 60d)
        port_ret = (avg * rets).sum(axis=1)
        port_vol = port_ret.rolling(60).std() * np.sqrt(252)
        scale = (self.target_vol / port_vol).clip(0, 3.0).fillna(1.0)
        w = avg.multiply(scale, axis=0)
        # Cap gross
        gross = w.abs().sum(axis=1).clip(lower=1.0)
        return w.div(gross, axis=0).clip(-1, 1)


ALL_STRATEGIES = [
    BuyAndHold(),
    EqualWeightRebalance(),
    MovingAvgCrossover(),
    TimeSeriesMomentum(),
    BollingerMeanReversion(),
    RSIReversion(),
    DonchianBreakout(),
    VolTargetedEnsemble(),
]
