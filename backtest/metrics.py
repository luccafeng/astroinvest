"""
metrics.py
==========
Performance & risk metrics. Crypto trades 365 days/year, so all annualization
uses 365 (NOT 252). Sortino uses downside deviation; Calmar = CAGR / |MaxDD|.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 365  # crypto trades 24/7


@dataclass
class PerfMetrics:
    cagr:         float
    total_return: float
    ann_vol:      float
    sharpe:       float
    sortino:      float
    calmar:       float
    max_drawdown: float
    avg_drawdown: float
    dd_duration_days: int
    skew:         float
    kurtosis:     float
    var_95:       float
    cvar_95:      float
    hit_ratio:    float
    pnl_per_trade_bps: float
    total_costs_bps:   float
    turnover_ann:      float

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        d = self.as_dict()
        rows = [
            ("CAGR",                    f"{d['cagr']:>9.2%}"),
            ("Total Return",            f"{d['total_return']:>9.2%}"),
            ("Annualised Volatility",   f"{d['ann_vol']:>9.2%}"),
            ("Sharpe Ratio (rf=0)",     f"{d['sharpe']:>9.2f}"),
            ("Sortino Ratio",           f"{d['sortino']:>9.2f}"),
            ("Calmar Ratio",            f"{d['calmar']:>9.2f}"),
            ("Max Drawdown",            f"{d['max_drawdown']:>9.2%}"),
            ("Avg Drawdown",            f"{d['avg_drawdown']:>9.2%}"),
            ("Max DD Duration (days)",  f"{d['dd_duration_days']:>9}"),
            ("Skewness",                f"{d['skew']:>9.2f}"),
            ("Excess Kurtosis",         f"{d['kurtosis']:>9.2f}"),
            ("Daily VaR 95%",           f"{d['var_95']:>9.2%}"),
            ("Daily CVaR 95%",          f"{d['cvar_95']:>9.2%}"),
            ("Hit Ratio",               f"{d['hit_ratio']:>9.2%}"),
            ("Avg PnL/day (bps)",       f"{d['pnl_per_trade_bps']:>9.1f}"),
            ("Total Costs (bps)",       f"{d['total_costs_bps']:>9.1f}"),
            ("Annualised Turnover",     f"{d['turnover_ann']:>9.1%}"),
        ]
        return "\n".join(f"  {k:<26} {v}" for k, v in rows)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _max_dd_duration(equity: pd.Series) -> int:
    dd = _drawdown_series(equity)
    in_dd = dd < 0
    if not in_dd.any():
        return 0
    # Run-length encode
    grp = (in_dd != in_dd.shift()).cumsum()
    return int(in_dd.groupby(grp).sum().max())


def compute_metrics(
    daily_ret: pd.Series,
    equity: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> PerfMetrics:
    r = daily_ret.dropna()
    eq = equity.dropna()
    n = len(r)
    if n == 0:
        raise ValueError("Empty returns series.")

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = n / PERIODS_PER_YEAR
    cagr = (1 + total_return) ** (1 / max(years, 1e-9)) - 1.0
    ann_vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (r.mean() * PERIODS_PER_YEAR) / (ann_vol if ann_vol > 0 else np.nan)

    downside = r[r < 0]
    down_vol = downside.std() * np.sqrt(PERIODS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = (r.mean() * PERIODS_PER_YEAR) / down_vol if down_vol and down_vol > 0 else np.nan

    dd = _drawdown_series(eq)
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    dd_dur = _max_dd_duration(eq)

    var95 = float(np.quantile(r, 0.05))
    cvar95 = float(r[r <= var95].mean())

    hit = float((r > 0).sum() / n)
    pnl_bps = float(r.mean() * 1e4)

    total_costs_bps = float(costs.sum() * 1e4) if costs is not None else float("nan")
    turnover_ann = float(turnover.mean() * PERIODS_PER_YEAR) if turnover is not None else float("nan")

    return PerfMetrics(
        cagr=float(cagr),
        total_return=float(total_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino) if sortino == sortino else float("nan"),
        calmar=float(calmar) if calmar == calmar else float("nan"),
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        dd_duration_days=int(dd_dur),
        skew=float(pd.Series(r).skew()),
        kurtosis=float(pd.Series(r).kurtosis()),
        var_95=var95,
        cvar_95=cvar95,
        hit_ratio=hit,
        pnl_per_trade_bps=pnl_bps,
        total_costs_bps=total_costs_bps,
        turnover_ann=turnover_ann,
    )
