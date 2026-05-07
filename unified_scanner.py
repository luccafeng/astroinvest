"""
unified_scanner.py
==================
Unified daily LONG/SHORT entry decision system with regime detection.

Workflow:
  1. Detect market regime (BULL / BEAR / NEUTRAL) using multi-factor logic
  2. In BULL regime: only consider LONG signals (longs have tailwind)
  3. In BEAR regime: only consider SHORT signals (shorts have tailwind)
  4. In NEUTRAL/TRANSITION: stay flat OR run market-neutral carry
  5. Apply confluence score for final entry decision
  6. Output position size, stop-loss, max risk

Critical principle:
  Never fight the regime. Most retail losses come from longing in bear
  markets ("bottom fishing") or shorting in bull markets ("top picking").
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def compute_rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def detect_regime(close):
    """
    Classify market into BULL / BEAR / NEUTRAL using 4 conditions.

    BULL = at least 3 of 4 bullish conditions met
    BEAR = at least 3 of 4 bearish conditions met
    NEUTRAL = transition / mixed signals -> stay flat or carry
    """
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()

    # 4 bullish conditions
    bull_score = pd.Series(0, index=close.index)
    bull_score += (close > ma200).astype(int)            # above 200MA
    bull_score += (ma50 > ma200).astype(int)             # 50 > 200
    bull_score += (macd > macd_signal).astype(int)       # MACD positive
    # Slope of 50MA positive
    ma50_slope = (ma50 - ma50.shift(20)) / ma50.shift(20)
    bull_score += (ma50_slope > 0).astype(int)

    # 4 bearish conditions
    bear_score = pd.Series(0, index=close.index)
    bear_score += (close < ma200).astype(int)            # below 200MA
    bear_score += (ma50 < ma200).astype(int)             # 50 < 200
    bear_score += (macd < macd_signal).astype(int)       # MACD negative
    bear_score += (ma50_slope < 0).astype(int)

    regime = pd.Series("NEUTRAL", index=close.index)
    regime[bull_score >= 3] = "BULL"
    regime[bear_score >= 3] = "BEAR"

    return regime, bull_score, bear_score


def long_confluence(close):
    """5-point long entry score (validated by entry_signals_analysis.py)."""
    rsi = compute_rsi(close)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    macd_sig = macd.ewm(span=9).mean()

    s = pd.Series(0, index=close.index)
    s += (rsi < 40).astype(int)
    s += (close > ma200).astype(int)
    s += (ma50 > ma200).astype(int)
    s += (macd > macd_sig).astype(int)
    s += (close > ma50 * 0.95).astype(int)
    return s


def short_confluence(close):
    """5-point short entry score (validated by short_signals_analysis.py)."""
    rsi = compute_rsi(close)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    macd_sig = macd.ewm(span=9).mean()

    s = pd.Series(0, index=close.index)
    s += (rsi > 60).astype(int)               # not oversold
    s += (close < ma200).astype(int)          # below 200MA
    s += (ma50 < ma200).astype(int)           # 50 < 200
    s += (macd < macd_sig).astype(int)        # MACD bearish
    s += (close < ma50 * 1.05).astype(int)    # not breaking out
    return s


def make_decision(close, account_size=100_000.0):
    """Returns full decision context for the latest bar."""
    regime, bull_sc, bear_sc = detect_regime(close)
    long_sc = long_confluence(close)
    short_sc = short_confluence(close)

    high_20 = close.rolling(20).max()
    high_55 = close.rolling(55).max()
    low_20 = close.rolling(20).min()
    low_55 = close.rolling(55).min()

    bo_20 = close >= high_20 * 0.999
    bo_55 = close >= high_55 * 0.999
    bd_20 = close <= low_20 * 1.001
    bd_55 = close <= low_55 * 1.001

    rsi = compute_rsi(close)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    latest = {
        "date": close.index[-1],
        "price": float(close.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "ma50": float(ma50.iloc[-1]),
        "ma200": float(ma200.iloc[-1]),
        "regime": regime.iloc[-1],
        "bull_score": int(bull_sc.iloc[-1]),
        "bear_score": int(bear_sc.iloc[-1]),
        "long_score": int(long_sc.iloc[-1]),
        "short_score": int(short_sc.iloc[-1]),
        "breakout_20d": bool(bo_20.iloc[-1]),
        "breakout_55d": bool(bo_55.iloc[-1]),
        "breakdown_20d": bool(bd_20.iloc[-1]),
        "breakdown_55d": bool(bd_55.iloc[-1]),
    }

    # Decision logic with regime gating
    action = "STAY FLAT"
    side = None
    position_pct = 0.0
    urgency = "Wait for clearer signals"
    reason = []

    if latest["regime"] == "BULL":
        # Only longs in bull regime
        if latest["breakout_55d"] and latest["long_score"] >= 4:
            action = "STRONG BUY"
            side = "LONG"
            position_pct = 0.25
            urgency = "Enter today, full position"
            reason.append("55d high breakout in BULL regime + confluence ≥4")
        elif latest["breakout_20d"] and latest["long_score"] >= 4:
            action = "BUY"
            side = "LONG"
            position_pct = 0.18
            urgency = "Enter today, standard position"
            reason.append("20d high breakout in BULL regime + confluence ≥4")
        elif latest["long_score"] == 5:
            action = "ACCUMULATE"
            side = "LONG"
            position_pct = 0.15
            reason.append("Max long confluence in BULL regime")
        elif latest["long_score"] == 4:
            action = "STARTER LONG"
            side = "LONG"
            position_pct = 0.08
            reason.append("Good long confluence; await breakout")
        else:
            action = "WAIT (BULL regime, no entry)"
            reason.append(f"BULL regime but long_score={latest['long_score']}")

    elif latest["regime"] == "BEAR":
        # Only shorts in bear regime
        if latest["breakdown_55d"] and latest["short_score"] >= 4:
            action = "STRONG SHORT"
            side = "SHORT"
            position_pct = 0.20
            urgency = "Enter today, but watch for short squeeze"
            reason.append("55d low breakdown in BEAR regime + confluence ≥4")
        elif latest["breakdown_20d"] and latest["short_score"] >= 4:
            action = "SHORT"
            side = "SHORT"
            position_pct = 0.13
            urgency = "Enter today, smaller position than longs"
            reason.append("20d low breakdown in BEAR regime + confluence ≥4")
        elif latest["short_score"] == 5:
            action = "ACCUMULATE SHORT"
            side = "SHORT"
            position_pct = 0.10
            reason.append("Max short confluence in BEAR regime")
        elif latest["short_score"] == 4:
            action = "STARTER SHORT"
            side = "SHORT"
            position_pct = 0.06
            reason.append("Good short confluence; await breakdown")
        else:
            action = "WAIT (BEAR regime, no entry)"
            reason.append(f"BEAR regime but short_score={latest['short_score']}")

    else:  # NEUTRAL
        action = "FLAT or CARRY (NEUTRAL regime)"
        position_pct = 0.0
        reason.append("Regime is transitioning - dont fight it")
        reason.append("Consider funding rate carry (market-neutral)")

    # Position sizing & stop-loss
    if side == "LONG":
        stop_dist_pct = 0.08
        stop_price = latest["price"] * (1 - stop_dist_pct)
    elif side == "SHORT":
        stop_dist_pct = 0.10  # wider stop for shorts (asymmetric vol)
        stop_price = latest["price"] * (1 + stop_dist_pct)
    else:
        stop_dist_pct = 0
        stop_price = None

    capital = account_size * position_pct
    units = capital / latest["price"] if latest["price"] > 0 else 0
    max_risk_pct = position_pct * stop_dist_pct * 100

    decision = {
        **latest,
        "action": action,
        "side": side,
        "position_pct": position_pct,
        "capital_usd": capital,
        "units": units,
        "stop_price": stop_price,
        "stop_dist_pct": stop_dist_pct,
        "max_risk_pct_of_account": max_risk_pct,
        "urgency": urgency,
        "reason": " | ".join(reason),
    }
    return decision


def print_report(asset, d):
    regime_emoji = {"BULL": "🟢 BULL", "BEAR": "🔴 BEAR", "NEUTRAL": "🟡 NEUTRAL"}
    print("\n" + "=" * 76)
    print(f"  UNIFIED L/S SCANNER — {asset}   |   {d['date'].date()}")
    print("=" * 76)
    print(f"  Price:        ${d['price']:>12,.2f}")
    print(f"  RSI(14):      {d['rsi']:>12.1f}")
    print(f"  50-day MA:    ${d['ma50']:>12,.2f}  ({(d['price']/d['ma50']-1)*100:+.1f}%)")
    print(f"  200-day MA:   ${d['ma200']:>12,.2f}  ({(d['price']/d['ma200']-1)*100:+.1f}%)")
    print()
    print(f"  REGIME:       {regime_emoji.get(d['regime'], d['regime'])}")
    print(f"     bull score: {d['bull_score']}/4   bear score: {d['bear_score']}/4")
    print()
    print(f"  LONG  confluence:  {d['long_score']}/5    "
          f"breakouts: 20d={'✓' if d['breakout_20d'] else '✗'}  "
          f"55d={'✓' if d['breakout_55d'] else '✗'}")
    print(f"  SHORT confluence:  {d['short_score']}/5    "
          f"breakdowns: 20d={'✓' if d['breakdown_20d'] else '✗'}  "
          f"55d={'✓' if d['breakdown_55d'] else '✗'}")
    print()
    print(f"  >>> ACTION: {d['action']}")
    print(f"      Side:        {d['side'] if d['side'] else 'NONE'}")
    print(f"      Reason:      {d['reason']}")
    if d["urgency"]:
        print(f"      Urgency:     {d['urgency']}")
    print()
    if d["position_pct"] > 0:
        print(f"  POSITION SIZING (account = ${d['capital_usd']/(d['position_pct']):,.0f}):")
        print(f"      Side:           {d['side']}")
        print(f"      Position size:  {d['position_pct']:.0%}  (${d['capital_usd']:,.0f})")
        print(f"      Units:          {d['units']:.4f} {asset}")
        print(f"      Entry price:    ${d['price']:,.2f}")
        print(f"      Stop loss:      ${d['stop_price']:,.2f}  "
              f"({'-' if d['side']=='LONG' else '+'}{d['stop_dist_pct']:.0%})")
        print(f"      Max risk:       {d['max_risk_pct_of_account']:.2f}% of account")

        if d["side"] == "SHORT":
            print()
            print(f"  ⚠️  SHORT-SPECIFIC WARNINGS:")
            print(f"      - Watch funding rate. If >0.05%/8h (>54%/yr), reduce size.")
            print(f"      - Use tighter trailing stop on profit (asymmetric vol).")
            print(f"      - Halve position if BTC dominance starts falling rapidly.")
            print(f"      - Cover immediately if regime flips to NEUTRAL.")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="both", choices=["BTC", "ETH", "both"])
    parser.add_argument("--account", type=float, default=100_000)
    args = parser.parse_args()

    from data.data_loader import get_prices
    try:
        prices = get_prices(use_real=True)
        print(f"[Real OKX data: {prices.index[0].date()} -> {prices.index[-1].date()}]")
    except Exception as e:
        print(f"[Real data unavailable, using calibrated demo: {e}]")
        prices = get_prices(use_real=False)

    assets = ["BTC", "ETH"] if args.asset == "both" else [args.asset]
    for asset in assets:
        d = make_decision(prices[asset], account_size=args.account)
        print_report(asset, d)

    print("\nNOTE: Single-asset signals only. Always sanity-check with macro context.")
    print("      Never put more than 1.5% of account at risk on any single trade.")
    print("      In SHORT positions, monitor funding rate and short-interest daily.")


if __name__ == "__main__":
    main()
