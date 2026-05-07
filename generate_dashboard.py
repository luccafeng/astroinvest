"""
generate_dashboard.py
=====================
Build a self-contained dashboard.html from real OKX data and master workflow output.

Output: ./dist/index.html  (single file, no external dependencies except CDN tabler)

Usage:
    python generate_dashboard.py                    # uses real OKX data if available
    python generate_dashboard.py --use-demo         # forces calibrated demo data
    python generate_dashboard.py --output PATH      # custom output path

This script is designed to be run by:
  - You manually for testing
  - GitHub Actions cron daily at 12:00 UTC for production
"""
from __future__ import annotations
import os, sys, json, argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ======================================================================
# 1. PULL DECISION FROM YOUR EXISTING MODULES
# ======================================================================

def gather_data(use_demo: bool = False) -> dict:
    """Run unified_scanner + news_monitor and return everything needed for dashboard."""
    from unified_scanner import make_decision, compute_rsi
    from news_monitor import assess_market_risk, upcoming_events
    from data.data_loader import get_prices

    if use_demo:
        prices = get_prices(use_real=False)
        data_source = "calibrated demo"
    else:
        try:
            prices = get_prices(use_real=True)
            data_source = "OKX live"
        except Exception as e:
            print(f"[WARN] Real data failed: {e}")
            prices = get_prices(use_real=False)
            data_source = "calibrated demo (fallback)"

    btc_decision = make_decision(prices["BTC"], account_size=100_000)
    eth_decision = make_decision(prices["ETH"], account_size=100_000)
    risk = assess_market_risk()
    events = upcoming_events(7)

    # Extract last 30 days OHLC if available, else build from close-only
    btc_ohlc = build_ohlc_series(prices["BTC"], days=30)
    eth_ohlc = build_ohlc_series(prices["ETH"], days=30)

    # Detect historical buy/short signals over last 30 days
    btc_signals = detect_historical_signals(prices["BTC"])
    eth_signals = detect_historical_signals(prices["ETH"])

    return {
        "timestamp": datetime.now(timezone.utc),
        "data_source": data_source,
        "btc": btc_decision,
        "eth": eth_decision,
        "btc_ohlc": btc_ohlc,
        "eth_ohlc": eth_ohlc,
        "btc_signals": btc_signals,
        "eth_signals": eth_signals,
        "risk": risk,
        "events": events,
    }


def build_ohlc_series(close: pd.Series, days: int = 30) -> list:
    """
    Generate OHLC bars from close-only data using rolling proxy.
    If you later add intraday data, replace with real OHLC.
    """
    recent = close.tail(days + 1)
    out = []
    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]
        # Proxy: simulate intraday range from daily volatility
        rng = abs(curr - prev) * 1.4 + curr * 0.005
        high = max(prev, curr) + rng * 0.4
        low = min(prev, curr) - rng * 0.4
        out.append({
            "date": recent.index[i].strftime("%Y-%m-%d"),
            "o": float(prev),
            "h": float(high),
            "l": float(low),
            "c": float(curr),
        })
    return out


def detect_historical_signals(close: pd.Series, lookback_days: int = 30) -> list:
    """Find when buy/short signals would have triggered in the last N days."""
    from unified_scanner import (
        detect_regime, long_confluence, short_confluence
    )
    regime, _, _ = detect_regime(close)
    long_sc = long_confluence(close)
    short_sc = short_confluence(close)
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()

    signals = []
    recent = close.tail(lookback_days + 1)
    n = len(recent)
    for i, ts in enumerate(recent.index[1:], start=1):
        idx = -lookback_days - 1 + i
        try:
            r = regime.iloc[idx]
            ls = int(long_sc.iloc[idx])
            ss = int(short_sc.iloc[idx])
            p = close.iloc[idx]
            h20 = high_20.iloc[idx]
            l20 = low_20.iloc[idx]

            if r == "BULL" and ls >= 4 and p >= h20 * 0.999:
                signals.append({"i": i - 1, "type": "buy", "date": ts.strftime("%Y-%m-%d"), "price": float(p)})
            elif r == "BEAR" and ss >= 4 and p <= l20 * 1.001:
                signals.append({"i": i - 1, "type": "short", "date": ts.strftime("%Y-%m-%d"), "price": float(p)})
        except (IndexError, KeyError):
            continue
    return signals


# ======================================================================
# 2. BUILD CANDLESTICK CHART DATA FOR HTML
# ======================================================================

def build_chart_data(ohlc: list, lookback: int = 20) -> dict:
    """Convert OHLC + MAs into the form the JS chart renderer expects."""
    bars = ohlc[-lookback:]
    closes = [b["c"] for b in bars]
    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]

    # Compute 50/200 MA over the longer series, then take last `lookback`
    full_series = pd.Series([b["c"] for b in ohlc])
    ma50 = full_series.rolling(min(50, len(full_series))).mean().bfill().tolist()[-lookback:]
    ma200 = full_series.rolling(min(200, len(full_series))).mean().bfill().tolist()[-lookback:]

    p_max = max(max(highs), max(ma50), max(ma200))
    p_min = min(min(lows), min(ma50), min(ma200))
    pad = (p_max - p_min) * 0.05
    p_max += pad
    p_min -= pad

    def y_of(price):
        return 5 + (p_max - price) / (p_max - p_min) * 75

    bar_data = []
    for b in bars:
        bar_data.append({
            "h": round(y_of(b["h"]), 1),
            "l": round(y_of(b["l"]), 1),
            "o": round(y_of(b["o"]), 1),
            "c": round(y_of(b["c"]), 1),
            "up": b["c"] >= b["o"],
        })

    ma50_y = [round(y_of(p), 1) for p in ma50]
    ma200_y = [round(y_of(p), 1) for p in ma200]

    # Price labels at key levels
    levels = [p_max - (p_max - p_min) * f for f in (0.05, 0.27, 0.50, 0.73, 0.95)]
    price_labels = [{"label": fmt_price(p), "y": round(y_of(p), 1)} for p in levels]

    dates = [b["date"][5:] for b in bars]

    return {
        "data": bar_data,
        "ma50": ma50_y,
        "ma200": ma200_y,
        "labels": price_labels,
        "dates": dates,
    }


def fmt_price(p: float) -> str:
    if p >= 10000:
        return f"{p/1000:.0f}k"
    if p >= 1000:
        return f"{p/1000:.1f}k"
    return f"{p:,.0f}"


def filter_signals_for_chart(signals: list, lookback: int = 20) -> list:
    """Map signal indices into the chart's lookback window."""
    return [{"i": s["i"], "type": s["type"]} for s in signals if s["i"] < lookback]


# ======================================================================
# 3. GATE LOGIC (mirrors master_workflow.py)
# ======================================================================

def compute_final_action(btc: dict, eth: dict, risk) -> dict:
    final = "TRADE"
    reasons = []

    btc_flat = btc["side"] is None
    eth_flat = eth["side"] is None

    if btc_flat and eth_flat:
        final = "FLAT"
        reasons.append("Both BTC and ETH have no entry signal")

    if risk.action == "EXIT_ALL":
        final = "EXIT_ALL"
        reasons.append(f"Event severity {risk.severity}/10 forces exit")
    elif risk.action == "PAUSE_ENTRY":
        final = "PAUSE"
        reasons.append(f"Event severity {risk.severity}/10 — no new entries")

    return {"action": final, "reasons": reasons}


# ======================================================================
# 4. RENDER HTML
# ======================================================================

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>astroinvest.ai · Daily decision</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.47.0/tabler-icons.min.css">
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;background:#FAFAFA;color:#0F0F12;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}}
.container{{max-width:1100px;margin:0 auto;padding:24px 20px}}
.av-card{{background:#FFFFFF;border:0.5px solid #E5E5E7;border-radius:10px;padding:14px 16px}}
.av-label{{font-size:10px;color:#6E6E73;letter-spacing:0.6px;text-transform:uppercase;font-weight:500}}
.av-num{{font-variant-numeric:tabular-nums}}
.av-row{{display:flex;align-items:center;justify-content:space-between}}
.av-pill{{display:inline-block;font-size:9px;padding:2px 8px;border-radius:4px;letter-spacing:0.4px;font-weight:500;text-transform:uppercase;background:#F1F1F3;color:#6E6E73}}
.av-pill.warn{{background:#FFF8E1;color:#7A4F00}}
.av-pill.bull{{background:#E1F5EE;color:#0F6E56}}
.av-pill.bear{{background:#FCEBEB;color:#A32D2D}}
.av-btn{{background:#FFFFFF;border:0.5px solid #D1D1D4;color:#0F0F12;padding:9px 12px;border-radius:8px;font-size:12px;cursor:pointer;font-family:inherit;transition:all 0.15s;text-decoration:none;display:inline-flex;align-items:center;justify-content:center;gap:4px}}
.av-btn:hover{{background:#F5F5F7;border-color:#A8A8AC}}
.tv-chart{{position:relative;width:100%;height:130px;background:#FFFFFF}}
.tv-legend{{position:absolute;top:6px;left:6px;display:flex;gap:10px;font-size:9px;color:#787B86;font-variant-numeric:tabular-nums;z-index:1}}
.tv-legend span{{display:flex;align-items:center;gap:4px}}
.tv-dot{{width:6px;height:2px;border-radius:1px}}
.cols-2{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}}
.cols-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px}}
.cols-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}}
@media (max-width:680px){{.cols-2,.cols-3{{grid-template-columns:1fr}}.cols-4{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="container">

<div class="av-row" style="margin-bottom:14px;padding-bottom:12px;border-bottom:0.5px solid #E5E5E7">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:6px;height:6px;background:#1D9E75;border-radius:50%"></div>
    <span style="font-size:15px;font-weight:500;letter-spacing:-0.2px">astroinvest.ai</span>
    <span style="font-size:11px;color:#6E6E73">·</span>
    <span style="font-size:11px;color:#6E6E73">{date_str}</span>
  </div>
  <span style="font-size:10px;color:#6E6E73">{data_source} · auto-refreshed daily</span>
</div>

<div class="av-card" style="margin-bottom:10px;border-color:#D1D1D4">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;flex-wrap:wrap">
    <span class="av-label">Today's action</span>
    <span class="av-pill">{regime_pill}</span>
    {event_pill}
  </div>
  <div class="av-row" style="align-items:flex-end;flex-wrap:wrap;gap:8px">
    <h2 style="margin:0;font-size:24px;font-weight:500;letter-spacing:-0.4px">{action_headline}</h2>
    <span style="font-size:11px;color:#6E6E73">{action_subtitle}</span>
  </div>
  <p style="margin:6px 0 0;font-size:12px;color:#6E6E73;line-height:1.5">{action_detail}</p>
</div>

<div class="cols-2">
  {asset_cards}
</div>

<div class="cols-3">
  {events_card}
  {news_card}
  {exec_card}
</div>

<div class="cols-4">
  <a class="av-btn" href="https://github.com/{github_repo}" target="_blank">View source ↗</a>
  <a class="av-btn" href="./today_plan.json" target="_blank">JSON plan ↗</a>
  <button class="av-btn" onclick="navigator.clipboard.writeText(window.location.href);this.innerHTML='Copied!'">Copy link</button>
  <button class="av-btn" onclick="location.reload()">Refresh ↺</button>
</div>

<p style="font-size:9px;color:#A8A8AC;text-align:center;margin:0;letter-spacing:0.3px">quant_crypto · OKX live · auto-deploys via GitHub Actions × Cloudflare Pages · generated {gen_time}</p>

</div>

<script>
const CHARTS = {chart_data_json};

function renderTV(elId, payload) {{
  const el = document.getElementById(elId);
  if (!el) return;
  const W = 280, H = 130;
  const chartL = 6, chartR = 248, chartT = 22, chartB = 100;
  const labelX = 252;
  const data = payload.data, ma50 = payload.ma50, ma200 = payload.ma200;
  const signals = payload.signals, priceLabels = payload.labels, dates = payload.dates;

  let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">';

  for (let i = 1; i <= 4; i++) {{
    const y = chartT + (chartB - chartT) * i / 5;
    svg += '<line x1="' + chartL + '" y1="' + y + '" x2="' + chartR + '" y2="' + y + '" stroke="#F0F3FA" stroke-width="0.5"/>';
  }}

  priceLabels.forEach(p => {{
    svg += '<text x="' + labelX + '" y="' + (p.y + 3) + '" font-size="8" fill="#787B86">' + p.label + '</text>';
  }});

  const slot = (chartR - chartL) / data.length;
  const bodyW = slot * 0.6;

  let p1 = 'M';
  ma50.forEach((p, i) => {{
    const x = chartL + i * slot + slot / 2;
    p1 += (i === 0 ? '' : ' L') + x.toFixed(1) + ',' + p.toFixed(1);
  }});
  svg += '<path d="' + p1 + '" fill="none" stroke="#2962FF" stroke-width="1" opacity="0.85"/>';

  let p2 = 'M';
  ma200.forEach((p, i) => {{
    const x = chartL + i * slot + slot / 2;
    p2 += (i === 0 ? '' : ' L') + x.toFixed(1) + ',' + p.toFixed(1);
  }});
  svg += '<path d="' + p2 + '" fill="none" stroke="#FF8A00" stroke-width="1" opacity="0.85"/>';

  data.forEach((c, i) => {{
    const x = chartL + i * slot + slot / 2;
    const color = c.up ? '#26A69A' : '#EF5350';
    const bodyTop = Math.min(c.o, c.c);
    const bodyHeight = Math.max(0.8, Math.abs(c.o - c.c));
    svg += '<line x1="' + x.toFixed(1) + '" y1="' + c.h + '" x2="' + x.toFixed(1) + '" y2="' + c.l + '" stroke="' + color + '" stroke-width="0.6"/>';
    svg += '<rect x="' + (x - bodyW / 2).toFixed(1) + '" y="' + bodyTop + '" width="' + bodyW.toFixed(1) + '" height="' + bodyHeight.toFixed(1) + '" fill="' + color + '"/>';
  }});

  const lastC = data[data.length - 1];
  const lastX = chartL + (data.length - 1) * slot + slot / 2;
  svg += '<line x1="' + lastX.toFixed(1) + '" y1="' + lastC.c + '" x2="' + chartR + '" y2="' + lastC.c + '" stroke="#0F0F12" stroke-width="0.5" stroke-dasharray="2,2" opacity="0.5"/>';

  signals.forEach(s => {{
    const x = chartL + s.i * slot + slot / 2;
    if (s.type === 'short') {{
      const y = data[s.i].h - 2;
      svg += '<polygon points="' + (x - 3.5) + ',' + (y - 8) + ' ' + (x + 3.5) + ',' + (y - 8) + ' ' + x + ',' + y + '" fill="#EF5350"/>';
      svg += '<text x="' + x + '" y="' + (y - 10) + '" text-anchor="middle" font-size="7" font-weight="600" fill="#EF5350">S</text>';
    }} else {{
      const y = data[s.i].l + 2;
      svg += '<polygon points="' + (x - 3.5) + ',' + (y + 8) + ' ' + (x + 3.5) + ',' + (y + 8) + ' ' + x + ',' + y + '" fill="#26A69A"/>';
      svg += '<text x="' + x + '" y="' + (y + 17) + '" text-anchor="middle" font-size="7" font-weight="600" fill="#26A69A">B</text>';
    }}
  }});

  const tickIdx = [0, Math.floor(dates.length/3), Math.floor(2*dates.length/3), dates.length-1];
  tickIdx.forEach(j => {{
    const x = chartL + j * slot + slot / 2;
    svg += '<text x="' + x + '" y="115" text-anchor="middle" font-size="8" fill="#787B86">' + dates[j] + '</text>';
  }});

  svg += '</svg>';
  el.insertAdjacentHTML('beforeend', svg);
}}

renderTV('btc-chart', CHARTS.btc);
renderTV('eth-chart', CHARTS.eth);
</script>
</body>
</html>
'''


def render_asset_card(asset: str, decision: dict, chart_id: str) -> str:
    regime = decision["regime"]
    pill_class = {"BULL": "bull", "BEAR": "bear", "NEUTRAL": ""}.get(regime, "")
    side = decision["side"] or "FLAT"

    if side == "LONG":
        rec = "Build long position" if decision["position_pct"] >= 0.15 else "Starter long"
    elif side == "SHORT":
        rec = "Build short position" if decision["position_pct"] >= 0.13 else "Starter short"
    else:
        rec = "Stay flat — no entry signal"

    vs_200 = (decision["price"] / decision["ma200"] - 1) * 100
    vs_color = "#A32D2D" if vs_200 < 0 else "#0F6E56"

    return f'''
  <div class="av-card">
    <div class="av-row" style="margin-bottom:8px">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-weight:500;font-size:14px">{asset}</span>
        <span class="av-pill {pill_class}">{regime}</span>
      </div>
      <span class="av-num" style="font-size:16px;font-weight:500">${decision["price"]:,.0f}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px">
      <div><div class="av-label">Long</div><div class="av-num" style="font-size:14px;font-weight:500;margin-top:2px">{decision["long_score"]}/5</div></div>
      <div><div class="av-label">Short</div><div class="av-num" style="font-size:14px;font-weight:500;margin-top:2px">{decision["short_score"]}/5</div></div>
      <div><div class="av-label">RSI</div><div class="av-num" style="font-size:14px;font-weight:500;margin-top:2px">{decision["rsi"]:.1f}</div></div>
      <div><div class="av-label">200MA</div><div class="av-num" style="font-size:14px;font-weight:500;margin-top:2px;color:{vs_color}">{vs_200:+.1f}%</div></div>
    </div>
    <div class="tv-chart" id="{chart_id}">
      <div class="tv-legend">
        <span><span class="tv-dot" style="background:#2962FF"></span>50MA</span>
        <span><span class="tv-dot" style="background:#FF8A00"></span>200MA</span>
      </div>
    </div>
    <div class="av-row" style="font-size:10px;color:#6E6E73;margin-top:4px">
      <span>20d candles</span><span style="color:#0F0F12;font-weight:500">{rec}</span>
    </div>
  </div>'''


def render_events_card(events: list) -> str:
    if not events:
        items = '<div style="font-size:11px;color:#6E6E73">No major events in next 7 days</div>'
    else:
        items = ""
        now = datetime.now(timezone.utc)
        for i, ev in enumerate(events[:3]):
            ev_dt = datetime.strptime(ev.date, "%Y-%m-%d").replace(
                hour=18 if "FOMC" in ev.name else 13, tzinfo=timezone.utc)
            delta_h = (ev_dt - now).total_seconds() / 3600
            if delta_h < 48:
                t_str = f"{delta_h:.0f}h"
            else:
                t_str = f"{delta_h/24:.0f}d"
            sep = "border-bottom:0.5px solid #F1F1F3;margin-bottom:8px;padding-bottom:8px" if i < min(len(events)-1, 2) else ""
            advice = "Halve 4h before" if ev.impact >= 8 else "Reduce 24h before" if ev.impact >= 6 else "Watch closely"
            items += f'''
    <div style="display:flex;gap:8px;align-items:flex-start;{sep}">
      <div style="width:36px;text-align:center;flex-shrink:0">
        <div class="av-num" style="font-size:14px;font-weight:500;color:#7A4F00">{t_str}</div>
        <div style="font-size:9px;color:#6E6E73">until</div>
      </div>
      <div style="flex:1">
        <div style="font-size:12px;font-weight:500">{ev.name} <span style="color:#6E6E73;font-weight:400;font-size:11px">· {ev.impact}/10</span></div>
        <div style="font-size:10px;color:#6E6E73">{advice}</div>
      </div>
    </div>'''
    return f'''
  <div class="av-card">
    <div class="av-label" style="margin-bottom:8px">Upcoming events</div>
    {items}
  </div>'''


def render_news_card(risk) -> str:
    sent_label = "neutral" if abs(risk.sentiment_score) < 4 else ("bullish" if risk.sentiment_score > 0 else "bearish")
    items = ""
    titles = (risk.top_positive[:2] + risk.top_negative[:2])[:3] or ["No news flagged"]
    for t in titles:
        score = "+1" if t in risk.top_positive else ("-1" if t in risk.top_negative else "0")
        color = "#3B6D11" if score.startswith("+") else "#A32D2D" if score.startswith("-") else "#6E6E73"
        items += f'''
      <div style="font-size:11px;line-height:1.4;display:flex;gap:6px">
        <span style="color:{color};font-weight:500;font-size:10px;flex-shrink:0;font-variant-numeric:tabular-nums">{score}</span>
        <span>{t[:60]}</span>
      </div>'''
    return f'''
  <div class="av-card">
    <div class="av-row" style="margin-bottom:8px">
      <span class="av-label">News pulse</span>
      <span style="font-size:10px;color:#6E6E73">{risk.sentiment_score:+d} · {sent_label}</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:5px">
      {items}
    </div>
  </div>'''


def render_exec_card(risk) -> str:
    return f'''
  <div class="av-card">
    <div class="av-label" style="margin-bottom:8px">Execution metrics</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div>
        <div style="font-size:10px;color:#6E6E73">Risk severity</div>
        <div class="av-num" style="font-size:15px;font-weight:500;margin-top:1px">{risk.severity}/10</div>
        <div style="font-size:9px;color:#6E6E73">{risk.action.lower().replace("_", " ")}</div>
      </div>
      <div>
        <div style="font-size:10px;color:#6E6E73">Sentiment</div>
        <div class="av-num" style="font-size:15px;font-weight:500;margin-top:1px">{risk.sentiment_score:+d}</div>
        <div style="font-size:9px;color:#6E6E73">news flow</div>
      </div>
      <div>
        <div style="font-size:10px;color:#6E6E73">VWAP edge</div>
        <div class="av-num" style="font-size:15px;font-weight:500;margin-top:1px">+30 bp</div>
        <div style="font-size:9px;color:#6E6E73">last 24h</div>
      </div>
      <div>
        <div style="font-size:10px;color:#6E6E73">Exposure</div>
        <div class="av-num" style="font-size:15px;font-weight:500;margin-top:1px">0%</div>
        <div style="font-size:9px;color:#6E6E73">flat</div>
      </div>
    </div>
  </div>'''


def render_dashboard(payload: dict, github_repo: str = "yourname/astroinvest") -> str:
    btc = payload["btc"]
    eth = payload["eth"]
    risk = payload["risk"]
    final = compute_final_action(btc, eth, risk)

    if final["action"] == "FLAT":
        headline = "FLAT — do not trade"
        subtitle = "strategic + event override"
        detail = "Both BTC and ETH show no entry signal · funding-rate carry is the only viable position"
    elif final["action"] == "EXIT_ALL":
        headline = "EXIT ALL — risk override"
        subtitle = f"event severity {risk.severity}/10"
        detail = "Close all positions before market reaction"
    elif final["action"] == "PAUSE":
        headline = "PAUSE — no new entries"
        subtitle = f"event severity {risk.severity}/10"
        detail = "Hold existing positions but no new entries until risk clears"
    else:
        sides = []
        if btc["side"]: sides.append(f"BTC {btc['side']} {btc['position_pct']:.0%}")
        if eth["side"]: sides.append(f"ETH {eth['side']} {eth['position_pct']:.0%}")
        headline = "TRADE — " + " · ".join(sides)
        subtitle = "strategic confirmed"
        detail = "Execute through intraday slicing engine; obey stop-loss levels"

    regime_pill = f"BTC {btc['regime']} · ETH {eth['regime']}"

    next_ev = payload["events"][0] if payload["events"] else None
    if next_ev:
        now = datetime.now(timezone.utc)
        ev_dt = datetime.strptime(next_ev.date, "%Y-%m-%d").replace(
            hour=18 if "FOMC" in next_ev.name else 13, tzinfo=timezone.utc)
        delta_h = (ev_dt - now).total_seconds() / 3600
        if delta_h < 48:
            t_str = f"{delta_h:.0f}h"
        else:
            t_str = f"{delta_h/24:.0f}d"
        event_pill = f'<span class="av-pill warn">{next_ev.name.split()[1] if " " in next_ev.name else next_ev.name} in {t_str}</span>'
    else:
        event_pill = ""

    btc_chart = build_chart_data(payload["btc_ohlc"], lookback=20)
    btc_chart["signals"] = filter_signals_for_chart(payload["btc_signals"], lookback=20)
    eth_chart = build_chart_data(payload["eth_ohlc"], lookback=20)
    eth_chart["signals"] = filter_signals_for_chart(payload["eth_signals"], lookback=20)
    chart_data = {"btc": btc_chart, "eth": eth_chart}

    asset_cards = (
        render_asset_card("BTC", btc, "btc-chart") +
        render_asset_card("ETH", eth, "eth-chart")
    )

    return HTML_TEMPLATE.format(
        date_str=payload["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
        gen_time=payload["timestamp"].strftime("%H:%M UTC"),
        data_source=payload["data_source"],
        regime_pill=regime_pill,
        event_pill=event_pill,
        action_headline=headline,
        action_subtitle=subtitle,
        action_detail=detail,
        asset_cards=asset_cards,
        events_card=render_events_card(payload["events"]),
        news_card=render_news_card(risk),
        exec_card=render_exec_card(risk),
        chart_data_json=json.dumps(chart_data),
        github_repo=github_repo,
    )


# ======================================================================
# 5. MAIN
# ======================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-demo", action="store_true", help="Force calibrated demo data")
    p.add_argument("--output", default="./dist/index.html")
    p.add_argument("--github-repo", default=os.environ.get("GITHUB_REPO", "yourname/astroinvest"))
    args = p.parse_args()

    print(f"[generate_dashboard] Gathering data...")
    payload = gather_data(use_demo=args.use_demo)
    print(f"[generate_dashboard] Data source: {payload['data_source']}")
    print(f"[generate_dashboard] BTC: {payload['btc']['regime']} · {payload['btc']['action']}")
    print(f"[generate_dashboard] ETH: {payload['eth']['regime']} · {payload['eth']['action']}")
    print(f"[generate_dashboard] Risk: {payload['risk'].severity}/10 · {payload['risk'].action}")

    html = render_dashboard(payload, github_repo=args.github_repo)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[generate_dashboard] Wrote {out_path} ({len(html):,} bytes)")

    # Also write today_plan.json for the JSON button
    plan = {
        "timestamp": payload["timestamp"].isoformat(),
        "btc": {k: (v if not isinstance(v, (pd.Timestamp,)) else str(v))
                for k, v in payload["btc"].items() if k != "date"},
        "eth": {k: (v if not isinstance(v, (pd.Timestamp,)) else str(v))
                for k, v in payload["eth"].items() if k != "date"},
        "risk": {
            "severity": payload["risk"].severity,
            "action": payload["risk"].action,
            "sentiment": payload["risk"].sentiment_score,
        },
        "events": [{"date": e.date, "name": e.name, "impact": e.impact}
                   for e in payload["events"]],
    }
    json_path = out_path.parent / "today_plan.json"
    json_path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    print(f"[generate_dashboard] Wrote {json_path}")


if __name__ == "__main__":
    main()
