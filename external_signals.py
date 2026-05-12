"""
external_signals.py  (v2 — daily flow history for charting)
===================
Fetches non-technical signals missing from the technical scanner.

CHANGES vs v1:
  - fetch_etf_flows now returns a `daily_flows` array of (date, flow_musd) tuples
    for the last N days, enabling frontend bar-chart visualization.

Sources (all free):
  1. BTC spot ETF net flows  -- Farside Investors (HTML scrape)
  2. Perpetual funding rate  -- OKX public API
  3. Crypto Fear & Greed     -- alternative.me public API
  4. Macro (DXY + 10Y)       -- FRED API (free with key)
"""
from __future__ import annotations
import os, re, json
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional


def _http_get(url: str, timeout: float = 10.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; quant_crypto/2.0)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [_http_get] failed for {url[:60]}...: {e}")
        return None


def fetch_etf_flows(days_history: int = 14) -> dict:
    """
    Returns {
        'daily_flows': [(YYYY-MM-DD, flow_musd), ...]  # oldest-first
        'last_day_flow_musd': float,
        'sum_7d_flow_musd':   float,
        'sum_history_musd':   float,
        'days_positive_7d':   int,
        'score':              float,   # -1..+1
        'as_of':              str,
    }
    """
    raw = _http_get("https://farside.co.uk/bitcoin-etf-flow-all-data/")
    if not raw:
        return {"error": "fetch failed", "score": 0.0, "daily_flows": []}

    rows = re.findall(
        r"<tr[^>]*>\s*<td[^>]*>(\d{1,2}\s+\w{3}\s+\d{4})</td>(.*?)</tr>",
        raw, re.DOTALL
    )
    parsed = []
    for date_str, row_html in rows:
        cells = re.findall(r"<td[^>]*>([^<]*)</td>", row_html)
        cleaned = []
        for c in cells:
            c = c.strip().replace(",", "")
            if c in ("", "-", "—"):
                cleaned.append(0.0)
            elif c.startswith("(") and c.endswith(")"):
                try: cleaned.append(-float(c[1:-1]))
                except ValueError: pass
            else:
                try: cleaned.append(float(c))
                except ValueError: pass
        if not cleaned:
            continue
        total = cleaned[-1]
        try:
            dt = datetime.strptime(date_str.strip(), "%d %b %Y")
            parsed.append((dt, total))
        except ValueError:
            continue

    if not parsed:
        return {"error": "no data parsed", "score": 0.0, "daily_flows": []}

    parsed.sort(key=lambda x: x[0], reverse=True)
    last_n = parsed[:days_history]
    last_7 = parsed[:7]

    last_day = last_n[0][1] if last_n else 0
    sum_7   = sum(f for _, f in last_7)
    sum_n   = sum(f for _, f in last_n)
    days_pos_7 = sum(1 for _, f in last_7 if f > 0)

    score = 0.0
    if   sum_7 > 1500:  score = +1.0
    elif sum_7 > 500:   score = +0.5
    elif sum_7 > 100:   score = +0.25
    elif sum_7 < -1500: score = -1.0
    elif sum_7 < -500:  score = -0.5
    elif sum_7 < -100:  score = -0.25

    daily_flows = [(d.strftime("%Y-%m-%d"), round(f, 1)) for d, f in reversed(last_n)]

    return {
        "daily_flows": daily_flows,
        "last_day_flow_musd": round(last_day, 1),
        "sum_7d_flow_musd": round(sum_7, 1),
        "sum_history_musd": round(sum_n, 1),
        "days_positive_7d": days_pos_7,
        "score": score,
        "as_of": last_n[0][0].strftime("%Y-%m-%d") if last_n else None,
    }


def fetch_funding_rate(inst_id: str = "BTC-USDT-SWAP") -> dict:
    raw = _http_get(f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}")
    if not raw:
        return {"error": "fetch failed", "score": 0.0}
    try:
        data = json.loads(raw)
        if data.get("code") != "0" or not data.get("data"):
            return {"error": f"OKX: {data.get('msg')}", "score": 0.0}
        d = data["data"][0]
        rate_8h = float(d["fundingRate"])
        rate_pct = rate_8h * 100
        annualized = rate_8h * 3 * 365 * 100
        score = 0.0
        if rate_pct > 0.10:    score = -1.0
        elif rate_pct > 0.05:  score = -0.5
        elif rate_pct < -0.10: score = +1.0
        elif rate_pct < -0.05: score = +0.5
        return {"rate_8h_pct": round(rate_pct, 4),
                "annualized_pct": round(annualized, 1),
                "score": score,
                "as_of": datetime.fromtimestamp(int(d["fundingTime"])/1000,
                                                tz=timezone.utc).isoformat()}
    except Exception as e:
        return {"error": str(e), "score": 0.0}


def fetch_fear_greed() -> dict:
    raw = _http_get("https://api.alternative.me/fng/?limit=1")
    if not raw:
        return {"error": "fetch failed", "score": 0.0}
    try:
        data = json.loads(raw)
        d = data["data"][0]
        v = int(d["value"])
        label = d["value_classification"]
        if v <= 20:   score = +1.0
        elif v <= 35: score = +0.5
        elif v >= 80: score = -1.0
        elif v >= 65: score = -0.5
        else:         score = 0.0
        return {"value": v, "label": label, "score": score,
                "as_of": datetime.fromtimestamp(int(d["timestamp"]),
                                                tz=timezone.utc).isoformat()}
    except Exception as e:
        return {"error": str(e), "score": 0.0}


def fetch_macro(fred_api_key: Optional[str] = None) -> dict:
    if not fred_api_key:
        fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        return {"error": "no FRED_API_KEY", "score": 0.0}

    def fetch_series(sid, days=20):
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
               f"&api_key={fred_api_key}&file_type=json&sort_order=desc&limit={days}")
        raw = _http_get(url)
        if not raw: return None
        try:
            data = json.loads(raw)
            return [(o["date"], float(o["value"])) for o in data["observations"]
                    if o["value"] not in (".", "")]
        except Exception:
            return None

    dxy = fetch_series("DTWEXBGS", 20)
    y10 = fetch_series("DGS10", 20)
    if not dxy or not y10:
        return {"error": "FRED fetch failed", "score": 0.0}

    dxy_chg = (dxy[0][1] / dxy[5][1] - 1) * 100 if len(dxy) > 5 else 0
    y10_chg_bps = (y10[0][1] - y10[5][1]) * 100 if len(y10) > 5 else 0
    score = 0.0
    if dxy_chg < -1.5 and y10_chg_bps < -20: score = +1.0
    elif dxy_chg < -0.5 or y10_chg_bps < -10: score = +0.5
    elif dxy_chg > 1.5 and y10_chg_bps > 20:  score = -1.0
    elif dxy_chg > 0.5 or y10_chg_bps > 10:   score = -0.5

    return {"dxy_chg_5d": round(dxy_chg, 2),
            "yield_10y_chg_5d_bps": round(y10_chg_bps, 1),
            "score": score}


@dataclass
class MacroBias:
    timestamp: str
    etf: dict
    funding: dict
    fng: dict
    macro: dict
    total_score: float
    bias_direction: str
    confidence: str


def assess_macro_bias() -> MacroBias:
    print("[external] Fetching ETF flows...")
    etf = fetch_etf_flows(days_history=14)
    print(f"  ETF 7d: ${etf.get('sum_7d_flow_musd', 'N/A')}M  score: {etf.get('score', 0):+.2f}")
    print("[external] Fetching funding rate...")
    fr = fetch_funding_rate()
    print(f"  Funding: {fr.get('annualized_pct', 'N/A')}%/yr  score: {fr.get('score', 0):+.2f}")
    print("[external] Fetching Fear & Greed...")
    fng = fetch_fear_greed()
    print(f"  F&G: {fng.get('value', 'N/A')} ({fng.get('label', '')})  score: {fng.get('score', 0):+.2f}")
    print("[external] Fetching macro...")
    macro = fetch_macro()
    print(f"  Macro score: {macro.get('score', 0):+.2f}")

    total = sum(x.get("score", 0) for x in [etf, fr, fng, macro])
    direction = "BULLISH" if total >= 1.5 else "BEARISH" if total <= -1.5 else "NEUTRAL"
    conf = "STRONG" if abs(total) >= 2.0 else "MODERATE" if abs(total) >= 1.0 else "WEAK"
    return MacroBias(timestamp=datetime.now(timezone.utc).isoformat(),
                     etf=etf, funding=fr, fng=fng, macro=macro,
                     total_score=round(total, 2),
                     bias_direction=direction, confidence=conf)


def main():
    bias = assess_macro_bias()
    print(f"\n  >>> {bias.bias_direction} ({bias.confidence})  total: {bias.total_score:+.2f}\n")
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/external_signals.json", "w") as f:
        json.dump({"timestamp": bias.timestamp, "total_score": bias.total_score,
                   "bias_direction": bias.bias_direction, "confidence": bias.confidence,
                   "etf": bias.etf, "funding": bias.funding,
                   "fng": bias.fng, "macro": bias.macro}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
