"""
external_signals.py  (v3 — bitbo.io primary, Farside fallback)
===================
Fetches non-technical signals missing from the technical scanner.

CHANGES vs v2:
  - PRIMARY ETF source switched to bitbo.io (BitcoinTreasuries.com).
    Farside started returning 403 to scraping; bitbo serves the same data
    without anti-bot blocking and in a cleaner HTML table format.
  - Falls back to Farside if bitbo fails.
  - Caches last successful ETF fetch to ./reports/etf_cache.json so dashboard
    can show "yesterday's data" if today's fetch fails completely.

Sources (all free, no API key):
  1. BTC spot ETF flows: bitbo.io / Farside (HTML scrape)
  2. Funding rate:       OKX public API
  3. Fear & Greed:       alternative.me public API
  4. Macro DXY+10Y:      FRED API (free, needs key in FRED_API_KEY env var)
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [_http_get] failed for {url[:60]}...: {e}")
        return None


# ============================================================
# 1A. BTC SPOT ETF FLOWS — PRIMARY: bitbo.io
# ============================================================
# bitbo.io renders a clean HTML table. Columns:
#   Date | IBIT | FBTC | GBTC | BTC | ARKB | BITB | HODL | BTCO | BRRR |
#   EZBC | MSBT | BTCW | DEFI | Totals
# Date format: "MMM DD, YYYY"  (e.g. "May 07, 2026")
# Last numeric cell per row = daily total (in $M)

def _fetch_etf_bitbo() -> list:
    """Returns [(datetime, total_musd), ...] newest first, or [] on failure."""
    raw = _http_get("https://bitbo.io/treasuries/etf-flows/")
    if not raw:
        return []

    # Pull every row that starts with a date cell
    rows = re.findall(
        r"<tr[^>]*>\s*<td[^>]*>(\w{3}\s+\d{1,2},\s+\d{4})</td>(.*?)</tr>",
        raw, re.DOTALL | re.IGNORECASE
    )
    out = []
    for date_str, row_html in rows:
        nums = re.findall(r"<td[^>]*>(-?[\d,]+\.?\d*)</td>", row_html)
        if not nums:
            continue
        try:
            total = float(nums[-1].replace(",", ""))
            dt = datetime.strptime(date_str.strip(), "%b %d, %Y")
            out.append((dt, total))
        except (ValueError, IndexError):
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# ============================================================
# 1B. FALLBACK: Farside (kept in case bitbo breaks)
# ============================================================
def _fetch_etf_farside() -> list:
    raw = _http_get("https://farside.co.uk/bitcoin-etf-flow-all-data/")
    if not raw:
        return []
    rows = re.findall(
        r"<tr[^>]*>\s*<td[^>]*>(\d{1,2}\s+\w{3}\s+\d{4})</td>(.*?)</tr>",
        raw, re.DOTALL
    )
    out = []
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
        try:
            dt = datetime.strptime(date_str.strip(), "%d %b %Y")
            out.append((dt, cleaned[-1]))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# ============================================================
# Cache helpers
# ============================================================
_CACHE_PATH = "./reports/etf_cache.json"

def _save_cache(daily_records: list):
    """daily_records: [(date_str, flow_musd), ...] oldest first"""
    try:
        os.makedirs("./reports", exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump({"daily_flows": daily_records,
                       "cached_at": datetime.now(timezone.utc).isoformat()}, f)
    except Exception:
        pass


def _load_cache() -> Optional[dict]:
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def fetch_etf_flows(days_history: int = 14) -> dict:
    """
    Try bitbo first, fall back to Farside, fall back to cache.
    Returns dict with daily_flows (oldest-first for chart), 7d sum, score, etc.
    """
    sources_tried = []

    # Try bitbo first
    parsed = _fetch_etf_bitbo()
    if parsed:
        sources_tried.append("bitbo.io ✓")
    else:
        sources_tried.append("bitbo.io ✗")
        # Try Farside
        parsed = _fetch_etf_farside()
        if parsed:
            sources_tried.append("farside ✓")
        else:
            sources_tried.append("farside ✗")

    if not parsed:
        # Final fallback: use cached data if available
        cache = _load_cache()
        if cache and cache.get("daily_flows"):
            print(f"  [etf] live sources failed ({', '.join(sources_tried)}), using cache from {cache.get('cached_at')}")
            daily = cache["daily_flows"]
            # Recompute from cache
            sum_7 = sum(f for _, f in daily[-7:])
            last_day = daily[-1][1] if daily else 0
            days_pos_7 = sum(1 for _, f in daily[-7:] if f > 0)
            score = _score_from_sum(sum_7)
            return {
                "daily_flows": daily,
                "last_day_flow_musd": round(last_day, 1),
                "sum_7d_flow_musd": round(sum_7, 1),
                "sum_history_musd": round(sum(f for _, f in daily), 1),
                "days_positive_7d": days_pos_7,
                "score": score,
                "as_of": daily[-1][0] if daily else None,
                "source": f"cache ({cache.get('cached_at', '')[:10]})",
            }
        return {"error": f"all sources failed ({', '.join(sources_tried)})",
                "score": 0.0, "daily_flows": []}

    # Process live data
    last_n = parsed[:days_history]
    last_7 = parsed[:7]
    last_day = last_n[0][1] if last_n else 0
    sum_7 = sum(f for _, f in last_7)
    sum_n = sum(f for _, f in last_n)
    days_pos_7 = sum(1 for _, f in last_7 if f > 0)
    score = _score_from_sum(sum_7)

    daily_flows = [(d.strftime("%Y-%m-%d"), round(f, 1))
                   for d, f in reversed(last_n)]   # oldest-first for chart

    # Save cache (last 30 days)
    cache_records = [(d.strftime("%Y-%m-%d"), round(f, 1))
                     for d, f in reversed(parsed[:30])]
    _save_cache(cache_records)

    return {
        "daily_flows": daily_flows,
        "last_day_flow_musd": round(last_day, 1),
        "sum_7d_flow_musd": round(sum_7, 1),
        "sum_history_musd": round(sum_n, 1),
        "days_positive_7d": days_pos_7,
        "score": score,
        "as_of": last_n[0][0].strftime("%Y-%m-%d") if last_n else None,
        "source": sources_tried[-1].split()[0],  # "bitbo.io" or "farside"
    }


def _score_from_sum(sum_7: float) -> float:
    if   sum_7 > 1500:  return +1.0
    elif sum_7 > 500:   return +0.5
    elif sum_7 > 100:   return +0.25
    elif sum_7 < -1500: return -1.0
    elif sum_7 < -500:  return -0.5
    elif sum_7 < -100:  return -0.25
    return 0.0


# ============================================================
# 2. FUNDING RATE (OKX)
# ============================================================
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


# ============================================================
# 3. FEAR & GREED INDEX
# ============================================================
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


# ============================================================
# 4. MACRO (FRED)
# ============================================================
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


# ============================================================
# AGGREGATE
# ============================================================
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
    print("[external] Fetching ETF flows (bitbo primary)...")
    etf = fetch_etf_flows(days_history=14)
    if "error" in etf:
        print(f"  ETF: ERROR — {etf['error']}")
    else:
        print(f"  ETF source: {etf.get('source','?')}  7d: ${etf.get('sum_7d_flow_musd', 0):+,.0f}M  "
              f"score: {etf.get('score', 0):+.2f}")
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
