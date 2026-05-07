"""
news_monitor.py
===============
Real-time news + macro event monitor for crypto trading.

DESIGN PHILOSOPHY:
  This is a RISK MANAGEMENT tool, not an alpha generator.
  - Pre-event: warn about upcoming macro events (FOMC, CPI, NFP)
  - Real-time: flag breaking news with high impact
  - Post-event: highlight regime-changing items for review

WHY: News -> price relationship has 5-30 min half-life. Retail traders
can't beat HFT firms reacting to news in milliseconds. But retail CAN
avoid catastrophic events by NOT trading into them.

DATA SOURCES (all free / public):
  1. CryptoPanic public API (no key required for basic, free tier available)
  2. Macro calendar: hardcoded FOMC/CPI dates (publicly announced years ahead)
  3. RSS fallback: CoinDesk, CoinTelegraph, The Block
  4. Built-in keyword sentiment scoring (no external sentiment API needed)

OUTPUT:
  - Severity score 0-10 for current market state
  - Action recommendation: TRADE_NORMAL / REDUCE_SIZE / PAUSE_ENTRY / EXIT_ALL
  - Upcoming events in next 7 days
"""

from __future__ import annotations
import os
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


# ======================================================================
# 1. MACRO EVENT CALENDAR
# ======================================================================
# All FOMC meetings are scheduled by the Fed for the entire year.
# CPI/NFP/PCE dates follow standard BLS/BEA release schedules.
# This list should be updated annually from Fed calendar:
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

FOMC_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

CPI_2026 = [
    "2026-01-13", "2026-02-12", "2026-03-12", "2026-04-10",
    "2026-05-13", "2026-06-11", "2026-07-15", "2026-08-12",
    "2026-09-11", "2026-10-15", "2026-11-13", "2026-12-10",
]

NFP_2026 = [  # Non-Farm Payrolls (first Friday of month, mostly)
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]


@dataclass
class MacroEvent:
    date: str          # YYYY-MM-DD
    name: str
    impact: int        # 1-10, where 10 = highest impact
    notes: str = ""

MACRO_CALENDAR = (
    [MacroEvent(d, "FOMC Rate Decision",  10, "Largest single-event mover for BTC") for d in FOMC_2026] +
    [MacroEvent(d, "US CPI Release",       9, "Inflation data; Fed reaction proxy") for d in CPI_2026] +
    [MacroEvent(d, "US NFP",               7, "Employment; secondary Fed input") for d in NFP_2026]
)


def upcoming_events(days_ahead: int = 7) -> list[MacroEvent]:
    """Return macro events in the next N days."""
    today = datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=days_ahead)
    out = []
    for ev in MACRO_CALENDAR:
        ev_date = datetime.strptime(ev.date, "%Y-%m-%d").date()
        if today <= ev_date <= horizon:
            out.append(ev)
    return sorted(out, key=lambda e: e.date)


def hours_to_next_event() -> tuple[Optional[MacroEvent], Optional[float]]:
    """Returns (next_event, hours_until_event) or (None, None) if no event in 14 days."""
    now = datetime.now(timezone.utc)
    for ev in MACRO_CALENDAR:
        # Assume FOMC at 18:00 UTC, CPI/NFP at 12:30 UTC (release standard times)
        hour = 18 if "FOMC" in ev.name else 13
        ev_dt = datetime.strptime(ev.date, "%Y-%m-%d").replace(
            hour=hour, tzinfo=timezone.utc)
        delta_h = (ev_dt - now).total_seconds() / 3600
        if 0 < delta_h < 24 * 14:
            return ev, delta_h
    return None, None


# ======================================================================
# 2. NEWS FETCHING (CryptoPanic public + RSS fallback)
# ======================================================================

CRYPTOPANIC_PUBLIC_URL = "https://cryptopanic.com/api/v1/posts/"
RSS_SOURCES = {
    "CoinDesk":     "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "TheBlock":     "https://www.theblock.co/rss.xml",
    "Decrypt":      "https://decrypt.co/feed",
}


def _http_get(url: str, timeout: float = 8.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quant_crypto/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return None


def fetch_cryptopanic(api_key: Optional[str] = None,
                      currencies: str = "BTC,ETH",
                      filter_kind: str = "news") -> list[dict]:
    """
    CryptoPanic public API.
    Free tier: 500 requests/day, no key required for basic queries.
    Get free API key at https://cryptopanic.com/developers/api/ for higher limits.
    """
    params = f"?currencies={currencies}&filter={filter_kind}"
    if api_key:
        params = f"?auth_token={api_key}&currencies={currencies}&filter={filter_kind}"
    raw = _http_get(CRYPTOPANIC_PUBLIC_URL + params)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data.get("results", [])
    except Exception:
        return []


def fetch_rss(source_name: str, url: str) -> list[dict]:
    """Minimal RSS parser, no external deps."""
    raw = _http_get(url)
    if not raw:
        return []
    items = []
    # Simple regex-based RSS extraction (works for most feeds)
    item_blocks = re.findall(r"<item[^>]*>(.*?)</item>", raw, re.DOTALL | re.IGNORECASE)
    for block in item_blocks[:30]:
        title_m = re.search(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.DOTALL)
        date_m = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", block, re.DOTALL)
        link_m = re.search(r"<link[^>]*>(.*?)</link>", block, re.DOTALL)
        if title_m:
            items.append({
                "source": source_name,
                "title": title_m.group(1).strip(),
                "date": date_m.group(1).strip() if date_m else "",
                "url": link_m.group(1).strip() if link_m else "",
            })
    return items


# ======================================================================
# 3. KEYWORD-BASED SENTIMENT (built-in, no external API)
# ======================================================================

NEGATIVE_KEYWORDS = {
    # Strong negatives (-3 each)
    "hack": -3, "exploit": -3, "stolen": -3, "rugpull": -3, "scam": -3,
    "ban": -3, "banned": -3, "outlawed": -3, "criminal": -3,
    "bankruptcy": -3, "insolvent": -3, "collapse": -3,
    "lawsuit": -2, "sec sues": -3, "doj": -2, "indictment": -3,
    "liquidation": -2, "crash": -2, "plunge": -2, "tumble": -2,
    "regulation": -1, "regulatory": -1, "crackdown": -3,
    "fraud": -3, "manipulation": -2, "investigation": -2,
    "delisting": -2, "halted": -2, "suspended": -2,
    "selloff": -2, "dump": -2, "panic": -2, "fear": -1,
    # Macro negatives
    "rate hike": -2, "raising rates": -2, "hawkish": -2,
    "inflation rising": -2, "recession": -2,
}

POSITIVE_KEYWORDS = {
    # Strong positives (+3 each)
    "etf approved": 3, "etf approval": 3, "spot etf": 2,
    "rate cut": 2, "dovish": 2, "easing": 2,
    "institutional adoption": 2, "blackrock": 1, "fidelity": 1,
    "all-time high": 2, "ath": 1, "rally": 1, "surge": 1,
    "breakout": 1, "bullish": 1, "uptrend": 1,
    "approved": 1, "regulatory clarity": 2, "legal framework": 2,
    "partnership": 1, "integration": 1, "adoption": 1,
    "halving": 2, "upgrade": 1, "merge": 1,
}


def score_text(text: str) -> int:
    """Return sentiment score for a text. Negative = bearish, positive = bullish."""
    if not text:
        return 0
    text_lower = text.lower()
    score = 0
    for kw, w in NEGATIVE_KEYWORDS.items():
        if kw in text_lower:
            score += w
    for kw, w in POSITIVE_KEYWORDS.items():
        if kw in text_lower:
            score += w
    return score


# ======================================================================
# 4. AGGREGATED MARKET RISK SCORE
# ======================================================================

@dataclass
class NewsAssessment:
    timestamp: str
    severity: int           # 0-10
    sentiment_score: int    # negative = bearish, positive = bullish
    next_event: Optional[MacroEvent]
    hours_to_event: Optional[float]
    top_negative: list[str] = field(default_factory=list)
    top_positive: list[str] = field(default_factory=list)
    action: str = "TRADE_NORMAL"
    reason: list[str] = field(default_factory=list)


def assess_market_risk(api_key: Optional[str] = None) -> NewsAssessment:
    """
    Comprehensive risk assessment combining:
      1. Upcoming macro events (time-based)
      2. Recent news sentiment (content-based)
      3. Action recommendation
    """
    # Macro proximity
    next_ev, hours = hours_to_next_event()
    severity = 0
    reasons = []

    if next_ev and hours is not None:
        if hours < 4:
            severity += min(10, next_ev.impact)
            reasons.append(f"⚠️ {next_ev.name} in {hours:.1f}h - HALT new entries")
        elif hours < 24:
            severity += min(7, next_ev.impact - 2)
            reasons.append(f"⚠️ {next_ev.name} in {hours:.1f}h - reduce size 50%")
        elif hours < 72:
            severity += 2
            reasons.append(f"📅 {next_ev.name} in {hours:.0f}h - normal trade, keep stops tight")

    # News sentiment
    news_items = []
    cp = fetch_cryptopanic(api_key)
    for item in cp[:30]:
        title = item.get("title", "")
        s = score_text(title)
        news_items.append({"title": title, "score": s, "source": "CryptoPanic"})

    # RSS fallback if CryptoPanic returned nothing
    if not news_items:
        for src, url in RSS_SOURCES.items():
            for item in fetch_rss(src, url)[:10]:
                title = item["title"]
                s = score_text(title)
                news_items.append({"title": title, "score": s, "source": src})

    # Aggregate
    sent_total = sum(it["score"] for it in news_items)
    top_neg = sorted([it for it in news_items if it["score"] < 0],
                     key=lambda x: x["score"])[:3]
    top_pos = sorted([it for it in news_items if it["score"] > 0],
                     key=lambda x: -x["score"])[:3]

    if sent_total < -10:
        severity += 3
        reasons.append(f"📰 Strong negative news flow (score {sent_total})")
    elif sent_total < -5:
        severity += 1
        reasons.append(f"📰 Mildly negative news (score {sent_total})")
    elif sent_total > 10:
        reasons.append(f"📰 Strong positive news flow (score {sent_total}) - BEWARE crowd euphoria")
    elif sent_total > 5:
        reasons.append(f"📰 Mildly positive news (score {sent_total})")

    # Decide action
    severity = min(10, severity)
    if severity >= 8:
        action = "EXIT_ALL"
    elif severity >= 6:
        action = "PAUSE_ENTRY"
    elif severity >= 4:
        action = "REDUCE_SIZE_50"
    elif severity >= 2:
        action = "TRADE_NORMAL_WIDER_STOPS"
    else:
        action = "TRADE_NORMAL"

    return NewsAssessment(
        timestamp=datetime.now(timezone.utc).isoformat(),
        severity=severity,
        sentiment_score=sent_total,
        next_event=next_ev,
        hours_to_event=hours,
        top_negative=[it["title"] for it in top_neg],
        top_positive=[it["title"] for it in top_pos],
        action=action,
        reason=reasons,
    )


# ======================================================================
# 5. CLI
# ======================================================================

def print_assessment(a: NewsAssessment):
    print("\n" + "=" * 78)
    print(f"  NEWS & EVENT RISK ASSESSMENT   |   {a.timestamp[:19]}Z")
    print("=" * 78)
    print(f"\n  RISK SEVERITY:    {a.severity}/10   "
          f"({'■' * a.severity}{'□' * (10 - a.severity)})")
    print(f"  NEWS SENTIMENT:   {a.sentiment_score:+d}   "
          f"({'BEARISH' if a.sentiment_score < -3 else 'BULLISH' if a.sentiment_score > 3 else 'NEUTRAL'})")
    print(f"\n  >>> ACTION:       {a.action}")
    if a.reason:
        print("  REASONS:")
        for r in a.reason:
            print(f"    • {r}")

    if a.next_event:
        print(f"\n  NEXT MACRO EVENT:")
        print(f"    {a.next_event.date}  {a.next_event.name:<25}  impact: {a.next_event.impact}/10")
        print(f"    in ~{a.hours_to_event:.1f} hours")
        print(f"    {a.next_event.notes}")

    if a.top_negative:
        print(f"\n  TOP NEGATIVE NEWS:")
        for t in a.top_negative:
            print(f"    🔴 {t[:90]}")
    if a.top_positive:
        print(f"\n  TOP POSITIVE NEWS:")
        for t in a.top_positive:
            print(f"    🟢 {t[:90]}")

    # Upcoming events
    upcoming = upcoming_events(7)
    if upcoming:
        print(f"\n  EVENTS NEXT 7 DAYS:")
        for ev in upcoming:
            print(f"    {ev.date}  impact={ev.impact}/10  {ev.name}")

    print("=" * 78)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("CRYPTOPANIC_KEY"),
                   help="Optional CryptoPanic API key for higher rate limits")
    args = p.parse_args()

    a = assess_market_risk(api_key=args.api_key)
    print_assessment(a)

    # Save to JSON for downstream consumption
    out = {
        "timestamp": a.timestamp,
        "severity": a.severity,
        "sentiment_score": a.sentiment_score,
        "action": a.action,
        "reasons": a.reason,
        "top_negative": a.top_negative,
        "top_positive": a.top_positive,
    }
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/news_assessment.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> ./reports/news_assessment.json")
    print(f"  (Other systems can read this to gate trade decisions)\n")


if __name__ == "__main__":
    main()
