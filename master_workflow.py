"""
master_workflow.py
==================
Daily institutional workflow combining all 3 decision layers:

  1. Strategic Layer (unified_scanner)  - daily direction (LONG/SHORT/FLAT)
  2. Event Layer     (news_monitor)     - event/news risk gate
  3. Tactical Layer  (intraday_executor) - slice execution plan

Run this once per day in the morning. It will:
  - Decide if you should trade today (and which side)
  - Check upcoming events that could disrupt
  - If go, generate a slice execution plan
  - If event risk too high, output PAUSE / EXIT recommendation

The script writes a single JSON to ./reports/today_plan.json that fully
encodes the day's plan. Your trade execution code (manual or automated)
reads from that JSON.
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC", choices=["BTC", "ETH"])
    p.add_argument("--account", type=float, default=100_000)
    p.add_argument("--n-slices", type=int, default=8)
    p.add_argument("--cryptopanic-key", default=os.environ.get("CRYPTOPANIC_KEY"))
    args = p.parse_args()

    print("\n" + "█" * 72)
    print(f"  DAILY MASTER WORKFLOW — {args.asset}   {datetime.now(timezone.utc).date()}")
    print("█" * 72)

    # ========== LAYER 1: STRATEGIC (daily direction) ==========
    print("\n[Layer 1] Strategic decision (daily scanner)...")
    from data.data_loader import get_prices
    from unified_scanner import make_decision as make_strategic_decision

    try:
        prices = get_prices(use_real=True)
    except Exception as e:
        print(f"  Real data unavailable: {e}")
        prices = get_prices(use_real=False)

    strategic = make_strategic_decision(prices[args.asset], account_size=args.account)
    print(f"  Regime: {strategic['regime']}")
    print(f"  Action: {strategic['action']}")
    print(f"  Side:   {strategic['side']}   Position: {strategic['position_pct']:.0%}")
    print(f"  Capital to deploy: ${strategic['capital_usd']:,.0f}")

    # ========== LAYER 2: EVENT GATE (news + macro) ==========
    print("\n[Layer 2] Event/news risk assessment...")
    from news_monitor import assess_market_risk
    risk = assess_market_risk(api_key=args.cryptopanic_key)
    print(f"  Severity: {risk.severity}/10")
    print(f"  Action:   {risk.action}")
    if risk.next_event and risk.hours_to_event is not None and risk.hours_to_event < 24:
        print(f"  ⚠️  {risk.next_event.name} in {risk.hours_to_event:.1f}h")

    # ========== DECISION GATE ==========
    print("\n[Gate] Combining strategic + event signals...")
    final_action = "TRADE"
    final_reason = []

    if strategic["side"] is None:
        final_action = "FLAT"
        final_reason.append(f"Strategic layer says: {strategic['action']}")

    if risk.action == "EXIT_ALL":
        final_action = "EXIT_ALL"
        final_reason.append(f"Event risk forces exit: severity {risk.severity}")
    elif risk.action == "PAUSE_ENTRY":
        final_action = "PAUSE"
        final_reason.append(f"Event risk: severity {risk.severity}, no new entries")
    elif risk.action == "REDUCE_SIZE_50":
        if strategic["position_pct"] > 0:
            strategic["position_pct"] *= 0.5
            strategic["capital_usd"] *= 0.5
            final_reason.append("Position halved due to event risk")

    print(f"  >>> FINAL: {final_action}")
    for r in final_reason:
        print(f"      • {r}")

    # ========== LAYER 3: TACTICAL (slice plan) ==========
    plan_summary = None
    if final_action == "TRADE" and strategic["side"]:
        print("\n[Layer 3] Tactical execution plan...")
        try:
            from intraday_executor import (
                fetch_okx_intraday, SlicePlan, plan_slices, slice_summary
            )
            inst = f"{args.asset}-USDT"
            df = fetch_okx_intraday(inst, bar="15m", hours_back=24)
            if not df.empty:
                plan = SlicePlan(
                    side=strategic["side"],
                    total_size_usd=strategic["capital_usd"],
                    n_slices=args.n_slices,
                    bar="15m",
                    aggression="balanced",
                )
                decisions = plan_slices(plan, df)
                plan_summary = slice_summary(decisions)
                print(f"  Slices: {args.n_slices}, bar: 15m, aggression: balanced")
                print(f"  Last 24h fetched: {len(df)} bars")
                if plan_summary["n_executed"] > 0:
                    print(f"  Backtested savings vs VWAP: {plan_summary['savings_vs_vwap_bps']:+.1f} bps")
                    print(f"  (Use these settings for today's live execution)")
        except Exception as e:
            print(f"  Tactical layer skipped: {e}")
            plan_summary = {"error": str(e)}
    else:
        print("\n[Layer 3] Skipped (final action is not TRADE)")

    # ========== OUTPUT ==========
    out = {
        "date": datetime.now(timezone.utc).isoformat(),
        "asset": args.asset,
        "final_action": final_action,
        "final_reason": final_reason,
        "strategic": {
            "regime": strategic["regime"],
            "action": strategic["action"],
            "side": strategic["side"],
            "position_pct": strategic["position_pct"],
            "capital_usd": strategic["capital_usd"],
            "stop_price": strategic["stop_price"],
            "long_score": strategic["long_score"],
            "short_score": strategic["short_score"],
        },
        "event_risk": {
            "severity": risk.severity,
            "action": risk.action,
            "sentiment_score": risk.sentiment_score,
            "next_event": risk.next_event.name if risk.next_event else None,
            "hours_to_event": risk.hours_to_event,
        },
        "tactical_plan": plan_summary,
    }

    os.makedirs("./reports", exist_ok=True)
    out_path = f"./reports/today_plan_{args.asset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n  Full plan saved -> {out_path}")
    print("\n" + "█" * 72)
    print(f"  END OF DAILY WORKFLOW")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
