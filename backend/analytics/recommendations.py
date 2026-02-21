"""Recommendation text blocks used by UI and API.

These are intentionally short, non-judgmental, and actionable.
"""

from typing import Dict, List

DEFAULT_RECOMMENDATIONS: Dict[str, List[str]] = {
    "overtrading": [
        "Set a daily trade cap (e.g., 10–20 trades) and stop when you hit it.",
        "Add a 5-minute cooldown after each trade to avoid impulsive re-entries.",
        "Pre-define a setup checklist and only trade when all items are true.",
    ],
    "loss_aversion": [
        "Decide your stop-loss before entering a trade and respect it.",
        "Use a minimum risk/reward rule (e.g., R:R ≥ 1.5).",
        "Journal why you closed winners early (fear vs. plan).",
    ],
    "revenge_trading": [
        "After a loss streak, take a mandatory break (15–30 minutes).",
        "Reduce size after losses (never increase it) until you're back to baseline.",
        "Write a one-line reason before every trade: 'I trade because…'.",
    ],
    "fomo_trading": [
        "Avoid entering immediately after large green candles; wait for a pullback checklist.",
        "Limit FOMO entries to pre-defined setups and fixed risk size.",
        "If you feel urgency, delay execution by 3 minutes and re-check criteria.",
    ],
    "confirmation_bias": [
        "Before each trade, write one argument against your idea.",
        "Cap repeated same-direction entries on the same asset within a short window.",
        "Review losing streaks for pattern-fixation and broaden your scenario set.",
    ],
}

def merge_tips(bias_payload: dict) -> dict:
    """Append fallback tips if a detector returned too few."""
    for k, tips in DEFAULT_RECOMMENDATIONS.items():
        obj = bias_payload.get("biases", {}).get(k, {})
        existing = obj.get("tips", []) or []
        if len(existing) < 2:
            obj["tips"] = (existing + tips)[:6]
    return bias_payload
