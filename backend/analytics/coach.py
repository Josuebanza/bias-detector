"""Tiny 'coach' that turns scores into a short message.

No LLM needed (rules only) -> fast + deterministic for hackathon.
"""

def coaching_message(analysis: dict) -> str:
    overall = analysis.get("overall", {})
    score = overall.get("score", 0)
    level = overall.get("level", "LOW")

    if level == "HIGH":
        return (
            f"Your bias risk is HIGH (score {score}). Focus on discipline today: "
            "limit trades, avoid chasing losses, and follow pre-defined exits."
        )
    if level == "MEDIUM":
        return (
            f"Your bias risk is MEDIUM (score {score}). You’re close—use a checklist "
            "and take short breaks after wins/losses."
        )
    return f"Your bias risk is LOW (score {score}). Keep your process consistent and journal key decisions."
