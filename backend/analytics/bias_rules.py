from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Bias detection is RULE-BASED (fast, deterministic) for hackathon speed.
# The challenge asks for at least:
# - Overtrading
# - Loss Aversion
# - Revenge Trading
# We compute a score in [0, 100] + explanations + actionable tips.
# ---------------------------------------------------------------------

@dataclass
class BiasResult:
    score: float
    level: str
    signals: List[Dict[str, Any]]
    tips: List[str]

def _clamp(v: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, v)))

def _level(score: float) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"

def _zscore(x: pd.Series) -> pd.Series:
    if x.std(ddof=0) == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / x.std(ddof=0)

# --------------------------
# Core feature engineering
# --------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns used by multiple rules."""
    out = df.copy()
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.floor("H")
    out["is_win"] = out["pnl"] > 0
    out["abs_pnl"] = out["pnl"].abs()
    out["notional"] = out["quantity"].abs() * out["entry_price"].abs()
    # risk proxy: notional relative to balance (avoid divide by zero)
    out["risk_pct_balance"] = out["notional"] / out["balance"].replace(0, np.nan)
    out["risk_pct_balance"] = out["risk_pct_balance"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # streaks: loss streak length before each trade
    loss = (~out["is_win"]).astype(int)
    streak = []
    cur = 0
    for v in loss.tolist():
        if v == 1:
            cur += 1
        else:
            cur = 0
        streak.append(cur)
    out["loss_streak"] = streak
    return out

# --------------------------
# 1) OVERTRADING
# --------------------------
def detect_overtrading(df: pd.DataFrame) -> BiasResult:
    """Overtrading signals:
    - Trades per day relative to typical (baseline)
    - Time clustering: too many trades in 1 hour
    - Switching: rapid alternation of asset/side
    - Trading after large win/loss (emotional chasing)
    """
    if len(df) < 5:
        return BiasResult(score=0, level="LOW", signals=[{"msg": "Not enough trades to assess overtrading reliably."}], tips=[])

    d = add_features(df)

    # trades/day
    trades_by_day = d.groupby("date").size()
    mean_tpd = trades_by_day.mean()
    p90_tpd = trades_by_day.quantile(0.90)

    # clustering per hour
    trades_by_hour = d.groupby("hour").size()
    max_tph = trades_by_hour.max()
    p90_tph = trades_by_hour.quantile(0.90)

    # switching rate: fraction of consecutive trades changing asset OR side within short time
    d2 = d.sort_values("timestamp").reset_index(drop=True)
    change_asset = (d2["asset"].shift(1) != d2["asset"]).fillna(False)
    change_side = (d2["side"].shift(1) != d2["side"]).fillna(False)
    dt_minutes = (d2["timestamp"] - d2["timestamp"].shift(1)).dt.total_seconds().fillna(0) / 60.0
    rapid = dt_minutes <= 15
    switching_rate = float(((change_asset | change_side) & rapid).mean())

    # emotional follow-up: trade opened soon after big P/L move (|pnl| above 1.5 std)
    z_abs = _zscore(d2["abs_pnl"])
    big_move = z_abs > 1.5
    after_big = (big_move.shift(1).fillna(False)) & (dt_minutes <= 30)
    after_big_rate = float(after_big.mean())

    # Score composition (weights chosen for interpretability)
    # Scale trades/day and max trades/hour vs robust thresholds.
    # Absolute refs tuned to the provided hackathon datasets:
    # - 'calm' ~1.2k trades/day
    # - 'overtrader' ~4.9k trades/day
    TPD_REF = 2500.0  # trades/day reference where score starts saturating
    TPH_REF = 200.0   # trades/hour reference where score starts saturating
    tpd_score = _clamp((mean_tpd / TPD_REF) * 60.0)   # 0..60
    tph_score = _clamp((max_tph / TPH_REF) * 20.0)   # 0..20
    switch_score = _clamp(switching_rate * 100.0 * 0.15)       # 0..15
    chase_score = _clamp(after_big_rate * 100.0 * 0.10)        # 0..10

    score = _clamp(tpd_score + tph_score + switch_score + chase_score)

    signals = [
        {"metric": "avg_trades_per_day", "value": round(float(mean_tpd), 2)},
        {"metric": "90p_trades_per_day", "value": round(float(p90_tpd), 2)},
        {"metric": "max_trades_in_1_hour", "value": int(max_tph)},
        {"metric": "switching_rate_<=15min", "value": round(switching_rate, 3)},
        {"metric": "trade_after_big_move_<=30min", "value": round(after_big_rate, 3)},
    ]

    tips = [
        "Set a daily trade limit (e.g., max 5–10) and stop when reached.",
        "Add a 10–15 minute cooldown after any large win/loss before placing a new trade.",
        "Pre-define entry criteria; avoid switching symbols/side impulsively within minutes.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

# --------------------------
# 2) LOSS AVERSION
# --------------------------
def detect_loss_aversion(df: pd.DataFrame) -> BiasResult:
    """Loss aversion signals:
    - Average loss magnitude > average win magnitude
    - Win rate vs payoff ratio imbalance
    - Winners closed early / losers held longer (proxy using trade duration if available)
      NOTE: We don't always have explicit duration; we approximate using time between trades per asset.
    """
    if len(df) < 5:
        return BiasResult(score=0, level="LOW", signals=[{"msg": "Not enough trades to assess loss aversion reliably."}], tips=[])

    d = add_features(df)

    wins = d[d["is_win"]]
    losses = d[~d["is_win"]]

    avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0  # negative
    avg_loss_abs = abs(avg_loss)

    win_rate = float(d["is_win"].mean())
    payoff = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else 0.0  # reward-to-risk proxy
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and abs(losses["pnl"].sum()) > 0 else 0.0

    # approximate holding time: time delta between consecutive trades on same asset
    d2 = d.sort_values(["asset", "timestamp"]).copy()
    d2["dt_asset_min"] = d2.groupby("asset")["timestamp"].diff().dt.total_seconds().fillna(0) / 60.0
    # winners closed early => winners have smaller dt than losses (proxy)
    win_dt = float(d2[d2["is_win"]]["dt_asset_min"].median()) if len(wins) else 0.0
    loss_dt = float(d2[~d2["is_win"]]["dt_asset_min"].median()) if len(losses) else 0.0

    # Rules -> score
    # 1) avg loss abs bigger than avg win
    mag_ratio = (avg_loss_abs / avg_win) if avg_win > 0 else (2.0 if avg_loss_abs > 0 else 0.0)
    mag_score = _clamp((mag_ratio - 1.0) * 35.0)  # if ratio=2 => 35
    # 2) payoff too small (<1)
    payoff_score = _clamp((1.0 - payoff) * 35.0) if payoff < 1.0 else 0.0
    # 3) winners closed earlier than losses
    dt_score = 0.0
    if loss_dt > 0 and win_dt > 0:
        dt_ratio = loss_dt / win_dt  # >1 means losses held longer
        dt_score = _clamp((dt_ratio - 1.0) * 20.0)
    # 4) profit factor low
    pf_score = _clamp((1.2 - profit_factor) * 20.0) if profit_factor < 1.2 else 0.0

    score = _clamp(mag_score + payoff_score + dt_score + pf_score)

    signals = [
        {"metric": "win_rate", "value": round(win_rate, 3)},
        {"metric": "avg_win", "value": round(avg_win, 2)},
        {"metric": "avg_loss", "value": round(avg_loss, 2)},
        {"metric": "reward_to_risk_payoff", "value": round(payoff, 3)},
        {"metric": "profit_factor", "value": round(profit_factor, 3)},
        {"metric": "median_dt_win_min", "value": round(win_dt, 1)},
        {"metric": "median_dt_loss_min", "value": round(loss_dt, 1)},
    ]

    tips = [
        "Define a stop-loss before entering; respect it (no moving it away).",
        "Use a minimum reward:risk rule (e.g., only take trades with >= 1.5R potential).",
        "Consider partial take-profit instead of closing winners too early.",
        "Journal: after each loss, write what rule was violated (if any) and what to change.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

# --------------------------
# 3) REVENGE TRADING
# --------------------------
def detect_revenge_trading(df: pd.DataFrame) -> BiasResult:
    """Revenge trading signals:
    - Notional/risk increases immediately after a loss
    - Larger trade sizes after loss streaks
    - Short cooldown after losses
    """
    if len(df) < 8:
        return BiasResult(score=0, level="LOW", signals=[{"msg": "Not enough trades to assess revenge trading reliably."}], tips=[])

    d = add_features(df).sort_values("timestamp").reset_index(drop=True)

    prev_pnl = d["pnl"].shift(1)
    after_loss = prev_pnl < 0

    # risk increase after loss
    risk = d["risk_pct_balance"]
    risk_after_loss = float(risk[after_loss].mean()) if after_loss.any() else 0.0
    risk_after_win = float(risk[~after_loss].mean()) if (~after_loss).any() else 0.0
    risk_ratio = (risk_after_loss / risk_after_win) if risk_after_win > 0 else (1.0 if risk_after_loss > 0 else 0.0)

    # size (notional) increase after loss streak >=2
    streak_ge2 = d["loss_streak"].shift(1).fillna(0) >= 2
    notional = d["notional"]
    notional_streak = float(notional[streak_ge2].mean()) if streak_ge2.any() else 0.0
    notional_baseline = float(notional[~streak_ge2].mean()) if (~streak_ge2).any() else 0.0
    notional_ratio = (notional_streak / notional_baseline) if notional_baseline > 0 else (1.0 if notional_streak > 0 else 0.0)

    # cooldown after loss (minutes)
    dt_minutes = (d["timestamp"] - d["timestamp"].shift(1)).dt.total_seconds().fillna(0) / 60.0
    cooldown_after_loss = float(dt_minutes[after_loss].median()) if after_loss.any() else 0.0
    too_fast_rate = float((after_loss & (dt_minutes <= 10)).mean())

    # Score
    risk_score = _clamp((risk_ratio - 1.0) * 45.0)  # ratio 2 => 45
    size_score = _clamp((notional_ratio - 1.0) * 35.0)
    fast_score = _clamp(too_fast_rate * 100.0 * 0.30)  # up to 30

    score = _clamp(risk_score + size_score + fast_score)

    signals = [
        {"metric": "risk_after_loss_avg", "value": round(risk_after_loss, 4)},
        {"metric": "risk_after_nonloss_avg", "value": round(risk_after_win, 4)},
        {"metric": "risk_ratio_after_loss", "value": round(risk_ratio, 3)},
        {"metric": "notional_ratio_after_loss_streak>=2", "value": round(notional_ratio, 3)},
        {"metric": "median_cooldown_after_loss_min", "value": round(cooldown_after_loss, 1)},
        {"metric": "after_loss_trade_within_10min_rate", "value": round(too_fast_rate, 3)},
    ]

    tips = [
        "Enforce a mandatory cooling-off period after a loss (e.g., 15–30 minutes).",
        "Cap position size after a losing streak (e.g., half-size until next win).",
        "Use a checklist: if you can't state the setup in one sentence, do not trade.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

# --------------------------
# Aggregate
# --------------------------
def analyze_biases(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-serializable analysis payload."""
    over = detect_overtrading(df)
    loss = detect_loss_aversion(df)
    rev = detect_revenge_trading(df)

    overall = _clamp((over.score*0.35 + loss.score*0.35 + rev.score*0.30))

    return {
        "overall": {"score": round(overall, 1), "level": _level(overall)},
        "biases": {
            "overtrading": over.__dict__,
            "loss_aversion": loss.__dict__,
            "revenge_trading": rev.__dict__,
        },
        "meta": {
            "n_trades": int(len(df)),
            "start": df["timestamp"].min().isoformat() if len(df) else None,
            "end": df["timestamp"].max().isoformat() if len(df) else None,
        }
    }