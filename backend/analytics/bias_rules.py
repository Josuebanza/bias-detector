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
    out["hour"] = out["timestamp"].dt.floor("h")
    out["is_win"] = out["pnl"] > 0
    out["abs_pnl"] = out["pnl"].abs()
    out["notional"] = out["quantity"].abs() * out["entry_price"].abs()
    # risk proxy: notional relative to absolute balance (avoid divide by zero)
    out["risk_pct_balance"] = out["notional"] / out["balance"].abs().replace(0, np.nan)
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

    # Score composition (weights chosen for dataset separation):
    # - We prioritize trade cadence (day/hour intensity), which clearly separates
    #   calm-like behavior from true overtrading in the provided datasets.
    # - Switching/chasing are kept as small bonus signals only when clearly elevated.
    TPD_BASELINE = 1000.0  # around "normal-active" cadence
    TPH_BASELINE = 50.0    # around "normal-active" hourly burst

    tpd_ratio = (mean_tpd / TPD_BASELINE) if TPD_BASELINE > 0 else 0.0
    tph_ratio = (max_tph / TPH_BASELINE) if TPH_BASELINE > 0 else 0.0

    # Excess-only scoring: no penalty/score until baseline is exceeded.
    tpd_score = _clamp((tpd_ratio - 1.0) * 55.0, 0.0, 55.0)  # 0..55
    tph_score = _clamp((tph_ratio - 1.0) * 30.0, 0.0, 30.0)  # 0..30

    # Bonus signals only when behavior is above a high watermark.
    switch_score = _clamp((switching_rate - 0.95) * 100.0 * 0.5, 0.0, 5.0)  # 0..5
    chase_score = _clamp((after_big_rate - 0.10) * 100.0 * 0.5, 0.0, 10.0)  # 0..10

    score = _clamp(tpd_score + tph_score + switch_score + chase_score)

    signals = [
        {"metric": "avg_trades_per_day", "value": round(float(mean_tpd), 2)},
        {"metric": "90p_trades_per_day", "value": round(float(p90_tpd), 2)},
        {"metric": "max_trades_in_1_hour", "value": int(max_tph)},
        {"metric": "tpd_ratio_vs_baseline", "value": round(float(tpd_ratio), 3)},
        {"metric": "tph_ratio_vs_baseline", "value": round(float(tph_ratio), 3)},
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
    risk_after_nonloss = float(risk[~after_loss].mean()) if (~after_loss).any() else 0.0
    risk_ratio = (risk_after_loss / risk_after_nonloss) if risk_after_nonloss > 0 else (1.0 if risk_after_loss > 0 else 0.0)

    # size (notional) increase after loss streak >=2
    streak_ge2 = d["loss_streak"].shift(1).fillna(0) >= 2
    notional = d["notional"]
    notional_streak = float(notional[streak_ge2].mean()) if streak_ge2.any() else 0.0
    notional_baseline = float(notional[~streak_ge2].mean()) if (~streak_ge2).any() else 0.0
    notional_ratio = (notional_streak / notional_baseline) if notional_baseline > 0 else (1.0 if notional_streak > 0 else 0.0)

    # emotional escalation: losses become larger right after a prior loss
    abs_pnl = d["pnl"].abs()
    loss_intensity_after_loss = float(abs_pnl[after_loss].mean()) if after_loss.any() else 0.0
    loss_intensity_nonloss = float(abs_pnl[~after_loss].mean()) if (~after_loss).any() else 0.0
    loss_intensity_ratio = (
        loss_intensity_after_loss / loss_intensity_nonloss
        if loss_intensity_nonloss > 0
        else (1.0 if loss_intensity_after_loss > 0 else 0.0)
    )

    # persistence of loss sequences: next loss is as severe or worse than previous
    prev_abs_pnl = abs_pnl.shift(1)
    worse_after_loss = after_loss & (d["pnl"] < 0) & (abs_pnl >= prev_abs_pnl)
    worse_after_loss_rate = float(worse_after_loss[after_loss].mean()) if after_loss.any() else 0.0

    # cooldown after loss (minutes)
    dt_minutes = (d["timestamp"] - d["timestamp"].shift(1)).dt.total_seconds().fillna(0) / 60.0
    cooldown_after_loss = float(dt_minutes[after_loss].median()) if after_loss.any() else 0.0
    # immediate re-entry under 30s is a strong revenge marker (more than 10m in these datasets)
    ultra_fast_rate = float((after_loss & (dt_minutes <= 0.5)).mean())

    # Score
    # Calibrated to separate "revenge_trader" from calm/loss-averse datasets while
    # keeping overtrader mostly captured by overtrading instead of revenge.
    risk_score = _clamp((risk_ratio - 1.10) * 120.0, 0.0, 25.0)
    size_score = _clamp((notional_ratio - 1.10) * 100.0, 0.0, 15.0)
    escalation_score = _clamp((loss_intensity_ratio - 0.994) * 3000.0, 0.0, 45.0)
    persistence_score = _clamp((worse_after_loss_rate - 0.255) * 600.0, 0.0, 15.0)
    fast_score = _clamp(ultra_fast_rate * 100.0 * 0.25, 0.0, 10.0)

    score = _clamp(risk_score + size_score + escalation_score + persistence_score + fast_score)

    signals = [
        {"metric": "risk_after_loss_avg", "value": round(risk_after_loss, 4)},
        {"metric": "risk_after_nonloss_avg", "value": round(risk_after_nonloss, 4)},
        {"metric": "risk_ratio_after_loss", "value": round(risk_ratio, 3)},
        {"metric": "notional_ratio_after_loss_streak>=2", "value": round(notional_ratio, 3)},
        {"metric": "loss_intensity_ratio_after_loss", "value": round(loss_intensity_ratio, 3)},
        {"metric": "worse_loss_rate_after_loss", "value": round(worse_after_loss_rate, 3)},
        {"metric": "median_cooldown_after_loss_min", "value": round(cooldown_after_loss, 1)},
        {"metric": "after_loss_trade_within_30sec_rate", "value": round(ultra_fast_rate, 3)},
    ]

    tips = [
        "Enforce a mandatory cooling-off period after a loss (e.g., 15–30 minutes).",
        "Cap position size after a losing streak (e.g., half-size until next win).",
        "Use a checklist: if you can't state the setup in one sentence, do not trade.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

# --------------------------
# 4) FOMO TRADING
# --------------------------
def detect_fomo_trading(df: pd.DataFrame) -> BiasResult:
    """FOMO signals:
    - Entering quickly after large positive moves
    - Larger risk on those entries vs baseline
    - Weak quality after those entries (lower win rate)
    """
    if len(df) < 8:
        return BiasResult(score=0, level="LOW", signals=[{"msg": "Not enough trades to assess FOMO reliably."}], tips=[])

    d = add_features(df).sort_values("timestamp").reset_index(drop=True)
    dt_minutes = (d["timestamp"] - d["timestamp"].shift(1)).dt.total_seconds().fillna(0) / 60.0

    prev_pnl = d["pnl"].shift(1)
    prev_abs_z = _zscore(d["abs_pnl"]).shift(1)
    after_big_win_fast = (prev_pnl > 0) & (prev_abs_z > 1.3) & (dt_minutes <= 10)

    trigger_rate = float(after_big_win_fast.mean())

    risk = d["risk_pct_balance"]
    risk_fomo = float(risk[after_big_win_fast].mean()) if after_big_win_fast.any() else 0.0
    risk_base = float(risk[~after_big_win_fast].mean()) if (~after_big_win_fast).any() else 0.0
    risk_ratio = (risk_fomo / risk_base) if risk_base > 0 else (1.0 if risk_fomo > 0 else 0.0)

    win_rate_fomo = float((d.loc[after_big_win_fast, "pnl"] > 0).mean()) if after_big_win_fast.any() else 0.0
    win_rate_base = float((d.loc[~after_big_win_fast, "pnl"] > 0).mean()) if (~after_big_win_fast).any() else 0.0
    win_rate_gap = win_rate_base - win_rate_fomo

    trigger_score = _clamp((trigger_rate - 0.05) * 700.0, 0.0, 55.0)
    risk_score = _clamp((risk_ratio - 1.10) * 80.0, 0.0, 25.0)
    quality_score = _clamp(win_rate_gap * 100.0 * 0.40, 0.0, 20.0)
    score = _clamp(trigger_score + risk_score + quality_score)

    signals = [
        {"metric": "fast_entries_after_big_win_rate", "value": round(trigger_rate, 3)},
        {"metric": "risk_ratio_on_fomo_entries", "value": round(risk_ratio, 3)},
        {"metric": "win_rate_fomo_entries", "value": round(win_rate_fomo, 3)},
        {"metric": "win_rate_baseline", "value": round(win_rate_base, 3)},
    ]

    tips = [
        "After strong upside moves, wait for a predefined confirmation before entry.",
        "Use fixed risk for momentum re-entries; never increase size because of urgency.",
        "If a setup feels rushed, skip it and review after market close.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

# --------------------------
# 5) CONFIRMATION BIAS
# --------------------------
def detect_confirmation_bias(df: pd.DataFrame) -> BiasResult:
    """Confirmation-bias signals:
    - One-sided trading dominance (BUY or SELL)
    - High concentration on one asset
    - Repeating same thesis (side/asset) right after losses
    """
    if len(df) < 8:
        return BiasResult(score=0, level="LOW", signals=[{"msg": "Not enough trades to assess confirmation bias reliably."}], tips=[])

    d = add_features(df).sort_values("timestamp").reset_index(drop=True)
    prev_loss = d["pnl"].shift(1) < 0

    buy_share = float((d["side"] == "BUY").mean())
    sell_share = float((d["side"] == "SELL").mean())
    side_dominance = max(buy_share, sell_share)

    asset_share = d["asset"].value_counts(normalize=True)
    top_asset_share = float(asset_share.iloc[0]) if len(asset_share) else 0.0
    top_asset = str(asset_share.index[0]) if len(asset_share) else "-"

    same_side_after_loss = (d["side"] == d["side"].shift(1)) & prev_loss
    same_asset_after_loss = (d["asset"] == d["asset"].shift(1)) & prev_loss

    same_side_after_loss_rate = float(same_side_after_loss[prev_loss].mean()) if prev_loss.any() else 0.0
    same_asset_after_loss_rate = float(same_asset_after_loss[prev_loss].mean()) if prev_loss.any() else 0.0

    side_score = _clamp((side_dominance - 0.65) * 100.0, 0.0, 30.0)
    concentration_score = _clamp((top_asset_share - 0.45) * 100.0, 0.0, 35.0)
    same_side_score = _clamp((same_side_after_loss_rate - 0.55) * 120.0, 0.0, 20.0)
    same_asset_score = _clamp((same_asset_after_loss_rate - 0.55) * 120.0, 0.0, 15.0)

    score = _clamp(side_score + concentration_score + same_side_score + same_asset_score)

    signals = [
        {"metric": "side_dominance", "value": round(side_dominance, 3)},
        {"metric": "top_asset", "value": top_asset},
        {"metric": "top_asset_share", "value": round(top_asset_share, 3)},
        {"metric": "same_side_after_loss_rate", "value": round(same_side_after_loss_rate, 3)},
        {"metric": "same_asset_after_loss_rate", "value": round(same_asset_after_loss_rate, 3)},
    ]

    tips = [
        "Force a counter-thesis before repeating the same direction.",
        "Rotate watchlists and cap repeated entries on a single asset.",
        "After a losing trade, pause and validate one disconfirming signal before re-entry.",
    ]

    return BiasResult(score=score, level=_level(score), signals=signals, tips=tips)

def compute_risk_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute an explicit risk profile from drawdown, volatility and consistency."""
    if len(df) < 5:
        return {
            "score": 50.0,
            "level": "BALANCED",
            "max_drawdown_pct": 0.0,
            "pnl_volatility": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "stability": 50.0,
            "drawdown_resilience": 50.0,
            "consistency": 50.0,
        }

    balance = df["balance"].astype(float)
    running_max = balance.cummax().replace(0, np.nan)
    drawdown_pct = ((running_max - balance) / running_max).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_drawdown_pct = float(drawdown_pct.max() * 100.0)

    pnl = df["pnl"].astype(float)
    pnl_volatility = float(pnl.std(ddof=0))
    baseline_abs_pnl = float(pnl.abs().mean()) + 1e-9
    vol_ratio = pnl_volatility / baseline_abs_pnl

    win_rate = float((pnl > 0).mean() * 100.0)
    wins = float(pnl[pnl > 0].sum())
    losses_abs = abs(float(pnl[pnl <= 0].sum()))
    profit_factor = (wins / losses_abs) if losses_abs > 0 else (2.0 if wins > 0 else 1.0)

    stability = _clamp(100.0 - vol_ratio * 35.0, 0.0, 100.0)
    drawdown_resilience = _clamp(100.0 - max_drawdown_pct * 1.8, 0.0, 100.0)
    consistency = _clamp((win_rate * 0.55) + (_clamp(profit_factor * 40.0) * 0.45), 0.0, 100.0)

    risk_score = _clamp((stability * 0.40) + (drawdown_resilience * 0.35) + (consistency * 0.25))
    if risk_score >= 70:
        risk_level = "CONSERVATIVE"
    elif risk_score >= 45:
        risk_level = "BALANCED"
    else:
        risk_level = "AGGRESSIVE"

    return {
        "score": round(risk_score, 1),
        "level": risk_level,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "pnl_volatility": round(pnl_volatility, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(float(profit_factor), 3),
        "stability": round(float(stability), 1),
        "drawdown_resilience": round(float(drawdown_resilience), 1),
        "consistency": round(float(consistency), 1),
    }

# --------------------------
# Aggregate
# --------------------------
def analyze_biases(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-serializable analysis payload."""
    over = detect_overtrading(df)
    loss = detect_loss_aversion(df)
    rev = detect_revenge_trading(df)
    fomo = detect_fomo_trading(df)
    confirm = detect_confirmation_bias(df)
    risk_profile = compute_risk_profile(df)

    overall = _clamp(
        (over.score * 0.28)
        + (loss.score * 0.24)
        + (rev.score * 0.24)
        + (fomo.score * 0.12)
        + (confirm.score * 0.12)
    )
    all_bias_scores = [over.score, loss.score, rev.score, fomo.score, confirm.score]
    high_count = sum(1 for s in all_bias_scores if s >= 75)
    if max(all_bias_scores) >= 75 and overall < 45:
        overall = 45.0
    if high_count >= 2 and overall < 60:
        overall = 60.0

    return {
        "overall": {"score": round(overall, 1), "level": _level(overall)},
        "biases": {
            "overtrading": over.__dict__,
            "loss_aversion": loss.__dict__,
            "revenge_trading": rev.__dict__,
            "fomo_trading": fomo.__dict__,
            "confirmation_bias": confirm.__dict__,
        },
        "risk_profile": risk_profile,
        "meta": {
            "n_trades": int(len(df)),
            "start": df["timestamp"].min().isoformat() if len(df) else None,
            "end": df["timestamp"].max().isoformat() if len(df) else None,
        }
    }
