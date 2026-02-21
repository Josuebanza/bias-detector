import uuid
from typing import Dict, Any

import pandas as pd
from django.contrib import messages
from django.shortcuts import redirect, render
from django.urls import reverse

from analytics.bias_rules import analyze_biases
from analytics.coach import coaching_message
from analytics.parser import parse_trade_file
from analytics.recommendations import merge_tips
from api.models import Trade

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def upload_trades(request):
    """Handle CSV/Excel upload from the dashboard form and save to DB."""
    if request.method != "POST":
        return redirect("dashboard")

    f = request.FILES.get("file")
    if not f:
        messages.error(request, "Please choose a CSV/Excel file.")
        return redirect("dashboard")

    try:
        df = parse_trade_file(f)
    except Exception as e:
        messages.error(request, f"Upload failed: {e}")
        return redirect("dashboard")

    batch_id = uuid.uuid4().hex[:12]
    objs = []
    for row in df.to_dict(orient="records"):
        objs.append(
            Trade(
                timestamp=row["timestamp"].to_pydatetime(),
                side=str(row["side"]).upper(),
                asset=str(row["asset"]).upper(),
                quantity=float(row["quantity"]),
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                pnl=float(row["pnl"]),
                balance=float(row["balance"]),
                batch_id=batch_id,
            )
        )
    Trade.objects.bulk_create(objs, batch_size=2000)

    request.session["latest_batch_id"] = batch_id
    messages.success(request, f"Imported {len(objs)} trades (batch {batch_id}).")
    return redirect(reverse("dashboard") + f"?batch_id={batch_id}")


def _load_dataframe_for_batch(batch_id: str | None) -> pd.DataFrame | None:
    qs = Trade.objects.all().order_by("timestamp")
    if batch_id:
        qs = qs.filter(batch_id=batch_id)

    if not qs.exists():
        return None

    df = pd.DataFrame.from_records(
        qs.values(
            "timestamp",
            "side",
            "asset",
            "quantity",
            "entry_price",
            "exit_price",
            "pnl",
            "balance",
        )
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _serialize_hour_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for hour, row in frame.iterrows():
        rows.append(
            {
                "hour": f"{int(hour):02d}:00",
                "trades": int(row["trades"]),
                "win_rate": round(float(row["win_rate"]), 2),
                "avg_pnl": round(float(row["avg_pnl"]), 2),
            }
        )
    return rows


def _format_display(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "-"
    return ts.strftime("%d/%m/%Y %H:%M")


def _format_signal_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        v = float(value)
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _build_common_context(df: pd.DataFrame, batch_id: str | None, page: str) -> Dict[str, Any]:
    analysis = merge_tips(analyze_biases(df))
    coach = coaching_message(analysis)
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    return {
        "page": page,
        "empty": False,
        "batch_id": batch_id,
        "analysis": analysis,
        "coach": coach,
        "n_trades": len(df),
        "start": analysis["meta"]["start"],
        "end": analysis["meta"]["end"],
        "start_display": _format_display(start_ts),
        "end_display": _format_display(end_ts),
    }


def _build_dashboard_context(df: pd.DataFrame, batch_id: str | None) -> Dict[str, Any]:
    context = _build_common_context(df, batch_id=batch_id, page="dashboard")
    biases = context["analysis"].get("biases", {})

    bias_snapshot_cards = [
        {"label": "Overtrading", "icon": "bi-lightning-charge", "bias": biases.get("overtrading", {})},
        {"label": "Loss aversion", "icon": "bi-shield-exclamation", "bias": biases.get("loss_aversion", {})},
        {"label": "Revenge trading", "icon": "bi-fire", "bias": biases.get("revenge_trading", {})},
    ]
    recommendation_cards = [
        {"title": "Overtrading discipline", "bias": biases.get("overtrading", {})},
        {"title": "Loss aversion fixes", "bias": biases.get("loss_aversion", {})},
        {"title": "Revenge trading guardrails", "bias": biases.get("revenge_trading", {})},
    ]

    pnl_series = df[["timestamp", "pnl"]].copy()
    pnl_series["cum_pnl"] = pnl_series["pnl"].cumsum()

    intraday = (
        df.assign(hour_of_day=df["timestamp"].dt.hour)
        .groupby("hour_of_day")
        .agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda s: (s > 0).mean() * 100.0),
            avg_pnl=("pnl", "mean"),
        )
        .reindex(range(24), fill_value=0.0)
    )

    context.update(
        {
            "bias_snapshot_cards": bias_snapshot_cards,
            "recommendation_cards": recommendation_cards,
            "chart_cum_pnl_labels": [t.isoformat() for t in pnl_series["timestamp"].tolist()[-200:]],
            "chart_cum_pnl_values": pnl_series["cum_pnl"].tolist()[-200:],
            "chart_balance_labels": [t.isoformat() for t in df["timestamp"].tolist()[-200:]],
            "chart_balance_values": df["balance"].tolist()[-200:],
            "chart_intraday_labels": [f"{h:02d}:00" for h in intraday.index.tolist()],
            "chart_intraday_trades": [int(v) for v in intraday["trades"].tolist()],
            "chart_intraday_win_rate": [round(float(v), 2) for v in intraday["win_rate"].tolist()],
            "chart_intraday_avg_pnl": [round(float(v), 2) for v in intraday["avg_pnl"].tolist()],
            "win_values": df[df["pnl"] > 0]["pnl"].tolist()[:500],
            "loss_values": df[df["pnl"] <= 0]["pnl"].tolist()[:500],
        }
    )
    return context


def _build_diagnostics_context(df: pd.DataFrame, batch_id: str | None) -> Dict[str, Any]:
    context = _build_common_context(df, batch_id=batch_id, page="diagnostics")
    analysis = context["analysis"]
    biases = analysis.get("biases", {})

    intraday = (
        df.assign(hour_of_day=df["timestamp"].dt.hour)
        .groupby("hour_of_day")
        .agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda s: (s > 0).mean() * 100.0),
            avg_pnl=("pnl", "mean"),
        )
        .reindex(range(24), fill_value=0.0)
    )
    min_trades_threshold = max(5, min(40, int(len(df) * 0.005)))
    eligible_intraday = intraday[intraday["trades"] >= min_trades_threshold]
    if eligible_intraday.empty:
        eligible_intraday = intraday[intraday["trades"] > 0]

    best_hours = _serialize_hour_rows(
        eligible_intraday.sort_values(["win_rate", "avg_pnl"], ascending=[False, False]).head(3)
    )
    worst_hours = _serialize_hour_rows(
        eligible_intraday.sort_values(["win_rate", "avg_pnl"], ascending=[True, True]).head(3)
    )

    heat = (
        df.assign(day_of_week=df["timestamp"].dt.dayofweek, hour_of_day=df["timestamp"].dt.hour)
        .groupby(["day_of_week", "hour_of_day"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    heatmap_rows = [{"day": DAY_LABELS[d], "values": [int(v) for v in heat.loc[d].tolist()]} for d in range(7)]
    heatmap_max = max(int(heat.to_numpy().max()), 1)

    dedupe_cols = ["timestamp", "side", "asset", "quantity", "entry_price", "exit_price", "pnl", "balance"]
    duplicate_count = int(df.duplicated(subset=dedupe_cols).sum())

    dt_minutes = (df["timestamp"] - df["timestamp"].shift(1)).dt.total_seconds().fillna(0) / 60.0
    positive_dt = dt_minutes[dt_minutes > 0]
    median_gap_min = float(positive_dt.median()) if len(positive_dt) else 0.0
    gap_threshold_min = max(30.0, median_gap_min * 5.0) if median_gap_min > 0 else 30.0
    large_gap_count = int((dt_minutes > gap_threshold_min).sum())

    non_positive_balance_count = int((df["balance"] <= 0).sum())
    flat_balance_rate = float((df["balance"].diff().fillna(0) == 0).mean() * 100.0)
    data_quality = {
        "duplicate_count": duplicate_count,
        "large_gap_count": large_gap_count,
        "median_gap_min": round(median_gap_min, 2),
        "gap_threshold_min": round(gap_threshold_min, 2),
        "non_positive_balance_count": non_positive_balance_count,
        "flat_balance_rate": round(flat_balance_rate, 2),
    }

    hour_timeline = (
        df.assign(hour_bucket=df["timestamp"].dt.floor("h"))
        .groupby("hour_bucket")
        .size()
        .sort_index()
    )
    chart_hourly_labels = [t.isoformat() for t in hour_timeline.index.tolist()[-96:]]
    chart_hourly_values = [int(v) for v in hour_timeline.tolist()[-96:]]

    gap_bins = pd.cut(
        positive_dt,
        bins=[0.0, 1.0, 5.0, 30.0, float("inf")],
        labels=["<1 min", "1-5 min", "5-30 min", ">30 min"],
        include_lowest=True,
    )
    gap_dist = gap_bins.value_counts().reindex(["<1 min", "1-5 min", "5-30 min", ">30 min"], fill_value=0)
    gap_labels = gap_dist.index.tolist()
    gap_values = [int(v) for v in gap_dist.values.tolist()]

    bias_meta = [
        ("overtrading", "Overtrading", "bi-lightning-charge"),
        ("loss_aversion", "Loss aversion", "bi-shield-exclamation"),
        ("revenge_trading", "Revenge trading", "bi-fire"),
        ("fomo_trading", "FOMO trading", "bi-rocket-takeoff"),
        ("confirmation_bias", "Confirmation bias", "bi-bullseye"),
    ]
    bias_signal_cards = []
    for key, label, icon in bias_meta:
        bias_obj = biases.get(key, {})
        signals = bias_obj.get("signals", []) or []
        rendered_signals = [
            {"metric": str(s.get("metric", "-")).replace("_", " "), "value": _format_signal_value(s.get("value", "-"))}
            for s in signals[:5]
        ]
        bias_signal_cards.append(
            {
                "label": label,
                "icon": icon,
                "score": float(bias_obj.get("score", 0.0)),
                "level": str(bias_obj.get("level", "LOW")),
                "signals": rendered_signals,
            }
        )

    context.update(
        {
            "risk_profile": analysis.get("risk_profile", {}),
            "best_hours": best_hours,
            "worst_hours": worst_hours,
            "min_trades_threshold": min_trades_threshold,
            "heatmap_rows": heatmap_rows,
            "heatmap_max": heatmap_max,
            "hour_labels_24": [f"{h:02d}" for h in range(24)],
            "data_quality": data_quality,
            "bias_signal_cards": bias_signal_cards,
            "diag_hourly_labels": chart_hourly_labels,
            "diag_hourly_values": chart_hourly_values,
            "diag_gap_labels": gap_labels,
            "diag_gap_values": gap_values,
        }
    )
    return context


def _build_simulator_context(df: pd.DataFrame, batch_id: str | None) -> Dict[str, Any]:
    context = _build_common_context(df, batch_id=batch_id, page="simulator")
    biases = context["analysis"].get("biases", {})

    sim_df = df.sort_values("timestamp").tail(420).copy().reset_index(drop=True)
    sim_df["notional"] = sim_df["quantity"].abs() * sim_df["entry_price"].abs()
    sim_rows = []
    for _, row in sim_df.iterrows():
        sim_rows.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "display_time": row["timestamp"].strftime("%d/%m %H:%M:%S"),
                "asset": str(row["asset"]),
                "side": str(row["side"]),
                "pnl": round(float(row["pnl"]), 2),
                "balance": round(float(row["balance"]), 2),
                "notional": round(float(row["notional"]), 2),
            }
        )

    bucket_15 = df.set_index("timestamp").resample("15min").size()
    burst_baseline = float(bucket_15.median()) if len(bucket_15) else 1.0
    burst_warn = max(4.0, burst_baseline * 1.5)
    burst_alert = max(6.0, burst_baseline * 2.2)

    context.update(
        {
            "sim_stream_rows": sim_rows,
            "sim_thresholds": {
                "burst_warn": round(float(burst_warn), 2),
                "burst_alert": round(float(burst_alert), 2),
                "rapid_reentry_warn_min": 5.0,
                "rapid_reentry_alert_min": 2.0,
                "size_jump_warn_ratio": 1.25,
                "size_jump_alert_ratio": 1.60,
                "loss_ratio_warn": 1.20,
                "loss_ratio_alert": 1.60,
            },
            "sim_baseline_scores": {
                "overtrading": round(float(biases.get("overtrading", {}).get("score", 0.0)), 1),
                "loss_aversion": round(float(biases.get("loss_aversion", {}).get("score", 0.0)), 1),
                "revenge_trading": round(float(biases.get("revenge_trading", {}).get("score", 0.0)), 1),
            },
            "sim_watch_tips": [
                {
                    "title": "Overtrading watch",
                    "tip": (biases.get("overtrading", {}).get("tips", []) or ["Respect a max-trade cap each session."])[0],
                },
                {
                    "title": "Revenge watch",
                    "tip": (
                        biases.get("revenge_trading", {}).get("tips", [])
                        or ["After a loss, force a cooling-off period before re-entry."]
                    )[0],
                },
                {
                    "title": "Loss control watch",
                    "tip": (
                        biases.get("loss_aversion", {}).get("tips", [])
                        or ["Set your stop before entry and avoid moving it away."]
                    )[0],
                },
            ],
        }
    )
    return context


def dashboard(request):
    """Main overview page (upload + profile + key charts)."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")
    df = _load_dataframe_for_batch(batch_id)

    if df is None:
        return render(request, "dashboard.html", {"page": "dashboard", "batch_id": batch_id, "empty": True})

    context = _build_dashboard_context(df, batch_id=batch_id)
    return render(request, "dashboard.html", context)


def diagnostics(request):
    """Detailed diagnostics page (heatmap, data quality, extended bias panels)."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")
    df = _load_dataframe_for_batch(batch_id)

    if df is None:
        return render(request, "diagnostics.html", {"page": "diagnostics", "batch_id": batch_id, "empty": True})

    context = _build_diagnostics_context(df, batch_id=batch_id)
    return render(request, "diagnostics.html", context)


def simulator(request):
    """Replay view for near real-time behavioral drift and preventive alerts."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")
    df = _load_dataframe_for_batch(batch_id)

    if df is None:
        return render(request, "simulator.html", {"page": "simulator", "batch_id": batch_id, "empty": True})

    context = _build_simulator_context(df, batch_id=batch_id)
    return render(request, "simulator.html", context)
