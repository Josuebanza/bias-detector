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


def _build_context(df: pd.DataFrame, batch_id: str | None, page: str) -> Dict[str, Any]:
    analysis = merge_tips(analyze_biases(df))
    coach = coaching_message(analysis)
    biases = analysis.get("biases", {})

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
    extended_bias_cards = [
        {"label": "Overtrading", "icon": "bi-lightning-charge", "bias": biases.get("overtrading", {})},
        {"label": "Loss aversion", "icon": "bi-shield-exclamation", "bias": biases.get("loss_aversion", {})},
        {"label": "Revenge trading", "icon": "bi-fire", "bias": biases.get("revenge_trading", {})},
        {"label": "FOMO trading", "icon": "bi-rocket-takeoff", "bias": biases.get("fomo_trading", {})},
        {"label": "Confirmation bias", "icon": "bi-bullseye", "bias": biases.get("confirmation_bias", {})},
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
    intraday_labels = [f"{h:02d}:00" for h in intraday.index.tolist()]
    intraday_trades = [int(v) for v in intraday["trades"].tolist()]
    intraday_win_rate = [round(float(v), 2) for v in intraday["win_rate"].tolist()]
    intraday_avg_pnl = [round(float(v), 2) for v in intraday["avg_pnl"].tolist()]

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
    heat_win = (
        df.assign(day_of_week=df["timestamp"].dt.dayofweek, hour_of_day=df["timestamp"].dt.hour)
        .groupby(["day_of_week", "hour_of_day"])["pnl"]
        .apply(lambda s: float((s > 0).mean() * 100.0))
        .unstack(fill_value=0.0)
        .reindex(index=range(7), columns=range(24), fill_value=0.0)
    )
    heatmap_rows = [
        {
            "day": DAY_LABELS[d],
            "values": [int(v) for v in heat.loc[d].tolist()],
            "win_rates": [round(float(v), 1) for v in heat_win.loc[d].tolist()],
        }
        for d in range(7)
    ]
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

    win = df[df["pnl"] > 0]["pnl"].tolist()[:500]
    loss = df[df["pnl"] <= 0]["pnl"].tolist()[:500]
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    start_display = start_ts.strftime("%d/%m/%Y %H:%M")
    end_display = end_ts.strftime("%d/%m/%Y %H:%M")

    context = {
        "page": page,
        "empty": False,
        "batch_id": batch_id,
        "analysis": analysis,
        "risk_profile": analysis.get("risk_profile", {}),
        "coach": coach,
        "n_trades": len(df),
        "start": analysis["meta"]["start"],
        "end": analysis["meta"]["end"],
        "start_display": start_display,
        "end_display": end_display,
        "bias_snapshot_cards": bias_snapshot_cards,
        "extended_bias_cards": extended_bias_cards,
        "recommendation_cards": recommendation_cards,
        "best_hours": best_hours,
        "worst_hours": worst_hours,
        "min_trades_threshold": min_trades_threshold,
        "heatmap_rows": heatmap_rows,
        "heatmap_max": heatmap_max,
        "hour_labels_24": [f"{h:02d}" for h in range(24)],
        "data_quality": data_quality,
        "chart_cum_pnl_labels": [t.isoformat() for t in pnl_series["timestamp"].tolist()[-200:]],
        "chart_cum_pnl_values": pnl_series["cum_pnl"].tolist()[-200:],
        "chart_balance_labels": [t.isoformat() for t in df["timestamp"].tolist()[-200:]],
        "chart_balance_values": df["balance"].tolist()[-200:],
        "chart_intraday_labels": intraday_labels,
        "chart_intraday_trades": intraday_trades,
        "chart_intraday_win_rate": intraday_win_rate,
        "chart_intraday_avg_pnl": intraday_avg_pnl,
        "win_values": win,
        "loss_values": loss,
    }
    return context


def dashboard(request):
    """Main overview page (upload + profile + key charts)."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")
    df = _load_dataframe_for_batch(batch_id)

    if df is None:
        return render(request, "dashboard.html", {"page": "dashboard", "batch_id": batch_id, "empty": True})

    context = _build_context(df, batch_id=batch_id, page="dashboard")
    return render(request, "dashboard.html", context)


def diagnostics(request):
    """Detailed diagnostics page (heatmap, data quality, extended bias panels)."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")
    df = _load_dataframe_for_batch(batch_id)

    if df is None:
        return render(request, "diagnostics.html", {"page": "diagnostics", "batch_id": batch_id, "empty": True})

    context = _build_context(df, batch_id=batch_id, page="diagnostics")
    return render(request, "diagnostics.html", context)
