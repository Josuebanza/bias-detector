import uuid
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse

from analytics.parser import parse_trade_file
from analytics.bias_rules import analyze_biases
from analytics.recommendations import merge_tips
from analytics.coach import coaching_message

from api.models import Trade

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
        objs.append(Trade(
            timestamp=row["timestamp"].to_pydatetime(),
            side=str(row["side"]).upper(),
            asset=str(row["asset"]).upper(),
            quantity=float(row["quantity"]),
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            pnl=float(row["pnl"]),
            balance=float(row["balance"]),
            batch_id=batch_id,
        ))
    Trade.objects.bulk_create(objs, batch_size=2000)

    request.session["latest_batch_id"] = batch_id
    messages.success(request, f"Imported {len(objs)} trades (batch {batch_id}).")
    return redirect(reverse("dashboard") + f"?batch_id={batch_id}")

def dashboard(request):
    """Single-page dashboard (upload + insights)."""
    batch_id = request.GET.get("batch_id") or request.session.get("latest_batch_id")

    qs = Trade.objects.all().order_by("timestamp")
    if batch_id:
        qs = qs.filter(batch_id=batch_id)

    if not qs.exists():
        return render(request, "dashboard.html", {"batch_id": batch_id, "empty": True})

    df = pd.DataFrame.from_records(qs.values(
        "timestamp","side","asset","quantity","entry_price","exit_price","pnl","balance"
    ))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    analysis = analyze_biases(df)
    analysis = merge_tips(analysis)
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

    # Charts
    pnl_series = df[["timestamp","pnl"]].copy()
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

    win = df[df["pnl"] > 0]["pnl"].tolist()[:500]
    loss = df[df["pnl"] <= 0]["pnl"].tolist()[:500]
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    start_display = start_ts.strftime("%d/%m/%Y %H:%M")
    end_display = end_ts.strftime("%d/%m/%Y %H:%M")

    context = {
        "empty": False,
        "batch_id": batch_id,
        "analysis": analysis,
        "bias_snapshot_cards": bias_snapshot_cards,
        "recommendation_cards": recommendation_cards,
        "coach": coach,
        "n_trades": len(df),
        "start": analysis["meta"]["start"],
        "end": analysis["meta"]["end"],
        "start_display": start_display,
        "end_display": end_display,
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
    return render(request, "dashboard.html", context)
