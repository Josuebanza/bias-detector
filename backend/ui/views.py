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

    # Charts
    pnl_series = df[["timestamp","pnl"]].copy()
    pnl_series["cum_pnl"] = pnl_series["pnl"].cumsum()

    by_hour = df.groupby(df["timestamp"].dt.floor("H")).size()
    hour_labels = [t.isoformat() for t in by_hour.index.to_list()[-60:]]
    hour_values = by_hour.values.tolist()[-60:]

    win = df[df["pnl"] > 0]["pnl"].tolist()[:500]
    loss = df[df["pnl"] <= 0]["pnl"].tolist()[:500]

    context = {
        "empty": False,
        "batch_id": batch_id,
        "analysis": analysis,
        "coach": coach,
        "n_trades": len(df),
        "start": analysis["meta"]["start"],
        "end": analysis["meta"]["end"],
        "chart_cum_pnl_labels": [t.isoformat() for t in pnl_series["timestamp"].tolist()[-200:]],
        "chart_cum_pnl_values": pnl_series["cum_pnl"].tolist()[-200:],
        "chart_balance_labels": [t.isoformat() for t in df["timestamp"].tolist()[-200:]],
        "chart_balance_values": df["balance"].tolist()[-200:],
        "chart_hour_labels": hour_labels,
        "chart_hour_values": hour_values,
        "win_values": win,
        "loss_values": loss,
    }
    return render(request, "dashboard.html", context)
