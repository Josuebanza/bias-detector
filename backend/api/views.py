import uuid
import pandas as pd
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Trade
from .serializers import TradeSerializer
from analytics.parser import parse_trade_file
from analytics.bias_rules import analyze_biases

class UploadTradesAPIView(APIView):
    """Upload a CSV/Excel and persist trades to DB."""

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return Response({"error": "Missing file"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = parse_trade_file(f)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

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

        return Response({"batch_id": batch_id, "inserted": len(objs)})

class TradesListAPIView(APIView):
    """List trades; optionally filter by batch_id."""

    def get(self, request):
        batch_id = request.query_params.get("batch_id")
        qs = Trade.objects.all().order_by("timestamp")
        if batch_id:
            qs = qs.filter(batch_id=batch_id)

        data = TradeSerializer(qs[:5000], many=True).data  # protect UI
        return Response({"count": qs.count(), "results": data})

class AnalyzeAPIView(APIView):
    """Compute bias metrics from DB trades (batch_id optional)."""

    def get(self, request):
        batch_id = request.query_params.get("batch_id")
        qs = Trade.objects.all().order_by("timestamp")
        if batch_id:
            qs = qs.filter(batch_id=batch_id)

        if not qs.exists():
            return Response({"error": "No trades found. Upload first."}, status=status.HTTP_400_BAD_REQUEST)

        df = pd.DataFrame.from_records(qs.values(
            "timestamp","side","asset","quantity","entry_price","exit_price","pnl","balance"
        ))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        payload = analyze_biases(df)
        payload["batch_id"] = batch_id
        return Response(payload)
