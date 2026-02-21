from rest_framework import serializers
from .models import Trade

class TradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trade
        fields = [
            "id","timestamp","side","asset","quantity","entry_price","exit_price","pnl","balance","batch_id"
        ]
