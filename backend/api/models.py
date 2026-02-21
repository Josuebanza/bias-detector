from django.db import models

class Trade(models.Model):
    """A single trade record imported from a CSV/Excel or entered manually.

    Notes:
    - We store trades in DB so the dashboard can refresh quickly and the API
      can scale to larger datasets during judging (20x bigger per challenge docs).
    """
    timestamp = models.DateTimeField(db_index=True)
    side = models.CharField(max_length=4, choices=[("BUY", "BUY"), ("SELL", "SELL")])
    asset = models.CharField(max_length=50, db_index=True)
    quantity = models.FloatField()
    entry_price = models.FloatField()
    exit_price = models.FloatField()
    pnl = models.FloatField(help_text="Profit/Loss of the trade (can be negative)")
    balance = models.FloatField(help_text="Account balance after the trade")

    # Optional: group imports (useful if user uploads multiple files)
    batch_id = models.CharField(max_length=64, db_index=True, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.timestamp} {self.side} {self.asset} pnl={self.pnl}"
