from pathlib import Path

import pandas as pd
from django.test import SimpleTestCase

from .bias_rules import analyze_biases
from .parser import normalize_columns


class BiasCalibrationTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.datasets_dir = Path(__file__).resolve().parents[2] / "trading_datasets"

    def _load_dataset(self, filename: str) -> pd.DataFrame:
        df = pd.read_csv(self.datasets_dir / filename)
        df = normalize_columns(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["side"] = df["side"].astype(str).str.upper().str.strip()
        df["asset"] = df["asset"].astype(str).str.upper().str.strip()
        for c in ["quantity", "entry_price", "exit_price", "pnl", "balance"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["timestamp", "quantity", "entry_price", "exit_price", "pnl", "balance"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)

    def test_overtrading_separates_calm_vs_overtrader(self):
        calm = analyze_biases(self._load_dataset("calm_trader.csv"))
        over = analyze_biases(self._load_dataset("overtrader.csv"))

        self.assertEqual(calm["biases"]["overtrading"]["level"], "LOW")
        self.assertEqual(over["biases"]["overtrading"]["level"], "HIGH")

    def test_revenge_dataset_is_flagged(self):
        rev = analyze_biases(self._load_dataset("revenge_trader.csv"))
        score = rev["biases"]["revenge_trading"]["score"]

        self.assertGreaterEqual(score, 45.0)
        self.assertIn(rev["biases"]["revenge_trading"]["level"], ["MEDIUM", "HIGH"])

    def test_overall_guard_prevents_low_when_any_bias_high(self):
        over = analyze_biases(self._load_dataset("overtrader.csv"))
        loss = analyze_biases(self._load_dataset("loss_averse_trader.csv"))

        self.assertNotEqual(over["overall"]["level"], "LOW")
        self.assertNotEqual(loss["overall"]["level"], "LOW")

    def test_new_biases_present_in_payload(self):
        payload = analyze_biases(self._load_dataset("calm_trader.csv"))
        self.assertIn("fomo_trading", payload["biases"])
        self.assertIn("confirmation_bias", payload["biases"])
        self.assertIn("risk_profile", payload)
