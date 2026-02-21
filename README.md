# Bias Detector (Django + DRF)

A fast, **rule-based** trading psychology detector (Overtrading · Loss Aversion · Revenge Trading) built for the National Bank hackathon.

## Repo structure (matches your requested arborescence)

```
bias-detector/
├── backend/
│   ├── manage.py
│   ├── config/        # Django project
│   ├── api/           # DRF endpoints + Trade model
│   ├── analytics/     # parser + rules engine
│   └── ui/            # Django templates dashboard (Bootstrap + Chart.js)
└── docs/
    └── math_and_thresholds.md
```

---

## 1) Setup (local)

```bash
# from repo root
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt

# env (optional)
cp .env.example .env
```

## 2) Run migrations + server

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Open:
- Dashboard: http://127.0.0.1:8000/
- API docs (Browsable API): http://127.0.0.1:8000/api/

---

## 3) Test with the provided datasets

You were given 4 CSV files:
- `calm_trader.csv`
- `overtrader.csv`
- `loss_averse_trader.csv`
- `revenge_trader.csv`

### Expected columns
The parser accepts the “official” schema:

- `timestamp, side, asset, quantity, entry_price, exit_price, pnl, balance`

But the provided dataset uses **`profit_loss`** instead of `pnl` — we map that automatically.

### Option A — via Dashboard (fastest)
1. Open http://127.0.0.1:8000/
2. Upload one of the CSV files (e.g., `overtrader.csv`)
3. You should be redirected to the dashboard with:
   - Batch ID in the navbar
   - Charts + bias scores + tips

Repeat with other CSVs to see different bias profiles.

### Option B — via API (curl)

**Upload**
```bash
curl -X POST http://127.0.0.1:8000/api/upload/ \
  -F "file=@/path/to/overtrader.csv"
```

Response:
```json
{"batch_id":"ab12cd34ef56","inserted":9800}
```

**Analyze**
```bash
curl "http://127.0.0.1:8000/api/analyze/?batch_id=ab12cd34ef56"
```

**List trades** (limited to 5000 for safety)
```bash
curl "http://127.0.0.1:8000/api/trades/?batch_id=ab12cd34ef56"
```

---

## 4) Where the core logic lives

- File parsing + normalization: `backend/analytics/parser.py`
- Rule engine (math + thresholds): `backend/analytics/bias_rules.py`
- Extra tips + coach message: `backend/analytics/recommendations.py`, `backend/analytics/coach.py`
- DRF endpoints: `backend/api/views.py`
- Dashboard UI: `backend/ui/templates/dashboard.html`

For the exact formulas and thresholds, see:
- `docs/math_and_thresholds.md`

---

## 5) Notes for hackathon speed

- Uses `bulk_create()` for fast inserts.
- Uses vectorized Pandas features for scoring.
- No external services required (SQLite by default).
