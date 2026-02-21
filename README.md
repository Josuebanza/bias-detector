# TradeDNA (Django + DRF)

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

## 0) How the full code works (FR/EN)

1. **Upload / Import**
- FR: L'utilisateur charge un CSV/Excel depuis l'UI (`/`) ou l'API (`/api/upload/`).
- EN: The user uploads a CSV/Excel through the UI (`/`) or API (`/api/upload/`).

2. **Parsing + normalization**
- FR: `backend/analytics/parser.py` valide les colonnes, convertit les types, harmonise les noms (ex: `profit_loss` -> `pnl`).
- EN: `backend/analytics/parser.py` validates columns, converts types, and normalizes names (e.g., `profit_loss` -> `pnl`).

3. **Persistence**
- FR: Les trades sont stockés en base via le modèle `Trade` avec un `batch_id` pour isoler chaque import.
- EN: Trades are stored in DB via the `Trade` model with a `batch_id` to isolate each upload.

4. **Bias analysis engine**
- FR: `backend/analytics/bias_rules.py` calcule les features (cadence, risque, streaks, etc.) puis les scores de biais + niveau + signaux + tips.
- EN: `backend/analytics/bias_rules.py` computes features (cadence, risk, streaks, etc.) then bias scores + level + signals + tips.

5. **Coaching layer**
- FR: `recommendations.py` complète les tips manquants, `coach.py` produit un message synthétique.
- EN: `recommendations.py` enriches missing tips, `coach.py` generates a concise coaching message.

6. **Presentation layer**
- FR: `ui/views.py` prépare des contextes séparés pour `dashboard`, `diagnostics` et `simulator`; les templates affichent les graphiques (Chart.js).
- EN: `ui/views.py` builds separate contexts for `dashboard`, `diagnostics`, and `simulator`; templates render charts (Chart.js).

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

## 4) Core files with logic (FR/EN)

- `backend/analytics/parser.py`
  - FR: pipeline d'import/normalisation des datasets.
  - EN: dataset import/normalization pipeline.
- `backend/analytics/bias_rules.py`
  - FR: coeur du scoring (Overtrading, Loss Aversion, Revenge, FOMO, Confirmation + risk profile).
  - EN: scoring core (Overtrading, Loss Aversion, Revenge, FOMO, Confirmation + risk profile).
- `backend/analytics/recommendations.py`
  - FR: fusion/complément des recommandations actionnables.
  - EN: merges/enriches actionable recommendations.
- `backend/analytics/coach.py`
  - FR: génération du message coach final.
  - EN: generates the final coach message.
- `backend/api/views.py`
  - FR: endpoints DRF (`upload`, `analyze`, `trades`).
  - EN: DRF endpoints (`upload`, `analyze`, `trades`).
- `backend/ui/views.py`
  - FR: logique UI serveur + préparation des contextes Overview/Diagnostics/Simulator.
  - EN: server-side UI logic + Overview/Diagnostics/Simulator contexts.
- `backend/ui/templates/dashboard.html`
  - FR: vue exécutive (profil, KPI, graphiques clés, safeguards).
  - EN: executive view (profile, KPIs, key charts, safeguards).
- `backend/ui/templates/diagnostics.html`
  - FR: vue d'investigation (qualité données, heatmap, signaux détaillés).
  - EN: investigation view (data quality, heatmap, detailed signals).
- `backend/ui/templates/simulator.html`
  - FR: replay temps réel + alertes préventives.
  - EN: real-time replay + preventive alerts.
- `backend/ui/static/styles.css`
  - FR: identité visuelle et responsive.
  - EN: visual identity and responsive behavior.

Formulas and thresholds:
- `docs/math_and_thresholds.md`

---

## 5) Quick defense answers (FR/EN)

- **Pourquoi rule-based au lieu de full IA ? / Why rule-based instead of full AI?**  
  FR: déterministe, explicable, rapide à calibrer pendant hackathon.  
  EN: deterministic, explainable, and fast to calibrate during a hackathon.

- **Pourquoi `batch_id` ? / Why `batch_id`?**  
  FR: isoler chaque import pour comparer plusieurs profils sans collisions.  
  EN: isolate each upload to compare multiple profiles without collisions.

- **Performance ? / Performance?**  
  FR: `bulk_create` + pandas vectorisé => ingestion/analyse rapide sur gros CSV.  
  EN: `bulk_create` + vectorized pandas => fast ingestion/analysis on large CSVs.

---

## 6) Notes for hackathon speed

- Uses `bulk_create()` for fast inserts.
- Uses vectorized Pandas features for scoring.
- No external services required (SQLite by default).
