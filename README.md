# Community Mental Health Crisis Predictor

Predict community-wide mental health crises from Reddit data — up to a week before they happen.

This system treats subreddits like r/depression and r/anxiety as weather systems. It analyzes weekly patterns in language, sentiment, and behavior to forecast whether the community's aggregate distress will escalate next week — and *how bad* it will get.

---

## What It Does

The pipeline takes a subreddit's post history and produces a week-by-week crisis forecast across **four escalation states**:

| State | Meaning | Sigma threshold |
|-------|---------|-----------------|
| 0 — Stable | Community distress is within normal range | < 0.5σ above baseline |
| 1 — Early Vulnerability Signal | Early warning signs present at a community level | 0.5σ – 1.0σ |
| 2 — Elevated Distress | Significant distress increase requiring closer monitoring | 1.0σ – 2.0σ |
| 3 — Severe Community Distress Signal | Sustained or extreme community-wide distress signal | > 2.0σ |

The model never sees future data. Walk-forward time-series cross-validation ensures predictions are always made on unseen weeks.

**Decision usefulness (Recall@K).** Training/evaluation also records how well the model supports a **fixed weekly alert budget**: among all walk-forward weeks with valid labels, weeks are ranked by predicted high-distress probability; the top **K** are treated as alerts. **Recall@K** is the fraction of **true** elevated-distress weeks (binary: actual state ≥ 2, same target as PR-AUC) captured in those K slots. Reported next to **expected recall under random selection** of K weeks and a **persistence** baseline (rank weeks by whether the *previous* week was elevated-distress). Stored in `eval_results.json` under `decision_usefulness` and shown in the Streamlit **Model Metrics** expander and the static HTML report.

---

## How It Works

1. **Collect** — Collect from configurable source (`zenodo_covid`, `reddit_api`, or `synthetic`)
2. **Extract** — Build weekly feature vectors: linguistic patterns, VADER sentiment, distress lexicon density, topic distributions (BERTopic), behavioral signals, JS-divergence topic drift (1-week and 4-week lookback)
3. **Label** — Score each week's distress; classify next week into one of 4 states using community-specific baselines
4. **Train** — Two models run in parallel:
   - **LSTM** (primary) — PyTorch sequence model; sees the last 8 weeks as context. Features are **MinMax-normalized per fold** (scaler fit on training window only, applied at prediction time — no data leakage)
   - **XGBoost** (baseline) — Binary crisis classifier; trained on the same walk-forward splits
5. **Monitor** — Rolling z-score drift detection flags sudden signal changes; an alert engine logs state transitions to SQLite
6. **Visualize** — Streamlit dashboard with an all-community card row, week replay, drift/SHAP/data-quality tabs, or static HTML reports
7. **EDA** — After feature extraction, an automated EDA report is generated per subreddit: IQR-based outlier detection, linear distress trend (rising/stable/declining), crisis rate by year, feature distribution table with missingness flags
7. **Weekly brief** — After evaluation, each week with a prediction can get a short text brief: structured JSON is built from model outputs + global SHAP top features (with per-week deltas), augmented with retrieved text from `config/intervention_playbook.md`, then sent to **Claude** (`claude-sonnet-4-20250514`) if `ANTHROPIC_API_KEY` is set, else **GPT-4o** if `OPENAI_API_KEY` is set, else a **template** string. This is deterministic retrieval over fixed sources (no vector database). Optional: set `WEEKLY_NARRATIVE_MAX_WEEKS` to only generate the most recent N weeks (saves API calls).

State semantics and dashboard/report copy are centralized in code:
- `src/core/domain_config.py` — canonical state names + threshold/semantics labels
- `src/core/ui_config.py` — colors, badge styles, chart labels, and pipeline/dashboard/report copy strings

---

## Quick Start (Synthetic Data — No API Keys Needed)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate      # macOS / Linux

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run the full pipeline with synthetic data
#    --skip-topics  skips BERTopic (much faster for testing)
#    --skip-search  skips XGBoost hyperparameter search
#    --force        forces feature rebuild even if cache says unchanged
python -m src.pipeline.run_all --config config/default.yaml --synthetic --skip-topics --skip-search --force

# 4. Launch the Streamlit dashboard
streamlit run src/dashboard/app.py
```

The dashboard opens in your browser. Use the **header week slider** (and **Back / Advance**) to replay all communities in sync; click a community card to focus the timeline, weekly brief, and tabs.

---

## Data Source Selection

Collection source is now controlled by config:

```yaml
collection:
  source: "zenodo_covid"   # zenodo_covid | reddit_api | synthetic
```

- `zenodo_covid`: primary mode for project experiments (Zenodo + Arctic Shift gap-fill, manifest-aware/idempotent)
- `reddit_api`: existing PullPush.io + PRAW fallback path
- `synthetic`: generated development data (can also be forced via `--synthetic`)

`run_collect` writes canonical raw schema with provenance:
- `post_id`, `created_utc`, `selftext`, `subreddit`, `author`, `data_source`

---

## Zenodo-First Collection Workflow

The Zenodo source configured in `config/default.yaml` points to the Low et al. COVID mental health dataset.

### PowerShell + venv bootstrap (Windows)

```powershell
# From repo root
python -m venv venv
venv/Scripts/Activate.ps1
pip install -e ".[dev]"

# Ensure config uses source: zenodo_covid
python -m src.pipeline.run_collect --config config/default.yaml
```

Behavior:
- Query Zenodo record metadata (`record_id`) and resolve matching file URLs
- Download only matching subreddit/timeframe CSV files into `data/staging/zenodo/` (no full 3.1GB bulk fetch)
- Download Arctic Shift gap-fill zip from Google Drive (once) into `data/staging/arctic_shift/` and ingest JSONL files for missing 2018/2020 windows
- Build per-subreddit raw parquet under `data/raw/{subreddit}/posts.parquet`
- Track file integrity + per-subreddit ingestion metadata in manifest (`collection.zenodo.manifest_path`)
- Track source-level ingest state in `data/ingestion_manifest.json` (`zenodo`, `arctic_shift`)
- Re-run is idempotent when manifest and outputs are valid

Important:
- Zenodo is treated as **raw post source only**
- Precomputed columns like `tfidf_*` / `liwc_*` are intentionally ignored
- Default downloader uses per-file Zenodo links from record `3941387` instead of assuming one dataset zip

---

## Using Real Reddit API Data (Free)

Set `collection.source: reddit_api` in config.

Real API collection uses [PullPush.io](https://pullpush.io) — a free, publicly accessible archive of Reddit posts. **No Reddit account, no API keys, and no credentials are required** for the PullPush path.

The only thing you need to set is a privacy salt (used to hash author usernames before storing):

```bash
cp .env.example .env
```

Edit `.env` — you only need one line:

```
PRIVACY_SALT=any_random_string_here
```

The `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` fields in `.env` are only used by the PRAW fallback, which activates automatically if PullPush.io is unreachable. You can leave them blank unless you specifically want to use PRAW.

For LLM weekly briefs during `run_evaluate`, set optionally:

```
ANTHROPIC_API_KEY=...   # preferred
OPENAI_API_KEY=...      # used if Anthropic is unavailable
# WEEKLY_NARRATIVE_MAX_WEEKS=12
```

If both are unset, briefs are still written using the template fallback.

### Run the pipeline (reddit_api)

```bash
# Collect real posts (range from reddit.date_range in config)
# This takes 20–40 minutes due to API rate limiting (~1 req/sec)
python -m src.pipeline.run_collect --config config/default.yaml

# Extract features (see note below on caching)
python -m src.pipeline.run_features --config config/default.yaml

# Train LSTM + XGBoost, print comparison table
python -m src.pipeline.run_train --config config/default.yaml

# Generate reports, SHAP importance, drift alerts
python -m src.pipeline.run_evaluate --config config/default.yaml

# Open the live dashboard
streamlit run src/dashboard/app.py
```

**Expected data volume:** ~6,000–15,000 posts per subreddit. The 2-year date range gives ~104 weeks of training data per subreddit.

**Feature extraction cache (`run_features` only):** By default, feature extraction skips when `data/features/feature_build_meta.json` matches:
- raw parquet signatures (`size` + `mtime`) for configured subreddits
- `--skip-topics` mode
- relevant config content (`reddit.subreddits`, `processing`, `features`, `modeling.lstm.sequence_length`)

This avoids unnecessary recompute after unrelated config edits. To **force** a full recompute (e.g. after changing feature code), run:

```bash
python -m src.pipeline.run_features --config config/default.yaml --force
```

### Demo split (train vs live)

For a convincing live demo:
- Edit `config/default.yaml` → set `date_range.end: "2024-12-31"` for the training run
- Re-run from `run_collect` through `run_train`
- Then set `date_range.start: "2025-01-01"`, `end: "2026-03-01"` and re-collect for the "live" weeks
- The dashboard's week slider replays the 2025–2026 weeks as if they were arriving in real time

---

## Project Structure

```
src/
├── collector/        Data collection
│   ├── historical_loader.py   PushshiftLoader — PullPush.io API client
│   ├── reddit_client.py       PRAW fallback collector
│   ├── redarcs_loader.py      Load pre-downloaded CSV dumps
│   ├── synthetic.py           Synthetic data generator
│   ├── privacy.py             PII stripping (hash authors, remove URLs)
│   └── storage.py             Parquet read/write helpers
├── processing/       Text cleaning + weekly aggregation
├── features/         Feature extraction
│   ├── linguistic.py          Pronoun ratios, readability, sentence stats
│   ├── sentiment.py           VADER sentiment distributions
│   ├── distress.py            Hopelessness / help-seeking lexicon density
│   ├── topics.py              BERTopic distributions + JSD drift (1w and 4w)
│   ├── behavioral.py          Post volume, engagement, new author rate
│   └── temporal.py            Rolling averages (2w, 4w windows)
├── core/             Shared constants and copy
│   ├── domain_config.py       Canonical state semantics + threshold labels
│   └── ui_config.py           Colors, badges, chart labels, and UI/report copy
├── labeling/         Distress scoring + 4-class target generation
│   ├── distress_score.py      Weighted composite distress score
│   └── target.py              CrisisLabeler — 4 states with community baseline
├── modeling/         Models + walk-forward evaluation
│   ├── train_xgb.py           XGBoost binary baseline
│   ├── train_rnn.py           PyTorch LSTM — LSTMNet + LSTMCrisisModel
│   ├── evaluate.py            evaluate_walk_forward (XGB) + evaluate_walk_forward_lstm
│   ├── splits.py              WalkForwardSplitter
│   └── explain.py             SHAP importance via TreeExplainer
├── monitoring/       Drift detection + alert engine
│   ├── drift_detector.py      Rolling z-score detection, 4 signals, 3 alert levels
│   └── alert_engine.py        SQLite-backed escalation logger (data/alerts.db)
├── reporting/        Analytics reports
│   └── eda.py                 EDA report — IQR outlier detection, trend, crisis rate, self-contained HTML
├── visualization/    Static HTML reports
│   ├── timeline.py            4-color backtesting timeline (Plotly)
│   ├── feature_importance.py  SHAP bar chart
│   ├── case_study.py          Narrative markdown case studies
│   └── dashboard.py           Combined HTML report
├── dashboard/
│   ├── app.py                 Streamlit entrypoint (layout + orchestration)
│   ├── data_access.py         Cached loaders for features/eval/reports/db
│   ├── briefs.py              Weekly brief rendering helpers
│   ├── charts.py              Reusable chart builders (sparkline/SHAP)
│   ├── components.py          Reusable UI blocks (drift table, metrics panel)
│   ├── state.py               Session/index/state helper functions
│   ├── types.py               Typed payload definitions for dashboard data
│   └── demo_utils.py          Demo-mode helpers (scenario mapping, event parsing)
└── pipeline/         CLI entry points
    ├── run_collect.py
    ├── run_features.py
    ├── run_train.py
    ├── run_evaluate.py
    └── run_all.py
```

Also at repo root:

```
serving/                       FastAPI inference service (deployed to Render.com)
├── main.py                    API endpoints: /health /predict /model-info /logs/summary
├── requirements.txt           Service-only dependencies (lean, no BERTopic/Optuna)
├── Procfile                   Render start command
├── .python-version            Pins Python 3.11.7 (avoids pandas/Python 3.13 issues)
├── README.md                  Local run instructions + deployed URL + cold-start note
├── models/                    Deprecated local cache dir (artifacts now read from `data/`)
└── tests/                     API test suite (29 tests, runs with MOCK_MODELS=true)

config/
└── intervention_playbook.md   Retrieved moderation copy for weekly narrative (with structured model outputs)

.github/workflows/
├── ci.yml                     CI: core tests + API tests on every push/PR
└── retrain.yml                Manual dispatch: synthetic retrain + auto-commit + redeploy
```

---

## Evaluation Metrics

Walk-forward evaluation reports the following per model per subreddit:

| Metric | What it measures |
|--------|-----------------|
| **Recall** | Fraction of true crisis weeks the model catches (sensitivity) |
| **Precision** | Of weeks flagged as crisis, what fraction actually were |
| **F1** | Harmonic mean of precision and recall |
| **PR-AUC** | Area under the Precision-Recall curve — the primary metric for imbalanced detection tasks; baseline = crisis rate % |
| **ROC-AUC** | Area under the ROC curve — 0.5 = random, 1.0 = perfect; shows overall discrimination ability |
| **Recall@K** | If an ops team can only investigate K weeks, what fraction of true crisis weeks are caught? |
| **Avg detection lead time** | How many weeks ahead on average the model flags a crisis before it peaks |

After training all subreddits, a **High / Medium / Low performance band table** is printed:
- **High** (PR-AUC ≥ 0.45): model reliably detects crises in these communities
- **Medium** (0.20 – 0.45): moderate signal; worth monitoring
- **Low** (< 0.20): crisis signal hard to detect — usually means more data or better feature coverage is needed

The table also shows which model family (LSTM vs XGBoost) wins per community and prints data-grounded cross-learning recommendations (which community's SHAP features to apply to poorly-performing ones).

---

## EDA Reports

After feature extraction (`run_features`), an exploratory data analysis report is generated per subreddit at `data/reports/{sub}/eda_summary.html`. It contains:

- **Feature distribution table** — mean, std, IQR, skew, % missing per feature (color-coded: green <5%, amber 5–20%, red >20%)
- **Outlier detection (IQR rule)** — flags specific weeks where a feature value falls outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]
- **Distress trend** — linear regression on the community distress score over time; classifies as *rising*, *stable*, or *declining* with % change over the data period
- **Crisis rate by year** — fraction of weeks that reached State 2 or 3 each year
- **Quality flags** — high-missingness features, top outlier-prone features, class imbalance warnings

Open the HTML directly in any browser — no server required.

---

## Configuration (`config/default.yaml`)

Key settings you may want to change:

```yaml
reddit:
  subreddits: [depression, anxiety]   # subreddits to monitor
  date_range:
    start: "2024-01-01"
    end: "2026-03-01"

labeling:
  crisis_thresholds_std: [0.5, 1.0, 2.0]   # sigma cutoffs for the 4 states

modeling:
  lstm:
    sequence_length: 8      # weeks of context per prediction
    hidden_size: 64
    epochs: 50
    walk_forward_epochs: 20 # faster epochs during walk-forward CV

synthetic:
  n_weeks: 104              # 2 years of synthetic data
  crisis_frequency: 0.12    # ~12% of weeks are crisis weeks
```

---

## Output Files

After running the full pipeline, `data/` contains:

```
data/
├── raw/{subreddit}/posts.parquet          Raw collected posts (+ data_source provenance column)
├── features/features.parquet             Weekly feature matrix
├── models/eval_results.json              XGB + LSTM walk-forward metrics
├── alerts.db                             SQLite log of state transitions
└── reports/
    ├── {sub}/
    │   ├── eda_report.json               EDA summary — outlier weeks, distress trend, crisis rate by year
    │   ├── eda_summary.html              Self-contained EDA HTML for the project report
    │   ├── timeline.html                 4-color interactive backtesting plot
    │   ├── feature_importance.html       SHAP top-20 feature chart
    │   ├── shap.csv                      SHAP values for dashboard
    │   ├── drift_alerts.json             Rolling z-score drift alerts
    │   ├── dashboard.html                Combined HTML report
    │   ├── case_studies/
    │   │   └── case_study_*.md           Narrative high-distress case studies
    │   ├── weekly_briefs.json            Weekly narrative brief store keyed by week (one file/subreddit)
    │   └── logs/
    │       └── weekly_brief_calls.jsonl  LLM/template source + fallback notes
    └── ...
```

  The Streamlit dashboard reads from configured paths in `config/default.yaml` (`paths.features`, `paths.models`, `paths.reports`, `paths.alerts_db`).

---

## Data Strategy (git-tracked vs ignored)

Raw and intermediate data remain gitignored to keep the repo lean:

| Path | Tracked? | Reason |
|------|----------|--------|
| `data/raw/`, `data/processed/`, `data/staging/`, `data/external/` | No | Large source files |
| `data/features/features.parquet` | **Yes** | Read by Streamlit Cloud dashboard |
| `data/models/eval_results.json` | **Yes** | Read by dashboard + serving layer |
| `data/models/{sub}_xgb.pkl`, `{sub}_lstm.pt`, `{sub}_feature_stats.json` | **Yes** | Loaded by serving layer |
| `data/reports/**` (shap, drift, briefs, quality) | **Yes** | Read by dashboard tabs |
| `data/alerts.db`, `data/quality.db` | No | SQLite DBs reset on deploy anyway |
| `serving/models/**` | No | Deprecated duplicate store; API reads directly from `data/` |
| `serving/logs/*.jsonl` | No | Ephemeral (resets on Render restart) |

After retraining, run `make prepare-deploy` then `git push` — both cloud platforms redeploy automatically.

---

## Production Deployment

The system is deployed as two hosted services following the Train → API → Deploy pattern:

| Service | Platform | URL |
|---------|----------|-----|
| FastAPI inference API | Render.com | https://community-crisis-predictor.onrender.com |
| Streamlit dashboard | Streamlit Cloud | https://community-crisis-predictor.streamlit.app |

> **Cold-start note (free Render tier):** the API sleeps after 15 min of inactivity. The first
> request after sleep takes ~30–60 s. Hit `/health` once before a live demo to wake it.

### Local → Cloud in 2 commands

```bash
# 1. Run pipeline to refresh tracked data artifacts (real data, or add --synthetic for quick test)
make prepare-deploy

# 2. Commit and push — Render + Streamlit Cloud auto-redeploy
git add . && git commit -m "Update model artifacts" && git push
```

### One-click retrain on GitHub Actions

Go to **Actions → Retrain (Synthetic)** → **Run workflow**. This retrains on synthetic data,
commits the artifacts back to the repo, and triggers both cloud platforms to redeploy automatically.

### Local API demo (Render fallback)

If the Render service is cold during a live demo, run the API locally instead:

```bash
make serve-local        # starts FastAPI at http://localhost:8000
# then visit http://localhost:8000/docs for interactive Swagger UI
```

Set `API_URL=http://localhost:8000` in the Streamlit run environment for the same demo experience.

### Streamlit Cloud secrets

In Streamlit Cloud → Advanced Settings → Secrets:

```toml
API_URL = "https://community-crisis-predictor.onrender.com"
API_MODE = "true"
```

When `API_MODE=true`, the dashboard sidebar shows a live API connection status indicator.
When the API is unreachable, the dashboard automatically falls back to local pipeline outputs.

### Local Streamlit config (no shell export needed)

For local runs, you can set the same values in Streamlit secrets:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit:

```toml
API_MODE = "true"
API_URL = "http://127.0.0.1:8000"
```

Config precedence in `src/dashboard/app.py` is:
1) `st.secrets` (local `.streamlit/secrets.toml` or Streamlit Cloud Secrets)
2) environment variables (`API_MODE`, `API_URL`)

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, loaded models list |
| `/predict` | POST | XGB + optional LSTM inference, drift warnings |
| `/model-info` | GET | Walk-forward metrics + top SHAP features |
| `/logs/summary` | GET | Aggregate prediction log statistics |
| `/docs` | GET | Interactive Swagger UI (auto-generated) |

See `serving/README.md` for full endpoint documentation and local run instructions.

---

## Step-by-Step Commands Reference

| Command | What it does |
|---------|--------------|
| `make collect-synthetic` | Generate 2 years of synthetic Reddit data |
| `make collect` | Collect from configured source in `collection.source` |
| `make features` | Build weekly feature matrix (skips if cache says inputs unchanged; use `--force` on the underlying command to rebuild) |
| `make train` | Train LSTM + XGBoost, save `eval_results.json` + model pkl/pt files |
| `make evaluate` | Generate structured per-subreddit reports (HTML, SHAP, drift, weekly briefs), populate alerts.db |
| `make all-synthetic` | Run the full pipeline end-to-end with synthetic data |
| `make prepare-deploy` | Run full pipeline and refresh `data/features`, `data/models`, `data/reports` |
| `make serve-local` | Start FastAPI inference service at http://localhost:8000 |
| `make test` | Run all unit tests |
| `make clean` | Delete all generated data files |
| `streamlit run src/dashboard/app.py` | Launch the live Streamlit dashboard |

For faster runs during development, append flags:
```bash
python -m src.pipeline.run_all --synthetic --skip-topics --skip-search
python -m src.pipeline.run_features --config config/default.yaml --force   # always rebuild features
python -m src.pipeline.run_train --skip-lstm    # XGBoost only
python -m src.pipeline.run_train --skip-search  # LSTM + XGBoost, no hyperparam search
```

---

## Running Tests

```bash
make test
# or
python -m pytest tests/ -v          # 72 core tests
pytest serving/tests/ -v            # 29 API tests (MOCK_MODELS=true)
```

Unit tests cover collectors, features, labeling, modeling splits, narration helpers, decision-usefulness metrics, dashboard state/ensemble helpers, demo_utils (scenario mapping tests), text processing, and the FastAPI inference service (all endpoints, validation, drift detection, log aggregation).
