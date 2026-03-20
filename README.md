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

---

## How It Works

1. **Collect** — Fetch posts from Reddit via PullPush.io (free, no payment needed) or generate synthetic data
2. **Extract** — Build weekly feature vectors: linguistic patterns, VADER sentiment, distress lexicon density, topic distributions (BERTopic), behavioral signals, JS-divergence topic drift (1-week and 4-week lookback)
3. **Label** — Score each week's distress; classify next week into one of 4 states using community-specific baselines
4. **Train** — Two models run in parallel:
   - **LSTM** (primary) — PyTorch sequence model; sees the last 8 weeks as context
   - **XGBoost** (baseline) — Binary crisis classifier; trained on the same walk-forward splits
5. **Monitor** — Rolling z-score drift detection flags sudden signal changes; an alert engine logs state transitions to SQLite
6. **Visualize** — Streamlit live dashboard with week-by-week replay, or static HTML reports

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
python -m src.pipeline.run_all --config config/default.yaml --synthetic --skip-topics --skip-search

# 4. Launch the Streamlit dashboard
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser. Use the **week slider** in the sidebar to replay predictions week by week, and the **Advance / Back** buttons to step through the timeline.

---

## Using Real Reddit Data (Free)

Real data is fetched from [PullPush.io](https://pullpush.io) — a free, publicly accessible archive of Reddit posts. **No Reddit account, no API keys, and no credentials are required.** The collector just sends plain HTTP requests to the PullPush API.

The only thing you need to set is a privacy salt (used to hash author usernames before storing):

```bash
cp .env.example .env
```

Edit `.env` — you only need one line:

```
PRIVACY_SALT=any_random_string_here
```

The `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` fields in `.env` are only used by the PRAW fallback, which activates automatically if PullPush.io is unreachable. You can leave them blank unless you specifically want to use PRAW.

### Run the pipeline

```bash
# Collect real posts (2024-01-01 to 2026-03-01 by default)
# This takes 20–40 minutes due to API rate limiting (~1 req/sec)
python -m src.pipeline.run_collect --config config/default.yaml

# Extract features
python -m src.pipeline.run_features --config config/default.yaml

# Train LSTM + XGBoost, print comparison table
python -m src.pipeline.run_train --config config/default.yaml

# Generate reports, SHAP importance, drift alerts
python -m src.pipeline.run_evaluate --config config/default.yaml

# Open the live dashboard
streamlit run src/dashboard/app.py
```

**Expected data volume:** ~6,000–15,000 posts per subreddit. The 2-year date range gives ~104 weeks of training data per subreddit.

### Demo split (train vs live)

For a convincing live demo:
- Edit `config/default.yaml` → set `date_range.end: "2024-12-31"` for the training run
- Re-run from `run_collect` through `run_train`
- Then set `date_range.start: "2025-01-01"`, `end: "2026-03-01"` and re-collect for the "live" weeks
- The dashboard's week slider replays the 2025–2026 weeks as if they were arriving in real time

---

## Step-by-Step Commands Reference

| Command | What it does |
|---------|--------------|
| `make collect-synthetic` | Generate 2 years of synthetic Reddit data |
| `make collect` | Fetch real posts via PullPush.io |
| `make features` | Build weekly feature matrix |
| `make train` | Train LSTM + XGBoost, save `eval_results.json` |
| `make evaluate` | Generate HTML reports, SHAP CSV, drift alerts, populate alerts.db |
| `make all-synthetic` | Run the full pipeline end-to-end with synthetic data |
| `make test` | Run all 33 unit tests |
| `make clean` | Delete all generated data files |
| `streamlit run src/dashboard/app.py` | Launch the live Streamlit dashboard |

For faster runs during development, append flags:
```bash
python -m src.pipeline.run_all --synthetic --skip-topics --skip-search
python -m src.pipeline.run_train --skip-lstm    # XGBoost only
python -m src.pipeline.run_train --skip-search  # LSTM + XGBoost, no hyperparam search
```

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
├── visualization/    Static HTML reports
│   ├── timeline.py            4-color backtesting timeline (Plotly)
│   ├── feature_importance.py  SHAP bar chart
│   ├── case_study.py          Narrative markdown case studies
│   └── dashboard.py           Combined HTML report
├── dashboard/
│   └── app.py                 Streamlit live replay dashboard
└── pipeline/         CLI entry points
    ├── run_collect.py
    ├── run_features.py
    ├── run_train.py
    ├── run_evaluate.py
    └── run_all.py
```

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
├── raw/{subreddit}/posts.parquet          Raw collected posts
├── features/features.parquet             Weekly feature matrix
├── models/eval_results.json              XGB + LSTM walk-forward metrics
├── alerts.db                             SQLite log of state transitions
└── reports/
    ├── {sub}_timeline.html               4-color interactive backtesting plot
    ├── {sub}_feature_importance.html     SHAP top-20 feature chart
    ├── {sub}_shap.csv                    SHAP values for dashboard
    ├── {sub}_drift_alerts.json           Rolling z-score drift alerts
    ├── {sub}_case_study_*.md             Narrative high-distress case studies
    └── {sub}_dashboard.html             Combined HTML report
```

The Streamlit dashboard reads from `data/features/`, `data/models/eval_results.json`, `data/reports/{sub}_shap.csv`, `data/reports/{sub}_drift_alerts.json`, and `data/alerts.db`.

---

## Running Tests

```bash
make test
# or
python -m pytest tests/ -v
```

33 unit tests covering collectors, features, labeling, modeling splits, and text processing.
