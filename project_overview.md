# Community Mental Health Crisis Predictor - Project Overview

This document serves as a detailed report and context guide for other AI agents working on this project. It outlines the project's goals, architecture, data flow, and current state.

## 1. Project Goal & Core Concept

**Goal:** Forecast community-wide mental health crises from Reddit data (such as `r/depression` and `r/anxiety`) up to a week before they escalate.

**Core Concept:** The system treats online subreddits like weather systems. Instead of looking at individual users, it analyzes weekly aggregate patterns—such as language readability, sentiment distributions, hopelessness/help-seeking lexicon density, and topic drift. By continuously monitoring these features over time, the system can predict whether the overall community distress level will escalate in the next week, and categorize it into one of four states based on standard deviations (σ) from the community baseline:
- State 0: Stable (< 0.5σ)
- State 1: Early Vulnerability Signal (0.5σ - 1.0σ)
- State 2: Elevated Distress (1.0σ - 2.0σ)
- State 3: Severe Community Distress Signal (> 2.0σ)

## 2. Architecture & Data Pipeline

The project follows a standard machine learning pipeline structured into several distinct phases:

1. **Collect (`src/collector/`)**:
   - Gathers historical text data.
   - Supported sources: `zenodo_covid` (primary academic dataset), `arctic_shift` (gap-fill dataset), `reddit_api` (via PullPush.io/PRAW fallback), and `synthetic` (auto-generated testing data).
   - Scripts: `run_collect.py`
2. **Feature Extraction (`src/features/`)**:
   - Compiles weekly feature vectors from the raw data.
   - Dimensions tracked: Linguistic patterns (pronouns, readability), Sentiment (VADER), Distress (hopelessness lexicons), Topics (BERTopic, JS-divergence for drift), and Temporal behavioral signals (post volume, active users).
   - Scripts: `run_features.py`
3. **Labeling (`src/labeling/`)**:
   - Scores each week's distress level.
   - Calculates the rolling community baseline and classifies the *next* week into one of the 4 crisis states.
4. **Training (`src/modeling/`)**:
   - Implements two parallel models inside a Walk-Forward Validation time-series splits model (to prevent data leakage):
     - **LSTM Modeler:** PyTorch-based sequential model (primary). Context window: last 8 weeks.
     - **XGBoost Modeler:** Tree-based binary baseline.
   - Metrics optimized: Recall@K (Decision Usefulness), PR-AUC.
   - Scripts: `run_train.py`
5. **Monitoring & Alerts (`src/monitoring/`)**:
   - Detects drift using rolling z-scores across weekly signals. Logged to `data/alerts.db`.
6. **Evaluation & Reporting (`src/visualization/` & `run_evaluate.py`)**:
   - Generates static dashboard HTML, SHAP top-feature explanations, drift JSON, and markdown timeline reports.
   - Creates a **Weekly Narrative Brief** using LLMs (Claude 3.5 Sonnet / GPT-4o) using retrieved system prompt instructions from `config/intervention_playbook.md`.
7. **Visualization Dashboard (`src/dashboard/app.py`)**:
   - A live Streamlit application that provides a unified, interactive view into all communities, timelines, predicted signals, models feature importances, and synthesized text briefs.

## 3. Technology Stack

- **Core Python:** Pandas, Numpy, Pydantic, SQLite
- **Machine Learning:** PyTorch (LSTM), XGBoost, scikit-learn
- **NLP & Features:** BERTopic, VADER (NLTK), textstat
- **Interface & Viz:** Streamlit, Plotly, HTML/CSS/Markdown
- **Automation / MLOps:** argparse CLI pipeline (`run_all.py`), Makefiles, Github Actions
- **LLM Integration:** Anthropic / OpenAI APIs

## 4. Key Considerations for Agents

- **Idempotency Strategy:** Features and data downloading try to avoid recomputation. `manifest.json` handles dataset state. `feature_build_meta.json` caches extraction state. If modifying `src/features/`, you might need to run `run_features.py --force`.
- **Git Strategy:** The ENTIRE `data/` directory is present in `.gitignore`. **Do not attempt to commit raw `.parquet`, `.db`, or `.html` generated assets to the `main` branch**. Only commit code, configuration, markdown, and tests.
- **Config-Driven:** The core behavior, thresholds, selected models, subreddits, date windows, and feature configurations are stored in `config/default.yaml`. Start any behavior changes there before modifying complex source code.
- **Naming Conventions:** Community is often used synonymously with Subreddit. The weekly dimension expects Monday-start weeks.
- **Running:** It is highly recommended to run tests or quick dry-runs using the `--synthetic` and `--skip-topics` and `--skip-search` flags to avoid long 10-minute extraction cycles when iterating code.

## 5. End-to-End Execution Flow

When testing full pipeline integration, use:
```bash
python -m src.pipeline.run_all --config config/default.yaml --synthetic --skip-topics --skip-search --force
```
Then start the resulting UI dashboard:
```bash
streamlit run src/dashboard/app.py
```

## 6. Latest Preprocessing Reliability Additions

- **Leak-proof distress scoring:** Labeling now builds distress scores without normalizing over the entire timeline; models normalize per training fold to avoid future leakage and keep thresholds stable.
- **Calendar-complete aggregator:** Weekly aggregation creates placeholder rows for missing ISO weeks, marks them with `is_missing_week`, and zero-fills counts so temporal indicators and downstream features remain deterministic.
- **Artifact schema guards:** Raw, weekly, and feature outputs enforce required columns/IDs before writing to `data/raw`, `data/processed`, and `data/features`, preventing silent corruption when new sources add fields.
- **Idempotent provenance logging:** The data-quality store now enforces a `(subreddit, week, source)` unique key and upserts timestamps, so repeated collection runs do not flood the provenance table.
- **Stable post identity & dedupe hardening:** Both `ArcticShiftLoader` and `ZenodoLoader` now enforce non-empty `post_id` values. Any record whose source ID is absent, `"nan"`, or `"null"` receives a deterministic `hash_<sha256[:16]>` fallback derived from `subreddit + created_utc + selftext[:200]`, ensuring `drop_duplicates(subset=["post_id"])` in `run_collect.py` is never defeated by colliding empty strings.
- **Token-aware lexicon matching:** `DistressScorer` now compiles `\b`-bounded regex patterns at init time for all three lexicons (hopelessness, help-seeking, distress). Matching uses `re.findall` instead of plain substring search, eliminating false positives such as "panic" firing inside "panicking" or unrelated compound tokens.
- **Selective `fillna` with core-input assertion:** `FeaturePipeline.run()` no longer applies a blanket `fillna(0.0)` over the entire feature matrix. Only `_delta` columns (which legitimately produce NaN on the first row of each subreddit series from `.diff()`) are zero-filled. A post-fill assertion raises `ValueError` if any core (non-meta, non-temporal) column contains nulls, surfacing upstream extractor bugs instead of silently masking them.
- **Content-hash cache invalidation:** The feature-build fingerprint in `run_features.py` now includes a SHA-256 hash of each raw `posts.parquet` file in addition to mtime/size. In-place overwrites that preserve file metadata but change content (e.g. re-collected data written to the same path) now correctly invalidate the cache and trigger a full feature rebuild.

## 7. Handoff Orientation Checklist

1. **Environment**
   - Python 3.12.11 via `pyenv` (see `.python-version`). Install with `pip install -e ".[dev]"`.
   - `streamlit` and other CLI tools already used in prior automation (run `pip install -r serving/requirements.txt` if needed).

2. **Data Artifacts**
   - Raw downloads live under `data/raw/{subreddit}/posts.parquet`.
   - Weekly aggregates are saved at `data/processed/weekly.parquet`.
   - Feature matrix and models stored in `data/features/features.parquet` and `data/models/{sub}_*.pkl/.pt`.
   - Reports (`data/reports/{sub}`) and quality DBs (`data/alerts.db`, `data/quality.db`) should stay out of commits unless explicitly requested.

3. **Core Commands**
   - Collection: `python -m src.pipeline.run_collect --config config/default.yaml`.
   - Feature extraction: `python -m src.pipeline.run_features --config config/default.yaml`.
   - Training: `python -m src.pipeline.run_train --config config/default.yaml`.
   - Evaluation: `python -m src.pipeline.run_evaluate --config config/default.yaml`.

4. **Reliability Notes**
   - If modifying `src/features` or `src/processing`, rerun `run_features` with `--force` to refresh fingerprints.
   - Always validate new collectors against the schema guard in `src/collector/storage.py`; failing to provide required columns will raise before overwriting artifacts.
   - Use the weekly `is_missing_week` flag to detect data gaps before training; downstream code now assumes missing weeks exist for temporal consistency.

5. **Next-to-Do Ideas**
   - Implement cross-source schema validation (source-aware compatibility) upstream in `run_collect.py`.
   - Add automated regression tests covering weekly gaps, schema violations, ID hash fallback paths, and lexicon boundary matching in CI or smoke scripts.
