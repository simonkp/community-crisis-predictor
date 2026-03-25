## Project & repo
- **Project**: `community-crisis-predictor` (NUS IS5126 group work) — Reddit → weekly features → XGBoost + LSTM crisis prediction, Streamlit dashboard.
- **Path**: `c:\dev\learn\NUS\IS5126\project\community-crisis-predictor`
- **Env**: Windows PowerShell; use **venv**: `.\venv\Scripts\Activate.ps1` before `python`/`pytest`/pip.

## Data source & architecture
- **Primary source**: Zenodo COVID Reddit dataset — **raw posts only** (`selftext`, `created_utc`, `subreddit`, etc.); ignore precomputed TF-IDF/LIWC-style columns.
- **Config-driven** source selection (`synthetic` / `zenodo` / `reddit_api`); paths centralized in `config/default.yaml` under `paths` (staging, archive, quality DB, alerts DB, etc.).
- **`data_source`** column for provenance (`zenodo_covid`, etc.).
- **Zenodo loader**: selective file download, manifest, `_normalize_schema` builds stable `post_id` as `zenodo_<subreddit>_<source_slug>_<row>` to avoid **cross-file ID collisions** and massive `drop_duplicates` loss.

## Pipeline behavior you should know
- **`run_collect`**: Verbose prints per file; avoid Unicode arrows in prints on Windows (use `->`).
- **`run_features`**: **Skips by default** when raw parquets + `--skip-topics` match + **fingerprint of relevant config** (`reddit.subreddits`, `processing`, `features`, `modeling.lstm.sequence_length`) matches `feature_build_meta.json`. **`--force`** recomputes. Consolidated **per-subreddit summary table** printed **before** heavy work and again after; explains overlapping calendar weeks vs union “414 weeks”.
- **Progress**: `tqdm` via `src/features/progress_util.py` in linguistic/sentiment/distress/behavioral/topics/temporal loops.
- **`run_train`**: Class-weighted **LSTM** `CrossEntropyLoss` in `train_rnn.py`; **anomaly flags** + **section 3 summary table** in `run_train.py`.
- **`run_all`**: Stage timing table; reports path from config.

## Dashboard
- **`app.py`**: all-community **card row** (severity order), global **week replay** in header, **model** selector (LSTM / XGBoost / Ensemble when both exist); **two-column** main (distress timeline + weekly brief / metric tiles); **tabs** for drift, SHAP, data quality. Session keys: `selected_sub`, `current_week`, `selected_model`. STePS / scenario demo UI **removed** (see README).
- Logic remains in `data_access.py`, `briefs.py`, `state.py` (incl. ensemble merge), `charts.py`, `components.py`, `types.py`.
- **Fixes**: `week_idx` clamped / empty-data guards; shorter subs use `week_idx_plot` without rewriting global slider.
- **Monitoring mode**: `MONITORING_MODE` subs (`lonely`, `mentalhealth`) show **Trend monitoring** pill when `n_crisis_actual` is below `monitoring_min_crisis_weeks`.

## Docs & repo hygiene
- **`README.md`**: Updated for Zenodo, feature cache semantics, dashboard module layout, `data/` gitignored.
- **`.gitignore`**: Broad ignore of `data/` (large/generated).

## Audit / temp scripts
- `scripts/tmp_zenodo_depression_lonely_audit.py` and `scripts/tmp_zenodo_all_subs_audit.py` — CSV vs parquet checks; guard `source_file` when column missing.

## Verification
- User expects **lint/build/tests** after substantive changes (`pytest`, etc. in venv).

## GitHub issues context (historical)
- **#26**: Pipeline reliability / quality gates.
- **#28**: Source selector + Zenodo ingestion (implemented earlier in the thread).

