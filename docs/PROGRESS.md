# Progress Tracker

## Phase 1: Foundation
- [x] Project directory structure
- [x] pyproject.toml with dependencies
- [x] .gitignore, .env.example, Makefile
- [x] Config YAML + lexicon files
- [x] Config loader module (src/config.py)
- [x] Privacy module (src/collector/privacy.py)
- [x] Storage module (src/collector/storage.py)

## Phase 2: Data Collection
- [x] Synthetic data generator (src/collector/synthetic.py)
- [x] Reddit client (src/collector/reddit_client.py)
- [x] Historical loader (src/collector/historical_loader.py)
- [x] Collection CLI (src/pipeline/run_collect.py)

## Phase 3: Processing & Features
- [x] Text cleaner (src/processing/text_cleaner.py)
- [x] Weekly aggregator (src/processing/weekly_aggregator.py)
- [x] Linguistic features (src/features/linguistic.py)
- [x] Sentiment features (src/features/sentiment.py)
- [x] Distress features (src/features/distress.py)
- [x] Behavioral features (src/features/behavioral.py)
- [x] Topic features (src/features/topics.py)
- [x] Temporal features (src/features/temporal.py)
- [x] Feature pipeline (src/features/pipeline.py)

## Phase 4: Labeling & Modeling
- [x] Distress score (src/labeling/distress_score.py)
- [x] Target labeling (src/labeling/target.py)
- [x] Walk-forward splitter (src/modeling/splits.py)
- [x] XGBoost training (src/modeling/train_xgb.py)
- [x] Evaluation (src/modeling/evaluate.py)
- [x] SHAP explainability (src/modeling/explain.py)

## Phase 5: Visualization
- [x] Backtesting timeline (src/visualization/timeline.py)
- [x] Feature importance plots (src/visualization/feature_importance.py)
- [x] Case study generator (src/visualization/case_study.py)
- [x] Dashboard (src/visualization/dashboard.py)

## Phase 6: Pipeline & Testing
- [x] CLI scripts (run_collect, run_features, run_train, run_evaluate, run_all)
- [x] Unit tests (33 tests, all passing)
- [x] End-to-end pipeline verified with synthetic data
- [x] README.md

## Results (Synthetic Data, skip-topics, skip-search)

| Subreddit | Recall | Precision | F1 | PR-AUC |
|-----------|--------|-----------|-----|--------|
| r/anxiety | 0.333 | 0.375 | 0.353 | 0.581 |
| r/depression | 0.444 | 0.727 | 0.552 | 0.656 |

Reports generated in `data/reports/`:
- Interactive timelines (HTML)
- Feature importance charts (SHAP)
- 6 case study narratives (markdown)
- 2 full dashboards (HTML)
