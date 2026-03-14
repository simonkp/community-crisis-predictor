# Project Plan

## Overview
Build an early warning system that treats Reddit communities (r/depression, r/anxiety) as weather systems. Predict whether next week's aggregate community distress will spike, using only this week's posts as input.

## Architecture
```
collect → clean → aggregate (weekly) → extract features → label → train XGBoost → evaluate → visualize
```

## Pipeline Stages

### 1. Data Collection
- Synthetic data generator for development (no API keys needed)
- PRAW + Pushshift for real Reddit data when credentials available
- Privacy-first: hash authors, strip PII before storage
- Parquet storage partitioned by subreddit/year

### 2. Processing
- Text cleaning (lowercase, URL removal, encoding normalization)
- Weekly aggregation by ISO week with author tracking

### 3. Feature Engineering (5 families)
- **Linguistic:** word count, type-token ratio, readability, pronoun ratios (I/me vs we/us)
- **Sentiment:** VADER scores, sentiment distribution buckets
- **Distress:** lexicon-based hopelessness/help-seeking/distress density
- **Behavioral:** post volume, comments, unique posters, posting-time entropy
- **Topics:** BERTopic topic distribution, topic entropy, week-over-week topic shift (JSD)
- **Temporal:** deltas, rolling averages (2w/4w), cyclical seasonality encoding

### 4. Labeling
- Composite distress score = weighted z-scores of neg_sentiment + hopelessness + help_seeking
- Crisis label: y[t] = 1 if distress_score[t+1] > mean + 1.5·std
- Threshold recomputed per CV fold to prevent leakage

### 5. Modeling
- XGBoost with walk-forward time-series cross-validation
- Min 26-week training window, 1-week gap, expanding window
- Hyperparameter search via RandomizedSearchCV
- SHAP for feature importance and per-prediction explanations

### 6. Evaluation & Visualization
- Metrics: recall (primary), precision, F1, PR-AUC, detection lead time
- Backtesting timeline (Plotly interactive HTML)
- Feature importance (SHAP bar + beeswarm)
- Case study reports (markdown narratives)
- HTML dashboard assembling all artifacts

## Future Work
- GRU/LSTM sequence models on weekly feature vectors
- Ensemble (XGBoost + GRU stacking)
- Real-time monitoring mode with streaming Reddit data
- Multi-subreddit generalization testing
