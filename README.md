# Community Mental Health Crisis Predictor

Predict community-wide mental health crises from Reddit data before they happen.

This system treats subreddits like r/depression and r/anxiety as weather systems — analyzing weekly patterns in language, sentiment, and behavior to forecast whether the community's aggregate distress will spike next week.

## How It Works

1. **Collect** posts from target subreddits (or generate synthetic data for development)
2. **Extract** weekly features: linguistic patterns, sentiment scores, distress lexicon density, topic distributions, behavioral signals
3. **Label** each week based on whether the *following* week's distress score exceeds a crisis threshold
4. **Train** an XGBoost model using walk-forward time-series cross-validation
5. **Evaluate** with SHAP explainability, backtesting timelines, and narrative case studies

The model never sees future data — it learns which combinations of signals (hopelessness keyword spikes, "I" vs "we" pronoun shifts, topic distribution changes) reliably precede community deterioration.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run full pipeline with synthetic data (no API keys needed)
make all-synthetic

# Or step by step:
make collect-synthetic   # Generate synthetic Reddit data
make features            # Extract feature matrix
make train               # Train model + evaluate
make evaluate            # Generate visualizations + reports
```

## Using Real Reddit Data

1. Create a Reddit app at https://www.reddit.com/prefs/apps (free)
2. Copy `.env.example` to `.env` and fill in credentials
3. Run `make collect` followed by `make features`, `make train`, `make evaluate`

## Project Structure

```
src/
├── collector/       Data collection (synthetic, PRAW, Pushshift)
├── processing/      Text cleaning + weekly aggregation
├── features/        Feature extraction (linguistic, sentiment, distress, topics, behavioral, temporal)
├── labeling/        Distress scoring + crisis target generation
├── modeling/        Walk-forward CV, XGBoost training, SHAP explainability
├── visualization/   Timeline plots, feature importance, case studies, HTML dashboard
└── pipeline/        CLI entry points for each stage
```

## Output

After running the pipeline, find results in `data/reports/`:
- **Timeline HTML** — Interactive backtesting plot showing predictions vs actual crises
- **Feature importance** — SHAP-based ranking of most predictive features
- **Case studies** — Markdown narratives for specific crisis events the model detected
- **Dashboard** — Single HTML page combining all artifacts

## Configuration

All parameters are in `config/default.yaml`:
- Subreddits, date ranges
- Feature extraction settings (sentiment bins, topic count, rolling windows)
- Labeling (distress weights, crisis threshold)
- Model hyperparameters (XGBoost grid, walk-forward splits)

## Future Work

- **GRU/LSTM sequence model** — Capture temporal dynamics by training on sequences of weekly feature vectors
- **Ensemble** — Combine XGBoost + RNN predictions via stacking
- **Real-time monitoring** — Stream Reddit data and flag emerging crises
- **Multi-subreddit generalization** — Test on r/mentalhealth, r/SuicideWatch, etc.

## Running Tests

```bash
make test
```
