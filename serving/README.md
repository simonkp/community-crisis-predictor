# Community Crisis Predictor — Inference Service

FastAPI REST service that loads trained XGBoost + LSTM models and serves predictions
for each monitored subreddit, with drift detection and inference logging.

---

## Live URLs

| Service | URL |
|---------|-----|
| API root / health | https://community-crisis-predictor.onrender.com/health |
| Swagger UI (interactive docs) | https://community-crisis-predictor.onrender.com/docs |
| Streamlit dashboard | https://community-crisis-predictor.streamlit.app |

> **Cold-start warning (free Render tier):** the service sleeps after 15 min of
> inactivity. The first request after sleep takes **30–60 seconds** to respond while
> the container wakes up. Subsequent requests are fast (< 1 s). Before a live demo,
> hit `/health` once to wake the service.

---

## Deployment Setup Guide (First Run)

### Accounts you need

Create/sign in to these accounts before first deployment:

1. **GitHub** (repo host)
2. **Render** ([render.com](https://render.com)) for FastAPI API hosting
3. **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io)) for dashboard hosting

### Repository prerequisites

Before connecting cloud services, ensure:

- Repo is pushed to GitHub
- Required tracked artifacts exist:
  - `data/features/features.parquet`
  - `data/models/*` (xgb/lstm/stats/eval)
  - `data/reports/*` (including `shap.csv`)
- If artifacts are stale, refresh first:

```bash
make prepare-deploy
git add . && git commit -m "Refresh deployment artifacts" && git push
```

### First-time API deployment (Render)

1. In Render dashboard: **New** -> **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Root directory**: `serving`
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance type**: Free (or paid if you want fewer cold starts)
4. Deploy and wait for first build
5. Verify:
   - `https://<your-render-app>.onrender.com/health`
   - `https://<your-render-app>.onrender.com/docs`

> No mandatory env vars for basic deployment because model/report artifacts are read from `../data`.

### First-time dashboard deployment (Streamlit Cloud)

1. In Streamlit Cloud: **New app**
2. Select the same GitHub repo/branch
3. Set entrypoint file to:
   - `src/dashboard/app.py`
4. In **Advanced settings -> Secrets**, add:

```toml
API_MODE = "true"
API_URL = "https://<your-render-app>.onrender.com"
```

5. Deploy and verify the sidebar shows API status.

---

## Ongoing Deploy Flow (After First Run)

### Typical update cycle

1. Rebuild artifacts locally:
   - real data: `make prepare-deploy`
   - synthetic fast path: `make prepare-deploy-synthetic`
2. Commit + push:

```bash
git add .
git commit -m "Update pipeline artifacts"
git push
```

3. Auto-redeploy behavior:
   - **Render** rebuilds API automatically from latest commit
   - **Streamlit Cloud** redeploys dashboard automatically from latest commit

### If only code changes (no data refresh)

Just commit/push code; no need to rerun full pipeline.

### If API URL changes

Update `API_URL` in Streamlit Cloud Secrets and redeploy/restart app.

---

## Run locally

```bash
# From the repo root — installs serving deps into current env
pip install -r serving/requirements.txt

# Start server (auto-reload on file changes)
cd serving
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Quick test (PowerShell)

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# Predict with a minimal feature vector
$body = @{
    subreddit = "depression"
    week_start = "2020-03-09"
    features = @{ hopelessness_density = 0.05; avg_negative_roll4w = 0.32 }
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $body -ContentType "application/json"
```

### Quick test (bash / curl)

```bash
curl http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subreddit":"depression","week_start":"2020-03-09","features":{"hopelessness_density":0.05}}'
```

---

## Model artifacts

Model files live under the repository `data/` directory and are committed to the
repo so Render can access them without a separate model store.

| File pattern | Source | Description |
|---|---|---|
| `{sub}_xgb.pkl` | `data/models/` | Serialized XGBClassifier (last walk-forward fold) |
| `{sub}_lstm.pt` | `data/models/` | PyTorch state dict + architecture metadata |
| `{sub}_feature_stats.json` | `data/models/` | Training distribution stats for drift detection |
| `eval_results.json` | `data/models/` | Walk-forward metrics for `/model-info` |
| `shap.csv` | `data/reports/{sub}/` | SHAP feature importance for `/model-info` |

Refresh artifacts after retraining:

```bash
make prepare-deploy   # runs full pipeline and refreshes data artifacts
git add . && git commit -m "Update model artifacts" && git push
```

---

## Endpoints

### `GET /health`
Returns service status and list of loaded subreddits.

### `POST /predict`
Runs inference on a weekly feature vector.

**Request body:**
```json
{
  "subreddit": "depression",
  "week_start": "2020-03-09",
  "features": { "hopelessness_density": 0.05, "avg_negative_roll4w": 0.32 },
  "feature_history": [ {...week-8 features...}, ..., {...current-week features...} ]
}
```

- `features` — current week's feature vector (required for XGB)
- `feature_history` — optional list of N=8 consecutive weekly feature dicts; enables LSTM

**Response:**
```json
{
  "subreddit": "depression",
  "week_start": "2020-03-09",
  "prediction_available": true,
  "xgb": { "predicted_state": 2, "predicted_state_label": "Elevated Distress", "crisis_probability": 0.73 },
  "lstm": { "predicted_state": 2, "predicted_state_label": "Elevated Distress", "class_probabilities": [0.1, 0.15, 0.6, 0.15] },
  "ensemble": { "predicted_state": 2, "predicted_state_label": "Elevated Distress", "crisis_probability": 0.67 },
  "drift_warnings": ["hopelessness_density: value 0.15 is 3.2 std from training mean (0.04)"],
  "latency_ms": 45
}
```

### `GET /model-info`
Walk-forward metrics (recall, precision, F1, PR-AUC) and top-5 SHAP features per subreddit.

### `GET /logs/summary`
Aggregate statistics from `logs/predictions.jsonl`.

---

## Deployment on Render.com

1. Push repo to GitHub (must be public for free tier)
2. Render → **New → Web Service** → connect repo
3. Set **Root directory:** `serving`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Instance type: **Free**
7. No environment variables required (artifacts are committed to repo)

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `../data/models` | Path to model artifact directory |
| `SHAP_DIR` | `../data/reports` | Path to subreddit report folders containing `shap.csv` |
| `MOCK_MODELS` | `false` | Set to `true` to start without real model files (CI/testing) |

---

## Ephemeral storage note

On the Render free tier, `logs/predictions.jsonl` **resets to empty each time the
service restarts** (ephemeral filesystem). This is documented and expected. The log
is useful for demo monitoring within a session but does not persist across deployments.
For persistent logging, upgrade to a paid Render plan or write to an external store.
