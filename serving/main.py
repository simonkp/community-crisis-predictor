"""
Community Mental Health Crisis Predictor — FastAPI Inference Service.

Endpoints
---------
GET  /health          Service status and loaded models list.
POST /predict         Run XGB + optional LSTM inference on a weekly feature vector.
GET  /model-info      Walk-forward metrics and top SHAP features per subreddit.
GET  /logs/summary    Aggregate statistics from the prediction log.

Run locally
-----------
    cd serving
    uvicorn main:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants (inlined — serving layer must not import from src/)
# ---------------------------------------------------------------------------

STATE_NAMES: dict[int, str] = {
    0: "Stable",
    1: "Early Vulnerability Signal",
    2: "Elevated Distress",
    3: "Severe Community Distress Signal",
}

# Canonical subreddit names as they appear in model filenames (lowercase from pipeline)
FULL_MODEL_SUBS = {"depression", "anxiety", "suicidewatch"}
MONITORING_ONLY_SUBS = {"lonely", "mentalhealth"}
ALL_SUBS = FULL_MODEL_SUBS | MONITORING_ONLY_SUBS

# Accept user-facing aliases (e.g. "SuicideWatch" -> "suicidewatch")
_SUB_ALIASES: dict[str, str] = {s.lower(): s for s in ALL_SUBS}


def _normalize_sub(name: str) -> str:
    """Normalize subreddit name to the canonical lowercase used in model files."""
    return _SUB_ALIASES.get(name.lower(), name.lower())
VERSION = "1.0.0"
SEQUENCE_LENGTH_DEFAULT = 8
DRIFT_THRESHOLD_STD = 2.5

# ---------------------------------------------------------------------------
# LSTMNet (replicated here so serving/ has no dependency on src/)
# ---------------------------------------------------------------------------


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout_layer(out[:, -1, :])
        return self.fc(out)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../data/models"))
SHAP_DIR = Path(os.environ.get("SHAP_DIR", "../data/reports"))
MOCK_MODELS = os.environ.get("MOCK_MODELS", "false").lower() == "true"
_REPORTS_DIR_MAP: dict[str, str] = {"suicidewatch": "SuicideWatch"}

# Loaded at startup
_xgb_models: dict = {}
_lstm_models: dict = {}
_feature_stats: dict = {}
_eval_results: dict = {}
_shap_data: dict = {}
_feature_columns: list[str] = []
_xgb_calibrators: dict[str, dict] = {}
_lstm_calibrators: dict[str, dict] = {}
STRICT_MAX_SIGMA = float(os.environ.get("STRICT_MAX_SIGMA", "5.0"))
CLIP_OUTLIER_FEATURES = os.environ.get("CLIP_OUTLIER_FEATURES", "false").lower() == "true"


def _load_models() -> None:
    """Load all model artifacts at startup. Fails fast on critical missing files."""
    global _feature_columns

    if MOCK_MODELS:
        logging.warning("MOCK_MODELS=true — running without real model files (test mode)")
        return

    # eval_results.json
    er_path = MODEL_DIR / "eval_results.json"
    if er_path.exists():
        with open(er_path, encoding="utf-8") as f:
            _eval_results.update(json.load(f))
    else:
        logging.warning("eval_results.json not found in %s", MODEL_DIR)

    for sub in ALL_SUBS:
        # XGB
        xgb_path = MODEL_DIR / f"{sub}_xgb.pkl"
        if xgb_path.exists():
            try:
                import joblib
                _xgb_models[sub] = joblib.load(xgb_path)
                logging.info("Loaded XGB model for %s", sub)
            except Exception as exc:
                logging.error("Failed to load XGB for %s: %s", sub, exc)
        xgb_cal_path = MODEL_DIR / f"{sub}_xgb_calibrator.json"
        if xgb_cal_path.exists():
            try:
                with open(xgb_cal_path, encoding="utf-8") as f:
                    _xgb_calibrators[sub] = json.load(f)
            except Exception as exc:
                logging.warning("Failed to load XGB calibrator for %s: %s", sub, exc)

        # LSTM
        lstm_path = MODEL_DIR / f"{sub}_lstm.pt"
        if lstm_path.exists():
            try:
                ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
                net = LSTMNet(
                    input_size=ckpt["feature_size"],
                    hidden_size=ckpt.get("hidden_size", 64),
                    num_layers=ckpt.get("num_layers", 2),
                    num_classes=ckpt.get("num_classes", 4),
                    dropout=ckpt.get("dropout", 0.2),
                )
                net.load_state_dict(ckpt["state_dict"])
                net.eval()
                _lstm_models[sub] = {
                    "net": net,
                    "feature_size": ckpt["feature_size"],
                    "sequence_length": ckpt.get("sequence_length", SEQUENCE_LENGTH_DEFAULT),
                }
                logging.info("Loaded LSTM model for %s", sub)
            except Exception as exc:
                logging.error("Failed to load LSTM for %s: %s", sub, exc)
        lstm_cal_path = MODEL_DIR / f"{sub}_lstm_calibrator.json"
        if lstm_cal_path.exists():
            try:
                with open(lstm_cal_path, encoding="utf-8") as f:
                    _lstm_calibrators[sub] = json.load(f)
            except Exception as exc:
                logging.warning("Failed to load LSTM calibrator for %s: %s", sub, exc)

        # Feature stats
        stats_path = MODEL_DIR / f"{sub}_feature_stats.json"
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                _feature_stats[sub] = json.load(f)

        # SHAP
        shap_subdir = _REPORTS_DIR_MAP.get(sub, sub)
        shap_path = SHAP_DIR / shap_subdir / "shap.csv"
        if shap_path.exists():
            try:
                _shap_data[sub] = pd.read_csv(shap_path)
            except Exception as exc:
                logging.warning("Failed to load SHAP for %s: %s", sub, exc)

    # Derive feature columns from the first available feature_stats entry
    for sub_stats in _feature_stats.values():
        if "features" in sub_stats:
            _feature_columns = list(sub_stats["features"].keys())
            break

    loaded = list(_xgb_models.keys())
    if not loaded and not MOCK_MODELS:
        logging.warning(
            "No XGB models loaded from %s. Run `make prepare-deploy` first.", MODEL_DIR
        )


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def _check_drift(sub: str, features: dict[str, float]) -> list[str]:
    """Return a warning string for each feature > DRIFT_THRESHOLD_STD stds from training mean."""
    warnings: list[str] = []
    stats = _feature_stats.get(sub, {}).get("features", {})
    for feat, value in features.items():
        if feat not in stats:
            continue
        mean = stats[feat]["mean"]
        std = stats[feat]["std"]
        if std > 0 and abs(value - mean) > DRIFT_THRESHOLD_STD * std:
            z = (value - mean) / std
            warnings.append(
                f"{feat}: value {value:.4f} is {abs(z):.1f} std from training mean ({mean:.4f})"
            )
    return warnings


def _apply_calibrator(prob: float, calibrator: dict | None) -> float:
    p = float(np.clip(prob, 0.0, 1.0))
    if not calibrator or calibrator.get("type") in {None, "identity"}:
        return p
    ctype = calibrator.get("type")
    if ctype == "platt":
        coef = float(calibrator.get("coef", 1.0))
        intercept = float(calibrator.get("intercept", 0.0))
        z = coef * p + intercept
        return float(1.0 / (1.0 + np.exp(-z)))
    if ctype == "isotonic":
        xs = np.asarray(calibrator.get("x", []), dtype=float)
        ys = np.asarray(calibrator.get("y", []), dtype=float)
        if len(xs) >= 2 and len(xs) == len(ys):
            return float(np.interp(np.clip(p, xs.min(), xs.max()), xs, ys))
    return p


def _validate_or_clip_features(sub: str, features: dict[str, float]) -> tuple[dict[str, float], list[str], list[str]]:
    """
    Validate incoming features against training distribution.
    Returns: (possibly clipped features, clipped_messages, violation_messages)
    """
    stats = _feature_stats.get(sub, {}).get("features", {})
    if not stats:
        return dict(features), [], []
    clipped: list[str] = []
    violations: list[str] = []
    out = dict(features)
    for feat, value in out.items():
        feat_stats = stats.get(feat)
        if not feat_stats:
            continue
        std = float(feat_stats.get("std", 0.0))
        mean = float(feat_stats.get("mean", 0.0))
        if std <= 0:
            continue
        z = abs((float(value) - mean) / std)
        if z <= STRICT_MAX_SIGMA:
            continue
        lo = mean - STRICT_MAX_SIGMA * std
        hi = mean + STRICT_MAX_SIGMA * std
        if CLIP_OUTLIER_FEATURES:
            out[feat] = float(np.clip(float(value), lo, hi))
            clipped.append(
                f"{feat} clipped to [{lo:.4f}, {hi:.4f}] from {value:.4f} ({z:.1f}σ)"
            )
        else:
            violations.append(
                f"{feat}={value:.4f} outside allowed ±{STRICT_MAX_SIGMA:.1f}σ range "
                f"[{lo:.4f}, {hi:.4f}]"
            )
    return out, clipped, violations


# ---------------------------------------------------------------------------
# Inference logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"

_log_handler = logging.FileHandler(PREDICTION_LOG, encoding="utf-8")
_pred_logger = logging.getLogger("predictions")
_pred_logger.setLevel(logging.INFO)
_pred_logger.addHandler(_log_handler)
_pred_logger.propagate = False


def _log_prediction(entry: dict) -> None:
    _pred_logger.info(json.dumps(entry))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: "FastAPI"):  # noqa: F821
    logging.basicConfig(level=logging.INFO)
    _load_models()
    yield


app = FastAPI(
    title="Community Mental Health Crisis Predictor API",
    description=(
        "Inference service for the Community Mental Health Crisis Predictor. "
        "Accepts weekly community feature vectors and returns crisis state predictions "
        "from XGBoost and LSTM models, with drift detection and inference logging."
    ),
    version=VERSION,
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    subreddit: str = Field(..., description="Target subreddit (e.g. 'depression')")
    week_start: str = Field(..., description="ISO date of week being predicted (YYYY-MM-DD)")
    features: dict[str, float] = Field(
        ..., description="Weekly feature vector — all values must be finite floats"
    )
    feature_history: Optional[list[dict[str, float]]] = Field(
        None,
        description=(
            "Optional ordered list of feature dicts for the last N weeks (including current). "
            "Must have length == sequence_length (8) to enable LSTM prediction."
        ),
    )

    @field_validator("subreddit")
    @classmethod
    def validate_subreddit(cls, v: str) -> str:
        normalized = _normalize_sub(v)
        if normalized not in ALL_SUBS:
            raise ValueError(
                f"Unknown subreddit '{v}'. Must be one of: {sorted(ALL_SUBS)} "
                f"(case-insensitive, e.g. 'SuicideWatch' or 'suicidewatch')"
            )
        return normalized

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k, val in v.items() if not np.isfinite(val)]
        if bad:
            raise ValueError(f"Non-finite feature values for: {bad}")
        return v

    @field_validator("week_start")
    @classmethod
    def validate_week_start(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("week_start must be in YYYY-MM-DD format")
        return v


class XGBResult(BaseModel):
    predicted_state: int
    predicted_state_label: str
    crisis_probability: float


class LSTMResult(BaseModel):
    predicted_state: int
    predicted_state_label: str
    class_probabilities: list[float]


class EnsembleResult(BaseModel):
    predicted_state: int
    predicted_state_label: str
    crisis_probability: float


class PredictResponse(BaseModel):
    subreddit: str
    week_start: str
    prediction_available: bool
    xgb: Optional[XGBResult] = None
    lstm: Optional[LSTMResult] = None
    ensemble: Optional[EnsembleResult] = None
    drift_warnings: list[str] = []
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Service health and loaded model inventory")
def health() -> dict:
    models_loaded = sorted(
        sub for sub in FULL_MODEL_SUBS if sub in _xgb_models or sub in _lstm_models
    )
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "monitoring_only": sorted(MONITORING_ONLY_SUBS),
        "feature_columns_loaded": len(_feature_columns),
        "mock_mode": MOCK_MODELS,
        "version": VERSION,
    }


@app.post("/predict", response_model=PredictResponse, summary="Predict crisis state for a community week")
def predict(req: PredictRequest) -> PredictResponse:
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())

    validated_features, clipped_msgs, violations = _validate_or_clip_features(req.subreddit, req.features)
    if violations:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input features outside validated training range",
                "violations": violations,
                "max_sigma": STRICT_MAX_SIGMA,
            },
        )

    # Monitoring-only subreddits — no model inference
    if req.subreddit in MONITORING_ONLY_SUBS:
        return PredictResponse(
            subreddit=req.subreddit,
            week_start=req.week_start,
            prediction_available=False,
            drift_warnings=clipped_msgs,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
        )

    # Validate that required features are present if we know the expected columns
    if _feature_columns and not MOCK_MODELS:
        missing = [c for c in _feature_columns if c not in validated_features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing {len(missing)} feature(s): {missing[:10]}{'...' if len(missing) > 10 else ''}",
            )

    drift_warnings = clipped_msgs + _check_drift(req.subreddit, validated_features)

    xgb_result: Optional[XGBResult] = None
    lstm_result: Optional[LSTMResult] = None

    # ── XGBoost inference ──────────────────────────────────────────────
    xgb_model = _xgb_models.get(req.subreddit)
    if xgb_model is not None:
        try:
            feat_order = _feature_columns if _feature_columns else sorted(validated_features.keys())
            X = pd.DataFrame([{k: validated_features.get(k, 0.0) for k in feat_order}])
            raw_prob = float(xgb_model.predict_proba(X)[0, 1])
            crisis_prob = _apply_calibrator(raw_prob, _xgb_calibrators.get(req.subreddit))
            state = int(crisis_prob >= 0.5) * 2  # XGB is binary; map to state 0 or 2
            xgb_result = XGBResult(
                predicted_state=state,
                predicted_state_label=STATE_NAMES.get(state, "Unknown"),
                crisis_probability=round(crisis_prob, 4),
            )
        except Exception as exc:
            logging.error("XGB inference failed for %s: %s", req.subreddit, exc)

    # ── LSTM inference (requires feature_history of length == sequence_length) ──
    lstm_info = _lstm_models.get(req.subreddit)
    if lstm_info is not None and req.feature_history is not None:
        seq_len = lstm_info["sequence_length"]
        if len(req.feature_history) >= seq_len:
            try:
                feat_order = _feature_columns if _feature_columns else sorted(validated_features.keys())
                history = req.feature_history[-seq_len:]
                arr = np.array(
                    [[h.get(k, 0.0) for k in feat_order] for h in history],
                    dtype=np.float32,
                )
                t = torch.tensor(arr).unsqueeze(0)  # (1, seq_len, features)
                net: LSTMNet = lstm_info["net"]
                with torch.no_grad():
                    logits = net(t)
                    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                state = int(np.argmax(probs))
                calibrated_crisis = _apply_calibrator(
                    float(probs[2] + probs[3]),
                    _lstm_calibrators.get(req.subreddit),
                )
                crisis_raw = float(probs[2] + probs[3])
                if crisis_raw > 0:
                    scale = calibrated_crisis / crisis_raw
                    probs[2] = float(np.clip(probs[2] * scale, 0.0, 1.0))
                    probs[3] = float(np.clip(probs[3] * scale, 0.0, 1.0))
                lstm_result = LSTMResult(
                    predicted_state=state,
                    predicted_state_label=STATE_NAMES.get(state, "Unknown"),
                    class_probabilities=[round(float(p), 4) for p in probs],
                )
            except Exception as exc:
                logging.error("LSTM inference failed for %s: %s", req.subreddit, exc)

    # ── Ensemble (average crisis probability when both models available) ──
    ensemble_result: Optional[EnsembleResult] = None
    if xgb_result is not None and lstm_result is not None:
        lstm_crisis_prob = sum(lstm_result.class_probabilities[2:])
        ens_prob = (xgb_result.crisis_probability + lstm_crisis_prob) / 2.0
        ens_state = int(ens_prob >= 0.5) * 2
        ensemble_result = EnsembleResult(
            predicted_state=ens_state,
            predicted_state_label=STATE_NAMES.get(ens_state, "Unknown"),
            crisis_probability=round(ens_prob, 4),
        )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # ── Inference log ──────────────────────────────────────────────────
    log_entry: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "subreddit": req.subreddit,
        "week_start": req.week_start,
        "feature_validation_clipped": clipped_msgs,
        "xgb_crisis_prob": xgb_result.crisis_probability if xgb_result else None,
        "lstm_predicted_state": lstm_result.predicted_state if lstm_result else None,
        "ensemble_crisis_prob": ensemble_result.crisis_probability if ensemble_result else None,
        "predicted_state_label": (
            (ensemble_result or xgb_result or lstm_result).predicted_state_label
            if (ensemble_result or xgb_result or lstm_result)
            else None
        ),
        "drift_warnings": drift_warnings,
        "has_drift": len(drift_warnings) > 0,
        "latency_ms": latency_ms,
    }
    _log_prediction(log_entry)

    return PredictResponse(
        subreddit=req.subreddit,
        week_start=req.week_start,
        prediction_available=True,
        xgb=xgb_result,
        lstm=lstm_result,
        ensemble=ensemble_result,
        drift_warnings=drift_warnings,
        latency_ms=latency_ms,
    )


@app.get("/model-info", summary="Walk-forward metrics and top SHAP features per subreddit")
def model_info() -> dict:
    info: dict = {}
    for sub in ALL_SUBS:
        sub_data: dict = {
            "models_available": {
                "xgb": sub in _xgb_models,
                "lstm": sub in _lstm_models,
            },
            "monitoring_only": sub in MONITORING_ONLY_SUBS,
        }

        # Walk-forward metrics from eval_results
        sub_results = _eval_results.get(sub, {})
        for model_key in ("xgb", "lstm"):
            model_res = sub_results.get(model_key, {})
            if model_res and "error" not in model_res:
                sub_data[f"{model_key}_metrics"] = {
                    k: model_res.get(k)
                    for k in ("recall", "precision", "f1", "pr_auc",
                              "avg_detection_lead_time_weeks", "n_folds",
                              "n_valid_predictions", "n_crisis_actual")
                    if k in model_res
                }

        # Top-5 SHAP features
        shap_df = _shap_data.get(sub)
        if shap_df is not None and not shap_df.empty:
            top5 = shap_df.head(5)
            sub_data["top_shap_features"] = [
                {"feature": row.get("feature", ""), "mean_abs_shap": round(float(row.get("mean_abs_shap", 0)), 5)}
                for _, row in top5.iterrows()
            ]

        # Feature stats summary
        fstats = _feature_stats.get(sub, {})
        if fstats:
            sub_data["n_training_weeks"] = fstats.get("n_training_weeks")
            sub_data["feature_stats_generated_at"] = fstats.get("generated_at")

        info[sub] = sub_data

    return info


@app.get("/logs/summary", summary="Aggregate statistics from the prediction log")
def logs_summary() -> dict:
    if not PREDICTION_LOG.exists():
        return {
            "total_predictions": 0,
            "by_subreddit": {},
            "overall_elevated_rate": 0.0,
            "overall_drift_rate": 0.0,
            "log_file": str(PREDICTION_LOG),
        }

    total = 0
    elevated = 0
    drifted = 0
    by_sub: dict[str, dict] = {}

    try:
        with open(PREDICTION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                sub = entry.get("subreddit", "unknown")
                if sub not in by_sub:
                    by_sub[sub] = {"count": 0, "elevated_count": 0, "drift_count": 0}
                by_sub[sub]["count"] += 1
                # Elevated = predicted state >= 2 (crisis probability >= 0.5)
                ens_prob = entry.get("ensemble_crisis_prob") or entry.get("xgb_crisis_prob")
                if ens_prob is not None and ens_prob >= 0.5:
                    elevated += 1
                    by_sub[sub]["elevated_count"] += 1
                if entry.get("has_drift"):
                    drifted += 1
                    by_sub[sub]["drift_count"] += 1
    except Exception as exc:
        logging.error("Failed to read prediction log: %s", exc)

    by_sub_rates = {
        s: {
            "count": d["count"],
            "elevated_rate": round(d["elevated_count"] / d["count"], 4) if d["count"] else 0.0,
            "drift_rate": round(d["drift_count"] / d["count"], 4) if d["count"] else 0.0,
        }
        for s, d in by_sub.items()
    }

    return {
        "total_predictions": total,
        "by_subreddit": by_sub_rates,
        "overall_elevated_rate": round(elevated / total, 4) if total else 0.0,
        "overall_drift_rate": round(drifted / total, 4) if total else 0.0,
        "log_file": str(PREDICTION_LOG),
    }
