"""
Tests for the FastAPI inference service.

All tests run with MOCK_MODELS=true so no real model files are required.
Tests that need real artifacts are marked with @pytest.mark.skipif.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make serving/ importable when tests run from that directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force mock mode before importing the app
os.environ.setdefault("MOCK_MODELS", "true")

from fastapi.testclient import TestClient
import main as app_module
from main import app, ALL_SUBS, FULL_MODEL_SUBS, MONITORING_ONLY_SUBS, STATE_NAMES

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FEATURES = {
    "hopelessness_density": 0.05,
    "avg_negative_roll4w": 0.32,
    "help_seeking_density": 0.03,
    "avg_sentiment_neg": 0.25,
    "post_count": 150.0,
    "avg_text_length": 200.0,
}

SAMPLE_HISTORY = [SAMPLE_FEATURES.copy() for _ in range(8)]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_structure(self):
        r = client.get("/health")
        d = r.json()
        assert d["status"] == "healthy"
        assert "models_loaded" in d
        assert "monitoring_only" in d
        assert "version" in d

    def test_monitoring_only_subs(self):
        r = client.get("/health")
        monitoring = set(r.json()["monitoring_only"])
        assert monitoring == MONITORING_ONLY_SUBS

    def test_mock_mode_flagged(self):
        r = client.get("/health")
        assert r.json()["mock_mode"] is True


# ---------------------------------------------------------------------------
# /predict — input validation
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_unknown_subreddit_returns_422(self):
        r = client.post(
            "/predict",
            json={"subreddit": "NotARealSub", "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 422

    def test_suicidewatch_alias_accepted(self):
        """SuicideWatch (mixed case) should be accepted and normalized to suicidewatch."""
        r = client.post(
            "/predict",
            json={"subreddit": "SuicideWatch", "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 200
        assert r.json()["subreddit"] == "suicidewatch"

    def test_bad_date_format_returns_422(self):
        r = client.post(
            "/predict",
            json={"subreddit": "depression", "week_start": "09-03-2020", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 422

    def test_non_finite_feature_returns_422(self):
        # Send null for a feature value — Pydantic rejects non-float types
        import json as _json
        raw = _json.dumps(
            {"subreddit": "depression", "week_start": "2020-03-09",
             "features": {"hopelessness_density": None}}
        )
        r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_missing_required_fields_returns_422(self):
        r = client.post("/predict", json={"subreddit": "depression"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict — monitoring-only subreddits
# ---------------------------------------------------------------------------


class TestPredictMonitoringOnly:
    @pytest.mark.parametrize("sub", sorted(MONITORING_ONLY_SUBS))
    def test_returns_200_prediction_unavailable(self, sub):
        r = client.post(
            "/predict",
            json={"subreddit": sub, "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["prediction_available"] is False
        assert body["subreddit"] == sub

    @pytest.mark.parametrize("sub", sorted(MONITORING_ONLY_SUBS))
    def test_no_xgb_or_lstm_in_response(self, sub):
        r = client.post(
            "/predict",
            json={"subreddit": sub, "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        body = r.json()
        assert body.get("xgb") is None
        assert body.get("lstm") is None


# ---------------------------------------------------------------------------
# /predict — full-model subreddits (mock mode, no real models)
# ---------------------------------------------------------------------------


class TestPredictFullSubsMockMode:
    @pytest.mark.parametrize("sub", sorted(FULL_MODEL_SUBS))
    def test_returns_200_with_latency(self, sub):
        r = client.post(
            "/predict",
            json={"subreddit": sub, "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        assert r.status_code == 200
        body = r.json()
        assert "latency_ms" in body
        assert body["latency_ms"] >= 0

    @pytest.mark.parametrize("sub", sorted(FULL_MODEL_SUBS))
    def test_has_drift_warnings_list(self, sub):
        r = client.post(
            "/predict",
            json={"subreddit": sub, "week_start": "2020-03-09", "features": SAMPLE_FEATURES},
        )
        body = r.json()
        assert isinstance(body["drift_warnings"], list)

    def test_week_start_echoed(self):
        r = client.post(
            "/predict",
            json={"subreddit": "depression", "week_start": "2019-05-13", "features": SAMPLE_FEATURES},
        )
        assert r.json()["week_start"] == "2019-05-13"


# ---------------------------------------------------------------------------
# /predict — drift detection logic
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_out_of_range_feature_rejected(self):
        """Features beyond strict sigma guardrail should be rejected."""
        sub = "depression"
        injected_stats = {
            "features": {
                "hopelessness_density": {"mean": 0.04, "std": 0.01, "min": 0.0, "max": 0.1}
            }
        }
        with patch.dict(app_module._feature_stats, {sub: injected_stats}):
            r = client.post(
                "/predict",
                json={
                    "subreddit": sub,
                    "week_start": "2020-03-09",
                    "features": {"hopelessness_density": 0.99},
                },
            )
        assert r.status_code == 422
        body = r.json()
        detail = body.get("detail", {})
        assert "violations" in detail
        assert any("hopelessness_density" in item for item in detail["violations"])

    def test_no_drift_for_normal_values(self):
        sub = "depression"
        injected_stats = {
            "features": {
                "hopelessness_density": {"mean": 0.04, "std": 0.01, "min": 0.0, "max": 0.1}
            }
        }
        with patch.dict(app_module._feature_stats, {sub: injected_stats}):
            r = client.post(
                "/predict",
                json={
                    "subreddit": sub,
                    "week_start": "2020-03-09",
                    "features": {"hopelessness_density": 0.04},
                },
            )
        body = r.json()
        assert body["drift_warnings"] == []


# ---------------------------------------------------------------------------
# /predict — with mocked XGB model
# ---------------------------------------------------------------------------


class TestPredictWithMockedXGB:
    def test_xgb_result_structure(self):
        mock_xgb = MagicMock()
        import numpy as np

        mock_xgb.predict_proba.return_value = np.array([[0.3, 0.7]])

        with patch.dict(app_module._xgb_models, {"depression": mock_xgb}):
            r = client.post(
                "/predict",
                json={
                    "subreddit": "depression",
                    "week_start": "2020-03-09",
                    "features": SAMPLE_FEATURES,
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert body["prediction_available"] is True
        xgb = body["xgb"]
        assert xgb is not None
        assert "predicted_state" in xgb
        assert "crisis_probability" in xgb
        assert xgb["predicted_state"] in STATE_NAMES
        assert xgb["predicted_state_label"] == STATE_NAMES[xgb["predicted_state"]]

    def test_xgb_high_prob_maps_to_elevated_state(self):
        mock_xgb = MagicMock()
        import numpy as np

        mock_xgb.predict_proba.return_value = np.array([[0.1, 0.9]])

        with patch.dict(app_module._xgb_models, {"depression": mock_xgb}):
            r = client.post(
                "/predict",
                json={
                    "subreddit": "depression",
                    "week_start": "2020-03-09",
                    "features": SAMPLE_FEATURES,
                },
            )
        body = r.json()
        # crisis_probability >= 0.5 -> state 2
        assert body["xgb"]["predicted_state"] == 2


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_returns_200(self):
        r = client.get("/model-info")
        assert r.status_code == 200

    def test_all_subs_present(self):
        r = client.get("/model-info")
        body = r.json()
        for sub in ALL_SUBS:
            assert sub in body, f"Missing subreddit {sub} in model-info response"

    def test_monitoring_only_flagged(self):
        r = client.get("/model-info")
        body = r.json()
        for sub in MONITORING_ONLY_SUBS:
            assert body[sub]["monitoring_only"] is True

    def test_full_subs_not_monitoring_only(self):
        r = client.get("/model-info")
        body = r.json()
        for sub in FULL_MODEL_SUBS:
            assert body[sub]["monitoring_only"] is False, f"{sub} should not be monitoring-only"


# ---------------------------------------------------------------------------
# /logs/summary
# ---------------------------------------------------------------------------


class TestLogsSummary:
    def test_returns_200_empty_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        try:
            with patch.object(app_module, "PREDICTION_LOG", log_path):
                r = client.get("/logs/summary")
            assert r.status_code == 200
            body = r.json()
            assert body["total_predictions"] == 0
        finally:
            log_path.unlink(missing_ok=True)

    def test_counts_predictions_correctly(self):
        entries = [
            {
                "subreddit": "depression",
                "xgb_crisis_prob": 0.8,
                "ensemble_crisis_prob": 0.75,
                "has_drift": True,
            },
            {
                "subreddit": "depression",
                "xgb_crisis_prob": 0.2,
                "ensemble_crisis_prob": 0.15,
                "has_drift": False,
            },
            {
                "subreddit": "anxiety",
                "xgb_crisis_prob": 0.6,
                "ensemble_crisis_prob": 0.65,
                "has_drift": False,
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            log_path = Path(f.name)
        try:
            with patch.object(app_module, "PREDICTION_LOG", log_path):
                r = client.get("/logs/summary")
            body = r.json()
            assert body["total_predictions"] == 3
            assert body["by_subreddit"]["depression"]["count"] == 2
            assert body["by_subreddit"]["anxiety"]["count"] == 1
            # 2 out of 3 are elevated (crisis_prob >= 0.5)
            assert abs(body["overall_elevated_rate"] - 2 / 3) < 0.01
            # 1 out of 3 have drift
            assert abs(body["overall_drift_rate"] - 1 / 3) < 0.01
        finally:
            log_path.unlink(missing_ok=True)
