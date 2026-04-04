from pathlib import Path

import pandas as pd

from src.pipeline.run_train import _build_ensemble_results, _select_features_for_subreddit


def test_select_features_for_subreddit_uses_shap_threshold(tmp_path: Path):
    reports = tmp_path / "reports"
    sub_dir = reports / "depression"
    sub_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "mean_abs_shap": [1.0, 0.3, 0.05],
        }
    ).to_csv(sub_dir / "shap.csv", index=False)
    config = {
        "paths": {"reports": str(reports)},
        "modeling": {
            "feature_selection": {
                "enabled": True,
                "shap_min_ratio_to_top": 0.2,
                "min_features": 2,
                "max_features": 10,
            }
        },
    }
    selected = _select_features_for_subreddit(config, "depression", ["f1", "f2", "f3", "f4"])
    assert "f1" in selected
    assert "f2" in selected
    assert "f4" not in selected


def test_build_ensemble_results_returns_metrics():
    xgb = {
        "per_week": {
            "probabilities": [0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6],
            "actuals": [0, 1, 0, 1, 1, 0, 1],
        }
    }
    lstm = {
        "per_week": {
            "probabilities": [0.2, 0.7, 0.3, 0.85, 0.65, 0.4, 0.55],
            "actuals": [0, 2, 0, 3, 2, 0, 2],
        }
    }
    config = {"evaluation": {"probability_threshold": 0.5}}
    out = _build_ensemble_results(xgb, lstm, config)
    assert out
    assert out["n_valid_predictions"] >= 5
    assert "pr_auc" in out
    assert "per_week" in out
