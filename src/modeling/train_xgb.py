import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


class XGBCrisisModel:
    def __init__(self, config: dict):
        xgb_cfg = config.get("modeling", {}).get("xgboost", {})
        self.param_grid = xgb_cfg.get("param_grid", {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.6, 0.8, 1.0],
        })
        self.n_search_iter = xgb_cfg.get("n_search_iter", 30)
        self.scale_pos_weight = xgb_cfg.get("scale_pos_weight", "auto")
        self.seed = config.get("random_seed", 42)
        self.model: xgb.XGBClassifier | None = None
        self.best_params: dict = {}

    def _compute_scale_pos_weight(self, y: pd.Series, max_weight: float = 8.0) -> float:
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        if n_pos == 0:
            return 1.0
        return min(n_neg / n_pos, max_weight)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              do_search: bool = True) -> None:
        spw = (
            self._compute_scale_pos_weight(y_train)
            if self.scale_pos_weight == "auto"
            else float(self.scale_pos_weight)
        )

        base_model = xgb.XGBClassifier(
            scale_pos_weight=spw,
            random_state=self.seed,
            eval_metric="logloss",
        )

        if do_search and len(X_train) > 30:
            tscv = TimeSeriesSplit(n_splits=3)
            search = RandomizedSearchCV(
                base_model,
                self.param_grid,
                n_iter=min(self.n_search_iter, 20),
                scoring="recall",
                cv=tscv,
                random_state=self.seed,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            base_model.set_params(
                max_depth=5, learning_rate=0.1, n_estimators=200
            )
            base_model.fit(X_train, y_train)
            self.model = base_model
            self.best_params = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 200}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Must train before predicting")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
