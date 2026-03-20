import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.core.ui_config import ALERT_ENGINE_COPY


class AlertEngine:
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    subreddit TEXT NOT NULL,
                    week_start TEXT NOT NULL,
                    from_state INTEGER NOT NULL,
                    to_state INTEGER NOT NULL,
                    distress_score REAL,
                    dominant_signal TEXT
                )
                """
            )
            conn.commit()

    def process_week_sequence(
        self,
        subreddit: str,
        weekly_states: list[int],
        weekly_scores: list[float],
        feature_df: pd.DataFrame | None = None,
    ) -> None:
        from src.monitoring.drift_detector import DRIFT_SIGNALS

        for i in range(1, len(weekly_states)):
            prev = weekly_states[i - 1]
            curr = weekly_states[i]
            if prev is None or curr is None:
                continue
            try:
                prev_int, curr_int = int(prev), int(curr)
            except (TypeError, ValueError):
                continue
            if np.isnan(prev) or np.isnan(curr):
                continue
            if curr_int <= prev_int:
                continue  # only log escalations

            dominant = ""
            if feature_df is not None and i < len(feature_df):
                row = feature_df.iloc[i]
                available = [s for s in DRIFT_SIGNALS if s in row.index]
                if available:
                    dominant = max(available, key=lambda s: float(row[s]))

            week_start = (
                str(feature_df.iloc[i]["week_start"])
                if feature_df is not None and "week_start" in feature_df.columns
                else str(i)
            )
            score = float(weekly_scores[i]) if i < len(weekly_scores) else 0.0

            record = {
                "subreddit": subreddit,
                "week_start": week_start,
                "from_state": prev_int,
                "to_state": curr_int,
                "distress_score": score,
                "dominant_signal": dominant,
            }
            self.fire_alert(record)

    def fire_alert(self, record: dict) -> None:
        from src.core.domain_config import STATE_NAMES

        from_name = STATE_NAMES.get(record["from_state"], str(record["from_state"]))
        to_name = STATE_NAMES.get(record["to_state"], str(record["to_state"]))

        colors = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m", 3: "\033[95m"}
        color = colors.get(record["to_state"], "\033[0m")
        reset = "\033[0m"
        print(
            f"{color}{ALERT_ENGINE_COPY['transition_prefix']}: r/{record['subreddit']} "
            f"week {record['week_start']}: {from_name} -> {to_name}{reset}"
        )

        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO transitions
                    (timestamp, subreddit, week_start, from_state, to_state,
                     distress_score, dominant_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    record["subreddit"],
                    record["week_start"],
                    record["from_state"],
                    record["to_state"],
                    record["distress_score"],
                    record["dominant_signal"],
                ),
            )
            conn.commit()

    def get_recent_transitions(self, n: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, subreddit, week_start, from_state, to_state,
                       distress_score, dominant_signal
                FROM transitions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (n,),
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
