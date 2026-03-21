from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DemoFeatureMap:
    hopelessness_feature: str | None
    post_volume_feature: str | None
    late_night_feature: str | None


def resolve_demo_feature_map(columns: list[str]) -> DemoFeatureMap:
    lower_map = {c.lower(): c for c in columns}

    def _pick(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        for c in columns:
            lc = c.lower()
            if any(token in lc for token in candidates):
                return c
        return None

    return DemoFeatureMap(
        hopelessness_feature=_pick(["hopelessness_density", "hopelessness", "distress_lexicon_density"]),
        post_volume_feature=_pick(["post_volume", "posts_count", "n_posts", "post_count", "volume"]),
        late_night_feature=_pick(["late_night_post_ratio", "late_night_ratio", "night_post_ratio"]),
    )


def apply_scenario_adjustments(
    row: pd.Series,
    feature_map: DemoFeatureMap,
    hopelessness_pct: float,
    post_volume_pct: float,
    late_night_pct: float,
) -> pd.Series:
    out = row.copy()
    if feature_map.hopelessness_feature in out.index:
        out[feature_map.hopelessness_feature] = float(out[feature_map.hopelessness_feature]) * (1 + hopelessness_pct / 100.0)
    if feature_map.post_volume_feature in out.index:
        out[feature_map.post_volume_feature] = float(out[feature_map.post_volume_feature]) * (1 + post_volume_pct / 100.0)
    if feature_map.late_night_feature in out.index:
        out[feature_map.late_night_feature] = float(out[feature_map.late_night_feature]) * (1 + late_night_pct / 100.0)
    return out


def parse_demo_events(events_cfg: list[dict] | None) -> list[tuple[pd.Timestamp, str]]:
    out: list[tuple[pd.Timestamp, str]] = []
    for e in events_cfg or []:
        label = str(e.get("label", "")).strip()
        date_raw = e.get("date")
        if not label or not date_raw:
            continue
        dt = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(dt):
            continue
        out.append((pd.Timestamp(dt), label))
    return out


def event_in_range(event_dt: pd.Timestamp, week_values: np.ndarray) -> bool:
    if len(week_values) == 0:
        return False
    s = pd.to_datetime(pd.Series(week_values), errors="coerce").dropna()
    if s.empty:
        return False
    return bool(s.min() <= event_dt <= s.max())
