import numpy as np
import pandas as pd
import pytest

from src.dashboard.demo_utils import (
    DemoFeatureMap,
    apply_scenario_adjustments,
    event_in_range,
    parse_demo_events,
    resolve_demo_feature_map,
)


def test_resolve_demo_feature_map_prefers_exact_names():
    cols = ["hopelessness_density", "post_volume", "late_night_post_ratio", "other"]
    fm = resolve_demo_feature_map(cols)
    assert fm.hopelessness_feature == "hopelessness_density"
    assert fm.post_volume_feature == "post_volume"
    assert fm.late_night_feature == "late_night_post_ratio"


def test_apply_scenario_adjustments_scales_values():
    row = pd.Series({"hopelessness_density": 10.0, "post_volume": 100.0, "late_night_post_ratio": 0.2})
    fm = DemoFeatureMap("hopelessness_density", "post_volume", "late_night_post_ratio")
    out = apply_scenario_adjustments(row, fm, hopelessness_pct=20, post_volume_pct=-10, late_night_pct=5)
    assert out["hopelessness_density"] == 12.0
    assert out["post_volume"] == 90.0
    assert out["late_night_post_ratio"] == pytest.approx(0.21)


def test_parse_demo_events_filters_invalid_entries():
    events = [
        {"label": "Exam", "date": "2026-04-15"},
        {"label": "Bad", "date": "not-a-date"},
        {"label": "", "date": "2026-05-01"},
    ]
    parsed = parse_demo_events(events)
    assert len(parsed) == 1
    assert parsed[0][1] == "Exam"


def test_event_in_range():
    weeks = np.array(["2026-04-01", "2026-04-08", "2026-04-15"])
    assert event_in_range(pd.Timestamp("2026-04-08"), weeks)
    assert not event_in_range(pd.Timestamp("2026-05-01"), weeks)
