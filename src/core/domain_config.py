"""Domain-level semantics and threshold metadata."""

STATE_NAMES = {
    0: "Stable",
    1: "Early Vulnerability Signal",
    2: "Elevated Distress",
    3: "Severe Community Distress Signal",
}

STATE_THRESHOLD_LABELS = [
    "Early Vulnerability Signal (0.5σ)",
    "Elevated Distress (1.0σ)",
    "Severe Community Distress Signal (2.0σ)",
]

CRISIS_THRESHOLD_LABEL = "Severe community distress threshold"
