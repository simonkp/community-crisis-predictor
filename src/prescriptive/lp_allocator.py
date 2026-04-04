"""
Prescriptive LP module — moderator resource allocation.

Formulates a linear programme that recommends how to allocate a fixed weekly
moderator-hour budget across subreddits to maximise expected crisis interceptions.

Problem formulation
-------------------
Decision variable : x[i]  — hours allocated to subreddit i
Objective         : maximise  sum_i ( p[i] * e[i] * x[i] )
                    where p[i] = predicted crisis probability for subreddit i
                          e[i] = intervention effectiveness coefficient (0–1)
Constraints       :
  (1)  sum_i x[i]  <= total_hours          (budget)
  (2)  x[i]        >= min_hours_per_sub    (every subreddit gets a floor)
  (3)  x[i]        >= 0

Solved via scipy.optimize.linprog (simplex / HiGHS backend).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_STATE_NAMES = {0: "Stable", 1: "Early Vulnerability", 2: "Elevated Distress", 3: "Severe"}


def _solve_lp(
    probabilities: dict[str, float],
    effectiveness: dict[str, float],
    total_hours: float,
    min_hours: float,
) -> dict[str, float]:
    """
    Solve the allocation LP for the given probabilities and budget.

    Returns a dict {subreddit: allocated_hours}.
    Falls back to pro-rata allocation if scipy is unavailable or the LP fails.
    """
    subs = list(probabilities.keys())
    n = len(subs)
    if n == 0:
        return {}

    p = np.array([probabilities[s] for s in subs], dtype=float)
    e = np.array([effectiveness.get(s, 0.7) for s in subs], dtype=float)

    # Feasibility guard: if n * min_hours > total_hours, scale min_hours down.
    safe_min = min(min_hours, total_hours / n)

    try:
        from scipy.optimize import linprog

        # linprog minimises; negate objective to maximise p*e*x.
        c = -(p * e)

        # Constraint (1): sum(x) <= total_hours
        A_ub = np.ones((1, n))
        b_ub = np.array([total_hours])

        # Bounds: safe_min <= x[i] <= total_hours
        bounds = [(safe_min, total_hours)] * n

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            return {subs[i]: round(float(res.x[i]), 2) for i in range(n)}
    except Exception:
        pass

    # Fallback: allocate proportionally to p*e, with floor applied first.
    allocation = {s: safe_min for s in subs}
    remaining = total_hours - safe_min * n
    weights = p * e
    total_w = weights.sum()
    if total_w > 0 and remaining > 0:
        for i, s in enumerate(subs):
            allocation[s] += round(remaining * weights[i] / total_w, 2)
    return allocation


def run_allocation(
    eval_results: dict,
    config: dict,
) -> dict:
    """
    Derive the latest per-subreddit crisis probabilities from eval_results,
    solve the LP, and return a structured allocation report.

    Parameters
    ----------
    eval_results : dict loaded from data/models/eval_results.json
    config       : pipeline config dict (reads prescriptive block)

    Returns
    -------
    dict with keys:
      subreddits    — per-sub {hours, probability, state, effectiveness}
      total_hours   — budget used
      objective     — achieved LP objective value (expected interceptions)
      sensitivity   — {budget: {sub: hours}} for budgets 5..20
    """
    presc_cfg = config.get("prescriptive", {})
    total_hours = float(presc_cfg.get("total_moderator_hours", 10.0))
    min_hours = float(presc_cfg.get("min_hours_per_sub", 0.5))
    default_eff = float(presc_cfg.get("default_effectiveness", 0.7))
    effectiveness_overrides: dict = presc_cfg.get("effectiveness", {})
    sensitivity_budgets: list[float] = [
        float(b) for b in presc_cfg.get("sensitivity_budgets", list(range(5, 21)))
    ]

    # Extract the most-recent non-NaN crisis probability for each subreddit.
    probabilities: dict[str, float] = {}
    latest_states: dict[str, int] = {}

    for sub, sub_res in eval_results.items():
        # Handle nested {xgb: ..., lstm: ...} or flat dict format.
        if "lstm" in sub_res or "xgb" in sub_res:
            model_res = sub_res.get("lstm") or sub_res.get("xgb", {})
        else:
            model_res = sub_res

        if not model_res or "error" in model_res:
            continue

        per_week = model_res.get("per_week", {})
        probs = per_week.get("probabilities", [])
        states = per_week.get("predictions", [])

        # Walk backwards to find the last valid probability.
        latest_prob = 0.0
        for v in reversed(probs):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                latest_prob = float(v)
                break

        latest_state = 0
        for v in reversed(states):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                latest_state = int(v) if int(v) in _STATE_NAMES else 0
                break

        probabilities[sub] = round(latest_prob, 4)
        latest_states[sub] = latest_state

    if not probabilities:
        return {"error": "no_valid_probabilities", "subreddits": {}}

    effectiveness = {s: effectiveness_overrides.get(s, default_eff) for s in probabilities}

    allocation = _solve_lp(probabilities, effectiveness, total_hours, min_hours)

    # Compute achieved objective value.
    objective = sum(
        probabilities[s] * effectiveness[s] * allocation.get(s, 0.0)
        for s in probabilities
    )

    subreddit_details = {
        s: {
            "hours": allocation.get(s, 0.0),
            "probability": probabilities[s],
            "state": latest_states.get(s, 0),
            "state_label": _STATE_NAMES.get(latest_states.get(s, 0), "Unknown"),
            "effectiveness": effectiveness[s],
        }
        for s in probabilities
    }

    # Sensitivity analysis: re-run LP at each budget level.
    sensitivity: dict[str, dict[str, float]] = {}
    for budget in sensitivity_budgets:
        sens_alloc = _solve_lp(probabilities, effectiveness, budget, min_hours)
        sensitivity[str(int(budget))] = sens_alloc

    return {
        "subreddits": subreddit_details,
        "total_hours": total_hours,
        "objective": round(objective, 4),
        "sensitivity": sensitivity,
    }


def format_allocation_text(report: dict) -> str:
    """Return a plain-text summary table of the allocation report."""
    if "error" in report:
        return f"Allocation unavailable: {report['error']}"

    total = report.get("total_hours", 0)
    rows = sorted(
        report["subreddits"].items(),
        key=lambda kv: kv[1]["hours"],
        reverse=True,
    )
    lines = [f"Weekly allocation ({total} hrs total)"]
    lines.append("-" * 54)
    for sub, d in rows:
        lines.append(
            f"  r/{sub:<18}  {d['hours']:>4.1f} hrs"
            f"  [{d['state_label']}, p={d['probability']:.2f}]"
        )
    lines.append("-" * 54)
    lines.append(f"  Expected interceptions (objective): {report['objective']:.4f}")
    return "\n".join(lines)


def save_allocation_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
