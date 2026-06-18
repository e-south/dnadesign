"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/problems.py

Review problem aggregation for DenseGen axis probe reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping


def campaign_review_problems(campaign_reviews: list[dict[str, Any]]) -> list[str]:
    """Return review-level problem tokens derived from campaign review rows."""

    problems: list[str] = []
    for row in campaign_reviews:
        run_key = str(row.get("run_key") or "unknown")
        warnings = row.get("warnings") or []
        stale_artifacts = row.get("stale_artifacts") or []
        if warnings:
            problems.append(f"opal_campaign_review_warnings:{run_key}:{len(warnings)}")
        if stale_artifacts:
            problems.append(f"opal_campaign_review_stale_artifacts:{run_key}:{len(stale_artifacts)}")
    return problems


def plot_quality_problems(plot_quality: Mapping[str, Any]) -> list[str]:
    """Return review-level problem tokens derived from configured plot quality."""

    if plot_quality.get("status") == "ok":
        return []
    return [
        f"configured_plot_quality:{problem.get('run_key', 'unknown')}:{problem.get('problem', 'unknown')}"
        for problem in plot_quality.get("problems") or []
    ]
