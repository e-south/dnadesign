"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/notebook_summary.py

Typed summary contract for the response-metastudy review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewSummary:
    """Reader-facing decision summary derived from one verified manifest."""

    decision: str
    basis: str
    primary_assay_summary: str
    evidence_base: str
    prospective_hill_climb: str


def build_review_summary(bundle_manifest: Mapping[str, object]) -> ReviewSummary:
    """Build the RMF-specific header independently of the SFXI verdict."""

    response_screen = _mapping(bundle_manifest, "response_metric_screen")
    status = str(response_screen.get("status") or "")
    if status != "screen_complete_not_promoted":
        raise ValueError(f"Unsupported response-screen status: {status!r}")

    best_model = _mapping(response_screen, "best_fixed_model_screen")
    weakest_ordering = float(best_model["weakest_target_view_response_separation_spearman"])
    primary_reduction_id = str(response_screen["primary_reduction_candidate"])
    protocol = _mapping(response_screen, "response_screen_protocol")
    reductions = protocol.get("reductions")
    if not isinstance(reductions, list):
        raise ValueError("Response screen reductions must be a list.")
    primary_reductions = [
        row
        for row in reductions
        if isinstance(row, Mapping)
        and str(row.get("id")) == primary_reduction_id
        and str(row.get("screen_role")) == "primary"
    ]
    if len(primary_reductions) != 1:
        raise ValueError("Response screen must declare exactly one primary reduction record.")
    primary = primary_reductions[0]
    method_labels = {"geometric_time_mean": "geometric-time mean"}
    method = str(primary["method"])
    if method not in method_labels:
        raise ValueError(f"Unsupported primary reduction method: {method!r}")
    if response_screen.get("prospective_hill_climb_demonstrated") is not False:
        raise ValueError("Inactive response screen cannot claim a prospective hill climb.")

    return ReviewSummary(
        decision="RMF remains inactive for production selection",
        basis=(
            "The leading fixed challenger has weakest target-view response-separation ordering of "
            f"{weakest_ordering:.2f}; prospective rank stability is not established"
        ),
        primary_assay_summary=(
            f"{float(primary['window_start_event_h']):g}-{float(primary['window_end_event_h']):g} h "
            f"after stress addition, {method_labels[method]} (`{primary_reduction_id}`)"
        ),
        evidence_base=(
            f"{int(response_screen['label_count'])} observed labels across "
            f"{int(response_screen['reader_event_experiment_count'])} Reader experiments"
        ),
        prospective_hill_climb="not yet measured",
    )


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Metastudy manifest field {key!r} must be a mapping.")
    return value


__all__ = ["ReviewSummary", "build_review_summary"]
