"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/stage_b_portfolio_fixtures.py

Fixture writers for TFBS Stage B portfolio tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path


def write_replicated_review_source(
    tmp_path: Path,
    *,
    source_id: str,
    label_name: str,
    replicate_count: int = 3,
    replicate_seeds: list[int] | None = None,
    control_role: str | None = None,
    profile_role: str | None = None,
    claim_ready: bool | None = None,
) -> Path:
    root = tmp_path / source_id
    root.mkdir(parents=True)
    trajectory_path = root / "trajectory.csv"
    pair_summary_path = root / "pair_summary.csv"
    trajectory_path.write_text("label_name,round,value\nlexA_in_slot0,0,1\n", encoding="utf-8")
    pair_summary_path.write_text("label_name,value\nlexA_in_slot0,1\n", encoding="utf-8")
    plot_path = root / f"{label_name}.png"
    plot_path.write_bytes(b"not-a-real-png-but-existing")
    plot_manifest_path = root / "plot_manifest.json"
    plot_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_review_plots.v1",
                "plot_count": 1,
                "plots": [
                    {
                        "kind": "realized_label_lift_trajectory",
                        "label_name": label_name,
                        "path": str(plot_path),
                        "interval_kind": "sample_sd",
                        **({"control_role": control_role} if control_role is not None else {}),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path = root / "summary.json"
    resolved_profile_role = profile_role or (
        "boundary_stage_b_count_fixed_minimal_placement_probe"
        if control_role == "count_fixed_shuffled_slot_negative_control"
        else "boundary_stage_b_sentinel_probe"
    )
    resolved_claim_ready = (
        claim_ready if claim_ready is not None else control_role == "count_fixed_shuffled_slot_negative_control"
    )
    claim_readiness = _claim_readiness(label_name, claim_ready=resolved_claim_ready)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_replicated_review.v1",
                "status": "PASS",
                "claim_readiness": claim_readiness,
                "interpretation_boundary": "Fixture interpretation boundary.",
                "replicate_count": replicate_count,
                "replicate_seeds": replicate_seeds or [7, 17, 29],
                "target_profile": {
                    "profile_id": "fixture_profile",
                    "profile_role": resolved_profile_role,
                    "label_names": [label_name],
                    "label_family_ids": ["tf_slot_family_presence"],
                    "canonical": False,
                    "interpretation_boundary": "Fixture interpretation boundary.",
                },
                "trajectory_csv_path": str(trajectory_path),
                "replicate_pair_summary_csv_path": str(pair_summary_path),
                "plot_manifest_json_path": str(plot_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def write_learning_loop_source(
    tmp_path: Path,
    *,
    review_id: str = "count_fraction_learning_loop",
    profile_id: str = "tfbs_count_fraction_probe_v1",
    visual_tier: str = "composition_learning_loop",
    comparison_set_label: str | None = None,
) -> Path:
    root = tmp_path / review_id
    root.mkdir(parents=True)
    trajectory_path = root / "trajectory.csv"
    endpoint_path = root / "endpoint.csv"
    claim_path = root / "claim.csv"
    trajectory_path.write_text("label_name,value\nlexA_count_fraction,1\n", encoding="utf-8")
    endpoint_path.write_text("label_name,value\nlexA_count_fraction,1\n", encoding="utf-8")
    claim_path.write_text("label_name,value\nlexA_count_fraction,1\n", encoding="utf-8")
    cumulative_path = root / "cumulative.png"
    endpoint_plot_path = root / "endpoint.png"
    cumulative_path.write_bytes(b"png")
    endpoint_plot_path.write_bytes(b"png")
    plot_manifest_path = root / "plot_manifest.json"
    plot_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline_plots.v1",
                "plot_count": 2,
                "plots": [
                    {"kind": "frozen_round0_cumulative_enrichment", "path": str(cumulative_path)},
                    {"kind": "frozen_round0_endpoint_adaptive_gain", "path": str(endpoint_plot_path)},
                    {"kind": "known_label_gain_recovery", "path": str(endpoint_plot_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "learning_loop_baseline_manifest.json"
    resolved_comparison_set_label = comparison_set_label or (
        "Placement learning loop" if visual_tier == "placement_learning_loop" else "Composition learning loop"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline.v1",
                "status": "PASS",
                "review_id": review_id,
                "comparison_set_key": f"{review_id}_baseline",
                "comparison_set_label": resolved_comparison_set_label,
                "visual_tier": visual_tier,
                "source_profile_ids": [profile_id],
                "claim_boundary": (
                    "This supports a harness-level active-learning claim for synthetic DenseGen count-fraction "
                    "metadata."
                ),
                "interpretation_boundary": "Fixture learning-loop interpretation boundary.",
                "trajectory_csv_path": str(trajectory_path),
                "endpoint_summary_csv_path": str(endpoint_path),
                "claim_interpretation_csv_path": str(claim_path),
                "plot_manifest_json_path": str(plot_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _claim_readiness(label_name: str, *, claim_ready: bool) -> dict[str, object]:
    if claim_ready:
        return {
            "blocked_or_limited_claim_count": 0,
            "blocked_or_limited_labels": [],
            "claim_readiness_status_counts": {"READY_AS_REPLICATED_VALID_NULL_LEARNABILITY_SIGNAL": 1},
            "ready_claim_count": 1,
            "ready_labels": [label_name],
        }
    return {
        "blocked_or_limited_claim_count": 1,
        "blocked_or_limited_labels": [label_name],
        "claim_readiness_status_counts": {"LIMITED_INVALID_NEGATIVE_CONTROL_REPLICATE": 1},
        "ready_claim_count": 0,
        "ready_labels": [],
    }
