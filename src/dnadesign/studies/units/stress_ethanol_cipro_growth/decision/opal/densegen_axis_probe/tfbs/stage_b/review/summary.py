"""Summary manifest builder for DenseGen TFBS Stage B realized-label review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...profiles import tfbs_target_profile_for_labels
from ...stage_a.manifests import file_sha256
from ..claims import summarize_tfbs_stage_b_claim_assessment
from .contracts import REALIZED_REVIEW_SCHEMA_VERSION


def summary_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    trajectory_path: Path,
    pair_summary_path: Path,
    claim_assessment_path: Path,
    plot_manifest_path: Path,
    slot_diagnostics_summary_path: Path | None,
    slot_diagnostics_plot_manifest_path: Path | None,
    notebook_visual_registration: Mapping[str, Any],
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
    claim_assessment: pd.DataFrame,
) -> dict[str, Any]:
    """Build the realized-label review summary JSON payload."""

    budget_failures = int((trajectory["selection_budget_status"] != "PASS").sum()) if not trajectory.empty else 0
    confounded_pairs = (
        int((pair_summary["peer_review_claim_status"] == "null_is_confound_control_only").sum())
        if not pair_summary.empty
        else 0
    )
    target_profile = manifest.get("target_profile")
    if not isinstance(target_profile, Mapping):
        target_profile = tfbs_target_profile_for_labels(_review_label_names(manifest)).to_manifest()
    return {
        "schema_version": REALIZED_REVIEW_SCHEMA_VERSION,
        "status": "PASS" if budget_failures == 0 else "FAIL_SELECTION_BUDGET",
        "source_config_manifest_path": str(manifest_path),
        "source_config_manifest_hash": file_sha256(manifest_path),
        "campaign_count": int(manifest["campaign_count"]),
        "pair_count": int(len(pair_summary)),
        "rounds": int(manifest["rounds"]),
        "target_profile": dict(target_profile),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "pair_summary_csv_path": str(pair_summary_path),
        "pair_summary_csv_hash": file_sha256(pair_summary_path),
        "claim_assessment_csv_path": str(claim_assessment_path),
        "claim_assessment_csv_hash": file_sha256(claim_assessment_path),
        "claim_readiness": summarize_tfbs_stage_b_claim_assessment(claim_assessment),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "slot_diagnostics_summary_json_path": (
            str(slot_diagnostics_summary_path) if slot_diagnostics_summary_path is not None else None
        ),
        "slot_diagnostics_summary_json_hash": (
            file_sha256(slot_diagnostics_summary_path) if slot_diagnostics_summary_path is not None else None
        ),
        "slot_diagnostics_plot_manifest_json_path": (
            str(slot_diagnostics_plot_manifest_path) if slot_diagnostics_plot_manifest_path is not None else None
        ),
        "slot_diagnostics_plot_manifest_json_hash": (
            file_sha256(slot_diagnostics_plot_manifest_path)
            if slot_diagnostics_plot_manifest_path is not None
            else None
        ),
        "notebook_visual_registration": dict(notebook_visual_registration),
        "budget_failure_count": budget_failures,
        "confounded_null_pair_count": confounded_pairs,
        "interpretation_boundary": (
            "Realized selected label enrichment ratio is the primary ML learnability endpoint. "
            "Predicted selected score is an acquisition diagnostic and must not be used alone as evidence "
            "that sequence-matched metadata is learned better than its row-shuffled control."
        ),
    }


def _review_label_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    labels = manifest.get("sentinel_labels")
    if isinstance(labels, list) and labels:
        return tuple(str(label) for label in labels)
    pairs = manifest.get("pairs")
    if isinstance(pairs, list):
        return tuple(
            dict.fromkeys(
                str(row.get("label_name")) for row in pairs if isinstance(row, Mapping) and row.get("label_name")
            )
        )
    raise ValueError("Stage B review manifest must expose sentinel_labels or pair label_name values")
