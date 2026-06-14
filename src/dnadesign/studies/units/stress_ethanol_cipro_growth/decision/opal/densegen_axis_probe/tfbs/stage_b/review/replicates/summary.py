"""Summary manifest builder for replicated Stage B realized-label reviews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ....profiles import tfbs_target_profile_for_labels, tfbs_target_profile_for_profile_id
from ....stage_a.manifests import file_sha256
from .claims import summarize_replicated_claim_assessment
from .contracts import REPLICATED_REVIEW_SCHEMA_VERSION, TfbsStageBReplicateManifest


def replicated_summary_payload(
    *,
    review_dir: Path,
    entries: Sequence[TfbsStageBReplicateManifest],
    trajectory_path: Path,
    pair_summary_path: Path,
    endpoint_summary_path: Path,
    claim_assessment_path: Path,
    plot_manifest_path: Path,
    notebook_visual_registration: Mapping[str, Any],
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    claim_assessment: pd.DataFrame,
) -> dict[str, Any]:
    """Build a machine-readable replicated review summary."""

    seeds = tuple(entry.seed for entry in entries)
    budget_failures = int((trajectory["selection_budget_status"] != "PASS").sum()) if not trajectory.empty else 0
    target_profile = _replicated_target_profile(entries)
    return {
        "schema_version": REPLICATED_REVIEW_SCHEMA_VERSION,
        "status": "PASS" if budget_failures == 0 else "FAIL_SELECTION_BUDGET",
        "review_dir": str(review_dir),
        "source_config_manifests": [
            {
                "seed": int(entry.seed),
                "path": str(entry.path),
                "sha256": file_sha256(entry.path),
            }
            for entry in entries
        ],
        "target_profile": target_profile,
        "interpretation_boundary": target_profile.get("interpretation_boundary"),
        "replicate_count": int(len(seeds)),
        "replicate_seeds": list(seeds),
        "label_count": int(endpoint_summary["label_name"].nunique()),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "replicate_pair_summary_csv_path": str(pair_summary_path),
        "replicate_pair_summary_csv_hash": file_sha256(pair_summary_path),
        "endpoint_summary_csv_path": str(endpoint_summary_path),
        "endpoint_summary_csv_hash": file_sha256(endpoint_summary_path),
        "claim_assessment_csv_path": str(claim_assessment_path),
        "claim_assessment_csv_hash": file_sha256(claim_assessment_path),
        "claim_readiness": summarize_replicated_claim_assessment(claim_assessment),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "notebook_visual_registration": dict(notebook_visual_registration),
        "trajectory_row_count": int(len(trajectory)),
        "replicate_pair_count": int(len(pair_summary)),
        "endpoint_row_count": int(len(endpoint_summary)),
        "budget_failure_count": budget_failures,
        "interval_boundary": (
            "Replicate bands use mean plus/minus sample standard deviation across deterministic seed pairs; "
            "they are descriptive spread, not inferential confidence intervals."
        ),
        "aggregation_boundary": (
            "Endpoint metrics aggregate one positive/null pair summary per replicate seed. Round rows are not "
            "pooled as independent endpoint observations."
        ),
    }


def _replicated_target_profile(entries: Sequence[TfbsStageBReplicateManifest]) -> dict[str, Any]:
    if not entries:
        raise ValueError("replicated review requires at least one config manifest")
    profile = entries[0].manifest.get("target_profile")
    if isinstance(profile, Mapping):
        profile_id = str(profile.get("profile_id") or "").strip()
        if profile_id:
            try:
                return tfbs_target_profile_for_profile_id(profile_id).to_manifest()
            except ValueError:
                pass
        payload = dict(profile)
        if not payload.get("interpretation_boundary"):
            labels = tuple(
                str(label) for label in payload.get("label_names") or entries[0].manifest.get("sentinel_labels", [])
            )
            payload["interpretation_boundary"] = tfbs_target_profile_for_labels(labels).interpretation_boundary
        return payload
    labels = tuple(str(label) for label in entries[0].manifest.get("sentinel_labels", []))
    return tfbs_target_profile_for_labels(labels).to_manifest()
