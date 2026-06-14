"""Materialize replicated Stage B realized-label review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ...notebook_visuals import register_tfbs_stage_b_realized_review_visuals
from ..plots import materialize_tfbs_stage_b_realized_review_plots
from .aggregation import build_replicated_review_frames
from .contracts import TfbsStageBReplicatedReviewResult
from .summary import replicated_summary_payload
from .validation import load_replicate_manifests


def build_tfbs_stage_b_replicated_realized_label_review(
    config_manifest_paths: Sequence[str | Path],
    *,
    out_dir: str | Path,
    collection_visual_index_path: str | Path | None = None,
) -> TfbsStageBReplicatedReviewResult:
    """Write replicated true-label trajectory, endpoint, claim, and plot artifacts."""

    entries = load_replicate_manifests(config_manifest_paths)
    review_dir = Path(out_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    trajectory, pair_summary, endpoint_summary, claim_assessment = build_replicated_review_frames(entries)

    trajectory_path = review_dir / "tfbs_stage_b_replicated_realized_label_trajectory.csv"
    pair_summary_path = review_dir / "tfbs_stage_b_replicated_positive_null_pair_summary.csv"
    endpoint_summary_path = review_dir / "tfbs_stage_b_replicated_endpoint_summary.csv"
    claim_assessment_path = review_dir / "tfbs_stage_b_replicated_claim_assessment.csv"
    summary_path = review_dir / "tfbs_stage_b_replicated_realized_label_review.json"
    trajectory.to_csv(trajectory_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    endpoint_summary.to_csv(endpoint_summary_path, index=False)
    claim_assessment.to_csv(claim_assessment_path, index=False)
    plot_manifest_path = materialize_tfbs_stage_b_realized_review_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        out_dir=review_dir / "plots",
    )
    notebook_visual_registration = _notebook_visual_registration(
        collection_visual_index_path=collection_visual_index_path,
        plot_manifest_path=plot_manifest_path,
        trajectory_path=trajectory_path,
        pair_summary_path=pair_summary_path,
    )
    summary = replicated_summary_payload(
        review_dir=review_dir,
        entries=entries,
        trajectory_path=trajectory_path,
        pair_summary_path=pair_summary_path,
        endpoint_summary_path=endpoint_summary_path,
        claim_assessment_path=claim_assessment_path,
        plot_manifest_path=plot_manifest_path,
        notebook_visual_registration=notebook_visual_registration,
        trajectory=trajectory,
        pair_summary=pair_summary,
        endpoint_summary=endpoint_summary,
        claim_assessment=claim_assessment,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TfbsStageBReplicatedReviewResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        trajectory_csv_path=trajectory_path,
        replicate_pair_summary_csv_path=pair_summary_path,
        endpoint_summary_csv_path=endpoint_summary_path,
        claim_assessment_csv_path=claim_assessment_path,
        plot_manifest_json_path=plot_manifest_path,
        notebook_visual_registration=notebook_visual_registration,
        summary_json_path=summary_path,
        replicate_count=len(entries),
        replicate_seeds=tuple(entry.seed for entry in entries),
    )


def _notebook_visual_registration(
    *,
    collection_visual_index_path: str | Path | None,
    plot_manifest_path: Path,
    trajectory_path: Path,
    pair_summary_path: Path,
) -> dict[str, object]:
    if collection_visual_index_path is None:
        return {
            "status": "SKIPPED_INDEX_NOT_PROVIDED",
            "collection_visual_index_path": None,
            "registered_visual_count": 0,
        }
    return register_tfbs_stage_b_realized_review_visuals(
        collection_visual_index_path=collection_visual_index_path,
        plot_manifest_json_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
    )
