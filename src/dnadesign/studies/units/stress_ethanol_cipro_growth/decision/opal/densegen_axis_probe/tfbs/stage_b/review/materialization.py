"""Materialize realized-label review artifacts for DenseGen TFBS Stage B campaigns."""

from __future__ import annotations

import json
from pathlib import Path

from ..claims import (
    build_tfbs_stage_b_claim_assessment,
)
from ..notebook_visuals import (
    maybe_register_tfbs_stage_b_realized_review_visuals,
    maybe_register_tfbs_stage_b_slot_diagnostic_visuals,
)
from ..slot_diagnostics.materialization import build_tfbs_stage_b_slot_diagnostics
from .contracts import TfbsStageBRealizedReviewResult
from .frames import pair_summary_frame, trajectory_frame
from .io import campaign_rows, has_slot_pairs, pair_rows, read_review_manifest
from .plots import materialize_tfbs_stage_b_realized_review_plots
from .summary import summary_payload


def build_tfbs_stage_b_realized_label_review(
    config_manifest_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    collection_visual_index_path: str | Path | None = None,
) -> TfbsStageBRealizedReviewResult:
    """Write true-label trajectory and paired positive/null review artifacts."""

    manifest_path = Path(config_manifest_path)
    manifest = read_review_manifest(manifest_path)
    campaigns = campaign_rows(manifest)
    review_dir = Path(out_dir) if out_dir is not None else manifest_path.parent.parent / "review" / "realized_labels"
    review_dir.mkdir(parents=True, exist_ok=True)

    trajectory = trajectory_frame(campaigns, rounds=int(manifest["rounds"]))
    pair_summary = pair_summary_frame(trajectory, campaigns=campaigns, pairs=pair_rows(manifest))
    claim_assessment = build_tfbs_stage_b_claim_assessment(pair_summary)
    trajectory_path = review_dir / "tfbs_stage_b_realized_label_trajectory.csv"
    pair_summary_path = review_dir / "tfbs_stage_b_positive_null_pair_summary.csv"
    claim_assessment_path = review_dir / "tfbs_stage_b_claim_assessment.csv"
    summary_path = review_dir / "tfbs_stage_b_realized_label_review.json"
    trajectory.to_csv(trajectory_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    claim_assessment.to_csv(claim_assessment_path, index=False)
    plot_manifest_path = materialize_tfbs_stage_b_realized_review_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        out_dir=review_dir / "plots",
    )
    slot_diagnostics = (
        build_tfbs_stage_b_slot_diagnostics(
            manifest_path,
            out_dir=review_dir,
        )
        if has_slot_pairs(manifest)
        else None
    )
    realized_visual_registration = maybe_register_tfbs_stage_b_realized_review_visuals(
        config_manifest_path=manifest_path,
        plot_manifest_json_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        collection_visual_index_path=collection_visual_index_path,
    )
    slot_visual_registration = (
        maybe_register_tfbs_stage_b_slot_diagnostic_visuals(
            config_manifest_path=manifest_path,
            plot_manifest_json_path=slot_diagnostics.plot_manifest_json_path,
            trajectory_csv_path=slot_diagnostics.trajectory_csv_path,
            pair_summary_csv_path=slot_diagnostics.pair_summary_csv_path,
            count_distribution_csv_path=slot_diagnostics.count_distribution_csv_path,
            collection_visual_index_path=collection_visual_index_path,
        )
        if slot_diagnostics is not None
        else {
            "status": "SKIPPED_NO_SLOT_LABELS",
            "collection_visual_index_path": None,
            "registered_visual_count": 0,
        }
    )
    notebook_visual_registration = {
        "realized_label_review": dict(realized_visual_registration),
        "slot_count_diagnostics": dict(slot_visual_registration),
    }
    summary = summary_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=trajectory_path,
        pair_summary_path=pair_summary_path,
        claim_assessment_path=claim_assessment_path,
        plot_manifest_path=plot_manifest_path,
        slot_diagnostics_summary_path=slot_diagnostics.summary_json_path if slot_diagnostics is not None else None,
        slot_diagnostics_plot_manifest_path=slot_diagnostics.plot_manifest_json_path
        if slot_diagnostics is not None
        else None,
        notebook_visual_registration=notebook_visual_registration,
        trajectory=trajectory,
        pair_summary=pair_summary,
        claim_assessment=claim_assessment,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TfbsStageBRealizedReviewResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        claim_assessment_csv_path=claim_assessment_path,
        plot_manifest_json_path=plot_manifest_path,
        slot_diagnostics_summary_json_path=slot_diagnostics.summary_json_path if slot_diagnostics is not None else None,
        slot_diagnostics_plot_manifest_json_path=slot_diagnostics.plot_manifest_json_path
        if slot_diagnostics is not None
        else None,
        notebook_visual_registration=notebook_visual_registration,
        summary_json_path=summary_path,
    )
