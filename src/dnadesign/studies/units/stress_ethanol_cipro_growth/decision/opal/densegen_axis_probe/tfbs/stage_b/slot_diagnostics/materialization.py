"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/slot_diagnostics/materialization.py

Slot-count confound diagnostics for DenseGen TFBS Stage B campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...stage_a.manifests import file_sha256
from .contracts import (
    POSITION_SIGNAL_AFTER_COUNT_RESTRICTION,
    SLOT_DIAGNOSTIC_SCHEMA_VERSION,
    TfbsStageBSlotDiagnosticResult,
)
from .io import _campaign_rows, _read_json, _slot_pair_rows
from .metrics import _slot_pair_summary_frame, _slot_trajectory_frames
from .plots import materialize_tfbs_stage_b_slot_diagnostic_plots


def build_tfbs_stage_b_slot_diagnostics(
    config_manifest_path: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> TfbsStageBSlotDiagnosticResult:
    """Write count-confound and count-stratified lift diagnostics for slot-label campaigns."""

    manifest_path = Path(config_manifest_path)
    manifest = _read_json(manifest_path)
    campaigns = _campaign_rows(manifest)
    pairs = _slot_pair_rows(manifest)
    review_dir = Path(out_dir) if out_dir is not None else manifest_path.parent.parent / "review" / "realized_labels"
    review_dir.mkdir(parents=True, exist_ok=True)

    trajectory, count_distribution = _slot_trajectory_frames(campaigns, pairs, rounds=int(manifest["rounds"]))
    pair_summary = _slot_pair_summary_frame(trajectory, pairs=pairs)

    trajectory_path = review_dir / "tfbs_stage_b_slot_count_diagnostic_trajectory.csv"
    count_distribution_path = review_dir / "tfbs_stage_b_slot_count_distribution.csv"
    pair_summary_path = review_dir / "tfbs_stage_b_slot_restricted_pair_summary.csv"
    summary_path = review_dir / "tfbs_stage_b_slot_diagnostics.json"
    trajectory.to_csv(trajectory_path, index=False)
    count_distribution.to_csv(count_distribution_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    plot_manifest_path = materialize_tfbs_stage_b_slot_diagnostic_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        count_distribution_csv_path=count_distribution_path,
        out_dir=review_dir / "plots",
    )
    summary = _summary_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=trajectory_path,
        count_distribution_path=count_distribution_path,
        pair_summary_path=pair_summary_path,
        plot_manifest_path=plot_manifest_path,
        trajectory=trajectory,
        pair_summary=pair_summary,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TfbsStageBSlotDiagnosticResult(
        status=str(summary["status"]),
        review_dir=review_dir,
        trajectory_csv_path=trajectory_path,
        count_distribution_csv_path=count_distribution_path,
        pair_summary_csv_path=pair_summary_path,
        plot_manifest_json_path=plot_manifest_path,
        summary_json_path=summary_path,
    )


def _summary_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    trajectory_path: Path,
    count_distribution_path: Path,
    pair_summary_path: Path,
    plot_manifest_path: Path,
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> dict[str, Any]:
    resolved = pair_summary.loc[
        pair_summary["slot_diagnostic_status"] == POSITION_SIGNAL_AFTER_COUNT_RESTRICTION, "label_name"
    ].astype(str)
    unresolved = pair_summary.loc[
        pair_summary["slot_diagnostic_status"] != POSITION_SIGNAL_AFTER_COUNT_RESTRICTION, "label_name"
    ].astype(str)
    return {
        "schema_version": SLOT_DIAGNOSTIC_SCHEMA_VERSION,
        "status": "PASS",
        "source_config_manifest_path": str(manifest_path),
        "source_config_manifest_hash": file_sha256(manifest_path),
        "campaign_count": int(manifest["campaign_count"]),
        "slot_label_count": int(pair_summary["label_name"].nunique()),
        "rounds": int(manifest["rounds"]),
        "trajectory_csv_path": str(trajectory_path),
        "trajectory_csv_hash": file_sha256(trajectory_path),
        "count_distribution_csv_path": str(count_distribution_path),
        "count_distribution_csv_hash": file_sha256(count_distribution_path),
        "pair_summary_csv_path": str(pair_summary_path),
        "pair_summary_csv_hash": file_sha256(pair_summary_path),
        "plot_manifest_json_path": str(plot_manifest_path),
        "plot_manifest_json_hash": file_sha256(plot_manifest_path),
        "resolved_position_signal_labels": resolved.tolist(),
        "unresolved_slot_labels": unresolved.tolist(),
        "slot_diagnostic_status_counts": {
            str(key): int(value)
            for key, value in pair_summary["slot_diagnostic_status"].value_counts().sort_index().to_dict().items()
        },
        "trajectory_row_count": int(len(trajectory)),
        "interpretation_boundary": (
            "The slot null preserves row-level target-family count, so raw slot lift can be count-confounded. "
            "Use the count-stratified diagnostics to decide whether selected rows show position signal beyond count."
        ),
    }
