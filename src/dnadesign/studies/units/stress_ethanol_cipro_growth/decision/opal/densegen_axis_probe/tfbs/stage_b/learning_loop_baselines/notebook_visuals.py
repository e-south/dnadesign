"""Notebook visual entries for TFBS learning-loop baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..notebook_visuals.io import csv_row_count, mapping_list, require_existing_file
from .contracts import (
    LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION,
    LEARNING_LOOP_BASELINE_SCHEMA_VERSION,
    LEARNING_LOOP_BASELINE_SURFACE_KIND,
    validate_learning_loop_source_profiles,
)


def learning_loop_visual_entries(manifest_path: Path) -> list[dict[str, Any]]:
    """Build OPAL collection visual entries from a frozen replay manifest."""

    manifest = _read_manifest(manifest_path)
    plot_manifest_path = Path(str(manifest["plot_manifest_json_path"]))
    plot_manifest = _read_plot_manifest(plot_manifest_path)
    trajectory_path = Path(str(manifest["trajectory_csv_path"]))
    claim_path = Path(str(manifest["claim_interpretation_csv_path"]))
    require_existing_file(trajectory_path, role="frozen replay trajectory CSV")
    require_existing_file(claim_path, role="frozen replay claim interpretation CSV")
    rows: list[dict[str, Any]] = []
    for plot in mapping_list(plot_manifest.get("plots"), field="plots"):
        plot_path = Path(str(plot.get("path") or ""))
        require_existing_file(plot_path, role="frozen replay plot")
        kind = str(plot.get("kind") or "").strip()
        if kind == "frozen_round0_cumulative_enrichment":
            rows.append(_visual_entry(plot, manifest, manifest_path, trajectory_path=trajectory_path))
        elif kind == "frozen_round0_endpoint_adaptive_gain":
            rows.append(_visual_entry(plot, manifest, manifest_path, trajectory_path=claim_path))
        elif kind == "known_label_gain_recovery":
            rows.append(_visual_entry(plot, manifest, manifest_path, trajectory_path=claim_path))
        else:
            raise ValueError(f"Unsupported frozen replay plot kind: {kind!r}")
    return rows


def _visual_entry(
    plot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    trajectory_path: Path,
) -> dict[str, Any]:
    kind = str(plot["kind"])
    is_endpoint = kind == "frozen_round0_endpoint_adaptive_gain"
    is_known_label_reference = kind == "known_label_gain_recovery"
    review_id = str(manifest["review_id"])
    source_profile_ids = [str(value) for value in manifest.get("source_profile_ids") or []]
    profile_id = source_profile_ids[0] if len(source_profile_ids) == 1 else "multiple_tfbs_profiles"
    return {
        "visual_id": f"tfbs_{review_id}__{kind}",
        "label": _visual_label(kind),
        "title": str(plot.get("title") or kind),
        "surface_kind": LEARNING_LOOP_BASELINE_SURFACE_KIND,
        "kind": LEARNING_LOOP_BASELINE_SURFACE_KIND,
        "view_kind": kind,
        "source_plot_name": kind,
        "source_plot_kind": LEARNING_LOOP_BASELINE_SURFACE_KIND,
        "comparison_scope": "study_review",
        "comparison_set_key": str(manifest["comparison_set_key"]),
        "comparison_set_label": str(manifest["comparison_set_label"]),
        "comparison_set_match": {
            "review_surface": "learning_loop_baseline",
            "profile_id": profile_id,
            "profile_ids": source_profile_ids,
        },
        "relationship_id": "active_retraining_vs_frozen_round0",
        "relationship_kind": "learning_loop_ablation",
        "group_key": "learning_loop_baseline",
        "metric": _metric_id(kind),
        "metric_label": "Cumulative enrichment vs candidate pool"
        if not (is_endpoint or is_known_label_reference)
        else ("Final active-minus-frozen cumulative lift" if is_endpoint else "Fraction of known-label gain recovered"),
        "metric_expression": _metric_expression(kind),
        "cohort": "selected",
        "summary": "cumulative_trajectory"
        if not (is_endpoint or is_known_label_reference)
        else "final_acquired_budget",
        "interval_kind": str(plot.get("interval_kind") or "none"),
        "interval": dict(plot.get("interval") or {}),
        "interpretation_note": str(manifest.get("interpretation_boundary") or ""),
        "premise": (
            "This asks whether retraining after each acquisition improves cumulative selected-label enrichment "
            "over a ranking frozen after the initial seed batch."
        ),
        "math_note": (
            "Lift is cumulative mean selected label divided by the candidate-pool mean for the same label table."
        ),
        "design_note": (
            "Replay uses the same completed campaigns, shared initial IDs, candidate scope, model config, and "
            "selection budget; only retraining after round 0 is removed."
        ),
        "claim_boundary": str(manifest.get("claim_boundary") or ""),
        "row_count": csv_row_count(trajectory_path, role="frozen replay tidy CSV"),
        "path": str(Path(str(plot["path"])).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "tidy_csv": str(trajectory_path.resolve()),
        "freshness": {"status": "current"},
        "caption": str(plot.get("caption") or ""),
        "alt_text": str(plot.get("alt_text") or ""),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEARNING_LOOP_BASELINE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported frozen replay manifest schema: {payload.get('schema_version')!r}")
    if payload.get("status") != "PASS":
        raise ValueError(f"Frozen replay visual source must be PASS: {path}")
    for key in ("review_id", "comparison_set_key", "comparison_set_label", "visual_tier", "source_profile_ids"):
        if key not in payload:
            raise ValueError(f"Frozen replay visual source is missing {key}: {path}")
    validate_learning_loop_source_profiles(
        visual_tier=payload.get("visual_tier"),
        source_profile_ids=payload.get("source_profile_ids"),
        path=path,
    )
    return payload


def _read_plot_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported frozen replay plot manifest schema: {payload.get('schema_version')!r}")
    return payload


def _visual_label(kind: str) -> str:
    if kind == "frozen_round0_endpoint_adaptive_gain":
        return "Final active-minus-frozen lift"
    if kind == "known_label_gain_recovery":
        return "Known-label gain recovered"
    return "Active vs frozen cumulative enrichment"


def _metric_id(kind: str) -> str:
    if kind == "frozen_round0_endpoint_adaptive_gain":
        return "active_minus_frozen_final_cumulative_lift"
    if kind == "known_label_gain_recovery":
        return "active_fraction_of_known_label_gain_recovered"
    return "cumulative_lift_ratio"


def _metric_expression(kind: str) -> str:
    if kind == "frozen_round0_endpoint_adaptive_gain":
        return "active final cumulative lift - frozen final cumulative lift"
    if kind == "known_label_gain_recovery":
        return "(active final cumulative lift - 1) / (same-budget known-label final cumulative lift - 1)"
    return "cumulative mean(selected label) / mean(candidate-pool label)"
