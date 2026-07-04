"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/selection_readiness.py

Link Eco1 panel-selection visuals into the review-deliverable manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import SECTION_FEASIBILITY_AND_HANDOFF
from .manifest import file_hashes, make_deliverable_row

_SELECTION_MANIFEST_RELATIVE_PATH = Path("design_classes/selection/selection_readiness_manifest.yaml")
_PANEL_TABLE_ARTIFACT_KEY = "candidate_selection_panel"
_PANEL_TABLE_DELIVERABLE_ID = "selection_panel_table"


def linked_selection_readiness_rows(output_root: Path) -> list[dict[str, Any]]:
    """Return review-manifest rows for materialized panel-selection plots."""

    manifest_path = output_root / _SELECTION_MANIFEST_RELATIVE_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {manifest_path}")
    rows: list[dict[str, Any]] = []
    rows.append(_panel_table_row(manifest_path=manifest_path, loaded=loaded))
    rows.append(_handoff_boundary_row(manifest_path=manifest_path))
    for plot in loaded.get("plots", []):
        if not isinstance(plot, dict):
            raise ValueError(f"Expected panel-selection plot row mappings in {manifest_path}")
        plot_id = str(plot.get("plot_id") or "")
        if not plot_id:
            raise ValueError(f"Panel-selection plot row is missing plot_id in {manifest_path}")
        plot_path = _resolve_manifest_path(manifest_path, str(plot.get("path") or ""))
        plot_status = str(plot.get("status") or "rendered")
        linked_status = "linked_existing" if plot_status == "rendered" and plot_path.exists() else plot_status
        if plot_status == "rendered" and not plot_path.exists():
            linked_status = "skipped_missing_input"
        rows.append(
            make_deliverable_row(
                deliverable_id=plot_id,
                section=SECTION_FEASIBILITY_AND_HANDOFF,
                artifact_kind=str(plot.get("artifact_kind") or "svg"),
                status=linked_status,
                path=plot_path,
                source_tables=[str(source) for source in plot.get("data_sources", [])],
                input_hashes={
                    **{str(key): str(value) for key, value in dict(plot.get("input_hashes") or {}).items()},
                    **file_hashes({"selection_readiness_manifest": manifest_path, "linked_plot": plot_path}),
                },
                alt_text=str(plot.get("alt_text") or ""),
                description=str(plot.get("description") or ""),
                interpretation_limit=str(plot.get("interpretation_limit") or ""),
                title=str(plot.get("title") or ""),
                role=str(plot.get("role") or "manuscript_facing"),
                render_mode=str(plot.get("render_mode") or "wide_visual"),
                skip_reason=""
                if linked_status != "skipped_missing_input"
                else f"Missing linked panel-selection visual: {plot_path}",
            )
        )
    return rows


def _panel_table_row(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    artifacts = loaded.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected artifacts mapping in {manifest_path}")
    table_value = str(artifacts.get(_PANEL_TABLE_ARTIFACT_KEY) or "")
    if not table_value:
        raise ValueError(f"Panel-selection manifest is missing artifacts.{_PANEL_TABLE_ARTIFACT_KEY}")
    table_path = _resolve_manifest_path(manifest_path, table_value)
    linked_status = "linked_existing" if table_path.exists() else "skipped_missing_input"
    manifest_hashes = {
        str(key): str(value)
        for key, value in dict(loaded.get("artifact_hashes") or {}).items()
        if str(key) == _PANEL_TABLE_ARTIFACT_KEY
    }
    return make_deliverable_row(
        deliverable_id=_PANEL_TABLE_DELIVERABLE_ID,
        section=SECTION_FEASIBILITY_AND_HANDOFF,
        artifact_kind="selection_panel_table",
        status=linked_status,
        path=table_path,
        source_tables=["design_classes/selection/candidate_selection_panel.parquet"],
        input_hashes={
            **manifest_hashes,
            **file_hashes({"selection_readiness_manifest": manifest_path, "selection_panel_table": table_path}),
        },
        alt_text="Compact table of the six selected Eco1 design-class representatives.",
        description=(
            "Lists the selected fold-preserved representative for each design class with feasibility, fold, "
            "sequence-distance, MSA-support, mutation-geography, and local-chemistry fields."
        ),
        interpretation_limit=(
            "The table records the proposed computational panel. It does not establish activity, strand "
            "displacement, or structured-template readthrough."
        ),
        title="Six Eco1 variants selected for assay review",
        role="manuscript_facing",
        render_mode="table",
        skip_reason="" if linked_status != "skipped_missing_input" else f"Missing selection panel table: {table_path}",
    )


def _handoff_boundary_row(*, manifest_path: Path) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id="selection_handoff_boundary",
        section=SECTION_FEASIBILITY_AND_HANDOFF,
        artifact_kind="handoff_boundary",
        status="linked_existing",
        path=manifest_path,
        source_tables=["design_classes/selection/selection_readiness_manifest.yaml"],
        input_hashes=file_hashes({"selection_readiness_manifest": manifest_path}),
        alt_text=(
            "Text note stating that the computational Eco1 panel is selected but RT-only handoff is not materialized."
        ),
        description=(
            "The current artifacts identify six computational panel candidates for assay review. The RT-only "
            "candidate_handoff.yaml record is intentionally absent until the downstream handoff contract is "
            "reviewed and accepted."
        ),
        interpretation_limit=(
            "Panel selection is not construct creation and does not assert improved RT activity, strand "
            "displacement, or structured-template readthrough."
        ),
        title="Computational panel selection is complete; RT-only handoff remains separate",
        role="manuscript_facing",
        render_mode="text",
    )


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
