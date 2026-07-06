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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    normalize_handoff_readiness,
)

from .constants import SECTION_FEASIBILITY_AND_HANDOFF
from .manifest import file_hashes, make_deliverable_row

_SELECTION_MANIFEST_RELATIVE_PATH = Path("design_classes/selection/selection_readiness_manifest.yaml")
_PANEL_TABLE_ARTIFACT_KEY = "candidate_selection_panel"
_HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY = "candidate_handoff_sequences"
_FUNNEL_SUMMARY_DELIVERABLE_ID = "selection_funnel_summary"
_PANEL_TABLE_DELIVERABLE_ID = "selection_panel_table"
_HANDOFF_SEQUENCE_CSV_DELIVERABLE_ID = "selection_handoff_sequences"
_HANDOFF_READINESS_DELIVERABLE_ID = "selection_handoff_readiness"


def linked_selection_readiness_rows(output_root: Path) -> list[dict[str, Any]]:
    """Return review-manifest rows for materialized panel-selection plots."""

    manifest_path = output_root / _SELECTION_MANIFEST_RELATIVE_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {manifest_path}")
    rows: list[dict[str, Any]] = []
    rows.append(_funnel_summary_row(manifest_path=manifest_path, loaded=loaded))
    rows.append(_panel_table_row(manifest_path=manifest_path, loaded=loaded))
    rows.append(_handoff_sequence_csv_row(manifest_path=manifest_path, loaded=loaded))
    rows.append(_handoff_readiness_row(manifest_path=manifest_path, loaded=loaded))
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


def _funnel_summary_row(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    artifacts = loaded.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected artifacts mapping in {manifest_path}")
    input_paths = {"selection_readiness_manifest": manifest_path}
    triage_value = str(artifacts.get("candidate_triage_table") or "")
    if triage_value:
        input_paths["candidate_triage_table"] = _resolve_manifest_path(manifest_path, triage_value)
    panel_value = str(artifacts.get(_PANEL_TABLE_ARTIFACT_KEY) or "")
    if panel_value:
        input_paths["candidate_selection_panel"] = _resolve_manifest_path(manifest_path, panel_value)
    return make_deliverable_row(
        deliverable_id=_FUNNEL_SUMMARY_DELIVERABLE_ID,
        section=SECTION_FEASIBILITY_AND_HANDOFF,
        artifact_kind="selection_funnel_summary",
        status="linked_existing",
        path=manifest_path,
        source_tables=[
            "design_classes/selection/selection_readiness_manifest.yaml",
            "design_classes/selection/candidate_triage_table.parquet",
            "design_classes/selection/candidate_selection_panel.parquet",
        ],
        input_hashes=file_hashes(input_paths),
        alt_text="Selection-funnel summary table with row counts, gate counts, selected IDs, and policy notes.",
        description=(
            "Shows row counts, gate counts, selected IDs, and selection policy from selection_readiness_manifest.yaml."
        ),
        interpretation_limit=(
            "ESMC and SAE are review annotations, not panel-selection evidence. The summary does not establish "
            "activity, strand displacement, or structured-template readthrough."
        ),
        title="Panel selection keeps fold checks separate from activity claims",
        role="manuscript_facing",
        render_mode="table",
        evidence_summary={
            "selection_policy_id": str(loaded.get("selection_policy_id") or ""),
            "selected_candidate_count": len(list(loaded.get("selected_candidate_ids") or [])),
        },
    )


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
        title="Six Eco1 RT variants form a protein review panel",
        role="manuscript_facing",
        render_mode="table",
        skip_reason="" if linked_status != "skipped_missing_input" else f"Missing selection panel table: {table_path}",
    )


def _handoff_sequence_csv_row(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    artifacts = loaded.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected artifacts mapping in {manifest_path}")
    table_value = str(artifacts.get(_HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY) or "")
    if not table_value:
        return make_deliverable_row(
            deliverable_id=_HANDOFF_SEQUENCE_CSV_DELIVERABLE_ID,
            section=SECTION_FEASIBILITY_AND_HANDOFF,
            artifact_kind="candidate_handoff_sequence_csv",
            status="skipped_missing_input",
            path=manifest_path.parent / "candidate_handoff_sequences.csv",
            source_tables=["design_classes/selection/selection_readiness_manifest.yaml"],
            input_hashes=file_hashes({"selection_readiness_manifest": manifest_path}),
            alt_text="Selected RT protein sequence CSV was not declared by the panel-selection manifest.",
            description="Selected protein-sequence CSV is unavailable until selection readiness is regenerated.",
            interpretation_limit="Missing sequence CSV cannot support handoff review.",
            title="Selected protein sequences keep handoff scope explicit",
            role="manuscript_facing",
            render_mode="table",
            skip_reason="selection_readiness_manifest.yaml has no candidate_handoff_sequences artifact",
        )
    table_path = _resolve_manifest_path(manifest_path, table_value)
    linked_status = "linked_existing" if table_path.exists() else "skipped_missing_input"
    manifest_hashes = {
        str(key): str(value)
        for key, value in dict(loaded.get("artifact_hashes") or {}).items()
        if str(key) == _HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY
    }
    return make_deliverable_row(
        deliverable_id=_HANDOFF_SEQUENCE_CSV_DELIVERABLE_ID,
        section=SECTION_FEASIBILITY_AND_HANDOFF,
        artifact_kind="candidate_handoff_sequence_csv",
        status=linked_status,
        path=table_path,
        source_tables=[
            "design_classes/selection/candidate_handoff_sequences.csv",
            "design_classes/selection/candidate_selection_panel.parquet",
        ],
        input_hashes={
            **manifest_hashes,
            **file_hashes({"selection_readiness_manifest": manifest_path, "candidate_handoff_sequences": table_path}),
        },
        alt_text="Flat CSV table of selected mapped RT-chain protein sequences and explicit non-DNA status fields.",
        description=(
            "Lists selected mapped_rt_chain_protein sequences with sequence hashes, selection slots, feasibility "
            "state, and explicit non-DNA, non-codon-optimized, and not-screened status fields."
        ),
        interpretation_limit=(
            "This CSV is a protein-sequence handoff table. It is not an E. coli codon-optimized DNA design and it has "
            "not passed DNA restriction-site screening."
        ),
        title="Selected protein sequences keep handoff scope explicit",
        role="manuscript_facing",
        render_mode="table",
        skip_reason="" if linked_status != "skipped_missing_input" else f"Missing handoff sequence CSV: {table_path}",
    )


def _handoff_readiness_row(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    readiness = _normalized_handoff_readiness(manifest_path=manifest_path, loaded=loaded)
    handoff_path = _resolve_manifest_path(manifest_path, str(readiness["candidate_handoff_path"]))
    handoff_state = "present" if handoff_path.exists() else "absent"
    return make_deliverable_row(
        deliverable_id=_HANDOFF_READINESS_DELIVERABLE_ID,
        section=SECTION_FEASIBILITY_AND_HANDOFF,
        artifact_kind="handoff_readiness",
        status="linked_existing",
        path=manifest_path,
        source_tables=["design_classes/selection/selection_readiness_manifest.yaml"],
        input_hashes=file_hashes({"selection_readiness_manifest": manifest_path, "candidate_handoff": handoff_path}),
        alt_text="Checklist for RT-only candidate-handoff readiness.",
        description=(
            f"candidate_handoff.yaml is {handoff_state}; panel selection remains separate from construct subject "
            "creation."
        ),
        interpretation_limit=(
            "The readiness state has no assay acceptance gate and does not create an RT-lnRNA construct subject."
        ),
        title="Candidate handoff remains blocked until candidate_handoff.yaml exists",
        role="manuscript_facing",
        render_mode="text",
        evidence_summary={str(key): value for key, value in readiness.items()},
    )


def _normalized_handoff_readiness(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, object]:
    raw = loaded.get("handoff_readiness") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected handoff_readiness mapping in {manifest_path}")
    return normalize_handoff_readiness(selection_root=manifest_path.parent, raw=raw)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
