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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DEFAULT_GENERATION_POLICIES_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    normalize_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_METADATA,
)

from .constants import SECTION_PANEL_SELECTION
from .manifest import file_hashes, make_deliverable_row

_SELECTION_MANIFEST_RELATIVE_PATH = (
    Path(DEFAULT_GENERATION_POLICIES_ROOT.name) / "selection/selection_readiness_manifest.yaml"
)
_PANEL_TABLE_ARTIFACT_KEY = "candidate_selection_panel"
_HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY = "candidate_handoff_sequences"
_FUNNEL_SUMMARY_DELIVERABLE_ID = "selection_funnel_summary"
_PANEL_TABLE_DELIVERABLE_ID = "selection_panel_table"
_HANDOFF_SEQUENCE_CSV_DELIVERABLE_ID = "selection_handoff_sequences"
_HANDOFF_READINESS_DELIVERABLE_ID = "selection_handoff_readiness"
_TWIST_HANDOFF_DELIVERABLE_ID = "twist_full_cds_handoff"


def linked_selection_readiness_rows(output_root: Path, *, selection_root: Path | None = None) -> list[dict[str, Any]]:
    """Return review-manifest rows for materialized panel-selection plots."""

    manifest_path = (
        (selection_root / "selection_readiness_manifest.yaml")
        if selection_root is not None
        else output_root / _SELECTION_MANIFEST_RELATIVE_PATH
    )
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {manifest_path}")
    rows: list[dict[str, Any]] = []
    rows.append(_funnel_summary_row(output_root=output_root, manifest_path=manifest_path, loaded=loaded))
    rows.append(_panel_table_row(output_root=output_root, manifest_path=manifest_path, loaded=loaded))
    rows.append(_handoff_sequence_csv_row(output_root=output_root, manifest_path=manifest_path, loaded=loaded))
    twist_handoff_path = manifest_path.parent.parent / "twist_handoff/twist_handoff_manifest.yaml"
    if twist_handoff_path.exists():
        rows.append(_twist_handoff_row(output_root=output_root, manifest_path=twist_handoff_path))
    rows.append(_handoff_readiness_row(output_root=output_root, manifest_path=manifest_path, loaded=loaded))
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
            raise FileNotFoundError(f"Panel-selection visual declared rendered but missing: {plot_path}")
        row = make_deliverable_row(
            deliverable_id=plot_id,
            section=SECTION_PANEL_SELECTION,
            artifact_kind=str(plot.get("artifact_kind") or "svg"),
            status=linked_status,
            path=plot_path,
            source_tables=_plot_source_tables(output_root=output_root, manifest_path=manifest_path, plot=plot),
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
        row.update(_selection_plot_metadata(plot_id=plot_id, plot=plot))
        rows.append(row)
    return rows


def _selection_plot_metadata(*, plot_id: str, plot: dict[str, Any]) -> dict[str, object]:
    metadata = dict(SELECTION_PLOT_METADATA.get(plot_id, {}))
    for key in ("selection_role", "funnel_stage_id", "notebook_group", "notebook_group_label", "not_a_selector_reason"):
        if plot.get(key):
            metadata[key] = str(plot[key])
    return metadata


def _funnel_summary_row(*, output_root: Path, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
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
        section=SECTION_PANEL_SELECTION,
        artifact_kind="selection_funnel_summary",
        status="linked_existing",
        path=manifest_path,
        source_tables=[
            _source_table_label(output_root=output_root, path=manifest_path),
            _source_table_label(output_root=output_root, path=manifest_path.parent / "candidate_triage_table.parquet"),
            _source_table_label(
                output_root=output_root,
                path=manifest_path.parent / "candidate_selection_panel.parquet",
            ),
        ],
        input_hashes=file_hashes(input_paths),
        alt_text=(
            "Selection-flow summary from complete sequences through local-geometry review and three design groups "
            "to eight selected sequences."
        ),
        description=(
            "Shows complete sequences, the local-geometry screen, the distal, peripheral, and combined groups, "
            "and the final eight-sequence panel."
        ),
        interpretation_limit=(
            "ESMC and SAE are review annotations, not panel-selection evidence. The summary does not establish "
            "activity, strand displacement, or structured-template readthrough."
        ),
        title="Selection flow and panel summary",
        role="manuscript_facing",
        render_mode="table",
        evidence_summary={
            "selection_policy_id": str(loaded.get("selection_policy_id") or ""),
            "selected_candidate_count": len(list(loaded.get("selected_candidate_ids") or [])),
            "selection_summary": loaded.get("selection_summary") or {},
        },
    )


def _panel_table_row(*, output_root: Path, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    artifacts = loaded.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected artifacts mapping in {manifest_path}")
    table_value = str(artifacts.get(_PANEL_TABLE_ARTIFACT_KEY) or "")
    if not table_value:
        raise ValueError(f"Panel-selection manifest is missing artifacts.{_PANEL_TABLE_ARTIFACT_KEY}")
    table_path = _resolve_manifest_path(manifest_path, table_value)
    if not table_path.exists():
        raise FileNotFoundError(f"Panel-selection table declared by manifest is missing: {table_path}")
    manifest_hashes = {
        str(key): str(value)
        for key, value in dict(loaded.get("artifact_hashes") or {}).items()
        if str(key) == _PANEL_TABLE_ARTIFACT_KEY
    }
    return make_deliverable_row(
        deliverable_id=_PANEL_TABLE_DELIVERABLE_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="selection_panel_table",
        status="linked_existing",
        path=table_path,
        source_tables=[_source_table_label(output_root=output_root, path=table_path)],
        input_hashes={
            **manifest_hashes,
            **file_hashes({"selection_readiness_manifest": manifest_path, "selection_panel_table": table_path}),
        },
        alt_text="Compact table of eight selected Eco1 RT sequence hypotheses.",
        description=(
            "Lists complete sequence hypotheses with design group, selection rank, exact F10/R13 annotations, "
            "RT-msDNA assembly-state review status, fold metrics, "
            "within-group mutation-set distance, regional MSA support, mutation "
            "geography, and peripheral DNA/RNA chemistry."
        ),
        interpretation_limit=(
            "The table records the proposed computational panel. It does not establish activity, strand "
            "displacement, or structured-template readthrough."
        ),
        title="Eight selected Eco1 RT sequences",
        role="manuscript_facing",
        render_mode="table",
        skip_reason="",
    )


def _handoff_sequence_csv_row(*, output_root: Path, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    artifacts = loaded.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected artifacts mapping in {manifest_path}")
    table_value = str(artifacts.get(_HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY) or "")
    if not table_value:
        raise ValueError(f"Panel-selection manifest is missing artifacts.{_HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY}")
    table_path = _resolve_manifest_path(manifest_path, table_value)
    if not table_path.exists():
        raise FileNotFoundError(f"Panel-selection sequence CSV declared by manifest is missing: {table_path}")
    manifest_hashes = {
        str(key): str(value)
        for key, value in dict(loaded.get("artifact_hashes") or {}).items()
        if str(key) == _HANDOFF_SEQUENCE_CSV_ARTIFACT_KEY
    }
    return make_deliverable_row(
        deliverable_id=_HANDOFF_SEQUENCE_CSV_DELIVERABLE_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="candidate_handoff_sequence_csv",
        status="linked_existing",
        path=table_path,
        source_tables=[
            _source_table_label(output_root=output_root, path=table_path),
            _source_table_label(
                output_root=output_root,
                path=manifest_path.parent / "candidate_selection_panel.parquet",
            ),
        ],
        input_hashes={
            **manifest_hashes,
            **file_hashes({"selection_readiness_manifest": manifest_path, "candidate_handoff_sequences": table_path}),
        },
        alt_text="Flat CSV table of selected canonical 320-aa RT protein sequences and explicit non-DNA status fields.",
        description=(
            "Lists selected canonical_rt_protein sequences with mapped-source hashes, selection slots, and explicit "
            "non-DNA, non-codon-optimized, and not-screened status fields."
        ),
        interpretation_limit=(
            "This CSV is a protein-sequence handoff table. It is not an E. coli codon-optimized DNA design and it has "
            "not passed DNA restriction-site screening."
        ),
        title="Selected protein sequences keep handoff scope explicit",
        role="manuscript_facing",
        render_mode="table",
        skip_reason="",
    )


def _handoff_readiness_row(*, output_root: Path, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    readiness = _normalized_handoff_readiness(manifest_path=manifest_path, loaded=loaded)
    handoff_path = _resolve_manifest_path(manifest_path, str(readiness["candidate_handoff_path"]))
    handoff_file_present = bool(readiness["candidate_handoff_file_present"])
    handoff_materialized = bool(readiness["candidate_handoff_materialized"])
    if handoff_materialized:
        title = "RT-only candidate handoff is materialized"
        handoff_state = "materialized and valid"
    elif handoff_file_present:
        title = "RT-only candidate handoff needs a valid candidate_handoff.yaml"
        handoff_state = "present but not valid"
    else:
        title = "RT-only candidate handoff still needs candidate_handoff.yaml"
        handoff_state = "absent"
    return make_deliverable_row(
        deliverable_id=_HANDOFF_READINESS_DELIVERABLE_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="handoff_readiness",
        status="linked_existing",
        path=manifest_path,
        source_tables=[_source_table_label(output_root=output_root, path=manifest_path)],
        input_hashes=file_hashes({"selection_readiness_manifest": manifest_path, "candidate_handoff": handoff_path}),
        alt_text="Checklist for RT-only candidate-handoff readiness.",
        description=(
            f"candidate_handoff.yaml is {handoff_state}; panel selection remains separate from construct subject "
            "creation."
        ),
        interpretation_limit=(
            "The readiness state has no assay acceptance gate and does not create an RT-lnRNA construct subject."
        ),
        title=title,
        role="manuscript_facing",
        render_mode="text",
        evidence_summary={str(key): value for key, value in readiness.items()},
    )


def _twist_handoff_row(*, output_root: Path, manifest_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema_id") != "eco1_rt.twist_full_cds_handoff"
        or loaded.get("schema_version") != 2
    ):
        raise ValueError(f"Expected eco1_rt.twist_full_cds_handoff schema version 2 at {manifest_path}")
    sequence_count = len(list(loaded.get("sequences") or []))
    return make_deliverable_row(
        deliverable_id=_TWIST_HANDOFF_DELIVERABLE_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="twist_handoff_manifest",
        status="linked_existing",
        path=manifest_path,
        source_tables=[_source_table_label(output_root=output_root, path=manifest_path)],
        input_hashes=file_hashes({"twist_handoff_manifest": manifest_path}),
        alt_text=(
            f"Twist handoff summary for {sequence_count} full-length Eco1 RT CDS designs, with mutation labels and "
            "sequence-quality checks."
        ),
        description=(
            "Lists the exact 963-bp CDS designs, compact amino-acid substitutions, GenBank files, sequence hashes, "
            f"and restriction-site checks for all {sequence_count} selected hypotheses."
        ),
        interpretation_limit=(
            "The sequences are ready for a vendor complexity check and quote. Cloning remains blocked until "
            "assembly flanks and junctions are declared."
        ),
        title="Twist full-CDS handoff",
        role="manuscript_facing",
        render_mode="table",
    )


def _normalized_handoff_readiness(*, manifest_path: Path, loaded: dict[str, Any]) -> dict[str, object]:
    raw = loaded.get("handoff_readiness") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected handoff_readiness mapping in {manifest_path}")
    return normalize_handoff_readiness(selection_root=manifest_path.parent, raw=raw)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _source_table_label(*, output_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)


def _plot_source_tables(*, output_root: Path, manifest_path: Path, plot: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for source in plot.get("data_sources", []):
        labels.append(str(source))
    return labels
