"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_rows.py

Structure-browser row loading and lookup helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)

from .notebook_structure_dashboard import format_float

_MASK_STRUCTURE_BROWSER_DELIVERABLE_ID = "mask_structure_browser_manifest"
_SAE_STRUCTURE_BROWSER_DELIVERABLE_ID = "biohub_esmc_sae_structure_browser_manifest"
_SAE_HIGHLIGHT_SOURCE_DELIVERABLE_IDS = frozenset({"interactive_structure_browser_manifest"})


def load_structure_browser_rows(
    *,
    manifest_root: Path,
    deliverables: list[dict[str, Any]],
    selected_deliverable_id: str = "",
    source_candidate_id: str = "",
) -> list[dict[str, Any]]:
    """Load rows for one interactive structure-browser manifest, or all when unfiltered."""

    rows: list[dict[str, Any]] = []
    for manifest_row in _interactive_structure_manifest_rows(deliverables):
        deliverable_id = str(manifest_row.get("deliverable_id") or "")
        if selected_deliverable_id and deliverable_id != selected_deliverable_id:
            continue
        if str(manifest_row.get("status") or "") != "rendered":
            continue
        manifest_path = resolve_manifest_path(manifest_root, str(manifest_row["path"]))
        if not manifest_path.exists():
            continue
        stat = manifest_path.stat()
        payload = _load_manifest_mapping(
            str(manifest_path),
            modified_time_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
        )
        if not payload:
            continue
        rows.extend(
            _enriched_structure_rows(
                payload=payload,
                manifest_row=manifest_row,
                deliverable_id=deliverable_id,
                browser_root=manifest_path.parent,
                source_candidate_id=source_candidate_id,
            )
        )
    return rows


@lru_cache(maxsize=16)
def _load_manifest_mapping(
    path: str,
    *,
    modified_time_ns: int,
    size_bytes: int,
) -> dict[str, Any]:
    """Parse a structure manifest once per observed file revision."""

    del modified_time_ns, size_bytes
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_structure_highlight_rows(
    *,
    manifest_root: Path,
    deliverables: list[dict[str, Any]],
    selected_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Load only the reference-mask and candidate SAE rows needed for residue highlights."""

    if selected_row is None or str(selected_row.get("structure_view_mode") or "") == "reference_selection":
        return []
    candidate_id = str(selected_row.get("source_candidate_id") or selected_row.get("candidate_id") or "")
    if not candidate_id:
        return []
    rows = load_structure_browser_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
        selected_deliverable_id=_MASK_STRUCTURE_BROWSER_DELIVERABLE_ID,
    )
    if _supports_sae_highlights(selected_row):
        rows.extend(
            load_structure_browser_rows(
                manifest_root=manifest_root,
                deliverables=deliverables,
                selected_deliverable_id=_SAE_STRUCTURE_BROWSER_DELIVERABLE_ID,
                source_candidate_id=candidate_id,
            )
        )
    return rows


def structure_browser_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_section: str,
    selected_deliverable_id: str = "",
    selected_group: str = "",
) -> dict[str, dict[str, Any]]:
    """Build dropdown labels for the selected interactive structure deliverable."""

    return {
        _structure_browser_label(row): row
        for row in rows
        if str(row.get("_section") or "") == selected_section
        and str(row.get("_deliverable_id") or "") == selected_deliverable_id
        and (not selected_group or str(row.get("group") or "") == selected_group)
    }


def structure_group_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_section: str,
    selected_deliverable_id: str = "",
) -> dict[str, str]:
    """Build structure group dropdown labels for the selected structure deliverable."""

    groups: dict[str, str] = {}
    for row in rows:
        if str(row.get("_section") or "") != selected_section:
            continue
        if str(row.get("_deliverable_id") or "") != selected_deliverable_id:
            continue
        group = str(row.get("group") or "Ungrouped structures")
        groups.setdefault(group, group)
    return {group: groups[group] for group in sorted(groups, key=_structure_group_sort_key)}


def structure_highlight_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_row: dict[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
    """Build residue-highlight options for the currently selected structure."""

    if selected_row is None or str(selected_row.get("structure_view_mode") or "") == "reference_selection":
        return {}
    candidate_id = str(selected_row.get("source_candidate_id") or selected_row.get("candidate_id") or "")
    if not candidate_id:
        return {}
    options: dict[str, dict[str, Any] | None] = {"No residue highlight": None}
    for row in rows:
        if str(row.get("_deliverable_id") or "") != _MASK_STRUCTURE_BROWSER_DELIVERABLE_ID:
            continue
        if str(row.get("structure_view_mode") or "") != "reference_selection":
            continue
        options[_structure_browser_label(row)] = row
    for row in rows:
        if str(row.get("_deliverable_id") or "") != _SAE_STRUCTURE_BROWSER_DELIVERABLE_ID:
            continue
        if str(row.get("source_candidate_id") or row.get("candidate_id") or "") != candidate_id:
            continue
        if row.get("feature_index") is None:
            continue
        options[_structure_highlight_label(row)] = row
    return options


def _supports_sae_highlights(selected_row: dict[str, Any]) -> bool:
    return str(selected_row.get("_deliverable_id") or "") in _SAE_HIGHLIGHT_SOURCE_DELIVERABLE_IDS


def _structure_group_sort_key(group: str) -> tuple[int, str]:
    prefix = group.partition(" ")[0]
    return (int(prefix), group) if prefix.isdigit() else (10_000, group)


def _enriched_structure_rows(
    *,
    payload: dict[str, Any],
    manifest_row: dict[str, Any],
    deliverable_id: str,
    browser_root: Path,
    source_candidate_id: str,
) -> list[dict[str, Any]]:
    reference = dict(payload.get("reference") or {})
    alignment = dict(payload.get("alignment") or {})
    rows: list[dict[str, Any]] = []
    for row in payload.get("structures") or []:
        if not isinstance(row, dict):
            continue
        row_candidate_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "")
        if source_candidate_id and row_candidate_id != source_candidate_id:
            continue
        enriched = dict(row)
        enriched["_browser_root"] = str(browser_root)
        enriched["_reference"] = reference
        enriched["_alignment"] = alignment
        enriched["_deliverable_id"] = deliverable_id
        enriched["_section"] = str(manifest_row.get("section") or "")
        enriched["_control_label"] = str(payload.get("control_label") or "Structure view")
        enriched["_deliverable_description"] = str(manifest_row.get("description") or "")
        enriched["_deliverable_alt_text"] = str(manifest_row.get("alt_text") or "")
        enriched["_interpretation_limit"] = str(
            payload.get("interpretation_limit") or manifest_row.get("interpretation_limit") or ""
        )
        rows.append(enriched)
    return rows


def _interactive_structure_manifest_rows(deliverables: list[dict[str, Any]]) -> Any:
    for row in deliverables:
        if str(row.get("artifact_kind") or "") == "structure_browser_manifest":
            yield row


def _structure_browser_label(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        residue_count = row.get("selection_residue_count")
        if residue_count is not None:
            return f"{label} | {int(residue_count)} residues"
        return label
    rmsd = row.get("wt_runtime_ca_rmsd")
    plddt = row.get("plddt")
    if rmsd is not None and plddt is not None:
        return f"{label} | WT RMSD {float(rmsd):.2f} A | pLDDT {float(plddt):.1f}"
    return label


def _structure_highlight_label(row: dict[str, Any]) -> str:
    feature_index = int(row["feature_index"])
    rank_text = ""
    display_label = str(row.get("display_label") or "")
    if "peak order " in display_label:
        rank_text = display_label.split("peak order ", 1)[1].split("|", 1)[0].strip()
    activation_max = format_float(row.get("activation_max"), decimals=3)
    suffix = f" | max {activation_max}" if activation_max else ""
    if rank_text:
        return f"SAE F{feature_index} | peak order {rank_text}{suffix}"
    return f"SAE F{feature_index}{suffix}"
