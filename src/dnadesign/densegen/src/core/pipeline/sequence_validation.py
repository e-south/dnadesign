"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/sequence_validation.py

Sequence validation helpers for pipeline constraints and postprocess checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..artifacts.library import LibraryRecord


def _apply_pad_offsets(used_tfbs_detail: list[dict], pad_meta: dict) -> list[dict]:
    pad_left = 0
    if pad_meta.get("used") and pad_meta.get("end") == "5prime":
        pad_left = int(pad_meta.get("bases") or 0)
    for entry in used_tfbs_detail:
        offset_raw = int(entry.get("offset_raw", entry.get("offset", 0)))
        length = int(entry.get("length", len(entry.get("tfbs") or "")))
        offset = offset_raw + pad_left
        entry["offset_raw"] = offset_raw
        entry["pad_left"] = pad_left
        entry["length"] = length
        entry["offset"] = offset
        entry["end"] = offset + length
    return used_tfbs_detail


def _validate_library_constraints(
    record: LibraryRecord,
    *,
    groups: list,
    min_count_by_regulator: dict[str, int],
    input_name: str,
    plan_name: str,
) -> None:
    required_regulators_selected = list(record.required_regulators_selected or [])
    if groups:
        if not required_regulators_selected:
            raise RuntimeError(
                f"Library artifact missing required_regulators_selected for {input_name}/{plan_name} "
                f"(library_index={record.library_index}). Rebuild libraries with the current version."
            )
        library_tf_set = {tf for tf in record.library_tfs if tf}
        missing = [tf for tf in required_regulators_selected if tf not in library_tf_set]
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Library artifact required_regulators_selected includes TFs not in the library: {preview}."
            )
        group_members = {m for g in groups for m in g.members}
        invalid = [tf for tf in required_regulators_selected if tf not in group_members]
        if invalid:
            preview = ", ".join(invalid[:10])
            raise RuntimeError(
                f"Library artifact required_regulators_selected includes TFs not in regulator groups: {preview}."
            )
        for group in groups:
            selected = [tf for tf in required_regulators_selected if tf in group.members]
            if len(selected) < int(group.min_required):
                raise RuntimeError(
                    f"Library artifact required_regulators_selected does not satisfy group '{group.name}' "
                    f"min_required={group.min_required}."
                )
    for tf, min_count in min_count_by_regulator.items():
        found = sum(1 for t in record.library_tfs if t == tf)
        if found < int(min_count):
            raise RuntimeError(
                f"Library artifact for {input_name}/{plan_name} (library_index={record.library_index}) "
                f"has tf={tf} count={found} < min_count_by_regulator={min_count}."
            )
