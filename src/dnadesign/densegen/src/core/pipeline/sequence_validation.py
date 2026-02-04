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


def _find_motif_positions(seq: str, motif: str, bounds: tuple[int, int] | list[int] | None) -> list[int]:
    seq = str(seq)
    motif = str(motif)
    if not motif:
        return []
    lo = None
    hi = None
    if bounds is not None:
        try:
            lo = int(bounds[0])
            hi = int(bounds[1])
        except Exception:
            lo = None
            hi = None
    positions: list[int] = []
    start = 0
    while start < len(seq):
        idx = seq.find(motif, start)
        if idx == -1:
            break
        if lo is not None and idx < lo:
            start = idx + 1
            continue
        if hi is not None and idx > hi:
            start = idx + 1
            continue
        if idx + len(motif) <= len(seq):
            positions.append(idx)
        start = idx + 1
    return positions


def _promoter_windows(seq: str, fixed_elements_dump: dict) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    pcs = fixed_elements_dump.get("promoter_constraints") or []
    for pc in pcs:
        if not isinstance(pc, dict):
            continue
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        if not upstream or not downstream:
            continue
        spacer = pc.get("spacer_length")
        if isinstance(spacer, (list, tuple)) and spacer:
            spacer_min = int(min(spacer))
            spacer_max = int(max(spacer))
        elif spacer is not None:
            spacer_min = int(spacer)
            spacer_max = int(spacer)
        else:
            spacer_min = None
            spacer_max = None
        up_positions = _find_motif_positions(seq, upstream, pc.get("upstream_pos"))
        down_positions = _find_motif_positions(seq, downstream, pc.get("downstream_pos"))
        if not up_positions or not down_positions:
            continue
        matched = False
        for up_start in sorted(up_positions):
            up_end = up_start + len(upstream)
            for down_start in sorted(down_positions):
                if down_start < up_end:
                    continue
                spacer_len = down_start - up_end
                if spacer_min is not None and spacer_len < spacer_min:
                    continue
                if spacer_max is not None and spacer_len > spacer_max:
                    continue
                windows.append((up_start, up_end))
                windows.append((down_start, down_start + len(downstream)))
                matched = True
                break
            if matched:
                break
    return windows


def _find_forbidden_kmer(
    seq: str,
    kmers: list[str],
    allowed_windows: list[tuple[int, int]],
) -> tuple[str, int] | None:
    if not kmers:
        return None
    for kmer in kmers:
        start = 0
        while start < len(seq):
            idx = seq.find(kmer, start)
            if idx == -1:
                break
            end = idx + len(kmer)
            allowed = False
            for win_start, win_end in allowed_windows:
                if idx >= win_start and end <= win_end:
                    allowed = True
                    break
            if not allowed:
                return kmer, idx
            start = idx + 1
    return None


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
