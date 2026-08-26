"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/dense_arrays/playback.py

Translate persisted DenseGen records into public playback contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

from dense_arrays.playback import PlaybackPlan, reconstruct_playback
from dense_arrays.realized import (
    DeclaredConstraint,
    Orientation,
    Placement,
    PlacementKind,
    RealizedArray,
)

_DNA_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return text


def _details(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = record.get("densegen__used_tfbs_detail")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        msg = "densegen__used_tfbs_detail must be a sequence or JSON array"
        raise TypeError(msg)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"densegen__used_tfbs_detail[{index}] must be an object"
            raise TypeError(msg)
        result.append(item)
    if not result:
        msg = "densegen__used_tfbs_detail must contain at least one placement"
        raise ValueError(msg)
    return result


def _placement_kind(value: object) -> PlacementKind:
    text = str(value or PlacementKind.TFBS.value).strip().casefold()
    if text == PlacementKind.TFBS.value:
        return PlacementKind.TFBS
    if text == PlacementKind.FIXED_ELEMENT.value:
        return PlacementKind.FIXED_ELEMENT
    return PlacementKind.OTHER


def _orientation(value: object) -> Orientation:
    text = str(value or "").strip().casefold()
    if text == Orientation.FORWARD.value:
        return Orientation.FORWARD
    if text == Orientation.REVERSE.value:
        return Orientation.REVERSE
    return Orientation.UNSPECIFIED


def _oriented_sequence(sequence: str, orientation: Orientation) -> str:
    normalized = sequence.upper()
    if orientation is Orientation.REVERSE:
        return normalized.translate(_DNA_COMPLEMENT)[::-1]
    return normalized


def _feature_id(detail: Mapping[str, object], *, index: int) -> str:
    for key in ("tfbs_id", "site_id", "motif_id"):
        value = detail.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    constraint_name = detail.get("constraint_name")
    if constraint_name is not None and str(constraint_name).strip():
        parts = (
            str(constraint_name).strip(),
            str(detail.get("placement_index") or 0),
            str(detail.get("role") or "element"),
            str(detail.get("variant_id") or "default"),
        )
        return ":".join(parts)
    return f"densegen-feature:{index}"


def _metadata(detail: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "offset_raw",
        "pad_left",
        "part_index",
        "placement_index",
        "constraint_name",
        "role",
        "variant_id",
        "regulator",
        "tfbs_id",
        "site_id",
        "motif_id",
        "spacer_length",
    )
    return {key: detail[key] for key in keys if detail.get(key) is not None}


def _start(
    detail: Mapping[str, object],
    *,
    realized_sequence: str,
    placement_sequence: str,
) -> tuple[int, str]:
    candidates: list[tuple[str, int]] = []
    offset = detail.get("offset")
    raw = detail.get("offset_raw")
    pad_left = int(detail.get("pad_left") or 0)
    if offset is not None:
        candidates.append(("offset", int(offset)))
    if raw is not None:
        candidates.append(("offset_raw_plus_pad", int(raw) + pad_left))
        candidates.append(("offset_raw", int(raw)))
    if not candidates:
        msg = "each DenseGen placement requires offset or offset_raw"
        raise ValueError(msg)
    seen: set[int] = set()
    for source, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate < 0:
            continue
        observed = realized_sequence[candidate : candidate + len(placement_sequence)]
        if observed == placement_sequence:
            return candidate, source
    described = {source: candidate for source, candidate in candidates}
    msg = f"no persisted DenseGen coordinate places the feature sequence on the realized sequence: {described}"
    raise ValueError(msg)


def realized_array_from_densegen_record(
    record: Mapping[str, object],
    *,
    source_ref: str,
    source_sha256: str | None = None,
) -> RealizedArray:
    """Translate one DenseGen output row without inferring solver-selected edges."""
    record_id = _required_text(record.get("id"), field_name="record.id")
    sequence = _required_text(record.get("sequence"), field_name="record.sequence")
    source_ref = _required_text(source_ref, field_name="source_ref")
    detail_rows = _details(record)
    placements: list[Placement] = []
    detail_by_placement_id: dict[str, Mapping[str, object]] = {}
    for index, detail in enumerate(detail_rows):
        feature_id = _feature_id(detail, index=index)
        placement_id = f"{record_id}:{index}:{feature_id}"
        library_sequence = _required_text(
            detail.get("sequence"),
            field_name=f"densegen__used_tfbs_detail[{index}].sequence",
        )
        orientation = _orientation(detail.get("orientation"))
        placement_sequence = _oriented_sequence(library_sequence, orientation)
        declared_end = detail.get("end")
        start, coordinate_source = _start(
            detail,
            realized_sequence=sequence,
            placement_sequence=placement_sequence,
        )
        if (
            coordinate_source == "offset"
            and declared_end is not None
            and int(declared_end) != start + len(placement_sequence)
        ):
            msg = f"DenseGen placement {index} has inconsistent offset, length, and end"
            raise ValueError(msg)
        label_value = detail.get("regulator") or detail.get("role")
        metadata = _metadata(detail)
        metadata["library_sequence"] = library_sequence
        metadata["coordinate_source"] = coordinate_source
        if declared_end is not None:
            metadata["declared_end"] = int(declared_end)
        placement = Placement(
            placement_id=placement_id,
            feature_id=feature_id,
            kind=_placement_kind(detail.get("part_kind")),
            sequence=placement_sequence,
            start=start,
            orientation=orientation,
            label=None if label_value is None else str(label_value),
            metadata=metadata,
        )
        placements.append(placement)
        detail_by_placement_id[placement_id] = detail

    fixed_groups: dict[tuple[str, int], list[Placement]] = defaultdict(list)
    for placement in placements:
        detail = detail_by_placement_id[placement.placement_id]
        constraint_name = detail.get("constraint_name")
        if placement.kind is not PlacementKind.FIXED_ELEMENT or constraint_name is None:
            continue
        group_key = (str(constraint_name), int(detail.get("placement_index") or 0))
        fixed_groups[group_key].append(placement)

    constraints: list[DeclaredConstraint] = []
    for (constraint_name, placement_index), group in sorted(fixed_groups.items()):
        by_role = {str(detail_by_placement_id[item.placement_id].get("role")): item for item in group}
        if "upstream" not in by_role or "downstream" not in by_role:
            continue
        spacer_values = {
            int(detail_by_placement_id[item.placement_id]["spacer_length"])
            for item in group
            if detail_by_placement_id[item.placement_id].get("spacer_length") is not None
        }
        if len(spacer_values) != 1:
            msg = f"fixed-element group {constraint_name!r}/{placement_index} must declare one shared spacer_length"
            raise ValueError(msg)
        spacer = spacer_values.pop()
        constraints.append(
            DeclaredConstraint(
                constraint_id=f"{constraint_name}:{placement_index}",
                upstream_placement_id=by_role["upstream"].placement_id,
                downstream_placement_id=by_role["downstream"].placement_id,
                min_distance_bp=spacer,
                max_distance_bp=spacer,
                label=constraint_name,
                metadata={"placement_index": placement_index},
            )
        )

    provenance_keys = (
        "densegen__schema_version",
        "densegen__run_id",
        "densegen__plan",
        "densegen__input_name",
        "densegen__sampling_library_hash",
        "densegen__sampling_library_index",
        "densegen__pad_used",
        "densegen__pad_bases",
        "densegen__pad_end",
    )
    provenance = {key: record[key] for key in provenance_keys if record.get(key) is not None}
    provenance["producer"] = "dnadesign.densegen"
    provenance["source_ref"] = source_ref
    return RealizedArray(
        source_id=f"{source_ref}#{record_id}",
        source_digest=source_sha256,
        sequence=sequence,
        placements=tuple(placements),
        constraints=tuple(constraints),
        coordinate_space="realized_sequence",
        provenance=provenance,
    )


def playback_plan_from_densegen_record(
    record: Mapping[str, object],
    *,
    source_ref: str,
    source_sha256: str | None = None,
) -> PlaybackPlan:
    """Translate and compile one persisted DenseGen record for playback."""
    return reconstruct_playback(
        realized_array_from_densegen_record(
            record,
            source_ref=source_ref,
            source_sha256=source_sha256,
        )
    )
