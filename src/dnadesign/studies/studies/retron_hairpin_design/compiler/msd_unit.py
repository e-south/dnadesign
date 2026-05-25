"""
Pure Retron MSD unit compilation for downstream source promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dnadesign.contracts.sequence import MsdDesignReferenceV1

from ..catalog.sequence_inputs import validate_dna_sequence
from ..outputs.layout import DEFAULT_FLANK_3P_SUFFIX, DEFAULT_FLANK_5P_PREFIX, SNAPBACK_FOLDBACK_SEGMENT_ID
from .exceptions import RetronMsdCompilerError


@dataclass(frozen=True, slots=True)
class MsdCompiledSegment:
    role: str
    sequence: str
    start_0: int
    end_0: int

    @property
    def span_0(self) -> tuple[int, int]:
        return (self.start_0, self.end_0)


@dataclass(frozen=True, slots=True)
class MsdCompiledUnitV1:
    contract: str
    schema_version: int
    msd_design_id: str
    construct_id: str
    sequence_5to3: str
    segments: tuple[MsdCompiledSegment, ...]
    provenance: Mapping[str, object]

    def segment_sequence(self, role: str) -> str:
        return self._segment(role).sequence

    def segment_span(self, role: str) -> tuple[int, int]:
        return self._segment(role).span_0

    def _segment(self, role: str) -> MsdCompiledSegment:
        matches = [segment for segment in self.segments if segment.role == role]
        if len(matches) != 1:
            raise RetronMsdCompilerError(f"Expected exactly one MSD segment with role {role!r}; found {len(matches)}.")
        return matches[0]


def compile_msd_design_unit(
    record: MsdDesignReferenceV1,
    *,
    payload_sequences: Mapping[str, str],
    cap_sequences: Mapping[str, str],
    flank_5p_prefix: str = DEFAULT_FLANK_5P_PREFIX,
    flank_3p_suffix: str = DEFAULT_FLANK_3P_SUFFIX,
) -> MsdCompiledUnitV1:
    """Compile one selected Retron MSD design into its 5'->3' MSD product unit.

    This is deliberately pure: it does not invoke Construct, Folding,
    BaseRender, or any filesystem materialization path.
    """

    payload_id = record.payload_or_target.id
    cap_id = record.cap.id
    if payload_id not in payload_sequences:
        raise RetronMsdCompilerError(f"payload_sequences is missing required payload id: {payload_id}.")
    if cap_id not in cap_sequences:
        raise RetronMsdCompilerError(f"cap_sequences is missing required cap id: {cap_id}.")

    payload_sequence = _dna(payload_sequences[payload_id], label=f"payload_sequences.{payload_id}")
    cap_sequence = _dna(cap_sequences[cap_id], label=f"cap_sequences.{cap_id}")
    _validate_cap_topology_length(record=record, cap_sequence=cap_sequence)

    parts = (
        ("flank_5p_prefix", _dna(flank_5p_prefix, label="flank_5p_prefix")),
        ("stem_base_left", _dna(record.scar_nick.left_base, label="stem_base_left")),
        ("payload_primary", payload_sequence),
        (SNAPBACK_FOLDBACK_SEGMENT_ID, cap_sequence),
        ("payload_complement", reverse_complement(payload_sequence)),
        ("stem_base_right", _dna(record.scar_nick.right_base, label="stem_base_right")),
        ("flank_3p_suffix", _dna(flank_3p_suffix, label="flank_3p_suffix")),
    )
    offset = 0
    segments: list[MsdCompiledSegment] = []
    for role, sequence in parts:
        start = offset
        offset += len(sequence)
        segments.append(MsdCompiledSegment(role=role, sequence=sequence, start_0=start, end_0=offset))

    sequence_5to3 = "".join(segment.sequence for segment in segments)
    topology = record.cap.snapback_topology
    provenance = {
        "construct_id": record.construct_id,
        "construct_label": record.construct_label,
        "msd_design_id": record.msd_design_id,
        "payload_id": payload_id,
        "cap_id": cap_id,
        "cap_source_construct": record.cap.source_construct,
        "snapback_topology_source": topology.source if topology is not None else None,
        "scar_nick_route_status": record.scar_nick.route_status,
        "scar_nick_profile_s3s2s1s0": record.scar_nick.profile_s3s2s1s0,
        "nick_orientation": record.scar_nick.nick_orientation,
        "nickase": record.scar_nick.nickase,
        "source_notes": record.source_notes,
    }
    return MsdCompiledUnitV1(
        contract="msd_compiled_unit_v1",
        schema_version=1,
        msd_design_id=record.msd_design_id,
        construct_id=record.construct_id,
        sequence_5to3=sequence_5to3,
        segments=tuple(segments),
        provenance=provenance,
    )


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def _dna(value: str, *, label: str) -> str:
    try:
        return validate_dna_sequence(str(value), label=label).upper()
    except ValueError as exc:
        raise RetronMsdCompilerError(str(exc)) from exc


def _validate_cap_topology_length(*, record: MsdDesignReferenceV1, cap_sequence: str) -> None:
    topology = record.cap.snapback_topology
    if topology is None:
        return
    expected_length = topology.foldback_return_span.end
    if len(cap_sequence) != expected_length:
        raise RetronMsdCompilerError(
            f"MSD cap '{record.cap.id}' sequence length {len(cap_sequence)} does not match "
            f"snapback_topology length {expected_length}."
        )


__all__ = [
    "MsdCompiledSegment",
    "MsdCompiledUnitV1",
    "compile_msd_design_unit",
    "reverse_complement",
]
