"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/ship_v3.py

Canonical YIU v3 state-graph builder for the ship-ready circularized workflow.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import _segment_rows, _state, _StateSegment
from dnadesign.cruncher.bio import reverse_complement_iupac
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
from dnadesign.cruncher.yiu.models import (
    SourceOligoSpecV3,
    YiuProcessSpecV3,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)

_PAYLOAD_ASSEMBLY_STATES = {
    "circularized_payload_candidate",
    "post_exonuclease_cleanup",
    "post_fragment_cleanup",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
}
_NT_BPU10I_STATES = {
    "post_fragment_cleanup",
    "snapback_adapter_complex",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
}
_SACRIFICIAL_FRAGMENTATION_STATES = {
    "post_sacrificial_fragmentation",
    "post_fragment_cleanup",
}
_SOURCE_OWNER_ORDER = [
    "source_fwd_primer_binding_region",
    "payload_left_half",
    "sacrificial_region_long",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
    "payload_right_half",
    "source_rev_primer_binding_region",
]
_CIRCULARIZED_OWNER_ORDER = [
    "payload_left_half",
    "payload_right_half",
    "sacrificial_region_long",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
]
_POST_FRAGMENT_OWNER_ORDER = [
    "payload_left_half",
    "payload_right_half",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
]


@dataclass(frozen=True)
class _FragmentDescriptor:
    fragment_id: str
    owner_id: str
    start: int
    end: int
    sequence: str

    @property
    def length_nt(self) -> int:
        return self.end - self.start

    def as_fragment_row(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "owner_id": self.owner_id,
            "start": self.start,
            "end": self.end,
            "length_nt": self.length_nt,
            "sequence": self.sequence,
        }


def _issue(code: str, message: str, *, step_id: str | None = None, state_id: str | None = None) -> YiuValidationIssue:
    return YiuValidationIssue(code=code, message=message, step_id=step_id, state_id=state_id)


def _owner_by_id(source_oligo: SourceOligoSpecV3) -> dict[str, Any]:
    return {owner.id: owner for owner in source_oligo.structural_owners}


def _effect_by_id(source_oligo: SourceOligoSpecV3) -> dict[str, Any]:
    return {tag.id: tag for tag in source_oligo.effect_tags}


def _sequence_for_owner(source_oligo: SourceOligoSpecV3, owner_id: str) -> str:
    owner = _owner_by_id(source_oligo)[owner_id]
    return (source_oligo.sequence or "")[owner.start : owner.end]


def _sequence_for_tag(source_oligo: SourceOligoSpecV3, tag_id: str) -> str:
    tag = _effect_by_id(source_oligo)[tag_id]
    return (source_oligo.sequence or "")[tag.start : tag.end]


def _joined_sequence(spec: YiuProcessSpecV3, owner_order: list[str]) -> str:
    return "".join(_sequence_for_owner(spec.source_oligo, owner_id) for owner_id in owner_order)


def _state_segments_from_owner_order(spec: YiuProcessSpecV3, owner_order: list[str]) -> list[_StateSegment]:
    cursor = 0
    segments: list[_StateSegment] = []
    owners = _owner_by_id(spec.source_oligo)
    for owner_id in owner_order:
        owner = owners[owner_id]
        length = owner.end - owner.start
        segments.append(
            _StateSegment(
                segment_id=owner_id,
                source_start=owner.start,
                source_end=owner.end,
                state_start=cursor,
                state_end=cursor + length,
            )
        )
        cursor += length
    return segments


def _segments_for_adapter_tail(*, start: int, length: int, owner_id: str) -> list[_StateSegment]:
    return [
        _StateSegment(
            segment_id=owner_id,
            source_start=0,
            source_end=length,
            state_start=start,
            state_end=start + length,
        )
    ]


def _state_contains(item: Any, *, state_id: str) -> bool:
    lifecycle = item.state_lifecycle
    first_index = spec_state_index(lifecycle.first_state)
    last_index = (
        spec_state_index(lifecycle.last_state) if lifecycle.last_state is not None else len(_SOURCE_OWNER_ORDER) + 100
    )
    return first_index <= spec_state_index(state_id) <= last_index


def spec_state_index(state_id: str) -> int:
    state_order = (
        "source_oligo_ssdna",
        "pcr_linear_duplex",
        "type_iis_digest_linear_duplex",
        "circularized_payload_candidate",
        "post_exonuclease_cleanup",
        "post_sacrificial_fragmentation",
        "post_fragment_cleanup",
        "snapback_adapter_complex",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    )
    return state_order.index(state_id)


def _annotation_provenance(item: Any, *, row_id: str, state_id: str) -> dict[str, Any]:
    provenance = item.provenance.model_dump(mode="json")
    if row_id == "primary":
        return provenance
    return {
        "origin_state": provenance["origin_state"],
        "origin_owner": provenance["origin_owner"],
        "derivation": {
            "kind": "reverse_complement_projection",
            "from_state": state_id,
            "from_owner": item.id,
        },
    }


def _project_source_annotations(
    spec: YiuProcessSpecV3,
    *,
    state_id: str,
    segments: list[_StateSegment],
    row_id: str,
    include_tags: set[str] | None = None,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    source_oligo = spec.source_oligo
    for owner in source_oligo.structural_owners:
        if not _state_contains(owner, state_id=state_id):
            continue
        for segment in segments:
            if segment.segment_id != owner.id:
                continue
            projected.append(
                {
                    "annotation_layer": "structural_owner",
                    "id": owner.id,
                    "annotation_class": owner.id,
                    "state_id": state_id,
                    "row_id": row_id,
                    "start": segment.state_start,
                    "end": segment.state_end,
                    "provenance": _annotation_provenance(owner, row_id=row_id, state_id=state_id),
                    "state_lifecycle": owner.state_lifecycle.model_dump(mode="json"),
                }
            )
    requested_tags = include_tags
    for tag in source_oligo.effect_tags:
        if requested_tags is not None and tag.id not in requested_tags:
            continue
        if not _state_contains(tag, state_id=state_id):
            continue
        for segment in segments:
            overlap_start = max(tag.start, segment.source_start)
            overlap_end = min(tag.end, segment.source_end)
            if overlap_start >= overlap_end:
                continue
            projected_start = segment.state_start + (overlap_start - segment.source_start)
            projected_end = segment.state_start + (overlap_end - segment.source_start)
            projected.append(
                {
                    "annotation_layer": "effect_tag",
                    "id": tag.id,
                    "annotation_class": tag.class_,
                    "state_id": state_id,
                    "row_id": row_id,
                    "start": projected_start,
                    "end": projected_end,
                    "provenance": tag.provenance.model_dump(mode="json"),
                    "state_lifecycle": tag.state_lifecycle.model_dump(mode="json"),
                }
            )
    projected.sort(key=lambda item: (item["row_id"], int(item["start"]), item["annotation_layer"], item["id"]))
    return projected


def _late_owner_annotation(
    *,
    state_id: str,
    owner_id: str,
    row_id: str,
    start: int,
    end: int,
    first_state: str = "snapback_adapter_complex",
) -> dict[str, Any]:
    return {
        "annotation_layer": "structural_owner",
        "id": owner_id,
        "annotation_class": owner_id,
        "state_id": state_id,
        "row_id": row_id,
        "start": start,
        "end": end,
        "provenance": {
            "origin_state": "introduced_late",
            "origin_owner": None,
            "derivation": {
                "kind": "late_introduction",
                "from_state": None,
                "from_owner": None,
            },
        },
        "state_lifecycle": {
            "first_state": first_state,
            "last_state": None,
            "disposition": "introduced",
        },
    }


def _late_tag_annotation(
    *,
    state_id: str,
    tag_id: str,
    annotation_class: str,
    row_id: str,
    start: int,
    end: int,
    first_state: str = "snapback_adapter_complex",
) -> dict[str, Any]:
    return {
        "annotation_layer": "effect_tag",
        "id": tag_id,
        "annotation_class": annotation_class,
        "state_id": state_id,
        "row_id": row_id,
        "start": start,
        "end": end,
        "provenance": {
            "origin_state": "introduced_late",
            "origin_owner": None,
            "derivation": {
                "kind": "late_introduction",
                "from_state": None,
                "from_owner": None,
            },
        },
        "state_lifecycle": {
            "first_state": first_state,
            "last_state": None,
            "disposition": "introduced",
        },
    }


def _build_payload_invariant(spec: YiuProcessSpecV3) -> dict[str, Any]:
    assembled = _sequence_for_owner(spec.source_oligo, "payload_left_half") + _sequence_for_owner(
        spec.source_oligo, "payload_right_half"
    )
    return {
        "id": "payload_assembly",
        "class": "payload_assembly",
        "status": "guaranteed" if assembled == spec.payload_goal.assembled_payload_sequence else "impossible",
        "observed": {
            "assembled_payload_sequence": assembled,
            "expected_payload_sequence": spec.payload_goal.assembled_payload_sequence,
            "payload_overhang_geometry": spec.payload_goal.payload_overhang_geometry.model_dump(mode="json"),
        },
    }


def _build_nt_bpu10i_invariant(spec: YiuProcessSpecV3) -> dict[str, Any]:
    invariant = next(item for item in spec.hard_invariants if item.id == "nt_bpu10i_snapback_site")
    local_context = _sequence_for_tag(spec.source_oligo, "nt_bpu10i_snapback_site")
    complement_side = reverse_complement_iupac(local_context)
    produced_geometry = {
        "tether_dock": complement_side[0:4],
        "tether_cap": complement_side[4:8],
        "tether_dock_complement": complement_side[8:12],
        "snapback_stem": complement_side[12:14],
    }
    return {
        "id": invariant.id,
        "class": invariant.class_,
        "status": "guaranteed" if local_context == str(invariant.params["local_context_sequence"]) else "impossible",
        "observed": {
            "local_context_sequence": local_context,
            "recognized_sequence": invariant.params["recognized_sequence"],
            "nicked_strand": invariant.params["nicked_strand"],
            "nick_offset": invariant.params["nick_offset"],
            "nick_position": 33,
            "produced_geometry": produced_geometry,
        },
    }


def _sacrificial_fragment_descriptors(spec: YiuProcessSpecV3) -> list[_FragmentDescriptor]:
    circularized_segments = _state_segments_from_owner_order(spec, _CIRCULARIZED_OWNER_ORDER)
    sacrificial_segment = next(
        segment for segment in circularized_segments if segment.segment_id == "sacrificial_region_long"
    )
    sequence = _sequence_for_owner(spec.source_oligo, "sacrificial_region_long")
    return [
        _FragmentDescriptor(
            fragment_id="sacrificial_fragment_1",
            owner_id="sacrificial_region_long",
            start=sacrificial_segment.state_start,
            end=sacrificial_segment.state_end,
            sequence=sequence,
        )
    ]


def _retained_fragment_descriptor(spec: YiuProcessSpecV3) -> _FragmentDescriptor:
    retained_sequence = _joined_sequence(spec, _POST_FRAGMENT_OWNER_ORDER)
    return _FragmentDescriptor(
        fragment_id="retained_product_1",
        owner_id="retained_region",
        start=0,
        end=len(retained_sequence),
        sequence=retained_sequence,
    )


def _build_fragmentation_invariant(spec: YiuProcessSpecV3) -> tuple[dict[str, Any], list[_FragmentDescriptor]]:
    invariant = next(item for item in spec.hard_invariants if item.id == "sacrificial_fragmentation")
    fragments = _sacrificial_fragment_descriptors(spec)
    fragment_lengths = [fragment.length_nt for fragment in fragments]
    status = (
        "guaranteed"
        if fragment_lengths and max(fragment_lengths) <= int(invariant.params["max_fragment_nt"])
        else "impossible"
    )
    return (
        {
            "id": invariant.id,
            "class": invariant.class_,
            "status": status,
            "observed": {
                "max_fragment_nt": invariant.params["max_fragment_nt"],
                "threshold_mode": invariant.params["threshold_mode"],
                "require_retained_survival": invariant.params["require_retained_survival"],
                "allow_single_payload_adjacent_retained_nt": invariant.params[
                    "allow_single_payload_adjacent_retained_nt"
                ],
                "fragment_lengths": fragment_lengths,
                "fragments": [fragment.as_fragment_row() for fragment in fragments],
            },
        },
        fragments,
    )


def _status_from_issues(issues: list[YiuValidationIssue]) -> str:
    return "unsatisfied" if issues else "satisfied"


def _validate_spec(spec: YiuProcessSpecV3, *, catalogs: LoadedYiuCatalogs) -> list[YiuValidationIssue]:
    issues: list[YiuValidationIssue] = []
    required_parts = {
        "source_pcr": [
            ("source forward primer part is missing", spec.steps.source_pcr.forward_primer_id),
            ("source reverse primer part is missing", spec.steps.source_pcr.reverse_primer_id),
        ],
        "hairpin_pcr": [
            ("hairpin PCR forward primer part is missing", spec.steps.hairpin_pcr.forward_primer_id),
            ("hairpin PCR reverse primer part is missing", spec.steps.hairpin_pcr.reverse_primer_id),
        ],
        "snapback_adapter_engagement": [
            ("Y adapter part is missing", spec.steps.snapback_adapter_engagement.adapter_id)
        ],
    }
    for step_id, entries in required_parts.items():
        for message, part_id in entries:
            if part_id not in catalogs.oligo_parts:
                issues.append(_issue("YIU_EXTERNAL_PART_MISSING", message, step_id=step_id))
    if spec.steps.hairpin_ligation.require_5p_phosphate:
        adapter = catalogs.oligo_parts.get(spec.steps.snapback_adapter_engagement.adapter_id)
        if adapter is None or not adapter.phosphorylated_5p:
            issues.append(
                _issue(
                    "YIU_ADAPTER_5P_PHOSPHATE_REQUIRED",
                    "hairpin ligation requires a 5' phosphorylated Y adapter part",
                    step_id="hairpin_ligation",
                )
            )
    left = _sequence_for_tag(spec.source_oligo, spec.steps.circularization.left_overhang_ref)
    right = _sequence_for_tag(spec.source_oligo, spec.steps.circularization.right_overhang_ref)
    if left != reverse_complement_iupac(right):
        issues.append(
            _issue(
                "YIU_CIRCULARIZATION_OVERHANG_MISMATCH",
                "payload overhangs do not satisfy exact-complement circularization",
                step_id="circularization",
            )
        )
    payload_invariant = _build_payload_invariant(spec)
    if payload_invariant["status"] != "guaranteed":
        issues.append(
            _issue(
                "YIU_PAYLOAD_ASSEMBLY_MISMATCH",
                "payload halves do not assemble to the declared assembled_payload_sequence",
                step_id="circularization",
            )
        )
    nt_bpu10i_invariant = _build_nt_bpu10i_invariant(spec)
    if nt_bpu10i_invariant["status"] != "guaranteed":
        issues.append(
            _issue(
                "YIU_NT_BPU10I_LOCAL_CONTEXT_MISMATCH",
                "Nt.Bpu10I local context does not match the declared exact invariant",
                step_id="sacrificial_digest",
            )
        )
    return issues


def _state_hard_invariants(
    *,
    state_id: str,
    payload_invariant: dict[str, Any],
    nt_bpu10i_invariant: dict[str, Any],
    fragmentation_invariant: dict[str, Any],
) -> list[dict[str, Any]]:
    invariants: list[dict[str, Any]] = []
    if state_id in _PAYLOAD_ASSEMBLY_STATES:
        invariants.append(payload_invariant)
    if state_id in _NT_BPU10I_STATES:
        invariants.append(nt_bpu10i_invariant)
    if state_id in _SACRIFICIAL_FRAGMENTATION_STATES:
        invariants.append(fragmentation_invariant)
    return invariants


def _build_yiu_report_v3(
    spec: YiuProcessSpecV3,
    *,
    catalogs: LoadedYiuCatalogs | None = None,
) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs()
    issues = _validate_spec(spec, catalogs=catalogs)
    status = _status_from_issues(issues)
    payload_invariant = _build_payload_invariant(spec)
    nt_bpu10i_invariant = _build_nt_bpu10i_invariant(spec)
    fragmentation_invariant, sacrificial_fragments = _build_fragmentation_invariant(spec)
    retained_fragment = _retained_fragment_descriptor(spec)

    source_sequence = spec.source_oligo.sequence or ""
    source_segments = _state_segments_from_owner_order(spec, _SOURCE_OWNER_ORDER)
    circularized_segments = _state_segments_from_owner_order(spec, _CIRCULARIZED_OWNER_ORDER)
    post_fragment_segments = _state_segments_from_owner_order(spec, _POST_FRAGMENT_OWNER_ORDER)
    circularized_primary = _joined_sequence(spec, _CIRCULARIZED_OWNER_ORDER)
    post_fragment_primary = _joined_sequence(spec, _POST_FRAGMENT_OWNER_ORDER)

    adapter_sequence = catalogs.oligo_parts[spec.steps.snapback_adapter_engagement.adapter_id].sequence
    adapter_complementary_length = 4
    hairpin_complex_primary = post_fragment_primary + adapter_sequence
    ligated_hairpin_primary = hairpin_complex_primary
    hairpin_insert_primary = post_fragment_primary + reverse_complement_iupac(post_fragment_primary)
    hairpin_insert_complement = reverse_complement_iupac(hairpin_insert_primary)

    hairpin_insert_primary_segments = _state_segments_from_owner_order(spec, _POST_FRAGMENT_OWNER_ORDER)
    tail_segments = _state_segments_from_owner_order(spec, list(reversed(_POST_FRAGMENT_OWNER_ORDER)))
    tail_offset = len(post_fragment_primary)
    hairpin_insert_primary_segments.extend(
        [
            _StateSegment(
                segment_id=segment.segment_id,
                source_start=segment.source_start,
                source_end=segment.source_end,
                state_start=tail_offset + segment.state_start,
                state_end=tail_offset + segment.state_end,
            )
            for segment in tail_segments
        ]
    )

    states: list[YiuStateRecord] = []

    def _metadata_for(state_id: str, *, fragments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        fragment_rows = fragments or []
        return {
            "hard_invariants": _state_hard_invariants(
                state_id=state_id,
                payload_invariant=payload_invariant,
                nt_bpu10i_invariant=nt_bpu10i_invariant,
                fragmentation_invariant=fragmentation_invariant,
            ),
            "fragment_lengths": [int(fragment["length_nt"]) for fragment in fragment_rows],
        }

    # source_oligo_ssdna
    states.append(
        _state(
            state_id="source_oligo_ssdna",
            step_id="source_oligo",
            kind="source_oligo_ssdna",
            state_kind="source_oligo_ssdna",
            topology_kind="linear_ssdna",
            status=status,
            primary_sequence=source_sequence,
            metadata=_metadata_for("source_oligo_ssdna"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(source_segments),
            annotations=_project_source_annotations(
                spec, state_id="source_oligo_ssdna", segments=source_segments, row_id="primary"
            ),
            pattern_label="pattern",
        )
    )

    pcr_annotations = _project_source_annotations(
        spec, state_id="pcr_linear_duplex", segments=source_segments, row_id="primary"
    )
    pcr_annotations.extend(
        _project_source_annotations(spec, state_id="pcr_linear_duplex", segments=source_segments, row_id="complement")
    )
    states.append(
        _state(
            state_id="pcr_linear_duplex",
            step_id="source_pcr",
            kind="pcr_linear_duplex",
            state_kind="pcr_linear_duplex",
            topology_kind="linear_dsdna",
            status=status,
            primary_sequence=source_sequence,
            complement_sequence=reverse_complement_iupac(source_sequence),
            metadata=_metadata_for("pcr_linear_duplex"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(source_segments),
            annotations=pcr_annotations,
            pattern_label="pattern",
        )
    )

    digest_annotations = _project_source_annotations(
        spec,
        state_id="type_iis_digest_linear_duplex",
        segments=source_segments,
        row_id="primary",
    )
    digest_annotations.extend(
        _project_source_annotations(
            spec,
            state_id="type_iis_digest_linear_duplex",
            segments=source_segments,
            row_id="complement",
        )
    )
    states.append(
        _state(
            state_id="type_iis_digest_linear_duplex",
            step_id="type_iis_digest",
            kind="type_iis_digest_linear_duplex",
            state_kind="type_iis_digest_linear_duplex",
            topology_kind="linear_dsdna",
            status=status,
            primary_sequence=source_sequence,
            complement_sequence=reverse_complement_iupac(source_sequence),
            metadata=_metadata_for("type_iis_digest_linear_duplex"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(source_segments),
            annotations=digest_annotations,
            cuts=[
                {
                    "id": spec.steps.type_iis_digest.left_site_ref,
                    "orientation": spec.steps.type_iis_digest.left_orientation,
                    "top_cut_offset": spec.steps.type_iis_digest.top_cut_offset,
                    "bottom_cut_offset": spec.steps.type_iis_digest.bottom_cut_offset,
                },
                {
                    "id": spec.steps.type_iis_digest.right_site_ref,
                    "orientation": spec.steps.type_iis_digest.right_orientation,
                    "top_cut_offset": spec.steps.type_iis_digest.top_cut_offset,
                    "bottom_cut_offset": spec.steps.type_iis_digest.bottom_cut_offset,
                },
            ],
            pattern_label="pattern",
        )
    )

    circularized_annotations = _project_source_annotations(
        spec,
        state_id="circularized_payload_candidate",
        segments=circularized_segments,
        row_id="primary",
    )
    circularized_annotations.extend(
        _project_source_annotations(
            spec,
            state_id="circularized_payload_candidate",
            segments=circularized_segments,
            row_id="complement",
        )
    )
    states.append(
        _state(
            state_id="circularized_payload_candidate",
            step_id="circularization",
            kind="circularized_payload_candidate",
            state_kind="circularized_payload_candidate",
            topology_kind="circular_dsdna_candidate",
            status=status,
            primary_sequence=circularized_primary,
            complement_sequence=reverse_complement_iupac(circularized_primary),
            metadata=_metadata_for("circularized_payload_candidate"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            annotations=circularized_annotations,
            junctions=[
                {"id": "circularized_payload_junction", "payload_goal": spec.payload_goal.model_dump(mode="json")}
            ],
            pattern_label="pattern",
        )
    )

    post_exonuclease_annotations = _project_source_annotations(
        spec,
        state_id="post_exonuclease_cleanup",
        segments=circularized_segments,
        row_id="primary",
    )
    post_exonuclease_annotations.extend(
        _project_source_annotations(
            spec,
            state_id="post_exonuclease_cleanup",
            segments=circularized_segments,
            row_id="complement",
        )
    )
    states.append(
        _state(
            state_id="post_exonuclease_cleanup",
            step_id="exonuclease_cleanup",
            kind="post_exonuclease_cleanup",
            state_kind="post_exonuclease_cleanup",
            topology_kind="circular_dsdna_candidate",
            status=status,
            primary_sequence=circularized_primary,
            complement_sequence=reverse_complement_iupac(circularized_primary),
            metadata=_metadata_for("post_exonuclease_cleanup"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            annotations=post_exonuclease_annotations,
            pattern_label="pattern",
        )
    )

    sacrificial_fragment_rows = [fragment.as_fragment_row() for fragment in sacrificial_fragments]
    states.append(
        _state(
            state_id="post_sacrificial_fragmentation",
            step_id="sacrificial_digest",
            kind="post_sacrificial_fragmentation",
            state_kind="post_sacrificial_fragmentation",
            topology_kind="fragment_pool",
            status=status,
            primary_sequence=circularized_primary,
            metadata=_metadata_for("post_sacrificial_fragmentation", fragments=sacrificial_fragment_rows),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            annotations=_project_source_annotations(
                spec,
                state_id="post_sacrificial_fragmentation",
                segments=circularized_segments,
                row_id="primary",
            ),
            fragments=sacrificial_fragment_rows,
            pattern_label="pattern",
        )
    )

    post_fragment_annotations = _project_source_annotations(
        spec,
        state_id="post_fragment_cleanup",
        segments=post_fragment_segments,
        row_id="primary",
    )
    states.append(
        _state(
            state_id="post_fragment_cleanup",
            step_id="fragment_cleanup",
            kind="post_fragment_cleanup",
            state_kind="post_fragment_cleanup",
            topology_kind="linear_ssdna",
            status=status,
            primary_sequence=post_fragment_primary,
            metadata=_metadata_for("post_fragment_cleanup", fragments=[retained_fragment.as_fragment_row()]),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(post_fragment_segments),
            annotations=post_fragment_annotations,
            fragments=[retained_fragment.as_fragment_row()],
            pattern_label="pattern",
        )
    )

    adapter_owner_start = len(post_fragment_primary)
    adapter_annotations = _project_source_annotations(
        spec,
        state_id="snapback_adapter_complex",
        segments=post_fragment_segments,
        row_id="primary",
    )
    adapter_annotations.extend(
        [
            _late_owner_annotation(
                state_id="snapback_adapter_complex",
                owner_id="y_adapter_complementary_arm",
                row_id="primary",
                start=adapter_owner_start,
                end=adapter_owner_start + adapter_complementary_length,
            ),
            _late_owner_annotation(
                state_id="snapback_adapter_complex",
                owner_id="y_adapter_noncomplementary_arm",
                row_id="primary",
                start=adapter_owner_start + adapter_complementary_length,
                end=len(hairpin_complex_primary),
            ),
            _late_tag_annotation(
                state_id="snapback_adapter_complex",
                tag_id="introduced_late::y_adapter",
                annotation_class="introduced_late",
                row_id="primary",
                start=adapter_owner_start,
                end=len(hairpin_complex_primary),
            ),
            _late_tag_annotation(
                state_id="snapback_adapter_complex",
                tag_id="y_adapter_binding::complementary_arm",
                annotation_class="y_adapter_binding",
                row_id="primary",
                start=adapter_owner_start,
                end=adapter_owner_start + adapter_complementary_length,
            ),
        ]
    )
    states.append(
        _state(
            state_id="snapback_adapter_complex",
            step_id="snapback_adapter_engagement",
            kind="snapback_adapter_complex",
            state_kind="snapback_adapter_complex",
            topology_kind="branched_y",
            status=status,
            primary_sequence=hairpin_complex_primary,
            metadata=_metadata_for("snapback_adapter_complex"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(
                post_fragment_segments
                + _segments_for_adapter_tail(
                    start=adapter_owner_start, length=len(adapter_sequence), owner_id="y_adapter_noncomplementary_arm"
                )
            ),
            annotations=adapter_annotations,
            pattern_label="pattern",
        )
    )

    ligated_annotations = [annotation | {"state_id": "ligated_ssdna_hairpin"} for annotation in adapter_annotations]
    states.append(
        _state(
            state_id="ligated_ssdna_hairpin",
            step_id="hairpin_ligation",
            kind="ligated_ssdna_hairpin",
            state_kind="ligated_ssdna_hairpin",
            topology_kind="hairpin_ssdna",
            status=status,
            primary_sequence=ligated_hairpin_primary,
            metadata=_metadata_for("ligated_ssdna_hairpin"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(
                post_fragment_segments
                + _segments_for_adapter_tail(
                    start=adapter_owner_start, length=len(adapter_sequence), owner_id="y_adapter_noncomplementary_arm"
                )
            ),
            annotations=ligated_annotations,
            pattern_label="pattern",
        )
    )

    hairpin_insert_annotations = _project_source_annotations(
        spec,
        state_id="hairpin_pcr_linear_insert",
        segments=hairpin_insert_primary_segments,
        row_id="primary",
    )
    hairpin_insert_annotations.extend(
        _project_source_annotations(
            spec,
            state_id="hairpin_pcr_linear_insert",
            segments=hairpin_insert_primary_segments,
            row_id="complement",
        )
    )
    states.append(
        _state(
            state_id="hairpin_pcr_linear_insert",
            step_id="hairpin_pcr",
            kind="hairpin_pcr_linear_insert",
            state_kind="hairpin_pcr_linear_insert",
            topology_kind="linear_dsdna",
            status=status,
            primary_sequence=hairpin_insert_primary,
            complement_sequence=hairpin_insert_complement,
            metadata=_metadata_for("hairpin_pcr_linear_insert"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(hairpin_insert_primary_segments),
            annotations=hairpin_insert_annotations,
            pattern_label="pattern",
        )
    )

    metadata = YiuReportMetadata(
        spec_schema_version=spec.schema_version,
        step_count=len(states),
        state_count=len(states),
        emitted_view_count=0,
        view_contract_version=spec.output.publish_contract_version,
        catalog_paths=[str(path) for path in catalogs.paths],
    )
    return YiuValidationReport(
        protocol="yiu_v3",
        protocol_template=spec.protocol_template,
        spec_name=spec.name,
        status=status,
        metadata=metadata,
        states=states,
        issues=issues,
    )
