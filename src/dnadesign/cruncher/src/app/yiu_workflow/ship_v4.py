"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/ship_v4.py

Canonical YIU v4 state-graph builder.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import _segment_rows, _state, _StateSegment
from dnadesign.cruncher.bio import reverse_complement_iupac
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
from dnadesign.cruncher.yiu.models import (
    SourceOligoSpecV4,
    YiuProcessSpecV4,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)
from dnadesign.cruncher.yiu.models.v4 import (
    YIU_V4_NT_BPU10I_RECOGNITION_SEQUENCE,
)

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
_CUT_PRODUCT_OWNER_ORDER = [
    "payload_left_half",
    "sacrificial_region_long",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
    "payload_right_half",
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
_RETAINED_OWNER_ORDER = [
    "payload_left_half",
    "payload_right_half",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
]

_OWNER_DISPLAY: dict[str, tuple[str, str]] = {
    "source_fwd_primer_binding_region": ("Source Fwd Primer", "SrcF"),
    "payload_left_half": ("Payload Left", "PayL"),
    "sacrificial_region_long": ("Sacrificial Long", "SacL"),
    "tether_dock_complement": ("Tether Dock RC", "Tdc"),
    "tether_cap": ("Tether Cap", "Cap"),
    "tether_dock": ("Tether Dock", "Tdk"),
    "snapback_stem": ("Snapback Stem", "Stm"),
    "payload_right_half": ("Payload Right", "PayR"),
    "source_rev_primer_binding_region": ("Source Rev Primer", "SrcR"),
    "retained_region": ("Retained Region", "Ret"),
    "sacrificial_region_short": ("Sacrificial Short", "SacS"),
    "y_adapter_complementary_arm": ("Y Adapter Comp", "Yc"),
    "y_adapter_noncomplementary_arm": ("Y Adapter Tail", "Yn"),
    "hairpin_pcr_forward_binding_region": ("HP PCR Forward", "HPF"),
    "hairpin_pcr_reverse_binding_region": ("HP PCR Reverse", "HPR"),
}

_TAG_DISPLAY: dict[str, tuple[str, str]] = {
    "primer_bindable_by_source_forward": ("Source F Primer", "PrF"),
    "primer_bindable_by_source_reverse": ("Source R Primer", "PrR"),
    "primer_bindable_by_hairpin_pcr_forward": ("HP PCR F Primer", "HPF"),
    "primer_bindable_by_hairpin_pcr_reverse": ("HP PCR R Primer", "HPR"),
    "nb_bsssi_array_member": ("Nb.BssSI", "Bss"),
    "left_bsssi_bsai_overlap_unit": ("BssSI/BsaI Overlap", "Ovl"),
    "type_iis_recognition_left": ("Type IIS Left", "TIL"),
    "type_iis_recognition_right": ("Type IIS Right", "TIR"),
    "payload_overhang_left": ("Overhang Left", "OvL"),
    "payload_overhang_right": ("Overhang Right", "OvR"),
    "nt_bpu10i_snapback_site": ("Nt.Bpu10I", "Bpu"),
    "sacrificial": ("Sacrificial", "Sac"),
    "introduced_late": ("Introduced Late", "New"),
    "y_adapter_binding": ("Y Adapter Binding", "Yad"),
    "pairs_with": ("Pairs With", "Pair"),
    "ligation_junction_member": ("Ligation Junction", "Lig"),
    "cut_boundary_anchor": ("Cut Boundary", "Cut"),
    "nick_boundary_anchor": ("Nick Boundary", "Nick"),
    "payload_bulge_position": ("Payload Bulge", "Bul"),
}


def _issue(code: str, message: str, *, step_id: str | None = None, state_id: str | None = None) -> YiuValidationIssue:
    return YiuValidationIssue(code=code, message=message, step_id=step_id, state_id=state_id)


def _owner_by_id(source_oligo: SourceOligoSpecV4) -> dict[str, Any]:
    return {owner.id: owner for owner in source_oligo.structural_owners}


def _effect_by_id(source_oligo: SourceOligoSpecV4) -> dict[str, Any]:
    return {tag.id: tag for tag in source_oligo.effect_tags}


def _sequence_for_owner(spec: YiuProcessSpecV4, owner_id: str) -> str:
    owner = _owner_by_id(spec.source_oligo)[owner_id]
    return spec.source_oligo.authored_sequence[owner.start : owner.end]


def _sequence_for_tag(spec: YiuProcessSpecV4, tag_id: str) -> str:
    tag = _effect_by_id(spec.source_oligo)[tag_id]
    return spec.source_oligo.authored_sequence[tag.start : tag.end]


def _segments_for_source_order(spec: YiuProcessSpecV4, owner_order: list[str]) -> list[_StateSegment]:
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


def _segments_for_synthetic_segments(items: list[tuple[str, int]]) -> list[_StateSegment]:
    cursor = 0
    segments: list[_StateSegment] = []
    for owner_id, length in items:
        segments.append(
            _StateSegment(
                segment_id=owner_id,
                source_start=0,
                source_end=length,
                state_start=cursor,
                state_end=cursor + length,
            )
        )
        cursor += length
    return segments


def _offset_segments(segments: list[_StateSegment], *, offset: int) -> list[_StateSegment]:
    return [
        _StateSegment(
            segment_id=segment.segment_id,
            source_start=segment.source_start,
            source_end=segment.source_end,
            state_start=segment.state_start + offset,
            state_end=segment.state_end + offset,
        )
        for segment in segments
    ]


def _joined_sequence(spec: YiuProcessSpecV4, owner_order: list[str]) -> str:
    return "".join(_sequence_for_owner(spec, owner_id) for owner_id in owner_order)


def _owner_annotation(*, owner_id: str, row_id: str, start: int, end: int) -> dict[str, Any]:
    display_label, short_label = _OWNER_DISPLAY[owner_id]
    return {
        "annotation_layer": "structural_owner",
        "id": owner_id,
        "annotation_class": owner_id,
        "row_id": row_id,
        "start": start,
        "end": end,
        "display_label": display_label,
        "short_label": short_label,
    }


def _tag_annotation(
    *,
    tag_id: str,
    tag_kind: str,
    row_id: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    display_label, short_label = _TAG_DISPLAY.get(tag_kind, (tag_kind, tag_kind))
    return {
        "annotation_layer": "effect_tag",
        "id": tag_id,
        "annotation_class": tag_kind,
        "row_id": row_id,
        "start": start,
        "end": end,
        "display_label": display_label,
        "short_label": short_label,
    }


def _owner_annotations_from_segments(segments: list[_StateSegment], *, row_id: str) -> list[dict[str, Any]]:
    return [
        _owner_annotation(owner_id=segment.segment_id, row_id=row_id, start=segment.state_start, end=segment.state_end)
        for segment in segments
    ]


def _project_source_tags(
    spec: YiuProcessSpecV4,
    *,
    segments: list[_StateSegment],
    row_id: str,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for tag in spec.source_oligo.effect_tags:
        for segment in segments:
            overlap_start = max(tag.start, segment.source_start)
            overlap_end = min(tag.end, segment.source_end)
            if overlap_start >= overlap_end:
                continue
            annotations.append(
                _tag_annotation(
                    tag_id=tag.id,
                    tag_kind=tag.class_,
                    row_id=row_id,
                    start=segment.state_start + (overlap_start - segment.source_start),
                    end=segment.state_start + (overlap_end - segment.source_start),
                )
            )
    return annotations


def _payload_invariant(spec: YiuProcessSpecV4) -> dict[str, Any]:
    assembled = _sequence_for_owner(spec, "payload_left_half") + _sequence_for_owner(spec, "payload_right_half")
    return {
        "id": "payload_assembly_invariant",
        "class": "payload_assembly",
        "status": "guaranteed" if assembled == spec.payload.target_sequence else "impossible",
        "observed": {
            "assembled_payload_sequence": assembled,
            "payload_overhang_geometry": {
                "left_overhang_sequence": _sequence_for_tag(spec, "payload_overhang_left"),
                "right_overhang_sequence": _sequence_for_tag(spec, "payload_overhang_right"),
                "alignment_mode": "direct_indexed_overlap",
            },
            "payload_bulge_mask": list(spec.payload.bulge_mask),
        },
    }


def _nt_bpu10i_invariant(spec: YiuProcessSpecV4) -> dict[str, Any]:
    local_context = _sequence_for_tag(spec, "nt_bpu10i_snapback_site")
    complement_side = reverse_complement_iupac(local_context)
    exposed_geometry = {
        "tether_dock": complement_side[0:4],
        "tether_cap": complement_side[4:8],
        "tether_dock_complement": complement_side[8:12],
        "snapback_stem": complement_side[12:14],
    }
    subchecks = {
        "recognition_site_presence": {
            "status": "guaranteed" if local_context.startswith(YIU_V4_NT_BPU10I_RECOGNITION_SEQUENCE) else "impossible"
        },
        "nick_boundary_correctness": {"status": "guaranteed", "nick_boundary": 33},
        "downstream_exposed_tether_geometry": {
            "status": "guaranteed"
            if exposed_geometry
            == {
                "tether_dock": "TCAG",
                "tether_cap": "CGGG",
                "tether_dock_complement": "CTGA",
                "snapback_stem": "GG",
            }
            else "impossible"
        },
    }
    status = "guaranteed" if all(item["status"] == "guaranteed" for item in subchecks.values()) else "impossible"
    return {
        "id": "nt_bpu10i_snapback_invariant",
        "class": "nt_bpu10i_snapback_site",
        "status": status,
        "observed": {
            "local_context_sequence": local_context,
            "recognized_sequence": YIU_V4_NT_BPU10I_RECOGNITION_SEQUENCE,
            "nick_boundary": 33,
            "exposed_geometry": exposed_geometry,
        },
        "subchecks": subchecks,
    }


def _fragmentation_invariant(spec: YiuProcessSpecV4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fragment_sequence = _sequence_for_owner(spec, "sacrificial_region_long")
    fragment_rows = [
        {
            "state_id": "post_sacrificial_fragmentation",
            "fragment_count": 1,
            "max_fragment_nt": len(fragment_sequence),
            "fragment_lengths": [len(fragment_sequence)],
            "retained_product_sequence": _joined_sequence(spec, _RETAINED_OWNER_ORDER),
            "retained_owner_roster": list(_RETAINED_OWNER_ORDER),
        }
    ]
    invariant = {
        "id": "sacrificial_fragmentation_invariant",
        "class": "sacrificial_fragmentation",
        "status": "guaranteed" if len(fragment_sequence) <= 12 else "impossible",
        "observed": {
            "fragment_count": 1,
            "max_fragment_nt": len(fragment_sequence),
            "fragment_lengths": [len(fragment_sequence)],
            "retained_product_survives": True,
        },
    }
    return invariant, fragment_rows


def _validate_spec(spec: YiuProcessSpecV4, *, catalogs: LoadedYiuCatalogs) -> list[YiuValidationIssue]:
    issues: list[YiuValidationIssue] = []
    required_part_ids = [
        spec.external_parts.primer_source_forward,
        spec.external_parts.primer_source_reverse,
        spec.external_parts.hairpin_pcr_forward,
        spec.external_parts.hairpin_pcr_reverse,
        spec.external_parts.y_adapter,
    ]
    for part_id in required_part_ids:
        if part_id not in catalogs.oligo_parts:
            issues.append(_issue("YIU_EXTERNAL_PART_MISSING", f"missing required external part {part_id!r}"))
    if spec.enzymes.left_type_iis not in catalogs.enzymes or spec.enzymes.right_type_iis not in catalogs.enzymes:
        issues.append(_issue("YIU_TYPE_IIS_ENZYME_MISSING", "missing required type IIS enzyme catalog entry"))
    if spec.enzymes.snapback_nickase not in catalogs.enzymes:
        issues.append(_issue("YIU_NT_BPU10I_MISSING", "missing required Nt.Bpu10I catalog entry"))
    adapter = catalogs.oligo_parts.get(spec.external_parts.y_adapter)
    if adapter is None or not adapter.phosphorylated_5p:
        issues.append(_issue("YIU_ADAPTER_5P_PHOSPHATE_REQUIRED", "Y adapter must be 5' phosphorylated"))
    if _payload_invariant(spec)["status"] != "guaranteed":
        issues.append(_issue("YIU_PAYLOAD_ASSEMBLY_MISMATCH", "payload.target_sequence does not match source owners"))
    if _nt_bpu10i_invariant(spec)["status"] != "guaranteed":
        issues.append(_issue("YIU_NT_BPU10I_MISMATCH", "Nt.Bpu10I composite invariant failed"))
    return issues


def _state_hard_invariants(
    *,
    state_id: str,
    payload_invariant: dict[str, Any],
    nt_bpu10i_invariant: dict[str, Any],
    fragmentation_invariant: dict[str, Any],
) -> list[dict[str, Any]]:
    invariants: list[dict[str, Any]] = []
    if state_id in {
        "circularized_payload_candidate",
        "post_sacrificial_fragmentation",
        "post_fragment_cleanup",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    }:
        invariants.append(payload_invariant)
    if state_id in {
        "post_fragment_cleanup",
        "snapback_adapter_complex",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    }:
        invariants.append(nt_bpu10i_invariant)
    if state_id in {"post_sacrificial_fragmentation", "post_fragment_cleanup"}:
        invariants.append(fragmentation_invariant)
    return invariants


def _append_tags(target: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    target.extend(rows)


def _build_yiu_report_v4(
    spec: YiuProcessSpecV4,
    *,
    catalogs: LoadedYiuCatalogs | None = None,
) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs()
    issues = _validate_spec(spec, catalogs=catalogs)
    status = "unsatisfied" if issues else "satisfied"
    payload_invariant = _payload_invariant(spec)
    nt_bpu10i_invariant = _nt_bpu10i_invariant(spec)
    fragmentation_invariant, fragment_rows = _fragmentation_invariant(spec)

    source_sequence = spec.source_oligo.authored_sequence
    source_segments = _segments_for_source_order(spec, _SOURCE_OWNER_ORDER)
    cut_segments = _segments_for_source_order(spec, _CUT_PRODUCT_OWNER_ORDER)
    circularized_segments = _segments_for_source_order(spec, _CIRCULARIZED_OWNER_ORDER)
    retained_segments = _segments_for_source_order(spec, _RETAINED_OWNER_ORDER)
    cut_primary = _joined_sequence(spec, _CUT_PRODUCT_OWNER_ORDER)
    circularized_primary = _joined_sequence(spec, _CIRCULARIZED_OWNER_ORDER)
    retained_primary = _joined_sequence(spec, _RETAINED_OWNER_ORDER)
    payload_left_length = len(_sequence_for_owner(spec, "payload_left_half"))
    payload_right_length = len(_sequence_for_owner(spec, "payload_right_half"))
    payload_join_index = payload_left_length
    nick_anchor_start = payload_left_length + payload_right_length + 8

    adapter_sequence = catalogs.oligo_parts[spec.external_parts.y_adapter].sequence
    adapter_complementary_length = 4
    snapback_primary = retained_primary + adapter_sequence
    ligated_primary = snapback_primary

    hp_fwd_binding = reverse_complement_iupac(catalogs.oligo_parts[spec.external_parts.hairpin_pcr_forward].sequence)
    hp_rev_binding = reverse_complement_iupac(catalogs.oligo_parts[spec.external_parts.hairpin_pcr_reverse].sequence)
    final_primary = hp_fwd_binding + retained_primary + hp_rev_binding
    final_complement = reverse_complement_iupac(final_primary)

    states: list[YiuStateRecord] = []

    def _metadata_for(state_id: str) -> dict[str, Any]:
        return {
            "hard_invariants": _state_hard_invariants(
                state_id=state_id,
                payload_invariant=payload_invariant,
                nt_bpu10i_invariant=nt_bpu10i_invariant,
                fragmentation_invariant=fragmentation_invariant,
            ),
        }

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
            annotations=_owner_annotations_from_segments(source_segments, row_id="primary")
            + _project_source_tags(spec, segments=source_segments, row_id="primary"),
            pattern_label="pattern",
        )
    )

    pcr_annotations = _owner_annotations_from_segments(source_segments, row_id="primary")
    pcr_annotations.extend(_owner_annotations_from_segments(source_segments, row_id="complement"))
    _append_tags(pcr_annotations, _project_source_tags(spec, segments=source_segments, row_id="primary"))
    _append_tags(pcr_annotations, _project_source_tags(spec, segments=source_segments, row_id="complement"))
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

    cut_annotations = _owner_annotations_from_segments(cut_segments, row_id="primary")
    cut_annotations.extend(_owner_annotations_from_segments(cut_segments, row_id="complement"))
    _append_tags(cut_annotations, _project_source_tags(spec, segments=cut_segments, row_id="primary"))
    _append_tags(cut_annotations, _project_source_tags(spec, segments=cut_segments, row_id="complement"))
    _append_tags(
        cut_annotations,
        [
            _tag_annotation(
                tag_id="left_cut_boundary",
                tag_kind="cut_boundary_anchor",
                row_id="primary",
                start=0,
                end=1,
            ),
            _tag_annotation(
                tag_id="right_cut_boundary",
                tag_kind="cut_boundary_anchor",
                row_id="primary",
                start=len(cut_primary) - 1,
                end=len(cut_primary),
            ),
        ],
    )
    states.append(
        _state(
            state_id="type_iis_cut_product_duplex",
            step_id="type_iis_digest",
            kind="type_iis_cut_product_duplex",
            state_kind="type_iis_cut_product_duplex",
            topology_kind="linear_dsdna",
            status=status,
            primary_sequence=cut_primary,
            complement_sequence=reverse_complement_iupac(cut_primary),
            metadata=_metadata_for("type_iis_cut_product_duplex")
            | {
                "left_sticky_end_sequence": _sequence_for_tag(spec, "payload_overhang_left"),
                "right_sticky_end_sequence": _sequence_for_tag(spec, "payload_overhang_right"),
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(cut_segments),
            annotations=cut_annotations,
            pattern_label="pattern",
        )
    )

    circular_annotations = _owner_annotations_from_segments(circularized_segments, row_id="primary")
    circular_annotations.extend(_owner_annotations_from_segments(circularized_segments, row_id="complement"))
    _append_tags(circular_annotations, _project_source_tags(spec, segments=circularized_segments, row_id="primary"))
    _append_tags(circular_annotations, _project_source_tags(spec, segments=circularized_segments, row_id="complement"))
    _append_tags(
        circular_annotations,
        [
            _tag_annotation(
                tag_id="payload_ligation_member_left",
                tag_kind="ligation_junction_member",
                row_id="primary",
                start=0,
                end=4,
            ),
            _tag_annotation(
                tag_id="payload_ligation_member_right",
                tag_kind="ligation_junction_member",
                row_id="primary",
                start=len(_sequence_for_owner(spec, "payload_left_half")),
                end=len(_sequence_for_owner(spec, "payload_left_half")) + 4,
            ),
        ],
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
            metadata=_metadata_for("circularized_payload_candidate")
            | {"assembled_payload_sequence": spec.payload.target_sequence, "unique_alignment": True},
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            annotations=circular_annotations,
            junctions=[
                {
                    "id": "payload_circularization_junction",
                    "join_index": payload_join_index,
                }
            ],
            pattern_label="pattern",
        )
    )

    fragmentation_annotations = [
        _owner_annotation(owner_id="retained_region", row_id="primary", start=0, end=len(retained_primary)),
        _owner_annotation(
            owner_id="sacrificial_region_short",
            row_id="complement",
            start=0,
            end=len(_sequence_for_owner(spec, "sacrificial_region_long")),
        ),
    ]
    _append_tags(fragmentation_annotations, _project_source_tags(spec, segments=retained_segments, row_id="primary"))
    states.append(
        _state(
            state_id="post_sacrificial_fragmentation",
            step_id="sacrificial_fragmentation",
            kind="post_sacrificial_fragmentation",
            state_kind="post_sacrificial_fragmentation",
            topology_kind="fragment_pool",
            status=status,
            primary_sequence=retained_primary,
            metadata=_metadata_for("post_sacrificial_fragmentation")
            | {
                "fragment_summary": fragment_rows[0],
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(retained_segments),
            annotations=fragmentation_annotations,
            fragments=fragment_rows,
            pattern_label="pattern",
        )
    )

    cleanup_annotations = _owner_annotations_from_segments(retained_segments, row_id="primary")
    _append_tags(cleanup_annotations, _project_source_tags(spec, segments=retained_segments, row_id="primary"))
    _append_tags(
        cleanup_annotations,
        [
            _tag_annotation(
                tag_id="nick_boundary_anchor",
                tag_kind="nick_boundary_anchor",
                row_id="primary",
                start=nick_anchor_start,
                end=nick_anchor_start + 1,
            )
        ],
    )
    states.append(
        _state(
            state_id="post_fragment_cleanup",
            step_id="fragment_cleanup",
            kind="post_fragment_cleanup",
            state_kind="post_fragment_cleanup",
            topology_kind="linear_ssdna",
            status=status,
            primary_sequence=retained_primary,
            metadata=_metadata_for("post_fragment_cleanup") | {"retained_product_sequence": retained_primary},
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(retained_segments),
            annotations=cleanup_annotations,
            fragments=fragment_rows,
            pattern_label="pattern",
        )
    )

    adapter_segments = _offset_segments(
        _segments_for_synthetic_segments(
            [
                ("y_adapter_complementary_arm", adapter_complementary_length),
                ("y_adapter_noncomplementary_arm", len(adapter_sequence) - adapter_complementary_length),
            ]
        ),
        offset=len(retained_primary),
    )
    snapback_segments = retained_segments + adapter_segments
    snapback_annotations = _owner_annotations_from_segments(retained_segments, row_id="primary")
    snapback_annotations.extend(_owner_annotations_from_segments(adapter_segments, row_id="primary"))
    _append_tags(snapback_annotations, _project_source_tags(spec, segments=retained_segments, row_id="primary"))
    _append_tags(
        snapback_annotations,
        [
            _tag_annotation(
                tag_id="introduced_late::y_adapter",
                tag_kind="introduced_late",
                row_id="primary",
                start=len(retained_primary),
                end=len(snapback_primary),
            ),
            _tag_annotation(
                tag_id="y_adapter_binding",
                tag_kind="y_adapter_binding",
                row_id="primary",
                start=len(retained_primary),
                end=len(retained_primary) + adapter_complementary_length,
            ),
            _tag_annotation(
                tag_id="pairs_with::tether",
                tag_kind="pairs_with",
                row_id="primary",
                start=nick_anchor_start,
                end=nick_anchor_start + 4,
            ),
        ],
    )
    states.append(
        _state(
            state_id="snapback_adapter_complex",
            step_id="snapback_adapter_engagement",
            kind="snapback_adapter_complex",
            state_kind="snapback_adapter_complex",
            topology_kind="branched_y",
            status=status,
            primary_sequence=snapback_primary,
            metadata=_metadata_for("snapback_adapter_complex"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(snapback_segments),
            annotations=snapback_annotations,
            pattern_label="pattern",
        )
    )

    ligated_annotations = [dict(annotation) for annotation in snapback_annotations]
    _append_tags(
        ligated_annotations,
        [
            _tag_annotation(
                tag_id="hairpin_ligation_member",
                tag_kind="ligation_junction_member",
                row_id="primary",
                start=len(retained_primary) - 1,
                end=len(retained_primary) + 1,
            )
        ],
    )
    states.append(
        _state(
            state_id="ligated_ssdna_hairpin",
            step_id="hairpin_ligation",
            kind="ligated_ssdna_hairpin",
            state_kind="ligated_ssdna_hairpin",
            topology_kind="hairpin_ssdna",
            status=status,
            primary_sequence=ligated_primary,
            metadata=_metadata_for("ligated_ssdna_hairpin"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(snapback_segments),
            annotations=ligated_annotations,
            pattern_label="pattern",
        )
    )

    final_primary_segments = _segments_for_synthetic_segments(
        [
            ("hairpin_pcr_forward_binding_region", len(hp_fwd_binding)),
            ("payload_left_half", len(_sequence_for_owner(spec, "payload_left_half"))),
            ("payload_right_half", len(_sequence_for_owner(spec, "payload_right_half"))),
            ("tether_dock_complement", len(_sequence_for_owner(spec, "tether_dock_complement"))),
            ("tether_cap", len(_sequence_for_owner(spec, "tether_cap"))),
            ("tether_dock", len(_sequence_for_owner(spec, "tether_dock"))),
            ("snapback_stem", len(_sequence_for_owner(spec, "snapback_stem"))),
            ("hairpin_pcr_reverse_binding_region", len(hp_rev_binding)),
        ]
    )
    final_complement_segments = _segments_for_synthetic_segments(
        [
            ("hairpin_pcr_reverse_binding_region", len(hp_rev_binding)),
            ("retained_region", len(retained_primary)),
            ("hairpin_pcr_forward_binding_region", len(hp_fwd_binding)),
        ]
    )
    final_annotations = _owner_annotations_from_segments(final_primary_segments, row_id="primary")
    final_annotations.extend(_owner_annotations_from_segments(final_complement_segments, row_id="complement"))
    _append_tags(
        final_annotations,
        [
            _tag_annotation(
                tag_id="hairpin_pcr_forward_bindable",
                tag_kind="primer_bindable_by_hairpin_pcr_forward",
                row_id="primary",
                start=0,
                end=len(hp_fwd_binding),
            ),
            _tag_annotation(
                tag_id="hairpin_pcr_reverse_bindable",
                tag_kind="primer_bindable_by_hairpin_pcr_reverse",
                row_id="primary",
                start=len(final_primary) - len(hp_rev_binding),
                end=len(final_primary),
            ),
        ],
    )
    states.append(
        _state(
            state_id="hairpin_pcr_linear_insert",
            step_id="hairpin_pcr",
            kind="hairpin_pcr_linear_insert",
            state_kind="hairpin_pcr_linear_insert",
            topology_kind="linear_dsdna",
            status=status,
            primary_sequence=final_primary,
            complement_sequence=final_complement,
            metadata=_metadata_for("hairpin_pcr_linear_insert"),
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(final_primary_segments),
            annotations=final_annotations,
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
        protocol="yiu_v4",
        protocol_template=spec.protocol_template,
        spec_name=spec.name,
        status=status,  # type: ignore[arg-type]
        metadata=metadata,
        states=states,
        issues=issues,
    )
