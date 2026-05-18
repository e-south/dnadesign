"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/outputs/composition_payload.py

Retron MSD single-unit composition payload construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from ..errors import RetronMsdCompilerError
from .layout import MSD_UNIT_REPEAT_COUNT, SNAPBACK_FOLDBACK_SEGMENT_ID


def require_sequence_subcomponents(
    catalog: MsdDesignCatalogV1,
    *,
    payload_sequences: Mapping[str, str],
    cap_sequences: Mapping[str, str],
) -> None:
    missing_payloads = sorted(
        {
            record.payload_or_target.id
            for record in catalog.records
            if not payload_sequences.get(record.payload_or_target.id)
        }
    )
    missing_caps = sorted({record.cap.id for record in catalog.records if not cap_sequences.get(record.cap.id)})
    if not missing_payloads and not missing_caps:
        return
    pieces: list[str] = []
    if missing_payloads:
        pieces.append(f"payload(s): {', '.join(missing_payloads)}")
    if missing_caps:
        pieces.append(f"cap(s): {', '.join(missing_caps)}")
    raise RetronMsdCompilerError(
        "MSD sequence artifact generation requires concrete sequence subcomponents for "
        f"{'; '.join(pieces)}. Provide --payload-sequence ID=ACGT and --cap-sequence ID=ACGT overrides, "
        "or route missing cap/shortening inputs to Snapback and missing base-junction inputs to scar-nick first."
    )


def normalize_render_formats(render_formats: Sequence[str]) -> list[str]:
    requested = list(render_formats) or ["png"]
    formats: list[str] = []
    for raw_format in requested:
        fmt = str(raw_format or "").strip().lower()
        if fmt not in {"png", "svg", "pdf"}:
            raise RetronMsdCompilerError(f"Unsupported render format '{raw_format}'. Expected png, svg, or pdf.")
        if fmt not in formats:
            formats.append(fmt)
    return formats


def render_formats_for_review(render_formats: Sequence[str]) -> list[str]:
    formats = list(render_formats)
    if "svg" not in formats:
        formats.append("svg")
    return formats


def _msd_unit_segments(
    record: MsdDesignReferenceV1,
    *,
    flank_5p: str,
    flank_3p: str,
    payload_sequence: str,
    cap_sequence: str,
) -> list[dict[str, object]]:
    return [
        {
            "segment_id": "flank_5p",
            "role": "flank_5p",
            "sequence": flank_5p,
            "source": {
                "kind": "study_record",
                "study_id": "retron_hairpin_design",
                "ref": record.construct_label,
            },
        },
        {
            "segment_id": "payload_primary",
            "role": "payload_primary",
            "sequence": payload_sequence,
            "source": {"kind": "literal", "label": f"{record.payload_or_target.id} override"},
        },
        {
            "segment_id": SNAPBACK_FOLDBACK_SEGMENT_ID,
            "role": "snapback_foldback_geometry",
            "sequence": cap_sequence,
            "source": {"kind": "literal", "label": f"{record.cap.id} override"},
        },
        {
            "segment_id": "payload_complement",
            "role": "payload_complement",
            "transform": {
                "kind": "reverse_complement",
                "source_segment_id": "payload_primary",
                "assert_expected_sequence": True,
            },
            "source": {"kind": "derived", "from_segment_id": "payload_primary"},
        },
        {
            "segment_id": "flank_3p",
            "role": "flank_3p",
            "sequence": flank_3p,
            "source": {
                "kind": "study_record",
                "study_id": "retron_hairpin_design",
                "ref": record.construct_label,
            },
        },
    ]


def _msd_unit_annotations(
    *,
    flank_5p_prefix: str,
    flank_5p: str,
    right_base: str,
    cap_topology,
) -> list[dict[str, object]]:
    return [
        {
            "annotation_id": "stem_base_left",
            "role": "stem_base_left",
            "location": {
                "basis": "segment",
                "segment_id": "flank_5p",
                "start": len(flank_5p_prefix),
                "end": len(flank_5p),
            },
        },
        {
            "annotation_id": "stem_base_right",
            "role": "stem_base_right",
            "location": {
                "basis": "segment",
                "segment_id": "flank_3p",
                "start": 0,
                "end": len(right_base),
            },
        },
        {
            "annotation_id": "snapback_retained_stem",
            "role": "snapback_retained_stem",
            "location": {
                "basis": "segment",
                "segment_id": SNAPBACK_FOLDBACK_SEGMENT_ID,
                "start": cap_topology.retained_stem_span.start,
                "end": cap_topology.retained_stem_span.end,
            },
        },
        {
            "annotation_id": "snapback_cap",
            "role": "snapback_cap",
            "location": {
                "basis": "segment",
                "segment_id": SNAPBACK_FOLDBACK_SEGMENT_ID,
                "start": cap_topology.cap_span.start,
                "end": cap_topology.cap_span.end,
            },
        },
        {
            "annotation_id": "snapback_foldback_return",
            "role": "snapback_foldback_return",
            "location": {
                "basis": "segment",
                "segment_id": SNAPBACK_FOLDBACK_SEGMENT_ID,
                "start": cap_topology.foldback_return_span.start,
                "end": cap_topology.foldback_return_span.end,
            },
        },
    ]


def _msd_display_profile(record: MsdDesignReferenceV1, *, payload_label: str) -> dict[str, object]:
    return {
        "title": f"{record.construct_id} {payload_label}",
        "component_labels": {
            "flank_5p": "5' Flanking",
            "payload_primary": payload_label,
            SNAPBACK_FOLDBACK_SEGMENT_ID: "Foldback",
            "payload_complement": f"{payload_label} complement",
            "flank_3p": "3' Flanking",
        },
        "annotation_labels": {
            "stem_base_left": "Left Base",
            "stem_base_right": "Right Base",
            "snapback_retained_stem": "Foldback stem",
            "snapback_cap": "Cap",
            "snapback_foldback_return": "Foldback return",
        },
        "scar_nick": {
            "left_base": record.scar_nick.left_base,
            "right_base": record.scar_nick.right_base,
            "profile_s3s2s1s0": record.scar_nick.profile_s3s2s1s0,
        },
        "component_hues": {
            "flank_5p": "#2563EB",
            "flank_3p": "#14B8A6",
            "payload_primary": "#F97316",
            "payload_complement": "#DC2626",
            SNAPBACK_FOLDBACK_SEGMENT_ID: "#64748B",
            "snapback_retained_stem": "#7C3AED",
            "snapback_cap": "#16A34A",
            "snapback_foldback_return": "#DB2777",
            "stem_base_left": "#7C3AED",
            "stem_base_right": "#A16207",
        },
        "component_styles": {
            "flank_5p": {"fill": "#BFDBFE", "alpha": 0.72, "edge_color": "#2563EB"},
            "payload_primary": {"fill": "#FDBA74", "alpha": 0.66, "edge_color": "#EA580C"},
            "snapback_cap": {"fill": "#86EFAC", "alpha": 0.78, "edge_color": "#16A34A"},
            "payload_complement": {"fill": "#FCA5A5", "alpha": 0.66, "edge_color": "#DC2626"},
            "flank_3p": {"fill": "#5EEAD4", "alpha": 0.72, "edge_color": "#0D9488"},
            "stem_base_left": {"fill": "#EDE9FE", "alpha": 0.76, "edge_color": "#7C3AED"},
            "stem_base_right": {"fill": "#FEF3C7", "alpha": 0.76, "edge_color": "#A16207"},
        },
    }


def composition_config_payload(
    record: MsdDesignReferenceV1,
    *,
    artifact_bundle: Path,
    payload_sequence: str,
    cap_sequence: str,
    flank_5p_prefix: str,
    flank_3p_suffix: str,
    render_formats: Sequence[str],
) -> dict[str, object]:
    left_base = record.scar_nick.left_base
    right_base = record.scar_nick.right_base
    cap_topology = _require_cap_topology(record=record, cap_sequence=cap_sequence)
    flank_5p = f"{flank_5p_prefix}{left_base}"
    flank_3p = f"{right_base}{flank_3p_suffix}"
    payload_label = record.payload_or_target.display_name or record.payload_or_target.id
    return {
        "contract": "linear_ssdna_composition_v1",
        "schema_version": 1,
        "composition_id": record.msd_design_id,
        "alphabet": "dna",
        "topology": "linear_ssdna",
        "coordinate_system": "zero_based_half_open",
        "case_policy": "preserve_input_display_case",
        "canonicalization": {
            "compare_sequences_case_insensitive": True,
            "output_sequence_preserves_case": True,
        },
        "units": [
            {
                "unit_id": f"{record.msd_design_id}_unit",
                "repeat_count": MSD_UNIT_REPEAT_COUNT,
                "segments": _msd_unit_segments(
                    record,
                    flank_5p=flank_5p,
                    flank_3p=flank_3p,
                    payload_sequence=payload_sequence,
                    cap_sequence=cap_sequence,
                ),
                "annotations": _msd_unit_annotations(
                    flank_5p_prefix=flank_5p_prefix,
                    flank_5p=flank_5p,
                    right_base=right_base,
                    cap_topology=cap_topology,
                ),
                "assertions": [
                    {
                        "assertion_id": "payload_rc",
                        "kind": "reverse_complement",
                        "left_segment_id": "payload_primary",
                        "right_segment_id": "payload_complement",
                        "severity": "error",
                    }
                ],
            }
        ],
        "qa": {
            "require_no_unknown_bases": True,
            "allow_degenerate_bases": False,
            "require_segment_span_coverage": True,
            "require_non_overlapping_physical_segments": True,
            "require_annotation_bounds": True,
            "require_declared_transform_checks": True,
            "allow_cross_copy_intended_pairings": False,
        },
        "folding": {
            "enabled": True,
            "required": True,
            "scope": "canonical_component_unit",
            "backend": {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "backend_contract": "secondary_structure_prediction_v1",
            },
            "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
        },
        "visual": {
            "emit": ["sequence_evidence_map_v1", "viennarna_secondary_structure_svg_v1"],
            "display_profile": _msd_display_profile(record, payload_label=payload_label),
            "render_exports": {"formats": list(render_formats)},
        },
        "benchling_export": {
            "enabled": True,
            "primary_format": "genbank",
            "sidecars": ["fasta", "features_csv"],
        },
        "output": {"artifact_bundle": artifact_bundle.as_posix(), "usr": {"enabled": False}},
    }


def _require_cap_topology(*, record: MsdDesignReferenceV1, cap_sequence: str):
    topology = record.cap.snapback_topology
    if topology is None:
        raise RetronMsdCompilerError(
            f"MSD cap '{record.cap.id}' is missing snapback_topology. "
            "Retron materialization requires explicit Snapback foldback geometry so the Cap label covers only "
            "the cap subsection, not the whole foldback segment."
        )
    expected_length = topology.foldback_return_span.end
    if len(cap_sequence) != expected_length:
        raise RetronMsdCompilerError(
            f"MSD cap '{record.cap.id}' sequence length {len(cap_sequence)} does not match "
            f"snapback_topology length {expected_length}."
        )
    return topology


__all__ = [
    "composition_config_payload",
    "normalize_render_formats",
    "render_formats_for_review",
    "require_sequence_subcomponents",
]
