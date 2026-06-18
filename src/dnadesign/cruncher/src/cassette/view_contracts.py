"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cassette/view_contracts.py

Neutral shared-view publication for cassette explicit and solve workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.contracts.visual import CassetteViewsManifestV1, HairpinTopologyViewV1, LinearDuplexViewV1
from dnadesign.cruncher.cassette.models import CassetteCandidateDesign, CassetteEvaluationReport


def _segment_payloads(candidate: CassetteCandidateDesign) -> list[dict[str, object]]:
    cassette_start = int(candidate.context_offset)
    cassette_end = cassette_start + int(candidate.cassette_length_nt)
    evaluation_length = len(candidate.evaluation_primary_sequence)
    segments: list[dict[str, object]] = []
    if cassette_start > 0:
        segments.append(
            {
                "id": "left_flank",
                "start": 0,
                "end": cassette_start,
                "semantic": "flank",
                "label": "Left flank",
            }
        )
    segments.extend(
        [
            {
                "id": "stem5p_arm",
                "start": cassette_start + int(candidate.stem5p_span.start),
                "end": cassette_start + int(candidate.stem5p_span.end),
                "semantic": "stem5p_arm",
                "label": "Stem 5' arm",
            },
            {
                "id": "loop",
                "start": cassette_start + int(candidate.loop_span.start),
                "end": cassette_start + int(candidate.loop_span.end),
                "semantic": "loop",
                "label": "Loop",
            },
            {
                "id": "stem3p_arm",
                "start": cassette_start + int(candidate.stem3p_span.start),
                "end": cassette_start + int(candidate.stem3p_span.end),
                "semantic": "stem3p_arm",
                "label": "Stem 3' arm",
            },
        ]
    )
    if cassette_end < evaluation_length:
        segments.append(
            {
                "id": "right_flank",
                "start": cassette_end,
                "end": evaluation_length,
                "semantic": "flank",
                "label": "Right flank",
            }
        )
    return segments


def _site_payloads(candidate: CassetteCandidateDesign) -> list[dict[str, object]]:
    payloads = [
        {
            "id": "left_site",
            "variant_id": candidate.intended_left_site.variant_id,
            "specificity_id": candidate.intended_left_site.specificity_id,
            "start": int(candidate.intended_left_site.start),
            "end": int(candidate.intended_left_site.end),
            "orientation": candidate.intended_left_site.orientation,
            "intent": "intended_left",
            "label": candidate.intended_left_site.variant_id,
            "site_target_strand": candidate.intended_left_nick.strand,
        },
        {
            "id": "right_site",
            "variant_id": candidate.intended_right_site.variant_id,
            "specificity_id": candidate.intended_right_site.specificity_id,
            "start": int(candidate.intended_right_site.start),
            "end": int(candidate.intended_right_site.end),
            "orientation": candidate.intended_right_site.orientation,
            "intent": "intended_right",
            "label": candidate.intended_right_site.variant_id,
            "site_target_strand": candidate.intended_right_nick.strand,
        },
    ]
    for index, extra in enumerate(candidate.extra_designated_strand_nicks, start=1):
        payloads.append(
            {
                "id": f"extra_site_{index}",
                "variant_id": extra.variant_id,
                "specificity_id": extra.specificity_id,
                "start": int(extra.source_site_start),
                "end": int(extra.source_site_end),
                "orientation": extra.source_site_orientation,
                "intent": "extra",
                "label": extra.variant_id,
                "site_target_strand": extra.strand,
            }
        )
    return payloads


def _nick_payloads(candidate: CassetteCandidateDesign) -> list[dict[str, object]]:
    payloads = [
        {
            "id": "left_nick",
            "boundary": int(candidate.intended_left_nick.boundary),
            "target_strand": candidate.intended_left_nick.strand,
            "source_site_id": "left_site",
            "intent": "intended_left",
            "label": "Nick",
        },
        {
            "id": "right_nick",
            "boundary": int(candidate.intended_right_nick.boundary),
            "target_strand": candidate.intended_right_nick.strand,
            "source_site_id": "right_site",
            "intent": "intended_right",
            "label": "Nick",
        },
    ]
    for index, extra in enumerate(candidate.extra_designated_strand_nicks, start=1):
        payloads.append(
            {
                "id": f"extra_nick_{index}",
                "boundary": int(extra.boundary),
                "target_strand": extra.strand,
                "source_site_id": f"extra_site_{index}",
                "intent": "extra",
                "label": "Extra nick",
            }
        )
    return payloads


def _project_feature_spans(candidate: CassetteCandidateDesign) -> list[dict[str, object]]:
    cassette_start = int(candidate.context_offset)
    projections: list[dict[str, object]] = []
    for site in _site_payloads(candidate):
        start = max(0, int(site["start"]) - cassette_start)
        end = min(int(candidate.cassette_length_nt), int(site["end"]) - cassette_start)
        if not (0 <= start < end <= int(candidate.cassette_length_nt)):
            continue
        label = str(site["label"])
        projections.append(
            {
                "id": f"{site['id']}_projection",
                "start": start,
                "end": end,
                "semantic": "motif_projection",
                "label": f"{label} motif",
            }
        )
    return projections


def build_linear_duplex_view(
    *,
    report: CassetteEvaluationReport,
    solution_id: str,
    title: str,
    rank: int | None = None,
    source_solve_id: str | None = None,
    explicit_design_id: str | None = None,
) -> LinearDuplexViewV1:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Cassette view publication requires a satisfied report with candidate details.")
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": f"{solution_id}.linear_duplex",
        "solution_id": solution_id,
        "title": title,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "primary_sequence_5to3": candidate.evaluation_primary_sequence,
        "sequence_span": {"start": 0, "end": len(candidate.evaluation_primary_sequence)},
        "cassette_span": {
            "start": int(candidate.context_offset),
            "end": int(candidate.context_offset) + int(candidate.cassette_length_nt),
        },
        "row_labels": {
            "primary": "5' -> 3' primary",
            "complement": "3' -> 5' complement",
        },
        "target_strand": candidate.target_strand,
        "segments": _segment_payloads(candidate),
        "site_instances": _site_payloads(candidate),
        "nick_events": _nick_payloads(candidate),
        "bounded_segment": {
            "start_boundary": int(candidate.bounded_nicked_segment.start_boundary),
            "end_boundary": int(candidate.bounded_nicked_segment.end_boundary),
            "target_strand": candidate.bounded_nicked_segment.strand,
            "label": "Bounded nicked segment",
        },
        "labels": [{"text": f"Target strand: {candidate.target_strand}", "placement": "header"}],
        "meta": {
            "rank": rank,
            "left_variant_id": candidate.intended_left_nick.variant_id,
            "right_variant_id": candidate.intended_right_nick.variant_id,
            "left_boundary": candidate.intended_left_nick.boundary,
            "right_boundary": candidate.intended_right_nick.boundary,
            "bounded_length_nt": candidate.bounded_nicked_segment.length_nt,
            "source_solve_id": source_solve_id,
            "explicit_design_id": explicit_design_id,
        },
    }
    return LinearDuplexViewV1.model_validate(payload)


def build_hairpin_topology_view(
    *,
    report: CassetteEvaluationReport,
    solution_id: str,
    title: str,
    rank: int | None = None,
    source_solve_id: str | None = None,
    explicit_design_id: str | None = None,
) -> HairpinTopologyViewV1:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Cassette view publication requires a satisfied report with candidate details.")
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": f"{solution_id}.ssdna_hairpin",
        "solution_id": solution_id,
        "title": title,
        "primary_sequence_5to3": candidate.cassette_sequence,
        "topology": {
            "stem5p_span": candidate.stem5p_span.model_dump(mode="json"),
            "loop_span": candidate.loop_span.model_dump(mode="json"),
            "stem3p_span": candidate.stem3p_span.model_dump(mode="json"),
        },
        "pair_map": [{"left_index": int(pair.left), "right_index": int(pair.right)} for pair in candidate.pair_map],
        "feature_spans": _project_feature_spans(candidate),
        "duplex_derived_annotations": [
            {
                "kind": "informational_note",
                "text": "Nicking is defined in the linear duplex interpretation.",
            }
        ],
        "meta": {
            "rank": rank,
            "left_variant_id": candidate.intended_left_nick.variant_id,
            "right_variant_id": candidate.intended_right_nick.variant_id,
            "source_solve_id": source_solve_id,
            "explicit_design_id": explicit_design_id,
        },
    }
    return HairpinTopologyViewV1.model_validate(payload)


def build_views_manifest(
    *,
    solution_id: str,
    rank: int | None,
    include_jobs: bool,
) -> CassetteViewsManifestV1:
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "cassette_views_manifest_v1",
        "solution_id": solution_id,
        "rank": rank,
        "views": [
            {"view_kind": "linear_duplex_v1", "path": "linear_duplex.v1.json"},
            {"view_kind": "ssdna_hairpin_v1", "path": "ssdna_hairpin.v1.json"},
        ],
        "recommended_jobs": [],
    }
    if include_jobs:
        payload["recommended_jobs"] = [
            {"name": "linear_duplex", "path": "../baserender_jobs/linear_duplex.job.yaml"},
            {"name": "ssdna_hairpin", "path": "../baserender_jobs/ssdna_hairpin.job.yaml"},
        ]
    return CassetteViewsManifestV1.model_validate(payload)


def build_single_view_job(
    *,
    input_filename: str,
    adapter_kind: str,
    renderer: str,
    style_preset: str,
    output_filename: str,
) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "json",
            "path": f"../views/{input_filename}",
            "adapter": {"kind": adapter_kind},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": renderer,
            "style": {
                "preset": style_preset,
                "overrides": {},
            },
        },
        "outputs": [
            {
                "kind": "images",
                "path": f"../renders/{output_filename}",
                "fmt": "pdf",
            }
        ],
        "run": {
            "strict": True,
            "fail_on_skips": True,
            "emit_report": False,
        },
    }


def build_top_hits_job(
    *,
    input_filename: str,
    adapter_kind: str,
    renderer: str,
    style_preset: str,
    output_filename: str,
) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "jsonl",
            "path": f"../views/{input_filename}",
            "adapter": {"kind": adapter_kind},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": renderer,
            "style": {
                "preset": style_preset,
                "overrides": {},
            },
        },
        "outputs": [
            {
                "kind": "images",
                "path": f"../renders/{output_filename}",
                "fmt": "pdf",
            }
        ],
        "run": {
            "strict": True,
            "fail_on_skips": True,
            "emit_report": False,
        },
    }
