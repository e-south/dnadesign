"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/view_contracts.py

Visual publication for v2 snapback designs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.contracts.visual import SnapbackVisualV1
from dnadesign.cruncher.nickases.models import reverse_complement
from dnadesign.cruncher.snapback.models import CoordinateSpan, SnapbackEvaluationReport
from dnadesign.cruncher.snapback.view_models import (
    SnapbackPostNickExposedViewV1,
    SnapbackPostNickFoldbackViewV1,
    SnapbackPreNickDuplexViewV1,
    SnapbackViewsManifestV1,
)


def _complement(sequence: str) -> str:
    return reverse_complement(sequence)[::-1]


def _protected_overlap_span(*, candidate) -> dict[str, int] | None:
    overlap_start = max(candidate.protected_region.start, candidate.retained_homology_window.start)
    overlap_end = min(candidate.protected_region.end, candidate.retained_homology_window.end)
    if overlap_end <= overlap_start:
        return None
    local_start = overlap_start - candidate.retained_homology_window.start
    local_end = overlap_end - candidate.retained_homology_window.start
    return {
        "start": candidate.post_nick_retained_homology_span.start + local_start,
        "end": candidate.post_nick_retained_homology_span.start + local_end,
    }


def _released_suffix_source_span(*, candidate) -> CoordinateSpan | None:
    suffix_start = candidate.retained_homology_window.end
    suffix_end = len(candidate.input_sequence)
    if suffix_end <= suffix_start:
        return None
    return CoordinateSpan(start=suffix_start, end=suffix_end)


def _absolute_primary_mismatch_positions(candidate) -> list[int]:
    return [candidate.post_nick_retained_homology_span.start + position for position in candidate.mismatch_positions]


def _absolute_foldback_partner_mismatch_positions(candidate) -> list[int]:
    return [candidate.post_nick_foldback_arm_span.end - 1 - position for position in candidate.mismatch_positions]


def build_pre_nick_duplex_view(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback visual publication requires a satisfied report with candidate details.")
    payload = {
        "version": 1,
        "kind": "snapback_pre_nick_duplex_v1",
        "view_id": f"{solution_id}.pre_nick_duplex",
        "solution_id": solution_id,
        "title": title,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "boundary_semantics": report.metadata.boundary_semantics,
        "sequence_span": {"start": 0, "end": len(candidate.designed_sequence)},
        "input_span": {"start": 0, "end": len(candidate.input_sequence)},
        "rows": {
            "top": {
                "label": "Canonical top strand 5' -> 3'",
                "direction": "5to3",
                "sequence": candidate.designed_sequence,
            },
            "complement": {
                "label": "Complement strand 3' -> 5'",
                "direction": "3to5",
                "sequence": _complement(candidate.designed_sequence),
            },
        },
        "nick_boundary": candidate.nick_boundary,
        "ligation_junction_boundary": candidate.retained_homology_window.start,
        "protected_region": candidate.protected_region.model_dump(mode="json"),
        "pre_nick_duplex_window": candidate.pre_nick_duplex_window.model_dump(mode="json"),
        "retained_homology_window": candidate.retained_homology_window.model_dump(mode="json"),
        "cap_span": candidate.cap_span.model_dump(mode="json"),
        "foldback_arm_span": candidate.foldback_arm_span.model_dump(mode="json"),
        "intended_site": candidate.intended_site.model_dump(mode="json"),
        "intended_nick": candidate.intended_nick.model_dump(mode="json"),
        "extra_target_strand_nicks": [event.model_dump(mode="json") for event in candidate.extra_target_strand_nicks],
        "extra_nick_events": [event.model_dump(mode="json") for event in candidate.extra_nick_events],
        "meta": {
            "nick_boundary_from_left": candidate.nick_boundary_from_left,
            "released_prefix_nt": candidate.released_prefix_nt,
            "retained_start_from_nick": candidate.retained_start_from_nick,
            "cap_nt": candidate.cap_nt,
            "paired_bp": candidate.paired_bp,
            "mismatch_count": candidate.mismatch_count,
            "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
            "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
        },
    }
    return SnapbackPreNickDuplexViewV1.model_validate(payload).model_dump(mode="json")


def build_post_nick_exposed_view(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback visual publication requires a satisfied report with candidate details.")
    released_suffix = _released_suffix_source_span(candidate=candidate)
    payload = {
        "version": 1,
        "kind": "snapback_post_nick_exposed_v1",
        "view_id": f"{solution_id}.post_nick_exposed",
        "solution_id": solution_id,
        "title": title,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "boundary_semantics": report.metadata.boundary_semantics,
        "sequence_span": {"start": 0, "end": len(candidate.designed_sequence)},
        "rows": {
            "top": {
                "label": "Canonical top strand after nick 5' -> 3'",
                "direction": "5to3",
                "sequence": candidate.designed_sequence,
            },
            "complement": {
                "label": "Complement strand 3' -> 5'",
                "direction": "3to5",
                "sequence": _complement(candidate.designed_sequence),
            },
        },
        "nick_boundary": candidate.nick_boundary,
        "ligation_junction_boundary": candidate.retained_homology_window.start,
        "topology": {
            "anchored_top_span": {"start": 0, "end": candidate.nick_boundary},
            "released_top_span": {"start": candidate.nick_boundary, "end": len(candidate.designed_sequence)},
            "released_prefix_span": {
                "start": candidate.nick_boundary,
                "end": candidate.nick_boundary + candidate.released_prefix_nt,
            },
            "retained_homology_span": candidate.retained_homology_window.model_dump(mode="json"),
            "released_suffix_span": released_suffix.model_dump(mode="json") if released_suffix is not None else None,
            "cap_span": candidate.cap_span.model_dump(mode="json"),
            "foldback_arm_span": candidate.foldback_arm_span.model_dump(mode="json"),
        },
        "intended_site": candidate.intended_site.model_dump(mode="json"),
        "intended_nick": candidate.intended_nick.model_dump(mode="json"),
        "meta": {
            "released_prefix_sequence": candidate.released_prefix_sequence,
            "retained_homology_sequence": candidate.retained_homology_sequence,
            "cap_sequence": candidate.cap_sequence,
            "foldback_arm": candidate.foldback_arm,
            "released_prefix_nt": candidate.released_prefix_nt,
            "retained_start_from_nick": candidate.retained_start_from_nick,
            "cap_nt": candidate.cap_nt,
        },
    }
    return SnapbackPostNickExposedViewV1.model_validate(payload).model_dump(mode="json")


def build_post_nick_foldback_view(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback visual publication requires a satisfied report with candidate details.")
    payload = {
        "version": 1,
        "kind": "snapback_post_nick_foldback_v1",
        "view_id": f"{solution_id}.post_nick_foldback",
        "solution_id": solution_id,
        "title": title,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "boundary_semantics": report.metadata.boundary_semantics,
        "source_nick_boundary": candidate.nick_boundary,
        "ligation_junction_boundary": candidate.post_nick_retained_homology_span.start,
        "primary_sequence_5to3": candidate.post_nick_sequence,
        "topology": {
            "released_prefix_span": candidate.post_nick_released_prefix_span.model_dump(mode="json"),
            "retained_homology_span": candidate.post_nick_retained_homology_span.model_dump(mode="json"),
            "cap_span": candidate.post_nick_cap_span.model_dump(mode="json"),
            "foldback_arm_span": candidate.post_nick_foldback_arm_span.model_dump(mode="json"),
            "protected_overlap_span": _protected_overlap_span(candidate=candidate),
        },
        "pair_map": [pair.model_dump(mode="json") for pair in candidate.pair_map],
        "primary_mismatch_positions": _absolute_primary_mismatch_positions(candidate),
        "foldback_partner_mismatch_positions": _absolute_foldback_partner_mismatch_positions(candidate),
        "meta": {
            "protected_region": candidate.protected_region.model_dump(mode="json"),
            "released_prefix_sequence": candidate.released_prefix_sequence,
            "retained_homology_sequence": candidate.retained_homology_sequence,
            "cap_sequence": candidate.cap_sequence,
            "foldback_arm": candidate.foldback_arm,
            "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
            "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
        },
    }
    return SnapbackPostNickFoldbackViewV1.model_validate(payload).model_dump(mode="json")


def build_pre_nick_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback public visual publication requires a satisfied report with candidate details.")
    released_suffix = _released_suffix_source_span(candidate=candidate)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.pre_nick_duplex",
            "state_kind": "pre_nick_duplex",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.designed_sequence,
            "complement_sequence": _complement(candidate.designed_sequence),
            "primary_row_label": "Top",
            "complement_row_label": "Partner",
            "nick_boundary": candidate.nick_boundary,
            "ligation_junction_boundary": candidate.retained_homology_window.start,
            "protected_region_span": candidate.protected_region.model_dump(mode="json"),
            "pre_nick_duplex_window_span": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            "intended_site_span": {
                "start": candidate.intended_site.start,
                "end": candidate.intended_site.end,
            },
            "released_prefix_span": (
                {"start": candidate.nick_boundary, "end": candidate.retained_homology_window.start}
                if candidate.released_prefix_nt > 0
                else None
            ),
            "retained_stem_span": candidate.retained_homology_window.model_dump(mode="json"),
            "released_suffix_span": released_suffix.model_dump(mode="json") if released_suffix is not None else None,
            "cap_span": candidate.cap_span.model_dump(mode="json"),
            "foldback_revcomp_span": candidate.foldback_arm_span.model_dump(mode="json"),
            "pairings": [],
            "meta": {
                "source_view_kind": "snapback_pre_nick_duplex_v1",
                "nick_boundary_from_left": candidate.nick_boundary_from_left,
                "retained_start_from_nick": candidate.retained_start_from_nick,
                "added_nt": candidate.added_nt,
            },
        }
    ).model_dump(mode="json")


def build_post_nick_exposed_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback public visual publication requires a satisfied report with candidate details.")
    released_suffix = _released_suffix_source_span(candidate=candidate)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.post_nick_exposed",
            "state_kind": "post_nick_exposed",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.designed_sequence,
            "complement_sequence": _complement(candidate.designed_sequence),
            "primary_row_label": "Top",
            "complement_row_label": "Partner",
            "nick_boundary": candidate.nick_boundary,
            "ligation_junction_boundary": candidate.retained_homology_window.start,
            "protected_region_span": candidate.protected_region.model_dump(mode="json"),
            "pre_nick_duplex_window_span": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            "intended_site_span": {
                "start": candidate.intended_site.start,
                "end": candidate.intended_site.end,
            },
            "anchored_duplex_span": {"start": 0, "end": candidate.nick_boundary},
            "released_prefix_span": (
                {"start": candidate.nick_boundary, "end": candidate.retained_homology_window.start}
                if candidate.released_prefix_nt > 0
                else None
            ),
            "retained_stem_span": candidate.retained_homology_window.model_dump(mode="json"),
            "released_suffix_span": released_suffix.model_dump(mode="json") if released_suffix is not None else None,
            "cap_span": candidate.cap_span.model_dump(mode="json"),
            "foldback_revcomp_span": candidate.foldback_arm_span.model_dump(mode="json"),
            "exposed_complement_span": {"start": candidate.nick_boundary, "end": len(candidate.designed_sequence)},
            "pairings": [],
            "meta": {
                "source_view_kind": "snapback_post_nick_exposed_v1",
                "released_prefix_sequence": candidate.released_prefix_sequence,
            },
        }
    ).model_dump(mode="json")


def build_post_nick_foldback_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback public visual publication requires a satisfied report with candidate details.")
    protected_overlap = _protected_overlap_span(candidate=candidate)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.post_nick_foldback",
            "state_kind": "post_nick_foldback",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.post_nick_sequence,
            "complement_sequence": _complement(candidate.post_nick_sequence),
            "primary_row_label": "Foldback",
            "complement_row_label": "Partner",
            "ligation_junction_boundary": candidate.post_nick_retained_homology_span.start,
            "protected_region_span": protected_overlap,
            "released_prefix_span": candidate.post_nick_released_prefix_span.model_dump(mode="json"),
            "retained_stem_span": candidate.post_nick_retained_homology_span.model_dump(mode="json"),
            "cap_span": candidate.post_nick_cap_span.model_dump(mode="json"),
            "foldback_revcomp_span": candidate.post_nick_foldback_arm_span.model_dump(mode="json"),
            "pairings": [{"left_index": pair.left, "right_index": pair.right} for pair in candidate.pair_map],
            "primary_mismatch_positions": _absolute_primary_mismatch_positions(candidate),
            "complement_mismatch_positions": _absolute_foldback_partner_mismatch_positions(candidate),
            "meta": {
                "source_view_kind": "snapback_post_nick_foldback_v1",
                "source_nick_boundary": candidate.nick_boundary,
                "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
                "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
            },
        }
    ).model_dump(mode="json")


def build_views_manifest(*, solution_id: str, include_jobs: bool) -> dict[str, Any]:
    payload = {
        "version": 1,
        "kind": "snapback_views_manifest_v1",
        "solution_id": solution_id,
        "views": [
            {
                "name": "pre_nick_duplex_qa",
                "path": "views/pre_nick_duplex.v1.json",
                "contract_kind": "snapback_pre_nick_duplex_v1",
            },
            {
                "name": "post_nick_exposed_qa",
                "path": "views/post_nick_exposed.v1.json",
                "contract_kind": "snapback_post_nick_exposed_v1",
            },
            {
                "name": "post_nick_foldback_qa",
                "path": "views/post_nick_foldback.v1.json",
                "contract_kind": "snapback_post_nick_foldback_v1",
            },
            {
                "name": "pre_nick_duplex_visual_contract",
                "path": "views/pre_nick_duplex.snapback_visual.v1.json",
                "contract_kind": "snapback_visual_v1",
            },
            {
                "name": "post_nick_exposed_visual_contract",
                "path": "views/post_nick_exposed.snapback_visual.v1.json",
                "contract_kind": "snapback_visual_v1",
            },
            {
                "name": "post_nick_foldback_visual_contract",
                "path": "views/post_nick_foldback.snapback_visual.v1.json",
                "contract_kind": "snapback_visual_v1",
            },
        ],
        "recommended_jobs": [],
    }
    if include_jobs:
        payload["recommended_jobs"] = [
            {"name": "pre_nick_duplex", "path": "../baserender_jobs/pre_nick_duplex.job.yaml"},
            {"name": "post_nick_exposed", "path": "../baserender_jobs/post_nick_exposed.job.yaml"},
            {"name": "post_nick_foldback", "path": "../baserender_jobs/post_nick_foldback.job.yaml"},
        ]
    return SnapbackViewsManifestV1.model_validate(payload).model_dump(mode="json")


def build_single_view_job(*, input_filename: str, output_filename: str) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "json",
            "path": f"../views/{input_filename}",
            "adapter": {"kind": "snapback_visual_v1"},
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "snapback_map",
            "style": {
                "preset": "presentation_default",
                "overrides": {
                    "legend": False,
                    "figure_scale": 1.05,
                    "font_size_seq": 13,
                    "font_size_label": 10,
                    "padding_x": 32.0,
                    "padding_y": 22.0,
                },
            },
        },
        "outputs": [
            {
                "kind": "images",
                "path": f"../renders/{output_filename}",
                "fmt": "png",
            }
        ],
        "run": {
            "strict": True,
            "fail_on_skips": True,
            "emit_report": False,
        },
    }
