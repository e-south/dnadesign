"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/view_contracts.py

QA view publication for v2 snapback designs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport
from dnadesign.cruncher.snapback.publication_support import (
    absolute_foldback_partner_mismatch_positions,
    absolute_primary_mismatch_positions,
    complement_sequence,
    protected_overlap_span,
)
from dnadesign.cruncher.snapback.view_models import (
    SnapbackPostNickExposedViewV1,
    SnapbackPostNickFoldbackViewV1,
    SnapbackPreNickDuplexViewV1,
)


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
                "sequence": complement_sequence(candidate.designed_sequence),
            },
        },
        "nick_boundary": candidate.nick_boundary,
        "ligation_junction_boundary": candidate.nick_boundary,
        "protected_region": candidate.protected_region.model_dump(mode="json"),
        "pre_nick_duplex_window": candidate.pre_nick_duplex_window.model_dump(mode="json"),
        "retained_homology_window": candidate.retained_homology_window.model_dump(mode="json"),
        "source_cap_window": candidate.source_cap_window.model_dump(mode="json"),
        "effective_cap_window": {
            "start": candidate.source_cap_window.start,
            "end": candidate.cap_span.end,
        },
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
            "cap_extension_nt": candidate.cap_extension_nt,
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
                "label": "Released top strand 5' -> 3'",
                "direction": "5to3",
                "sequence": candidate.designed_sequence,
            },
            "complement": {
                "label": "Active strand 3' -> 5'",
                "direction": "3to5",
                "sequence": complement_sequence(candidate.designed_sequence),
            },
        },
        "nick_boundary": candidate.nick_boundary,
        "ligation_junction_boundary": candidate.nick_boundary,
        "topology": {
            "anchored_top_span": {"start": 0, "end": candidate.nick_boundary},
            "released_top_span": {"start": candidate.nick_boundary, "end": len(candidate.designed_sequence)},
            "released_prefix_span": {
                "start": candidate.nick_boundary,
                "end": candidate.nick_boundary,
            },
            "retained_homology_span": candidate.retained_homology_window.model_dump(mode="json"),
            "source_cap_span": candidate.source_cap_window.model_dump(mode="json"),
            "cap_extension_span": candidate.cap_span.model_dump(mode="json"),
            "cap_span": {
                "start": candidate.source_cap_window.start,
                "end": candidate.cap_span.end,
            },
            "foldback_arm_span": candidate.foldback_arm_span.model_dump(mode="json"),
        },
        "intended_site": candidate.intended_site.model_dump(mode="json"),
        "intended_nick": candidate.intended_nick.model_dump(mode="json"),
        "meta": {
            "released_prefix_sequence": candidate.released_prefix_sequence,
            "retained_homology_sequence": candidate.retained_homology_sequence,
            "source_cap_sequence": candidate.source_cap_sequence,
            "effective_cap_sequence": candidate.effective_cap_sequence,
            "cap_sequence": candidate.cap_sequence,
            "foldback_arm": candidate.foldback_arm,
            "released_prefix_nt": candidate.released_prefix_nt,
            "retained_start_from_nick": candidate.retained_start_from_nick,
            "cap_nt": candidate.cap_nt,
            "cap_extension_nt": candidate.cap_extension_nt,
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
            "source_cap_span": candidate.post_nick_source_cap_span.model_dump(mode="json"),
            "cap_extension_span": candidate.post_nick_cap_extension_span.model_dump(mode="json"),
            "cap_span": candidate.post_nick_cap_span.model_dump(mode="json"),
            "foldback_arm_span": candidate.post_nick_foldback_arm_span.model_dump(mode="json"),
            "protected_overlap_span": protected_overlap_span(candidate=candidate),
        },
        "pair_map": [pair.model_dump(mode="json") for pair in candidate.pair_map],
        "primary_mismatch_positions": absolute_primary_mismatch_positions(candidate),
        "foldback_partner_mismatch_positions": absolute_foldback_partner_mismatch_positions(candidate),
        "meta": {
            "protected_region": candidate.protected_region.model_dump(mode="json"),
            "released_prefix_sequence": candidate.released_prefix_sequence,
            "retained_homology_sequence": candidate.retained_homology_sequence,
            "source_cap_sequence": candidate.source_cap_sequence,
            "effective_cap_sequence": candidate.effective_cap_sequence,
            "cap_sequence": candidate.cap_sequence,
            "foldback_arm": candidate.foldback_arm,
            "cap_nt": candidate.cap_nt,
            "cap_extension_nt": candidate.cap_extension_nt,
            "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
            "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
        },
    }
    return SnapbackPostNickFoldbackViewV1.model_validate(payload).model_dump(mode="json")
