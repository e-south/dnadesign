"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/public_visuals.py

Public snapback visual-contract publication for renderer-facing artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.contracts.visual import SnapbackVisualV1
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport
from dnadesign.cruncher.snapback.publication_support import (
    absolute_foldback_partner_mismatch_positions,
    absolute_primary_mismatch_positions,
    complement_sequence,
    effective_cap_span,
    foldback_loop_geometry,
    protected_overlap_span,
)


def _require_candidate(report: SnapbackEvaluationReport):
    candidate = report.candidate
    if candidate is None:
        raise ValueError("Snapback public visual publication requires a satisfied report with candidate details.")
    return candidate


def build_pre_nick_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = _require_candidate(report)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.pre_nick_duplex",
            "state_kind": "pre_nick_duplex",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.designed_sequence,
            "complement_sequence": complement_sequence(candidate.designed_sequence),
            "primary_row_label": "Top",
            "complement_row_label": "Partner",
            "nick_boundary": candidate.nick_boundary,
            "ligation_junction_boundary": candidate.nick_boundary,
            "protected_region_span": candidate.protected_region.model_dump(mode="json"),
            "pre_nick_duplex_window_span": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            "intended_site_span": {
                "start": candidate.intended_site.start,
                "end": candidate.intended_site.end,
            },
            "released_prefix_span": None,
            "retained_stem_span": candidate.retained_homology_window.model_dump(mode="json"),
            "released_suffix_span": None,
            "cap_span": effective_cap_span(candidate),
            "foldback_revcomp_span": candidate.foldback_arm_span.model_dump(mode="json"),
            "pairings": [],
            "meta": {
                "source_view_kind": "snapback_pre_nick_duplex_v1",
                "nick_boundary_from_left": candidate.nick_boundary_from_left,
                "retained_start_from_nick": candidate.retained_start_from_nick,
                "added_nt": candidate.added_nt,
                "cap_nt": candidate.cap_nt,
                "cap_extension_nt": candidate.cap_extension_nt,
            },
        }
    ).model_dump(mode="json")


def build_post_nick_exposed_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = _require_candidate(report)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.post_nick_exposed",
            "state_kind": "post_nick_exposed",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.designed_sequence,
            "complement_sequence": complement_sequence(candidate.designed_sequence),
            "primary_row_label": "Released top",
            "complement_row_label": "Active strand",
            "nick_boundary": candidate.nick_boundary,
            "ligation_junction_boundary": candidate.nick_boundary,
            "protected_region_span": candidate.protected_region.model_dump(mode="json"),
            "pre_nick_duplex_window_span": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            "intended_site_span": {
                "start": candidate.intended_site.start,
                "end": candidate.intended_site.end,
            },
            "anchored_duplex_span": {"start": 0, "end": candidate.nick_boundary},
            "released_prefix_span": None,
            "retained_stem_span": candidate.retained_homology_window.model_dump(mode="json"),
            "released_suffix_span": None,
            "cap_span": effective_cap_span(candidate),
            "foldback_revcomp_span": candidate.foldback_arm_span.model_dump(mode="json"),
            "exposed_complement_span": {"start": candidate.nick_boundary, "end": len(candidate.designed_sequence)},
            "pairings": [],
            "meta": {
                "source_view_kind": "snapback_post_nick_exposed_v1",
                "released_prefix_sequence": candidate.released_prefix_sequence,
                "source_cap_sequence": candidate.source_cap_sequence,
                "effective_cap_sequence": candidate.effective_cap_sequence,
                "cap_nt": candidate.cap_nt,
                "cap_extension_nt": candidate.cap_extension_nt,
            },
        }
    ).model_dump(mode="json")


def build_post_nick_foldback_snapback_visual(
    *,
    report: SnapbackEvaluationReport,
    solution_id: str,
    title: str,
) -> dict[str, Any]:
    candidate = _require_candidate(report)
    return SnapbackVisualV1.model_validate(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": f"{solution_id}.post_nick_foldback",
            "state_kind": "post_nick_foldback",
            "alphabet": "dna",
            "title": title,
            "primary_sequence": candidate.post_nick_sequence,
            "complement_sequence": complement_sequence(candidate.post_nick_sequence),
            "primary_row_label": "Retained stem",
            "complement_row_label": "Foldback arm",
            "ligation_junction_boundary": candidate.post_nick_retained_homology_span.start,
            "protected_region_span": protected_overlap_span(candidate=candidate),
            "released_prefix_span": None,
            "retained_stem_span": candidate.post_nick_retained_homology_span.model_dump(mode="json"),
            "cap_span": candidate.post_nick_cap_span.model_dump(mode="json"),
            "foldback_revcomp_span": candidate.post_nick_foldback_arm_span.model_dump(mode="json"),
            "loop_geometry": foldback_loop_geometry(candidate),
            "pairings": [{"left_index": pair.left, "right_index": pair.right} for pair in candidate.pair_map],
            "primary_mismatch_positions": absolute_primary_mismatch_positions(candidate),
            "complement_mismatch_positions": absolute_foldback_partner_mismatch_positions(candidate),
            "meta": {
                "source_view_kind": "snapback_post_nick_foldback_v1",
                "source_nick_boundary": candidate.nick_boundary,
                "source_cap_sequence": candidate.source_cap_sequence,
                "effective_cap_sequence": candidate.effective_cap_sequence,
                "cap_nt": candidate.cap_nt,
                "cap_extension_nt": candidate.cap_extension_nt,
                "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
                "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
            },
        }
    ).model_dump(mode="json")
