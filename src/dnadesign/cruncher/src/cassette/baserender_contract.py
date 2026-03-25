"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/baserender_contract.py

Data-only baserender handoff contracts for cassette solve hits.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from dnadesign.cruncher.cassette.selection import SelectedCandidate
from dnadesign.cruncher.cassette.solve_models import SolveReport


def _strand(value: Literal["forward", "reverse"] | None) -> str | None:
    if value == "forward":
        return "fwd"
    if value == "reverse":
        return "rev"
    return None


def _feature(
    *,
    feature_id: str,
    kind: str,
    start: int,
    end: int,
    label: str,
    strand: str | None = None,
    attrs: dict[str, object] | None = None,
) -> dict[str, object]:
    feature: dict[str, object] = {
        "id": feature_id,
        "kind": kind,
        "span": {
            "start": start,
            "end": end,
        },
        "label": label,
        "attrs": attrs or {},
        "render": {},
        "tags": [],
    }
    if strand is not None:
        feature["span"]["strand"] = strand
    return feature


def _boundary_span(boundary: int, *, sequence_length: int) -> tuple[int, int]:
    if sequence_length <= 1:
        return (0, 1)
    if boundary <= 0:
        return (0, 1)
    if boundary >= sequence_length:
        return (sequence_length - 1, sequence_length)
    return (boundary - 1, boundary)


def _record_payload(*, rank: int, selected_hit: SelectedCandidate, selection_policy: str | None) -> dict[str, object]:
    explicit_report = selected_hit.record.report
    candidate = getattr(explicit_report, "candidate", None)
    if candidate is None:
        raise ValueError("Cassette solve baserender contract requires satisfied hit reports with candidates.")
    sequence = str(candidate.evaluation_primary_sequence)
    cassette_start = int(candidate.context_offset)
    cassette_end = cassette_start + int(candidate.cassette_length_nt)
    left_nick_start, left_nick_end = _boundary_span(
        int(candidate.intended_left_nick.boundary),
        sequence_length=len(sequence),
    )
    right_nick_start, right_nick_end = _boundary_span(
        int(candidate.intended_right_nick.boundary),
        sequence_length=len(sequence),
    )

    features = [
        _feature(
            feature_id="cassette_region",
            kind="cassette_region",
            start=cassette_start,
            end=cassette_end,
            label="cassette",
        ),
        _feature(
            feature_id="stem5p",
            kind="cassette_stem_5p",
            start=cassette_start + int(candidate.stem5p_span.start),
            end=cassette_start + int(candidate.stem5p_span.end),
            label="stem5p",
        ),
        _feature(
            feature_id="loop",
            kind="cassette_loop",
            start=cassette_start + int(candidate.loop_span.start),
            end=cassette_start + int(candidate.loop_span.end),
            label="loop",
        ),
        _feature(
            feature_id="stem3p",
            kind="cassette_stem_3p",
            start=cassette_start + int(candidate.stem3p_span.start),
            end=cassette_start + int(candidate.stem3p_span.end),
            label="stem3p",
        ),
        _feature(
            feature_id="intended_left_site",
            kind="cassette_site_left",
            start=int(candidate.intended_left_site.start),
            end=int(candidate.intended_left_site.end),
            label=selected_hit.record.left_variant_id,
            strand=_strand(candidate.intended_left_site.orientation),
            attrs={"specificity_id": str(candidate.intended_left_site.specificity_id)},
        ),
        _feature(
            feature_id="intended_right_site",
            kind="cassette_site_right",
            start=int(candidate.intended_right_site.start),
            end=int(candidate.intended_right_site.end),
            label=selected_hit.record.right_variant_id,
            strand=_strand(candidate.intended_right_site.orientation),
            attrs={"specificity_id": str(candidate.intended_right_site.specificity_id)},
        ),
        _feature(
            feature_id="bounded_segment",
            kind="cassette_bounded_segment",
            start=int(candidate.bounded_nicked_segment.start_boundary),
            end=int(candidate.bounded_nicked_segment.end_boundary),
            label="bounded segment",
            attrs={"length_nt": int(candidate.bounded_nicked_segment.length_nt)},
        ),
        _feature(
            feature_id="left_nick",
            kind="cassette_nick_left",
            start=left_nick_start,
            end=left_nick_end,
            label=f"left nick @{candidate.intended_left_nick.boundary}",
            attrs={"strand": str(candidate.intended_left_nick.strand)},
        ),
        _feature(
            feature_id="right_nick",
            kind="cassette_nick_right",
            start=right_nick_start,
            end=right_nick_end,
            label=f"right nick @{candidate.intended_right_nick.boundary}",
            attrs={"strand": str(candidate.intended_right_nick.strand)},
        ),
    ]

    return {
        "id": selected_hit.record.hit_id,
        "sequence": sequence,
        "features": features,
        "effects": [],
        "display": {
            "overlay_text": (
                f"rank {rank}: {selected_hit.record.hit_id} "
                f"{selected_hit.record.left_variant_id}->{selected_hit.record.right_variant_id}"
            ),
            "tag_labels": {
                "selection_policy": selection_policy or "unknown",
                "target_strand": str(candidate.target_strand),
            },
        },
        "meta": {
            "workflow": "cassette_solve",
            "hit_id": selected_hit.record.hit_id,
            "rank": rank,
            "selection_policy": selection_policy,
            "selection_rank_reason": selected_hit.selection_rank_reason,
            "distance_to_previous_selected": selected_hit.distance_to_previous_selected,
            "left_variant_id": selected_hit.record.left_variant_id,
            "right_variant_id": selected_hit.record.right_variant_id,
            "left_nick_boundary": selected_hit.record.left_nick_boundary,
            "right_nick_boundary": selected_hit.record.right_nick_boundary,
            "score": list(selected_hit.record.score_tuple),
            "base_penalty_vector": list(selected_hit.record.base_penalty_vector),
            "gc_fraction": selected_hit.record.gc_fraction,
            "bounded_segment_length": selected_hit.record.bounded_segment_length,
            "cassette_views": getattr(explicit_report, "render_contract", None),
        },
    }


def build_solve_baserender_hits_contract(
    *,
    report: SolveReport,
    selected_hits: Sequence[SelectedCandidate],
) -> dict[str, Any]:
    selection_summary = (
        report.selection_summary.model_dump(mode="json") if report.selection_summary is not None else None
    )
    return {
        "schema_version": 1,
        "workflow": "cassette_solve_baserender",
        "contract_kind": "baserender_record_bundle_v1",
        "solve_id": report.solve_id,
        "source": {
            "run_dir": report.run_dir,
            "solve_report_json": "solve_report.json",
            "table_hits_csv": "table__hits.csv",
            "selection_summary": selection_summary,
        },
        "adapter": {
            "kind": "generic_features",
            "alphabet": "DNA",
            "columns": {
                "id": "id",
                "sequence": "sequence",
                "features": "features",
                "effects": "effects",
                "display": "display",
            },
        },
        "render": {
            "renderer_name": "sequence_rows",
            "recommended_public_api": {
                "module": "dnadesign.baserender",
                "function": "render_record_grid_figure",
            },
        },
        "records": [
            _record_payload(
                rank=index,
                selected_hit=selected_hit,
                selection_policy=report.selection_summary.policy if report.selection_summary is not None else None,
            )
            for index, selected_hit in enumerate(selected_hits, start=1)
        ],
    }
