"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/view_contracts.py

QA view and BaseRender contract publication for scar-nick candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.contracts.visual import ScarNickVisualV1
from dnadesign.cruncher.scar_nick.models import ScarNickCandidate
from dnadesign.cruncher.scar_nick.view_models import ScarNickTerminalNickViewV1, ScarNickViewsManifestV1
from dnadesign.cruncher.scar_nick.visual_geometry import (
    ScarNickVisualContext,
    build_visual_context,
    complement_sequence,
    nickase_downstream_symbols,
    pairing_complement_sequence,
    protected_sequence_spans,
    recognition_nt,
    shift_optional_span,
    shift_span,
)

_TYPE_IIS_FILL = "#F0E442"
_OFFSET_FILL = "#FFF6B3"
_SCAR_FILL = "#009E73"
_NICKASE_FILL = "#56B4E9"
_PANEL_SPACER_NT = 4
_COMBINED_VISUAL_STATE_KIND = "pre_post_terminal_nick"
_VISUAL_JSONL_FILENAME = "scar_nick_terminal_nick.scar_nick_visual.v1.jsonl"
_JOB_RELATIVE_PATH = "../../baserender_jobs/scar_nick_terminal_nick.job.yaml"


@dataclass(frozen=True)
class ScarNickCandidateVisualBundle:
    terminal_nick_view: dict[str, Any]
    terminal_nick_visual_contract: dict[str, Any]
    terminal_nick_visual_contracts: list[dict[str, Any]]
    views_manifest: dict[str, Any]
    baserender_job: dict[str, Any]


def _state_title(candidate: ScarNickCandidate, state_kind: str) -> str:
    rank = f"{int(candidate.rank):02d}" if candidate.rank is not None else candidate.candidate_id
    left_right = f"L={candidate.left_base}/R={candidate.right_base}"
    return f"{rank} | {left_right} | {candidate.profile_s3s2s1s0}"


def _panel_coordinate_fields(
    *,
    context: ScarNickVisualContext,
    candidate: ScarNickCandidate,
    panel_id: str,
    title: str,
    start: int,
    state_kind: str,
    nick_state: str,
    fragment_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "title": title,
        "state_kind": state_kind,
        "nick_state": nick_state,
        "start": start,
        "end": start + len(context.primary_sequence_5to3),
        "terminal_boundary": candidate.terminal_boundary - context.context_start + start,
        "nick_boundary": candidate.nick_boundary - context.context_start + start,
        "retained_product_span": shift_span(context.retained_product_span, start),
        "release_site_span": shift_span(context.release_site_span, start),
        "type_iis_offset_span": shift_optional_span(context.type_iis_offset_span, start),
        "retained_scar_span": shift_span(context.retained_scar_span, start),
        "nickase_site_span": shift_span(context.nickase_site_span, start),
        "fragment_spans": [] if fragment_spans is None else fragment_spans,
    }


def _rectangular_fills_for_panel(panel: dict[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    fills = [
        {
            "fill_id": f"{prefix}_type_iis_release_site",
            "semantic": "type_iis_release_site",
            "start": panel["release_site_span"]["start"],
            "end": panel["release_site_span"]["end"],
            "cover_rows": "both",
            "fill": _TYPE_IIS_FILL,
            "alpha": 0.34,
            "corner_radius": 0.0,
        }
    ]
    if panel["type_iis_offset_span"] is not None:
        fills.append(
            {
                "fill_id": f"{prefix}_type_iis_offset_spacer",
                "semantic": "type_iis_offset_spacer",
                "start": panel["type_iis_offset_span"]["start"],
                "end": panel["type_iis_offset_span"]["end"],
                "cover_rows": "both",
                "fill": _OFFSET_FILL,
                "alpha": 0.28,
                "corner_radius": 0.0,
            }
        )
    fills.append(
        {
            "fill_id": f"{prefix}_retained_type_iis_scar",
            "semantic": "retained_type_iis_scar",
            "start": panel["retained_scar_span"]["start"],
            "end": panel["retained_scar_span"]["end"],
            "cover_rows": "both",
            "fill": _SCAR_FILL,
            "alpha": 0.36,
            "corner_radius": 0.0,
        }
    )
    if panel["panel_id"] == "pre_release":
        fills.append(
            {
                "fill_id": f"{prefix}_nickase_footprint",
                "semantic": "nickase_footprint",
                "start": panel["nickase_site_span"]["start"],
                "end": panel["nickase_site_span"]["end"],
                "cover_rows": "both",
                "fill": _NICKASE_FILL,
                "alpha": 0.24,
                "corner_radius": 0.0,
            }
        )
    return fills


def _mismatch_display_indices(*, panels: list[dict[str, Any]], pair_classes: list[dict[str, Any]]) -> list[int]:
    mismatch_offsets = [
        int(entry["position"]) for entry in pair_classes if str(entry.get("class_label") or "").upper() in {"W", "X"}
    ]
    indices: list[int] = []
    for panel in panels:
        if panel["panel_id"] != "post_release":
            continue
        scar_start = int(panel["retained_scar_span"]["start"])
        indices.extend(scar_start + offset for offset in mismatch_offsets)
    return sorted(set(indices))


def _release_placement_payload(candidate: ScarNickCandidate) -> dict[str, Any] | None:
    if candidate.release_placement is None:
        return None
    release = candidate.release_placement
    return {
        "variant_id": release.variant_id,
        "orientation": release.orientation,
        "recognition_sequence": release.recognition_sequence,
        "source_catalog_id": release.source_catalog_id,
        "source_url": release.source_url,
        "commercial_confidence": release.commercial_confidence,
        "warning_codes": list(release.warning_codes),
        "recognition_site_start": release.recognition_site_start,
        "recognition_site_end": release.recognition_site_end,
        "top_cut_boundary": release.top_cut_boundary,
        "bottom_cut_boundary": release.bottom_cut_boundary,
        "retained_scar_start": release.retained_scar_start,
        "retained_scar_end": release.retained_scar_end,
        "retained_scar_nt": release.retained_scar_nt,
        "recognition_site_excised": release.recognition_site_excised,
    }


def _nickase_payload(candidate: ScarNickCandidate) -> dict[str, Any] | None:
    if candidate.nickase_placement is None:
        return None
    placement = candidate.nickase_placement
    canonical_read_row = "complement" if placement.orientation == "reverse" else "primary"
    return {
        "variant_id": placement.variant_id,
        "specificity_id": placement.specificity_id,
        "orientation": placement.orientation,
        "canonical_read_row": canonical_read_row,
        "site": candidate.nickase_site,
        "motif_top_5to3": placement.motif_top_5to3,
        "recognition_nt": recognition_nt(placement.motif_top_5to3),
        "vendor": placement.vendor,
        "source_url": placement.source_url,
        "source_family": placement.source_family,
        "commercial_confidence": placement.commercial_confidence,
        "warning_codes": list(placement.warning_codes),
        "source_site_start": placement.source_site_start,
        "source_site_end": placement.source_site_end,
        "strand": placement.strand,
        "boundary": placement.boundary,
        "terminal_boundary": placement.terminal_boundary,
        "exact_terminal": placement.exact_terminal,
    }


def _base_view_payload(
    *,
    candidate: ScarNickCandidate,
    solution_id: str,
    state_kind: str,
) -> dict[str, Any]:
    context = build_visual_context(candidate)
    sequence = context.primary_sequence_5to3
    payload = {
        "version": 1,
        "kind": "scar_nick_terminal_nick_v1",
        "view_id": f"{solution_id}.{state_kind}",
        "solution_id": solution_id,
        "candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "title": _state_title(candidate, state_kind),
        "state_kind": state_kind,
        "event_scope": "terminal_nick",
        "coordinate_semantics": "half_open_zero_based_v1",
        "boundary_semantics": "closed_zero_based_boundary_v1",
        "primary_sequence_5to3": sequence,
        "complement_sequence_3to5": complement_sequence(sequence),
        "terminal_boundary": candidate.terminal_boundary - context.context_start,
        "nick_boundary": candidate.nick_boundary - context.context_start,
        "retained_product_span": context.retained_product_span,
        "release_site_span": context.release_site_span,
        "type_iis_offset_span": context.type_iis_offset_span,
        "retained_scar_span": context.retained_scar_span,
        "junction_partner_span": None,
        "nickase_site_span": context.nickase_site_span,
        "nickase_site_source_span": context.nickase_site_source_span,
        "nickase_site_span_clipped": False,
        "nick_state": "intact" if state_kind == "pre_terminal_nick" else "nicked",
        "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
        "profile_payload_outward": candidate.profile_payload_outward,
        "pair_classes": [entry.model_dump(mode="json") for entry in candidate.pair_classes],
        "release_placement": _release_placement_payload(candidate),
        "nickase_placement": _nickase_payload(candidate),
        "meta": {
            "processing_event_scope": "terminal_nick",
            "release_site_role": "excised_provenance",
            "retained_scar": candidate.retained_scar,
            "left_base": candidate.left_base,
            "right_base": candidate.right_base,
            "visual_context_start": context.context_start,
            "visual_context_end": context.context_end,
            "raw_terminal_boundary": candidate.terminal_boundary,
            "raw_nick_boundary": candidate.nick_boundary,
            "raw_retained_product_span": {"start": 0, "end": len(candidate.retained_product_sequence)},
            "right_base_role": "profile_reference_only_not_linear_downstream_sequence",
            "protected_sequence_spans": protected_sequence_spans(candidate, context),
            "nickase_downstream_symbols": nickase_downstream_symbols(candidate, context),
            "reference_distances": candidate.reference_distances,
            "reference_control_distance": candidate.reference_control_distance,
            "gc_fraction": candidate.gc_fraction,
            "nicked_strand": candidate.nicked_strand,
            "surviving_strand": candidate.surviving_strand,
            "retained_scar_source": candidate.retained_scar_source,
        },
    }
    return ScarNickTerminalNickViewV1.model_validate(payload).model_dump(mode="json")


def build_terminal_nick_view(
    *,
    candidate: ScarNickCandidate,
    solution_id: str,
    state_kind: str,
) -> dict[str, Any]:
    return _base_view_payload(candidate=candidate, solution_id=solution_id, state_kind=state_kind)


def build_terminal_nick_visual_contract(
    *,
    candidate: ScarNickCandidate,
    solution_id: str,
    state_kind: str,
) -> dict[str, Any]:
    if state_kind != _COMBINED_VISUAL_STATE_KIND:
        raise ValueError(f"scar-nick visual contracts require state_kind={_COMBINED_VISUAL_STATE_KIND!r}")
    if candidate.nicked_strand is None or candidate.surviving_strand is None:
        raise ValueError("scar-nick visual requires explicit nicked_strand and surviving_strand")

    view = build_terminal_nick_view(candidate=candidate, solution_id=solution_id, state_kind="post_terminal_nick")
    context = build_visual_context(candidate)
    panel_sequence = context.primary_sequence_5to3
    pre_panel_complement = complement_sequence(panel_sequence)
    post_panel_complement = pairing_complement_sequence(sequence=panel_sequence, context=context, candidate=candidate)
    aligned_right_display = "".join(pair.aligned_right_base for pair in candidate.pair_classes)
    raw_right_display = candidate.right_base[::-1]
    spacer = "N" * _PANEL_SPACER_NT
    post_offset = len(panel_sequence) + len(spacer)
    primary_sequence = panel_sequence + spacer + panel_sequence
    complement = pre_panel_complement + spacer + post_panel_complement

    post_fragment_row = "primary" if candidate.nicked_strand == "top" else "complement"
    post_fragment_span = {
        "row": post_fragment_row,
        "start": post_offset,
        "end": post_offset + context.retained_scar_span["start"],
    }
    panels = [
        _panel_coordinate_fields(
            context=context,
            candidate=candidate,
            panel_id="pre_release",
            title="before terminal nick",
            start=0,
            state_kind="pre_terminal_nick",
            nick_state="intact",
        ),
        _panel_coordinate_fields(
            context=context,
            candidate=candidate,
            panel_id="post_release",
            title="after terminal nick",
            start=post_offset,
            state_kind="post_terminal_nick",
            nick_state="nicked",
            fragment_spans=[post_fragment_span],
        ),
    ]
    rectangular_fills: list[dict[str, Any]] = []
    for panel in panels:
        rectangular_fills.extend(_rectangular_fills_for_panel(panel, prefix=panel["panel_id"]))

    nickase_payload = _nickase_payload(candidate)
    post_panel = panels[1]
    if nickase_payload is not None:
        nickase_payload = {
            **nickase_payload,
            "display_boundary": post_panel["nick_boundary"],
            "display_site_span": post_panel["nickase_site_span"],
        }
    mismatch_indices = _mismatch_display_indices(panels=panels, pair_classes=view["pair_classes"])
    spacer_indices = list(range(len(panel_sequence), post_offset))
    payload = {
        "contract_kind": "scar_nick_visual_v1",
        "state_id": f"{solution_id}.{state_kind}",
        "state_kind": state_kind,
        "event_scope": "terminal_nick",
        "alphabet": "iupac_dna",
        "title": view["title"],
        "primary_sequence": primary_sequence,
        "complement_sequence": complement,
        "primary_row_label": "Top",
        "complement_row_label": "Bottom",
        "terminal_boundary": post_panel["terminal_boundary"],
        "nick_boundary": post_panel["nick_boundary"],
        "retained_product_span": post_panel["retained_product_span"],
        "release_site_span": post_panel["release_site_span"],
        "type_iis_offset_span": post_panel["type_iis_offset_span"],
        "retained_scar_span": post_panel["retained_scar_span"],
        "junction_partner_span": view["junction_partner_span"],
        "nickase_site_span": post_panel["nickase_site_span"],
        "nickase_site_source_span": view["nickase_site_source_span"],
        "nickase_site_span_clipped": view["nickase_site_span_clipped"],
        "nick_state": "pre_post",
        "retained_scar": candidate.retained_scar,
        "left_base": candidate.left_base,
        "right_base": candidate.right_base,
        "nicked_strand": candidate.nicked_strand,
        "surviving_strand": candidate.surviving_strand,
        "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
        "profile_payload_outward": candidate.profile_payload_outward,
        "pair_classes": view["pair_classes"],
        "panels": panels,
        "rectangular_fills": rectangular_fills,
        "release_placement": view["release_placement"],
        "nickase": nickase_payload,
        "meta": {
            **view["meta"],
            "visual_state_kind": state_kind,
            "display_panels": panels,
            "profile_order": candidate.profile_order,
            "type_iis_label": (
                f"{candidate.release_placement.variant_id} {candidate.release_placement.recognition_sequence}"
                if candidate.release_placement is not None
                else ""
            ),
            "nickase_label": (
                f"{nickase_payload['variant_id']} {nickase_payload['motif_top_5to3']}"
                if nickase_payload is not None
                else ""
            ),
            "junction_label": "",
            "panel_spacer_indices": spacer_indices,
            "panel_transition_arrows": [{"start": panels[0]["end"], "end": panels[1]["start"]}],
            "fragment_spans": [post_fragment_span],
            "mismatch_indices": mismatch_indices,
            "right_base_display_order": raw_right_display,
            "right_base_raw_display_order": raw_right_display,
            "aligned_right_base_display_order": aligned_right_display,
        },
    }
    return ScarNickVisualV1.model_validate(payload).model_dump(mode="json")


def build_views_manifest(*, solution_id: str, include_jobs: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "kind": "scar_nick_views_manifest_v1",
        "solution_id": solution_id,
        "views": [
            {
                "name": "terminal_nick_view",
                "path": "analysis/views/post_terminal_nick.v1.json",
                "contract_kind": "scar_nick_terminal_nick_v1",
            },
            {
                "name": "terminal_nick_visual_contract",
                "path": "analysis/views/post_terminal_nick.scar_nick_visual.v1.json",
                "contract_kind": "scar_nick_visual_v1",
            },
            {
                "name": "scar_nick_terminal_nick_visual_contracts",
                "path": f"analysis/views/{_VISUAL_JSONL_FILENAME}",
                "contract_kind": "scar_nick_visual_v1",
            },
        ],
        "recommended_jobs": [],
        "meta": {"visual_event_scope": "terminal_nick"},
    }
    if include_jobs:
        payload["recommended_jobs"] = [{"name": "scar_nick_terminal_nick", "path": _JOB_RELATIVE_PATH}]
    return ScarNickViewsManifestV1.model_validate(payload).model_dump(mode="json")


def build_terminal_nick_job(*, output_format: str = "png") -> dict[str, Any]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "jsonl",
            "path": f"../analysis/views/{_VISUAL_JSONL_FILENAME}",
            "adapter": {"kind": "scar_nick_visual_v1"},
            "alphabet": "IUPAC_DNA",
        },
        "render": {
            "renderer": "nucleotide_evidence_map",
            "style": {
                "preset": "presentation_default",
                "overrides": {
                    "legend": False,
                    "figure_scale": 1.0,
                    "font_mono": "DejaVu Sans Mono",
                    "font_label": "DejaVu Sans Mono",
                    "font_size_seq": 12,
                    "font_size_label": 8,
                    "font_size_span_link_label": 8,
                    "padding_x": 34.0,
                    "padding_y": 48.0,
                    "baseline_spacing": 48.0,
                    "overlay_align": "center",
                    "overlay_title_color": "#4B5563",
                },
            },
        },
        "outputs": [
            {
                "kind": "images",
                "path": f"../plots/scar_nick_terminal_nick.{output_format}",
                "fmt": output_format,
            }
        ],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


def build_candidate_visual_bundle(
    *,
    candidate: ScarNickCandidate,
    solution_id: str,
    visual_contracts: list[dict[str, Any]] | None = None,
) -> ScarNickCandidateVisualBundle:
    terminal_view = build_terminal_nick_view(
        candidate=candidate,
        solution_id=solution_id,
        state_kind="post_terminal_nick",
    )
    terminal_visual = build_terminal_nick_visual_contract(
        candidate=candidate,
        solution_id=solution_id,
        state_kind=_COMBINED_VISUAL_STATE_KIND,
    )
    return ScarNickCandidateVisualBundle(
        terminal_nick_view=terminal_view,
        terminal_nick_visual_contract=terminal_visual,
        terminal_nick_visual_contracts=[terminal_visual] if visual_contracts is None else visual_contracts,
        views_manifest=build_views_manifest(solution_id=solution_id),
        baserender_job=build_terminal_nick_job(),
    )


__all__ = [
    "ScarNickCandidateVisualBundle",
    "build_candidate_visual_bundle",
    "build_terminal_nick_job",
    "build_terminal_nick_view",
    "build_terminal_nick_visual_contract",
    "build_views_manifest",
    "complement_sequence",
]
