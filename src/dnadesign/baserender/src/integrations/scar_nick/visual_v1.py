"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/scar_nick/visual_v1.py

Adapter from scar-nick visual contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import ScarNickVisualV1

from ...core import ContractError, Record, SchemaError
from ...core.record import Display, Effect

_TYPE_IIS_EDGE = "#B59B00"
_NICKASE_EDGE = "#0072B2"
_TYPE_IIS_TEXT = "#7A6500"
_NICKASE_TEXT = "#005A8D"
_MISMATCH_CONNECTOR = "#111827"
_FRAGMENT_TEXT = "#94A3B8"
_PANEL_EDGE = "#CBD5E1"
_CELL_WIDTH_SCALE = 1.12


def _span_indices(start: int, end: int) -> list[int]:
    return list(range(start, end))


def _post_mismatch_indices(contract: ScarNickVisualV1) -> list[int]:
    if "mismatch_indices" in contract.meta:
        return sorted({int(index) for index in contract.meta["mismatch_indices"]})
    mismatch_offsets = [
        int(entry["position"])
        for entry in contract.pair_classes
        if str(entry.get("class_label") or "").upper() in {"W", "X"}
    ]
    indices: list[int] = []
    for panel in contract.panels:
        if panel.panel_id != "post_release":
            continue
        indices.extend(panel.retained_scar_span.start + offset for offset in mismatch_offsets)
    return sorted(set(indices))


def _pre_release_panel(contract: ScarNickVisualV1):
    for panel in contract.panels:
        if panel.panel_id == "pre_release":
            return panel
    raise SchemaError("scar_nick_visual_v1 contract is missing pre_release panel")


def _post_release_panel(contract: ScarNickVisualV1):
    for panel in contract.panels:
        if panel.panel_id == "post_release":
            return panel
    raise SchemaError("scar_nick_visual_v1 contract is missing post_release panel")


def _nickase_display_row(contract: ScarNickVisualV1) -> str:
    return contract.nickase.canonical_read_row


def _pre_release_site_highlights(contract: ScarNickVisualV1) -> dict[str, list[int]]:
    panel = _pre_release_panel(contract)
    primary = _span_indices(panel.release_site_span.start, panel.release_site_span.end)
    complement: list[int] = []
    nickase_indices = _span_indices(panel.nickase_site_span.start, panel.nickase_site_span.end)
    if _nickase_display_row(contract) == "primary":
        primary.extend(nickase_indices)
    else:
        complement.extend(nickase_indices)
    return {"primary": sorted(set(primary)), "complement": sorted(set(complement))}


def _pre_release_site_highlight_colors(contract: ScarNickVisualV1) -> dict[str, dict[int, str]]:
    panel = _pre_release_panel(contract)
    primary = {
        index: _TYPE_IIS_TEXT for index in _span_indices(panel.release_site_span.start, panel.release_site_span.end)
    }
    complement: dict[int, str] = {}
    nickase_indices = _span_indices(panel.nickase_site_span.start, panel.nickase_site_span.end)
    if _nickase_display_row(contract) == "primary":
        for index in nickase_indices:
            primary.setdefault(index, _NICKASE_TEXT)
    else:
        for index in nickase_indices:
            complement[index] = _NICKASE_TEXT
    return {"primary": primary, "complement": complement}


def _fragment_dim_indices(contract: ScarNickVisualV1) -> dict[str, list[int]]:
    dim: dict[str, list[int]] = {"primary": [], "complement": []}
    for panel in contract.panels:
        for span in panel.fragment_spans:
            dim[span.row].extend(_span_indices(span.start, span.end))
    return {row: sorted(set(indices)) for row, indices in dim.items()}


def _hidden_indices(contract: ScarNickVisualV1) -> dict[str, list[int]]:
    spacer = [int(index) for index in contract.meta.get("panel_spacer_indices", [])]
    return {"primary": spacer, "complement": spacer}


def _protected_site_edge_markers(contract: ScarNickVisualV1) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    for panel in contract.panels:
        if panel.panel_id != "pre_release":
            if panel.start > 0:
                markers.append(
                    {
                        "start": panel.start,
                        "end": panel.start,
                        "cover_rows": "both",
                        "color": _PANEL_EDGE,
                        "alpha": 0.58,
                        "linewidth": 0.55,
                    }
                )
            continue
        markers.extend(
            [
                {
                    "start": panel.release_site_span.start,
                    "end": panel.release_site_span.end,
                    "cover_rows": "both",
                    "color": _TYPE_IIS_EDGE,
                    "alpha": 0.52,
                    "linewidth": 0.55,
                },
                {
                    "start": panel.nickase_site_span.start,
                    "end": panel.nickase_site_span.end,
                    "cover_rows": "both",
                    "color": _NICKASE_EDGE,
                    "alpha": 0.50,
                    "linewidth": 0.55,
                },
            ]
        )
    return markers


def _segment_labels(contract: ScarNickVisualV1) -> list[dict[str, object]]:
    pre_panel = _pre_release_panel(contract)
    post_panel = _post_release_panel(contract)
    release_label = f"{contract.release_placement.variant_id} {contract.release_placement.recognition_sequence}"
    nickase_label = f"{contract.nickase.variant_id} {contract.nickase.canonical_motif_top_5to3}"
    nickase_row = _nickase_display_row(contract)
    labels: list[dict[str, object]] = [
        {
            "text": release_label,
            "start": pre_panel.release_site_span.start,
            "end": pre_panel.release_site_span.end,
            "row_id": "primary",
            "label_side": "above",
            "color": _TYPE_IIS_TEXT,
            "label_offset_px": -12.0,
        },
        {
            "text": nickase_label,
            "start": pre_panel.nickase_site_span.start,
            "end": pre_panel.nickase_site_span.end,
            "row_id": nickase_row,
            "label_side": "below",
            "color": _NICKASE_TEXT,
            "label_offset_px": 0.0,
        },
        {
            "text": release_label,
            "start": post_panel.release_site_span.start,
            "end": post_panel.release_site_span.end,
            "row_id": "primary",
            "label_side": "above",
            "color": _TYPE_IIS_TEXT,
            "label_offset_px": -12.0,
        },
    ]
    for fragment_span in post_panel.fragment_spans:
        labels.append(
            {
                "text": "Y adaptor",
                "start": fragment_span.start,
                "end": fragment_span.end,
                "row_id": fragment_span.row,
                "label_side": "below",
                "color": _FRAGMENT_TEXT,
                "label_offset_px": 0.0,
            }
        )
    return labels


def _panel_transition_arrows(contract: ScarNickVisualV1) -> list[dict[str, int]]:
    raw_arrows = contract.meta.get("panel_transition_arrows", ())
    if isinstance(raw_arrows, list) and raw_arrows:
        arrows: list[dict[str, int]] = []
        for raw in raw_arrows:
            if not isinstance(raw, Mapping):
                continue
            try:
                start = int(raw.get("start"))
                end = int(raw.get("end"))
            except Exception:
                continue
            if end > start:
                arrows.append({"start": start, "end": end})
        if arrows:
            return arrows
    pre_panel = _pre_release_panel(contract)
    post_panel = _post_release_panel(contract)
    if post_panel.start <= pre_panel.end:
        return []
    return [{"start": pre_panel.end, "end": post_panel.start}]


def _overlay_text(contract: ScarNickVisualV1, *, row_index: int) -> str:
    return str(contract.title or "")


@dataclass(frozen=True)
class ScarNickVisualV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = ScarNickVisualV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid scar_nick_visual_v1 contract at row {row_index}: {exc}") from exc

        tag_labels = {
            "owner:type_iis_release_site": "Type IIS restriction site",
            "owner:retained_type_iis_scar": "Retained Type IIS scar",
            "effect:nickase_footprint": "Nicking endonuclease footprint",
            "effect:annealed_adapter_fragment": "Y adaptor",
            "effect:degenerate_nucleotide": "Degenerate nucleotide",
        }

        nick_lane = "primary" if contract.nicked_strand == "top" else "complement"
        effects = [
            Effect(
                kind="boundary_marker",
                target={
                    "boundary": panel.nick_boundary,
                    "lane": nick_lane,
                },
                params={
                    "label": "",
                    "semantic": "nick",
                    "intent": "terminal_nick",
                },
                render={},
            )
            for panel in contract.panels
        ]
        span_backdrops = []
        for fill in contract.rectangular_fills:
            backdrop = {
                "semantic": fill.semantic,
                "start": fill.start,
                "end": fill.end,
                "fill": fill.fill,
                "alpha": fill.alpha,
                "corner_radius": fill.corner_radius,
                "cover_rows": fill.cover_rows,
            }
            if fill.edge_linewidth > 0.0 and fill.edge_color is not None:
                backdrop.update(
                    {
                        "edge_color": fill.edge_color,
                        "edge_alpha": fill.edge_alpha,
                        "edge_linewidth": fill.edge_linewidth,
                    }
                )
            span_backdrops.append(backdrop)
        mismatch_indices = _post_mismatch_indices(contract)
        segment_labels = _segment_labels(contract)
        record = Record(
            id=contract.state_id,
            alphabet=self.alphabet,
            sequence=contract.primary_sequence,
            features=(),
            effects=tuple(effects),
            display=Display(overlay_text=_overlay_text(contract, row_index=row_index), tag_labels=tag_labels),
            meta={
                "adapter": "scar_nick_visual_v1",
                "contract": contract.model_dump(mode="json"),
                "view_meta": dict(contract.meta),
                "complement_sequence": contract.complement_sequence,
                "row_labels": {
                    "primary": contract.primary_row_label,
                    "complement": contract.complement_row_label,
                },
                "span_backdrops": span_backdrops,
                "span_edge_markers": _protected_site_edge_markers(contract),
                "panel_transition_arrows": _panel_transition_arrows(contract),
                "segment_labels": segment_labels,
                "base_highlights": _pre_release_site_highlights(contract),
                "base_highlight_colors": _pre_release_site_highlight_colors(contract),
                "base_highlight_color": {
                    "primary": _TYPE_IIS_TEXT,
                    "complement": _NICKASE_TEXT,
                },
                "dim_base_indices": _fragment_dim_indices(contract),
                "base_dim_color": _FRAGMENT_TEXT,
                "base_hidden_indices": _hidden_indices(contract),
                "connector_hidden_indices": _hidden_indices(contract)["primary"],
                "connector_cross_indices": mismatch_indices,
                "connector_cross_color": _MISMATCH_CONNECTOR,
                "connector_cross_linewidth": 1.05,
                "connector_cross_alpha": 0.98,
                "cell_width_scale": _CELL_WIDTH_SCALE,
                "grid_max_rows": 5,
                "panel_spans": [panel.model_dump(mode="json") for panel in contract.panels],
                "show_reverse_complement": True,
                "scar_nick": {
                    "state_kind": contract.state_kind,
                    "event_scope": contract.event_scope,
                    "profile_order": "S3_S2_S1_S0",
                    "terminal_boundary": contract.terminal_boundary,
                    "nick_boundary": contract.nick_boundary,
                    "nick_state": contract.nick_state,
                    "left_base": contract.left_base,
                    "right_base": contract.right_base,
                    "nicked_strand": contract.nicked_strand,
                    "surviving_strand": contract.surviving_strand,
                    "profile_s3s2s1s0": contract.profile_s3s2s1s0,
                    "profile_payload_outward": contract.profile_payload_outward,
                    "type_iis_variant_id": contract.release_placement.variant_id,
                    "type_iis_recognition_sequence": contract.release_placement.recognition_sequence,
                    "type_iis_top_cut_boundary": contract.release_placement.top_cut_boundary,
                    "type_iis_bottom_cut_boundary": contract.release_placement.bottom_cut_boundary,
                    "nickase_variant_id": contract.nickase.variant_id,
                    "nickase_motif_top_5to3": contract.nickase.motif_top_5to3,
                    "nickase_canonical_motif_top_5to3": contract.nickase.canonical_motif_top_5to3,
                },
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc


__all__ = ["ScarNickVisualV1Adapter"]
