"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/sequence_evidence_map_v1.py

Adapter from shared sequence-evidence contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import SequenceEvidenceMapV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


def _strand(row_id: str) -> str:
    return "fwd" if row_id == "primary" else "rev"


def _style_token_for_owner(owner_id: str) -> str:
    if "payload" in owner_id:
        return "segment_payload"
    if "adapter" in owner_id:
        return "segment_adapter"
    if "primer" in owner_id:
        return "segment_primer"
    if "retained" in owner_id:
        return "segment_retained"
    if "sacrificial" in owner_id:
        return "segment_sacrificial"
    return "segment"


def _style_token_for_tag(tag_kind: str) -> str:
    if "overhang" in tag_kind:
        return "site_overhang"
    if "recognition" in tag_kind:
        return "site_recognition"
    if "primer" in tag_kind:
        return "site_primer"
    if "adapter" in tag_kind:
        return "site_adapter"
    if "boundary" in tag_kind or "junction" in tag_kind:
        return "site_boundary"
    return "site_effect"


def _normalize_base_highlights(
    meta: Mapping[str, Any], *, primary_length: int, complement_length: int
) -> dict[str, tuple[int, ...]]:
    raw = meta.get("base_highlights")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SchemaError("sequence_evidence_map_v1 meta.base_highlights must be a mapping")
    normalized: dict[str, tuple[int, ...]] = {}
    expected = {
        "primary": primary_length,
        "complement": complement_length,
    }
    for row_id, limit in expected.items():
        values = raw.get(row_id)
        if values is None:
            continue
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise SchemaError(f"sequence_evidence_map_v1 meta.base_highlights.{row_id} must be a list of indices")
        try:
            indices = tuple(int(value) for value in values)
        except Exception as exc:
            raise SchemaError(
                f"sequence_evidence_map_v1 meta.base_highlights.{row_id} must contain integer indices"
            ) from exc
        if any(index < 0 or index >= limit for index in indices):
            raise SchemaError(
                f"sequence_evidence_map_v1 meta.base_highlights.{row_id} indices must be within row bounds"
            )
        normalized[row_id] = indices
    return normalized


def _normalize_row_index_map(
    meta: Mapping[str, Any],
    *,
    key: str,
    primary_length: int,
    complement_length: int,
) -> dict[str, tuple[int, ...]]:
    raw = meta.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SchemaError(f"sequence_evidence_map_v1 meta.{key} must be a mapping")
    normalized: dict[str, tuple[int, ...]] = {}
    expected = {
        "primary": primary_length,
        "complement": complement_length,
    }
    for row_id, limit in expected.items():
        values = raw.get(row_id)
        if values is None:
            continue
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise SchemaError(f"sequence_evidence_map_v1 meta.{key}.{row_id} must be a list of indices")
        try:
            indices = tuple(int(value) for value in values)
        except Exception as exc:
            raise SchemaError(f"sequence_evidence_map_v1 meta.{key}.{row_id} must contain integer indices") from exc
        if any(index < 0 or index >= limit for index in indices):
            raise SchemaError(f"sequence_evidence_map_v1 meta.{key}.{row_id} indices must be within row bounds")
        normalized[row_id] = indices
    return normalized


def _normalize_row_color_map(meta: Mapping[str, Any], *, key: str) -> dict[str, str]:
    raw = meta.get(key)
    if raw is None:
        return {}
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            raise SchemaError(f"sequence_evidence_map_v1 meta.{key} must be non-empty when provided")
        return {"primary": value, "complement": value}
    if not isinstance(raw, Mapping):
        raise SchemaError(f"sequence_evidence_map_v1 meta.{key} must be a string or a row-id mapping")
    normalized: dict[str, str] = {}
    for row_id in ("primary", "complement"):
        value = raw.get(row_id)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            raise SchemaError(f"sequence_evidence_map_v1 meta.{key}.{row_id} must be non-empty when provided")
        normalized[row_id] = text
    return normalized


def _normalize_index_list(meta: Mapping[str, Any], key: str, *, limit: int) -> tuple[int, ...]:
    raw = meta.get(key, ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise SchemaError(f"sequence_evidence_map_v1 meta.{key} must be a list of indices")
    try:
        values = tuple(int(value) for value in raw)
    except Exception as exc:
        raise SchemaError(f"sequence_evidence_map_v1 meta.{key} must contain integer indices") from exc
    if any(value < 0 or value >= limit for value in values):
        raise SchemaError(f"sequence_evidence_map_v1 meta.{key} indices must be within row bounds")
    return values


def _normalize_connector_overhang_spans(meta: Mapping[str, Any], *, limit: int) -> tuple[dict[str, int], ...]:
    raw = meta.get("connector_overhang_spans", ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise SchemaError("sequence_evidence_map_v1 meta.connector_overhang_spans must be a list")
    normalized: list[dict[str, int]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SchemaError("sequence_evidence_map_v1 meta.connector_overhang_spans entries must be mappings")
        try:
            start = int(entry.get("start"))
            end = int(entry.get("end"))
        except Exception as exc:
            raise SchemaError(
                "sequence_evidence_map_v1 meta.connector_overhang_spans entries must contain integer start/end"
            ) from exc
        if start < 0 or end > limit or end <= start:
            raise SchemaError("sequence_evidence_map_v1 meta.connector_overhang_spans entries must be within bounds")
        normalized.append({"start": start, "end": end})
    ordered = sorted(normalized, key=lambda entry: (entry["start"], entry["end"]))
    for previous, current in zip(ordered, ordered[1:]):
        if current["start"] < previous["end"]:
            raise SchemaError("sequence_evidence_map_v1 meta.connector_overhang_spans entries must not overlap")
    return tuple(normalized)


def _normalize_span_backdrops(
    meta: Mapping[str, Any],
    *,
    primary_length: int,
    complement_length: int,
) -> tuple[dict[str, object], ...]:
    raw = meta.get("span_backdrops", ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops must be a list")
    normalized: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries must be mappings")
        try:
            start = int(entry.get("start"))
            end = int(entry.get("end"))
        except Exception as exc:
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries require integer start/end") from exc
        cover_rows = str(entry.get("cover_rows", "both")).strip().lower()
        if cover_rows not in {"primary", "complement", "both"}:
            raise SchemaError(
                "sequence_evidence_map_v1 meta.span_backdrops entries cover_rows must be primary, complement, or both"
            )
        fill = str(entry.get("fill", "")).strip()
        if not fill:
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries require a non-empty fill")
        try:
            alpha = float(entry.get("alpha"))
        except Exception as exc:
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries require numeric alpha") from exc
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries alpha must be within [0, 1]")
        try:
            corner_radius = float(entry.get("corner_radius"))
        except Exception as exc:
            raise SchemaError(
                "sequence_evidence_map_v1 meta.span_backdrops entries require numeric corner_radius"
            ) from exc
        if not math.isfinite(corner_radius) or corner_radius < 0.0:
            raise SchemaError(
                "sequence_evidence_map_v1 meta.span_backdrops entries corner_radius must be finite and >= 0"
            )
        if cover_rows == "primary":
            limit = primary_length
        elif cover_rows == "complement":
            limit = complement_length
        else:
            limit = min(primary_length, complement_length)
        if start < 0 or end > limit or end <= start:
            raise SchemaError("sequence_evidence_map_v1 meta.span_backdrops entries must be within row bounds")
        normalized_entry: dict[str, object] = {
            "start": start,
            "end": end,
            "fill": fill,
            "alpha": alpha,
            "corner_radius": corner_radius,
            "cover_rows": cover_rows,
        }
        coordinate_space_raw = entry.get("coordinate_space")
        if coordinate_space_raw is not None:
            coordinate_space = str(coordinate_space_raw).strip()
            if not coordinate_space:
                raise SchemaError(
                    "sequence_evidence_map_v1 meta.span_backdrops entries coordinate_space must be non-empty"
                )
            normalized_entry["coordinate_space"] = coordinate_space
        normalized.append(normalized_entry)
    return tuple(normalized)


def _normalize_segment_labels(meta: Mapping[str, Any], *, primary_length: int) -> tuple[dict[str, object], ...]:
    raw = meta.get("segment_labels", ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise SchemaError("sequence_evidence_map_v1 meta.segment_labels must be a list")
    normalized: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SchemaError("sequence_evidence_map_v1 meta.segment_labels entries must be mappings")
        text = str(entry.get("text", "")).strip()
        if not text:
            raise SchemaError("sequence_evidence_map_v1 meta.segment_labels entries require text")
        try:
            start = int(entry.get("start"))
            end = int(entry.get("end"))
        except Exception as exc:
            raise SchemaError("sequence_evidence_map_v1 meta.segment_labels entries require integer start/end") from exc
        if start < 0 or end > primary_length or end <= start:
            raise SchemaError("sequence_evidence_map_v1 meta.segment_labels entries must be within primary bounds")
        normalized.append({"text": text, "start": start, "end": end, "row_id": "primary"})
    return tuple(normalized)


@dataclass(frozen=True)
class SequenceEvidenceMapV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = SequenceEvidenceMapV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid sequence_evidence_map_v1 contract at row {row_index}: {exc}") from exc

        features: list[Feature] = []
        tag_labels: dict[str, str] = {}

        for owner in contract.owners:
            tag = f"owner:{owner.owner_id}"
            tag_labels[tag] = owner.display_label
            features.append(
                Feature(
                    id=f"{owner.row_id}:{owner.owner_id}:{owner.start}:{owner.end}",
                    kind="interval_annotation",
                    span=Span(start=owner.start, end=owner.end, strand=_strand(owner.row_id)),
                    label=owner.short_label,
                    tags=(tag,),
                    attrs={
                        "lane": owner.row_id,
                        "shape": "band",
                        "semantic": owner.owner_id,
                        "intent": "owner",
                        "style_token": _style_token_for_owner(owner.owner_id),
                    },
                    render={"track": 0 if owner.row_id == "primary" else 1},
                )
            )

        for tag in contract.effect_tags:
            feature_tag = f"effect:{tag.tag_kind}"
            tag_labels[feature_tag] = tag.display_label
            features.append(
                Feature(
                    id=f"{tag.row_id}:{tag.tag_id}:{tag.start}:{tag.end}",
                    kind="interval_annotation",
                    span=Span(start=tag.start, end=tag.end, strand=_strand(tag.row_id)),
                    label=tag.short_label,
                    tags=(feature_tag,),
                    attrs={
                        "lane": tag.row_id,
                        "shape": "rounded_rect",
                        "semantic": tag.tag_kind,
                        "intent": "effect",
                        "style_token": _style_token_for_tag(tag.tag_kind),
                    },
                    render={},
                )
            )

        meta = dict(contract.meta)
        if "boundary_marker_style" in meta:
            raise SchemaError(
                "sequence_evidence_map_v1 meta.boundary_marker_style is no longer "
                "supported; encode semantics in boundaries.boundary_kind"
            )
        effects: list[Effect] = []
        for boundary in contract.boundaries:
            effects.append(
                Effect(
                    kind="boundary_marker",
                    target={"boundary": boundary.boundary, "lane": boundary.row_id},
                    params={
                        "label": boundary.short_label,
                        "semantic": boundary.boundary_kind,
                        "intent": "evidence_boundary",
                    },
                    render={},
                )
            )
        pairing_lane = "bottom" if contract.topology_kind == "hairpin_folded" else "top"
        pairing_track = 0 if contract.topology_kind == "hairpin_folded" else 4
        for pairing in contract.pairings:
            pairing_label = (
                "" if contract.topology_kind == "hairpin_folded" else pairing.short_label or pairing.display_label or ""
            )
            effects.append(
                Effect(
                    kind="span_link",
                    target={
                        "from_span": {"start": pairing.primary_start, "end": pairing.primary_end, "strand": "fwd"},
                        "to_span": {
                            "start": pairing.complement_start,
                            "end": pairing.complement_end,
                            "strand": "rev",
                        },
                    },
                    params={
                        "label": pairing_label,
                        "lane": pairing_lane,
                        "inner_margin_bp": 0.0,
                    },
                    render={"track": pairing_track},
                )
            )
        legend_exclude_tags_raw = meta.get("legend_exclude_tags", ())
        if isinstance(legend_exclude_tags_raw, Mapping):
            raise SchemaError("sequence_evidence_map_v1 meta.legend_exclude_tags must be a list of tag ids")
        if isinstance(legend_exclude_tags_raw, (str, bytes)):
            raise SchemaError("sequence_evidence_map_v1 meta.legend_exclude_tags must be a list of tag ids")
        try:
            legend_exclude_tags = tuple(str(tag).strip() for tag in legend_exclude_tags_raw if str(tag).strip())
        except TypeError as exc:
            raise SchemaError("sequence_evidence_map_v1 meta.legend_exclude_tags must be a list of tag ids") from exc
        row_labels = meta.get("row_labels")
        if not isinstance(row_labels, Mapping):
            row_labels = {"primary": "Primary", "complement": "Complement"}
        complement_length = (
            len(contract.complement_sequence)
            if contract.complement_sequence is not None
            else len(contract.primary_sequence)
        )
        base_highlights = _normalize_base_highlights(
            meta,
            primary_length=len(contract.primary_sequence),
            complement_length=complement_length,
        )
        base_highlight_color = _normalize_row_color_map(meta, key="base_highlight_color")
        dim_base_indices = _normalize_row_index_map(
            meta,
            key="dim_base_indices",
            primary_length=len(contract.primary_sequence),
            complement_length=complement_length,
        )
        connector_hidden_indices = _normalize_index_list(
            meta, "connector_hidden_indices", limit=len(contract.primary_sequence)
        )
        connector_cross_indices = _normalize_index_list(
            meta, "connector_cross_indices", limit=len(contract.primary_sequence)
        )
        connector_overhang_spans = _normalize_connector_overhang_spans(meta, limit=len(contract.primary_sequence))
        span_backdrops = _normalize_span_backdrops(
            meta,
            primary_length=len(contract.primary_sequence),
            complement_length=complement_length,
        )
        if connector_overhang_spans:
            overhang_indices = {
                index for span in connector_overhang_spans for index in range(span["start"], span["end"])
            }
            if any(index not in overhang_indices for index in connector_hidden_indices):
                raise SchemaError(
                    "sequence_evidence_map_v1 meta.connector_hidden_indices must lie within connector_overhang_spans"
                )
            if any(index not in overhang_indices for index in connector_cross_indices):
                raise SchemaError(
                    "sequence_evidence_map_v1 meta.connector_cross_indices must lie within connector_overhang_spans"
                )
        elif connector_hidden_indices or connector_cross_indices:
            raise SchemaError(
                "sequence_evidence_map_v1 connector hidden/cross indices require connector_overhang_spans"
            )
        segment_labels = _normalize_segment_labels(meta, primary_length=len(contract.primary_sequence))

        record = Record(
            id=contract.state_id,
            alphabet=self.alphabet,
            sequence=contract.primary_sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(overlay_text=str(contract.display.title or ""), tag_labels=tag_labels),
            meta={
                "adapter": "sequence_evidence_map_v1",
                "contract": contract.model_dump(mode="json"),
                "complement_sequence": contract.complement_sequence,
                "view_meta": meta,
                "base_highlights": base_highlights,
                "base_highlight_color": base_highlight_color,
                "dim_base_indices": dim_base_indices,
                "connector_hidden_indices": connector_hidden_indices,
                "connector_cross_indices": connector_cross_indices,
                "connector_overhang_spans": connector_overhang_spans,
                "span_backdrops": span_backdrops,
                "segment_labels": segment_labels,
                "legend_exclude_tags": legend_exclude_tags,
                "row_labels": dict(row_labels),
                "show_reverse_complement": contract.complement_sequence is not None,
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
