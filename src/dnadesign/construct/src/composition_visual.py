"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition_visual.py

Canonical visual and folding handoff helpers for linear ssDNA composition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from dnadesign.contracts.visual.sequence_evidence_map_v1 import (
    SequenceEvidenceEffectSpanV1,
    SequenceEvidenceMapV1,
    SequenceEvidenceOwnerSpanV1,
    SequenceEvidencePairingV1,
)

from .errors import ValidationError

SEQUENCE_EVIDENCE_MAP_PATH = Path("visual/sequence_evidence_map_v1.json")
CANONICAL_FOLDING_SEQUENCE_PATH = Path("folding/secondary_structure_input_sequence.json")
DEPRECATED_VISUAL_ARTIFACT_PATHS = (Path("visual/contracts/component_span_qa_sequence_evidence_map_v1.json"),)

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_DEFAULT_BACKDROP_STYLE = {"fill": "#CBD5E1", "alpha": 0.62, "edge_color": "#94A3B8"}
_SEGMENT_LABEL_COLOR = "#334155"
_ANNOTATION_LABEL_COLOR = "#475569"


def visual_contract_payload(composed: Any) -> dict[str, object]:
    view = _canonical_component_view(composed)
    display_profile = composed.config.visual.display_profile
    title = display_profile.title or _composition_display_name(composed.config.composition_id)
    contract = SequenceEvidenceMapV1(
        state_id=f"{composed.config.composition_id}.component_span_qa",
        topology_kind="linear_ssdna",
        alphabet="dna",
        primary_sequence=view["sequence"],
        complement_sequence=_complement(str(view["sequence"])),
        owners=[
            SequenceEvidenceOwnerSpanV1(
                owner_id=f"{span.unit_id}.{span.segment_id}",
                row_id="primary",
                start=span.start + view["copy_offset"],
                end=span.end + view["copy_offset"],
                display_label=_display_label(span.segment_id, display_profile.component_labels),
                short_label="",
            )
            for span in view["segment_spans"]
        ],
        effect_tags=[
            SequenceEvidenceEffectSpanV1(
                tag_id=f"{span.unit_id}.{span.annotation_id}",
                tag_kind=span.role,
                row_id="primary",
                start=span.start + view["copy_offset"],
                end=span.end + view["copy_offset"],
                display_label=_display_label(
                    span.annotation_id,
                    display_profile.annotation_labels,
                    fallback=span.role,
                ),
                short_label="",
            )
            for span in view["annotation_spans"]
        ],
        boundaries=[],
        pairings=_canonical_intended_pairings(composed, view),
        display={"title": f"{title} component span QA"},
        meta=_canonical_visual_meta(composed, view, title=title),
    )
    return contract.model_dump(mode="json")


def canonical_sequence_artifact_payload(composed: Any) -> dict[str, object]:
    view = _canonical_component_view(composed)
    sequence = str(view["sequence"])
    return {
        "contract": "linear_ssdna_canonical_component_sequence_v1",
        "schema_version": 1,
        "composition_id": composed.config.composition_id,
        "coordinate_system": composed.config.coordinate_system,
        "source_sequence": {
            "id": composed.config.composition_id,
            "length": len(composed.sequence),
            "sha256": composed.sequence_sha256,
        },
        "sequence": {
            "id": f"{composed.config.composition_id}.component_span_qa",
            "length": len(sequence),
            "sha256": _sha256_text(sequence),
            "sequence": sequence,
        },
        "unit_copies": _canonical_unit_copy_meta(view),
    }


def _canonical_component_view(composed: Any) -> dict[str, Any]:
    representative = _representative_unit_copy(composed)
    copy_offset = -representative.start
    sequence = composed.sequence[representative.start : representative.end]
    segment_spans = [
        span
        for span in composed.segment_spans
        if span.unit_id == representative.unit_id and span.copy_index == representative.copy_index
    ]
    annotation_spans = [
        span
        for span in composed.annotation_spans
        if span.unit_id == representative.unit_id
        and span.copy_index == representative.copy_index
        and not _annotation_exactly_duplicates_segment(span, segment_spans)
    ]
    return {
        "representative": representative,
        "copy_offset": copy_offset,
        "sequence": sequence,
        "segment_spans": segment_spans,
        "annotation_spans": annotation_spans,
    }


def _representative_unit_copy(composed: Any) -> Any:
    if len(composed.config.units) != 1:
        raise ValidationError(
            "canonical component visual/folding contracts require exactly one configured unit. "
            "Multi-unit visualization needs an explicit multi-record contract instead of concatenating units."
        )
    unit_id = composed.config.units[0].unit_id
    for copy in composed.unit_copies:
        if copy.unit_id == unit_id:
            return copy
    raise ValidationError(f"No expanded copy found for unit '{unit_id}'.")


def _annotation_exactly_duplicates_segment(annotation: Any, segment_spans: list[Any]) -> bool:
    return any(
        segment.unit_id == annotation.unit_id
        and segment.copy_index == annotation.copy_index
        and segment.start == annotation.start
        and segment.end == annotation.end
        for segment in segment_spans
    )


def _canonical_intended_pairings(
    composed: Any,
    view: dict[str, Any],
) -> list[SequenceEvidencePairingV1]:
    representative = view["representative"]
    offset = int(view["copy_offset"])
    spans_by_segment = {span.segment_id: span for span in view["segment_spans"]}
    pairings: list[SequenceEvidencePairingV1] = []
    unit = composed.config.units[0]
    for assertion in unit.assertions:
        if assertion.kind != "reverse_complement":
            continue
        left = spans_by_segment[assertion.left_segment_id]
        right = spans_by_segment[assertion.right_segment_id]
        pairings.append(
            SequenceEvidencePairingV1(
                pairing_id=f"{representative.unit_id}.{assertion.assertion_id}",
                primary_start=left.start + offset,
                primary_end=left.end + offset,
                complement_start=right.start + offset,
                complement_end=right.end + offset,
                display_label=assertion.assertion_id,
                short_label="intended RC",
            )
        )
    return pairings


def _canonical_visual_meta(composed: Any, view: dict[str, Any], *, title: str) -> dict[str, object]:
    representative = view["representative"]
    sequence = str(view["sequence"])
    segment_spans = list(view["segment_spans"])
    annotation_spans = list(view["annotation_spans"])
    display_profile = composed.config.visual.display_profile
    duplicate_annotations = [
        span
        for span in composed.annotation_spans
        if span.unit_id == representative.unit_id
        and span.copy_index == representative.copy_index
        and _annotation_exactly_duplicates_segment(span, segment_spans)
    ]
    meta: dict[str, object] = {
        "source_contract": "linear_ssdna_composition_v1",
        "sequence_sha256": _sha256_text(sequence),
        "source_sequence_sha256": composed.sequence_sha256,
        "validation_status": "ok",
        "interval_annotation_policy": "span_backdrops_only",
        "glyph_highlight_policy": "region_backdrops_only",
        "render_pairing_links": False,
        "row_labels": {"primary": "Top", "complement": "Bottom"},
        "base_highlights": {"primary": [], "complement": []},
        "structure_title": title,
        "component_palette": dict(display_profile.component_hues),
        "display_profile": _display_profile_meta(display_profile, title=title),
        "segment_label_gap_px": 6.0,
        "segment_label_tier_gap_px": 10.0,
        "visual_scope": {
            "mode": "canonical_component_span_qa",
            "source_sequence_id": composed.config.composition_id,
            "source_sequence_length": len(composed.sequence),
            "source_sequence_sha256": composed.sequence_sha256,
            "representative_copy_count": 1,
            "coordinate_transform": "representative_copy_to_canonical_zero_based_half_open",
            "repeat_expansion_rendered": False,
        },
        "unit_copies": _canonical_unit_copy_meta(view),
        "representative_unit_copies": [representative.to_dict()],
        "span_backdrops": [
            *_segment_span_backdrops(
                segment_spans,
                view,
                annotation_spans=annotation_spans,
                display_profile=display_profile,
            ),
            *_annotation_span_backdrops(annotation_spans, view, display_profile=display_profile),
        ],
        "segment_labels": [
            *_segment_label_entries(
                segment_spans,
                view,
                annotation_spans=annotation_spans,
                display_profile=display_profile,
            ),
            *_annotation_label_entries(annotation_spans, view, display_profile=display_profile),
        ],
        "legend_exclude_tags": _legend_exclude_tags(segment_spans, annotation_spans),
        "suppressed_exact_span_annotations": [
            {
                "annotation_id": span.annotation_id,
                "role": span.role,
                "start": span.start + view["copy_offset"],
                "end": span.end + view["copy_offset"],
                "reason": "duplicates_physical_segment_span",
            }
            for span in duplicate_annotations
        ],
    }
    scar_nick = _display_profile_scar_nick_meta(display_profile)
    if scar_nick:
        meta["scar_nick"] = scar_nick
    return meta


def _segment_span_backdrops(
    segment_spans: list[Any],
    view: dict[str, Any],
    *,
    annotation_spans: list[Any],
    display_profile: Any,
) -> list[dict[str, object]]:
    backdrops: list[dict[str, object]] = []
    has_snapback_subsections = _has_snapback_subsection_annotations(annotation_spans)
    for span in segment_spans:
        styles = display_profile.component_styles
        configured_style = styles.get(span.segment_id) or styles.get(span.role)
        if _suppresses_overview_markup(
            span.segment_id,
            span.role,
            has_snapback_subsections=has_snapback_subsections,
        ):
            continue
        if configured_style is None and span.segment_id not in display_profile.component_labels:
            continue
        style = _style_payload(configured_style)
        backdrops.append(
            {
                "semantic": span.segment_id,
                "start": span.start + view["copy_offset"],
                "end": span.end + view["copy_offset"],
                "cover_rows": "both",
                "fill": style["fill"],
                "alpha": style["alpha"],
                "corner_radius": 3.0,
                "edge_color": style["edge_color"],
                "edge_alpha": 0.72,
                "edge_linewidth": 0.36,
            }
        )
    return backdrops


def _segment_label_entries(
    segment_spans: list[Any],
    view: dict[str, Any],
    *,
    annotation_spans: list[Any],
    display_profile: Any,
) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    has_snapback_subsections = _has_snapback_subsection_annotations(annotation_spans)
    for span in segment_spans:
        if _suppresses_overview_markup(
            span.segment_id,
            span.role,
            has_snapback_subsections=has_snapback_subsections,
        ):
            continue
        text = display_profile.component_labels.get(span.segment_id)
        if not text:
            continue
        labels.append(
            {
                "text": text,
                "start": span.start + view["copy_offset"],
                "end": span.end + view["copy_offset"],
                "label_side": "above",
                "color": _SEGMENT_LABEL_COLOR,
            }
        )
    return labels


def _annotation_label_entries(
    annotation_spans: list[Any],
    view: dict[str, Any],
    *,
    display_profile: Any,
) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    for span in annotation_spans:
        if _suppresses_overview_markup(span.annotation_id, span.role):
            continue
        text = display_profile.annotation_labels.get(span.annotation_id)
        if not text:
            continue
        labels.append(
            {
                "text": text,
                "start": span.start + view["copy_offset"],
                "end": span.end + view["copy_offset"],
                "label_side": "below",
                "color": _ANNOTATION_LABEL_COLOR,
            }
        )
    return labels


def _has_snapback_subsection_annotations(annotation_spans: list[Any]) -> bool:
    subsection_ids = {"snapback_retained_stem", "snapback_cap", "snapback_foldback_return"}
    return any(span.annotation_id in subsection_ids or span.role in subsection_ids for span in annotation_spans)


def _suppresses_overview_markup(identifier: str, role: str, *, has_snapback_subsections: bool = False) -> bool:
    if identifier in {"snapback_retained_stem", "snapback_foldback_return"}:
        return True
    if role in {"snapback_retained_stem", "snapback_foldback_return"}:
        return True
    if identifier == "snapback_foldback_geometry" or role == "snapback_foldback_geometry":
        return has_snapback_subsections
    return False


def _annotation_span_backdrops(
    annotation_spans: list[Any],
    view: dict[str, Any],
    *,
    display_profile: Any,
) -> list[dict[str, object]]:
    backdrops: list[dict[str, object]] = []
    for span in annotation_spans:
        styles = display_profile.component_styles
        configured_style = styles.get(span.annotation_id) or styles.get(span.role)
        if configured_style is None:
            continue
        style = _style_payload(configured_style)
        backdrops.append(
            {
                "semantic": span.annotation_id,
                "start": span.start + view["copy_offset"],
                "end": span.end + view["copy_offset"],
                "cover_rows": "both",
                "fill": style["fill"],
                "alpha": style["alpha"],
                "corner_radius": 2.0,
                "edge_color": style["edge_color"],
                "edge_alpha": 0.82,
                "edge_linewidth": 0.44,
            }
        )
    return backdrops


def _canonical_unit_copy_meta(view: dict[str, Any]) -> list[dict[str, object]]:
    representative = view["representative"]
    return [
        {
            "unit_id": representative.unit_id,
            "copy_index": representative.copy_index,
            "span": {"start": 0, "end": representative.end - representative.start},
            "source_span": {"start": representative.start, "end": representative.end},
        }
    ]


def _legend_exclude_tags(segment_spans: list[Any], annotation_spans: list[Any]) -> list[str]:
    owner_tags = [f"owner:{span.unit_id}.{span.segment_id}" for span in segment_spans]
    effect_tags = [f"effect:{span.role}" for span in annotation_spans]
    return owner_tags + list(dict.fromkeys(effect_tags))


def _display_label(raw: str, mapping: dict[str, str], *, fallback: str | None = None) -> str:
    label = mapping.get(raw)
    if label is not None:
        return label
    return _pretty_display_label(fallback or raw)


def _style_payload(configured_style: Any | None) -> dict[str, object]:
    if configured_style is None:
        return dict(_DEFAULT_BACKDROP_STYLE)
    return {
        "fill": configured_style.fill or _DEFAULT_BACKDROP_STYLE["fill"],
        "alpha": configured_style.alpha if configured_style.alpha is not None else _DEFAULT_BACKDROP_STYLE["alpha"],
        "edge_color": configured_style.edge_color or _DEFAULT_BACKDROP_STYLE["edge_color"],
    }


def _display_profile_meta(display_profile: Any, *, title: str) -> dict[str, object]:
    meta: dict[str, object] = {
        "title": title,
        "component_labels": dict(display_profile.component_labels),
        "annotation_labels": dict(display_profile.annotation_labels),
        "component_hues": dict(display_profile.component_hues),
    }
    scar_nick = _display_profile_scar_nick_meta(display_profile)
    if scar_nick:
        meta["scar_nick"] = scar_nick
    return meta


def _display_profile_scar_nick_meta(display_profile: Any) -> dict[str, object]:
    scar_nick = getattr(display_profile, "scar_nick", None)
    if scar_nick is None:
        return {}
    payload = scar_nick.model_dump(exclude_none=True) if hasattr(scar_nick, "model_dump") else dict(scar_nick)
    return {key: value for key, value in payload.items() if str(value).strip()}


def _pretty_display_label(raw: str) -> str:
    text = str(raw).replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", text).capitalize() if text else str(raw)


def _composition_display_name(composition_id: str) -> str:
    text = str(composition_id).replace("_", " ").replace("-", " ")
    text = re.sub(r"\bmanual\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return str(composition_id)
    return " ".join(_pretty_display_token(token) for token in text.split())


def _pretty_display_token(token: str) -> str:
    lowered = token.lower()
    if re.fullmatch(r"x\d+", lowered):
        return lowered
    if token.isdigit():
        return token
    return token[:1].upper() + token[1:].lower()


def _complement(sequence: str) -> str:
    return sequence.translate(_DNA_COMPLEMENT)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "CANONICAL_FOLDING_SEQUENCE_PATH",
    "DEPRECATED_VISUAL_ARTIFACT_PATHS",
    "SEQUENCE_EVIDENCE_MAP_PATH",
    "canonical_sequence_artifact_payload",
    "visual_contract_payload",
]
