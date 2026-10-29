"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/yiu/payload_motif_overlay.py

Motif-overlay assembly helpers for YIU payload visual contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.contracts.visual import YiuPayloadVisualV1
from dnadesign.contracts.visual.yiu_payload_visual_v1 import YiuPayloadMotifLayerV1

from ...core import Record, Span
from ...core.record import Effect, Feature


def _payload_wide_matrix(contract: YiuPayloadVisualV1, motif: YiuPayloadMotifLayerV1) -> list[list[float]]:
    payload_length = len(contract.selected_payload_sequence)
    padded = [[0.25, 0.25, 0.25, 0.25] for _ in range(payload_length)]
    visible_rows = motif.matrix if motif.reference_strand == "+" else list(reversed(motif.matrix))
    for offset, row in enumerate(visible_rows):
        padded[motif.start + offset] = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    if motif.reference_strand == "+":
        return padded
    return [list(row) for row in reversed(padded)]


def _observed_sequence_5to3(contract: YiuPayloadVisualV1, motif: YiuPayloadMotifLayerV1) -> str:
    if motif.reference_strand == "+":
        return contract.selected_payload_sequence
    return contract.selected_complement_sequence[::-1]


def _motif_style_token(motif: YiuPayloadMotifLayerV1) -> str:
    return f"tf:{motif.tf_name}"


def _build_motif_feature(
    *,
    motif: YiuPayloadMotifLayerV1,
    base_record: Record,
    feature_id: str,
    style_token: str,
    motif_tag: str,
) -> Feature:
    strand = "fwd" if motif.reference_strand == "+" else "rev"
    span = Span(start=motif.start, end=motif.end, strand=strand)
    return Feature(
        id=feature_id,
        kind="regulator_window",
        span=span,
        label=base_record.segment_for(span),
        tags=(motif_tag, style_token),
        attrs={
            "tf": motif.tf_name,
            "motif_name": motif.motif_name,
            "display_label": motif.label,
            "style_token": style_token,
            "lane": "primary" if motif.reference_strand == "+" else "complement",
        },
        render={"priority": 10},
    )


def _build_motif_effect(
    *,
    contract: YiuPayloadVisualV1,
    motif: YiuPayloadMotifLayerV1,
    feature_id: str,
) -> Effect:
    return Effect(
        kind="motif_logo",
        target={"feature_id": feature_id},
        params={
            "matrix": _payload_wide_matrix(contract, motif),
            "render_span": {"start": 0, "end": len(contract.selected_payload_sequence)},
            "observed_sequence_5to3": _observed_sequence_5to3(contract, motif),
        },
        render={"priority": 20},
    )


@dataclass(frozen=True)
class YiuPayloadMotifOverlay:
    features: tuple[Feature, ...]
    effects: tuple[Effect, ...]
    tag_labels: dict[str, str]


def _build_tag_labels(contract: YiuPayloadVisualV1) -> dict[str, str]:
    tag_labels: dict[str, str] = {}
    motif_tf_counts: dict[str, int] = {}
    for motif in contract.motif_layers:
        motif_tf_counts[motif.tf_name] = motif_tf_counts.get(motif.tf_name, 0) + 1
    for motif in contract.motif_layers:
        style_token = _motif_style_token(motif)
        motif_tag = f"motif:{motif.motif_instance_id}"
        tag_labels.setdefault(style_token, motif.tf_name)
        if motif_tf_counts[motif.tf_name] > 1:
            tag_labels.setdefault(motif_tag, motif.label)
    return tag_labels


def build_motif_overlay(contract: YiuPayloadVisualV1, *, base_record: Record) -> YiuPayloadMotifOverlay:
    features: list[Feature] = []
    effects: list[Effect] = []
    tag_labels = _build_tag_labels(contract)
    for motif in contract.motif_layers:
        feature_id = f"motif:{motif.motif_instance_id}"
        style_token = _motif_style_token(motif)
        motif_tag = f"motif:{motif.motif_instance_id}"
        features.append(
            _build_motif_feature(
                motif=motif,
                base_record=base_record,
                feature_id=feature_id,
                style_token=style_token,
                motif_tag=motif_tag,
            )
        )
        effects.append(_build_motif_effect(contract=contract, motif=motif, feature_id=feature_id))
    return YiuPayloadMotifOverlay(
        features=tuple(features),
        effects=tuple(effects),
        tag_labels=tag_labels,
    )


__all__ = [
    "YiuPayloadMotifOverlay",
    "build_motif_overlay",
]
