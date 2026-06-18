"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/hairpin_topology_v1.py

Adapter from shared hairpin-topology cassette contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import HairpinTopologyViewV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


@dataclass(frozen=True)
class HairpinTopologyV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            view = HairpinTopologyViewV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid ssdna_hairpin_v1 contract at row {row_index}: {exc}") from exc

        topology = view.topology
        features: list[Feature] = [
            Feature(
                id="stem5p_span",
                kind="interval_annotation",
                span=Span(start=topology.stem5p_span.start, end=topology.stem5p_span.end, strand="fwd"),
                label="Stem 5' arm",
                tags=("topology:stem5p_arm",),
                attrs={
                    "lane": "topology",
                    "shape": "band",
                    "semantic": "stem5p_arm",
                    "intent": "structural",
                    "style_token": "segment_stem5p",
                },
                render={},
            ),
            Feature(
                id="loop_span",
                kind="interval_annotation",
                span=Span(start=topology.loop_span.start, end=topology.loop_span.end, strand="fwd"),
                label="Loop",
                tags=("topology:loop",),
                attrs={
                    "lane": "topology",
                    "shape": "band",
                    "semantic": "loop",
                    "intent": "structural",
                    "style_token": "segment_loop",
                },
                render={},
            ),
            Feature(
                id="stem3p_span",
                kind="interval_annotation",
                span=Span(start=topology.stem3p_span.start, end=topology.stem3p_span.end, strand="fwd"),
                label="Stem 3' arm",
                tags=("topology:stem3p_arm",),
                attrs={
                    "lane": "topology",
                    "shape": "band",
                    "semantic": "stem3p_arm",
                    "intent": "structural",
                    "style_token": "segment_stem3p",
                },
                render={},
            ),
        ]
        for feature in view.feature_spans:
            features.append(
                Feature(
                    id=feature.id,
                    kind="interval_annotation",
                    span=Span(start=feature.start, end=feature.end, strand="fwd"),
                    label=feature.label,
                    tags=("feature_projection",),
                    attrs={
                        "lane": "topology",
                        "shape": "underline",
                        "semantic": feature.semantic,
                        "intent": "informational",
                        "style_token": "feature_projection",
                    },
                    render={},
                )
            )

        record = Record(
            id=view.view_id,
            alphabet=self.alphabet,
            sequence=view.primary_sequence_5to3,
            features=tuple(features),
            effects=(
                Effect(
                    kind="pair_map",
                    target={"pairs": [pair.model_dump(mode="json") for pair in view.pair_map]},
                    params={"semantic": "stem_pairing"},
                    render={},
                ),
            ),
            display=Display(
                overlay_text=view.title,
                tag_labels={
                    "topology:stem5p_arm": "Stem 5' arm",
                    "topology:loop": "Loop",
                    "topology:stem3p_arm": "Stem 3' arm",
                    "feature_projection": "Motif projection",
                },
            ),
            meta={
                "adapter": "hairpin_topology_v1",
                "hairpin_topology": topology.model_dump(mode="json"),
                "hairpin_notes": [annotation.model_dump(mode="json") for annotation in view.duplex_derived_annotations],
                "solution_id": view.solution_id,
                "title": view.title,
                "view_meta": view.meta,
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
