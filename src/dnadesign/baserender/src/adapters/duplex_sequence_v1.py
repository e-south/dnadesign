"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/duplex_sequence_v1.py

Adapter from shared linear-duplex cassette contracts to baserender Record v1.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import LinearDuplexViewV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


def _strand(value: str) -> str:
    return "fwd" if value == "primary" else "rev"


def _style_token_for_segment(semantic: str) -> str:
    return {
        "flank": "segment_flank",
        "stem5p_arm": "segment_stem5p",
        "loop": "segment_loop",
        "stem3p_arm": "segment_stem3p",
    }.get(semantic, "segment")


@dataclass(frozen=True)
class DuplexSequenceV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            view = LinearDuplexViewV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid linear_duplex_v1 contract at row {row_index}: {exc}") from exc

        features: list[Feature] = []
        for segment in view.segments:
            tag = f"segment:{segment.semantic}"
            features.append(
                Feature(
                    id=segment.id,
                    kind="interval_annotation",
                    span=Span(start=segment.start, end=segment.end, strand="fwd"),
                    label=segment.label,
                    tags=(tag,),
                    attrs={
                        "lane": "primary",
                        "shape": "band",
                        "semantic": segment.semantic,
                        "intent": "structural",
                        "style_token": _style_token_for_segment(segment.semantic),
                    },
                    render={},
                )
            )

        for site in view.site_instances:
            tag = "site:intended" if site.intent != "extra" else "site:extra"
            features.append(
                Feature(
                    id=site.id,
                    kind="interval_annotation",
                    span=Span(start=site.start, end=site.end, strand=_strand(site.site_target_strand)),
                    label=site.label,
                    tags=(tag,),
                    attrs={
                        "lane": site.site_target_strand,
                        "shape": "rounded_rect",
                        "semantic": "recognition_site",
                        "intent": site.intent,
                        "style_token": "site_extra" if site.intent == "extra" else "site_intended",
                    },
                    render={},
                )
            )

        if view.bounded_segment is not None and view.bounded_segment.end_boundary > view.bounded_segment.start_boundary:
            features.append(
                Feature(
                    id="bounded_segment",
                    kind="interval_annotation",
                    span=Span(
                        start=view.bounded_segment.start_boundary,
                        end=view.bounded_segment.end_boundary,
                        strand=_strand(view.bounded_segment.target_strand),
                    ),
                    label=view.bounded_segment.label,
                    tags=("bounded_segment",),
                    attrs={
                        "lane": view.bounded_segment.target_strand,
                        "shape": "band",
                        "semantic": "bounded_segment",
                        "intent": "structural",
                        "style_token": "bounded_segment",
                    },
                    render={},
                )
            )

        effects = tuple(
            Effect(
                kind="boundary_marker",
                target={"boundary": nick.boundary, "lane": nick.target_strand},
                params={"label": nick.label, "semantic": "nick_event", "intent": nick.intent},
                render={},
            )
            for nick in view.nick_events
        )

        tag_labels = {
            "site:intended": "Intended site",
            "site:extra": "Extra site",
            "segment:flank": "Flank",
            "segment:stem5p_arm": "Stem 5' arm",
            "segment:loop": "Loop",
            "segment:stem3p_arm": "Stem 3' arm",
            "bounded_segment": "Bounded nicked segment",
        }

        record = Record(
            id=view.view_id,
            alphabet=self.alphabet,
            sequence=view.primary_sequence_5to3,
            features=tuple(features),
            effects=effects,
            display=Display(overlay_text=view.title, tag_labels=tag_labels),
            meta={
                "adapter": "duplex_sequence_v1",
                "row_labels": view.row_labels.model_dump(mode="json"),
                "contract_labels": [label.model_dump(mode="json") for label in view.labels],
                "solution_id": view.solution_id,
                "target_strand": view.target_strand,
                "title": view.title,
                "view_meta": view.meta,
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
