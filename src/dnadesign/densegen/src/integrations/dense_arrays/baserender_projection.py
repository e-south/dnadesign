"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/dense_arrays/baserender_projection.py

Project DenseGen playback plans into BaseRender duplex frames.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from dense_arrays.playback.graph import step_color
from dense_arrays.playback.html import PlaybackDocument
from dense_arrays.playback.typography import PUBLICATION_NUCLEOTIDE_TYPOGRAPHY
from dense_arrays.realized import RealizedArray
from PIL import Image

from dnadesign.baserender import (
    Display,
    Effect,
    Feature,
    Palette,
    Record,
    Span,
    Style,
    compute_layout,
    initialize_runtime,
    render_record,
)

_SVG_RE = re.compile(r"(<svg\b.*</svg>)", re.DOTALL)
_DNA_COMPLEMENT = str.maketrans("ATCGRYSWKMBDHVN", "TAGCYRSWMKVHDBN")
_ANCHORED_ILLUSTRATION_TOP_BAND_PX = 550.0


@dataclass(frozen=True)
class AnchoredIllustrationPresentation:
    asset_id: str
    constraint_name: str
    reveal: str = "bindings_as_placed"

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not self.constraint_name.strip():
            raise ValueError("anchored illustration asset and constraint must be non-empty")
        if self.reveal != "bindings_as_placed":
            raise ValueError("anchored illustration reveal must be 'bindings_as_placed'")


@dataclass(frozen=True)
class DuplexPresentation:
    fixed_element_annotations: str = "none"
    consensus_suffix: str = "omit"
    anchored_illustration: AnchoredIllustrationPresentation | None = None

    def __post_init__(self) -> None:
        if self.fixed_element_annotations not in {"none", "variant"}:
            raise ValueError("fixed_element_annotations must be 'none' or 'variant'")
        if self.consensus_suffix not in {"omit", "include"}:
            raise ValueError("consensus_suffix must be 'omit' or 'include'")


class BaseRenderDuplexProjection:
    """Render truthful stepwise duplex frames with BaseRender sequence_rows."""

    def __init__(
        self,
        documents: tuple[PlaybackDocument, ...],
        *,
        realized_arrays: Mapping[str, RealizedArray] | None = None,
        presentation: DuplexPresentation | None = None,
    ):
        if not documents:
            raise ValueError("BaseRender projection requires at least one document")
        profiles = {document.presentation.color_profile for document in documents}
        if len(profiles) != 1:
            raise ValueError("BaseRender projection requires one presentation profile")
        self._color_profile = profiles.pop()
        self._presentation = presentation or DuplexPresentation()
        self.preferred_figure_height_inches = 3.0 if self._presentation.anchored_illustration is not None else 2.4
        realized_arrays = dict(realized_arrays or {})
        requires_realized_metadata = (
            self._presentation.fixed_element_annotations != "none"
            or self._presentation.anchored_illustration is not None
        )
        if requires_realized_metadata:
            missing = sorted(
                document.plan.realization_digest
                for document in documents
                if document.plan.realization_digest not in realized_arrays
            )
            if missing:
                raise ValueError(
                    f"duplex presentation requires realized placement metadata for every document: {missing}"
                )
        self._placement_metadata: dict[str, dict[str, Mapping[str, object]]] = {}
        for digest, realized in realized_arrays.items():
            self._placement_metadata[digest] = {
                placement.placement_id: placement.metadata for placement in realized.placements
            }
        initialize_runtime()
        self._style = Style(
            dpi=180,
            figure_scale=1.6,
            font_mono=PUBLICATION_NUCLEOTIDE_TYPOGRAPHY.family,
            font_label=PUBLICATION_NUCLEOTIDE_TYPOGRAPHY.family,
            font_size_seq=PUBLICATION_NUCLEOTIDE_TYPOGRAPHY.duplex_font_size_pt,
            font_size_label=PUBLICATION_NUCLEOTIDE_TYPOGRAPHY.duplex_font_size_pt,
            font_size_feature_label=30,
            font_size_annotation_label=30,
            font_size_span_link_label=30,
            color_sequence="#000000",
            padding_x=12.0,
            padding_y=12.0,
            track_spacing=38.0,
            baseline_spacing=72.0,
            show_reverse_complement=True,
            show_coordinate_ticks=False,
            show_pair_rungs=False,
            layout={"outer_pad_cells": 0.18},
            sequence={"strand_gap_cells": 0.18, "to_kmer_gap_cells": 0.30},
            legend=False,
            legend_mode="none",
            connectors=False,
            kmer={
                "box_height_cells": 1.12,
                "edge_width": 0.0,
                "fill_alpha": 0.94,
                "text_color": "#FFFFFF",
                "text_y_nudge_cells": 0.0,
                "to_logo_gap_cells": 0.12,
            },
            uniform_display_font_size=False,
            span_link_line_width=2.8,
            span_link_tick_line_width=2.5,
            span_link_color="#000000",
            span_link_label_color="#000000",
        )
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        cap_bounds = TextPath(
            (0, 0),
            "ACGT",
            prop=FontProperties(
                family=self._style.font_mono,
                size=self._style.font_size_seq,
                weight="normal",
            ),
        ).get_extents()
        self.native_nucleotide_cap_height_px = (
            float(cap_bounds.height) * float(self._style.dpi) * float(self._style.figure_scale) / 72.0
        )
        self._records: dict[str, tuple[Record, ...]] = {}
        self._palettes: dict[str, Palette] = {}
        self._rgba_cache: dict[tuple[str, int], np.ndarray] = {}
        self._svg_cache: dict[tuple[str, int], str] = {}
        for document in documents:
            records, palette = self._prepare_document(document)
            self._records[document.plan.realization_digest] = records
            self._palettes[document.plan.realization_digest] = palette

    @staticmethod
    def _tag(index: int) -> str:
        return f"playback:step:{index}"

    def _record_for_step(self, document: PlaybackDocument, step_index: int) -> tuple[Record, dict[str, str]]:
        plan = document.plan
        placed = plan.steps[: step_index + 1]
        step_by_id = {step.placement_id: step for step in plan.steps}
        fixed_pair_ids: set[str] = set()
        if document.presentation.show_distance_bracket != "never":
            constraint_results = plan.constraint_results
        else:
            constraint_results = ()
        for result in constraint_results:
            upstream = step_by_id[result.upstream_placement_id]
            downstream = step_by_id[result.downstream_placement_id]
            upstream_strand = "rev" if upstream.orientation == "rev" else "fwd"
            downstream_strand = "rev" if downstream.orientation == "rev" else "fwd"
            if upstream_strand != downstream_strand:
                raise ValueError(
                    "fixed-distance playback endpoints must share a BaseRender strand "
                    f"({upstream.placement_id}={upstream_strand}, "
                    f"{downstream.placement_id}={downstream_strand})"
                )
            fixed_pair_ids.update((result.upstream_placement_id, result.downstream_placement_id))
        palette: dict[str, str] = {}
        features: list[Feature] = []
        placement_metadata = self._placement_metadata.get(plan.realization_digest, {})
        for index, step in enumerate(placed):
            tag = self._tag(index)
            palette[tag] = step_color(step, index, document.presentation.color_profile)
            strand = "rev" if step.orientation == "rev" else "fwd"
            sequence_segment = plan.realized_sequence[step.start : step.end]
            feature_label = sequence_segment.translate(_DNA_COMPLEMENT)[::-1] if strand == "rev" else sequence_segment
            feature_render = {"priority": 8}
            if step.placement_id in fixed_pair_ids:
                feature_render["track"] = 0
            feature_attrs: dict[str, object] = {"style_token": tag}
            if step.placement_kind == "fixed_element" and self._presentation.fixed_element_annotations == "variant":
                metadata = placement_metadata.get(step.placement_id)
                if metadata is None:
                    raise ValueError(f"fixed element {step.placement_id!r} is missing realized metadata")
                role = str(metadata.get("role") or "").strip()
                variant = str(metadata.get("variant_id") or "").strip()
                if role not in {"upstream", "downstream"} or not variant:
                    raise ValueError(f"fixed element {step.placement_id!r} requires role and variant_id")
                base_label = str(
                    document.label_overrides.get(
                        role,
                        "fixed upstream element" if role == "upstream" else "fixed downstream element",
                    )
                ).strip()
                include_suffix = not (
                    role == "downstream"
                    and variant.casefold() == "consensus"
                    and self._presentation.consensus_suffix == "omit"
                )
                display_label = f"{base_label} ({variant})" if include_suffix else base_label
                feature_attrs.update(
                    {
                        "source": "densegen_promoter",
                        "component": role,
                        "variant_id": variant,
                        "display_label": display_label,
                        "annotation_color": "#6B7280",
                    }
                )
            features.append(
                Feature(
                    id=step.placement_id,
                    kind="kmer",
                    span=Span(start=step.start, end=step.end, strand=strand),
                    label=feature_label,
                    tags=(tag,),
                    attrs=feature_attrs,
                    render=feature_render,
                )
            )
        placed_ids = {feature.id for feature in features}
        effects: list[Effect] = []
        for result in constraint_results:
            if result.upstream_placement_id not in placed_ids or result.downstream_placement_id not in placed_ids:
                continue
            effects.append(
                Effect(
                    kind="span_link",
                    target={
                        "from_feature_id": result.upstream_placement_id,
                        "to_feature_id": result.downstream_placement_id,
                    },
                    params={
                        "label": f"{result.actual_distance_bp} bp",
                        "lane": "top",
                        "shrink_label_to_fit": False,
                    },
                    render={"priority": 8, "track": 0},
                )
            )
        overlay = self._presentation.anchored_illustration
        if overlay is not None:
            matching_results = tuple(
                result
                for result in plan.constraint_results
                if result.label == overlay.constraint_name
                or result.constraint_id.split(":", 1)[0] == overlay.constraint_name
            )
            if len(matching_results) != 1:
                raise ValueError(
                    f"anchored illustration constraint {overlay.constraint_name!r} "
                    f"must resolve exactly once, found {len(matching_results)}"
                )
            result = matching_results[0]
            bindings = []
            for anchor_id, placement_id in (
                ("upstream", result.upstream_placement_id),
                ("downstream", result.downstream_placement_id),
            ):
                target_step = step_by_id[placement_id]
                bindings.append(
                    {
                        "anchor_id": anchor_id,
                        "feature_id": placement_id,
                        "start": target_step.start,
                        "end": target_step.end,
                    }
                )
            if any(binding["feature_id"] in placed_ids for binding in bindings):
                effects.append(
                    Effect(
                        kind="anchored_illustration",
                        target={"bindings": bindings},
                        params={
                            "asset_id": overlay.asset_id,
                            "width_px": 1216.0,
                            "top_gap_px": 8.0,
                            "fill_color": "#DDE2E7",
                            "fill_alpha": 0.42,
                        },
                        render={"priority": 6},
                    )
                )
        reveal_end = max(step.end for step in placed)
        is_final_step = step_index == len(plan.steps) - 1
        reveals_right_terminus = reveal_end == len(plan.realized_sequence)
        if is_final_step and not reveals_right_terminus:
            raise ValueError(
                "final dense-array playback step must reveal the complete realized sequence "
                f"(revealed through {reveal_end}, sequence length {len(plan.realized_sequence)})"
            )
        hidden = tuple(range(reveal_end, len(plan.realized_sequence)))
        record = Record(
            id=f"{plan.realization_digest}:{step_index}",
            alphabet="IUPAC_DNA",
            sequence=plan.realized_sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(),
            meta={
                "base_hidden_indices": {"primary": hidden, "complement": hidden},
                "terminal_label_visibility": {"left": True, "right": reveals_right_terminus},
            },
        ).validate()
        return record, palette

    def _prepare_document(self, document: PlaybackDocument) -> tuple[tuple[Record, ...], Palette]:
        records: list[Record] = []
        colors: dict[str, str] = {}
        for step_index in range(len(document.plan.steps)):
            record, step_colors = self._record_for_step(document, step_index)
            records.append(record)
            colors.update(step_colors)
        final_layout = compute_layout(records[-1], self._style)
        pinned_records: list[Record] = []
        for record in records:
            pinned_features = tuple(
                replace(
                    feature,
                    render={
                        **dict(feature.render),
                        "track": final_layout.feature_track_by_id[feature.id],
                    },
                )
                for feature in record.features
            )
            pinned_records.append(replace(record, features=pinned_features).validate())
        records = pinned_records
        top_extent = 0.0
        bottom_extent = 0.0
        for record in records:
            layout = compute_layout(record, self._style)
            centerline = (float(layout.y_forward) + float(layout.y_reverse)) / 2.0
            top_extent = max(top_extent, float(layout.content_top) - centerline)
            bottom_extent = max(bottom_extent, centerline - float(layout.content_bottom))
        if self._presentation.anchored_illustration is not None:
            top_extent += _ANCHORED_ILLUSTRATION_TOP_BAND_PX
        prepared = []
        for record in records:
            meta = dict(record.meta)
            meta["fixed_content_top_extent_px"] = top_extent
            meta["fixed_content_bottom_extent_px"] = bottom_extent
            prepared.append(replace(record, meta=meta))
        return tuple(prepared), Palette(colors)

    def _figure(self, document: PlaybackDocument, step_index: int):
        digest = document.plan.realization_digest
        records = self._records[digest]
        if not 0 <= step_index < len(records):
            raise IndexError(f"step_index out of range: {step_index}")
        return render_record(
            records[step_index],
            renderer_name="sequence_rows",
            style=self._style,
            palette=self._palettes[digest],
        )

    def render_rgba(self, document: PlaybackDocument, step_index: int) -> np.ndarray:
        key = (document.plan.realization_digest, step_index)
        cached = self._rgba_cache.get(key)
        if cached is not None:
            return cached
        import matplotlib.pyplot as plt

        figure = self._figure(document, step_index)
        buffer = io.BytesIO()
        figure.savefig(
            buffer, format="png", dpi=self._style.dpi, bbox_inches="tight", pad_inches=0.01, facecolor="white"
        )
        plt.close(figure)
        buffer.seek(0)
        rgba = np.asarray(Image.open(buffer).convert("RGBA")).copy()
        self._rgba_cache[key] = rgba
        return rgba

    def render_svg(self, document: PlaybackDocument, step_index: int) -> str:
        key = (document.plan.realization_digest, step_index)
        cached = self._svg_cache.get(key)
        if cached is not None:
            return cached
        import matplotlib.pyplot as plt

        figure = self._figure(document, step_index)
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0.01, facecolor="white")
        plt.close(figure)
        match = _SVG_RE.search(buffer.getvalue())
        if match is None:
            raise ValueError("BaseRender did not emit an SVG root")
        svg = match.group(1)
        self._svg_cache[key] = svg
        return svg

    def attach_svg_frames(self, documents: tuple[PlaybackDocument, ...]) -> tuple[PlaybackDocument, ...]:
        return tuple(
            replace(
                document,
                duplex_svg_frames=tuple(
                    self.render_svg(document, step_index) for step_index in range(len(document.plan.steps))
                ),
            )
            for document in documents
        )
