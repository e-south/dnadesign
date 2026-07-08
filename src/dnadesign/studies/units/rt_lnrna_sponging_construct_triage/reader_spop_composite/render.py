"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/render.py

Render the Reader SPOP condition heatmap with retron MSD thumbnails.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .condition_matrix import ReaderSpopConditionMatrix, ReaderSpopConditionRow
from .conditions import short_condition_label
from .identifiers import variant_sort_key
from .paths import resolve_repo_root, sha256_file
from .row_categories import RetronRowCategorySpan, category_spans_for_variants
from .structure_manifest import RetronStructureThumbnailRow
from .structure_svg import oriented_structure_geometry


class CompositeRenderError(ValueError):
    """Raised when the composite plot input violates a render contract."""


HEATMAP_TILE_ASPECT = "square"
PLOT_PREMISE = "Retron edits shift activation and growth"
TYPOGRAPHY_PROFILE = "publication_dense_v1"
PLOT_TITLE_FONTSIZE = 14.5
PANEL_TITLE_FONTSIZE = 11.0
GROUP_TITLE_FONTSIZE = 10.0
CATEGORY_LABEL_FONTSIZE_SINGLE = 6.8
CATEGORY_LABEL_FONTSIZE_MULTI = 7.8
CONDITION_TICK_FONTSIZE = 8.7
VARIANT_LABEL_FONTSIZE = 9.2
Y_AXIS_LABEL_FONTSIZE = 10.2
PRIMITIVE_TEXT_FONTSIZE = 7.0
PRIMITIVE_FOLDBACK_TEXT_FONTSIZE = 6.4
COLORBAR_LABEL_FONTSIZE = 8.4
COLORBAR_TICK_FONTSIZE = 8.2
ROW_CATEGORY_PANEL_LABEL = "Experiment group"
ROW_CATEGORY_BAND_POSITION = "left_of_heatmaps"
ROW_CATEGORY_BAND_SHAPE = "rounded_rectangles"
OD600_PANEL_LABEL = "OD600 rel."
RFP_OD600_PANEL_LABEL = "RFP/OD600 activation"
PRIMITIVE_PANEL_LABEL = "MSD primitives"
PANEL_ORDER = (
    ROW_CATEGORY_PANEL_LABEL,
    OD600_PANEL_LABEL,
    RFP_OD600_PANEL_LABEL,
    PRIMITIVE_PANEL_LABEL,
    "MSD structure",
)
PRIMITIVE_COLUMN_ORDER = (
    "left_base_sequence",
    "stem_length_bp",
    "foldback_sequence",
    "right_base_sequence",
)
PRIMITIVE_COLUMN_LABELS = ("left", "stem", "foldback", "right")
PRIMITIVE_COLUMN_SOURCE = "retron_hairpin_materialized_features_and_decomposed_msd_region_records"
PRIMITIVE_STEM_LENGTH_BASIS = "payload_primary_interval_plus_snapback_foldback_return_bp"
VALUE_PALETTE = "pastel_cold_to_warm_activation"
OD600_PANEL_PALETTE = "pastel_cold_to_warm_growth"
MISSING_TILE_COLOR = "#ffffff"
ZERO_VALUE_COLOR = "#c7e5ef"
HIGH_VALUE_COLOR = "#d58a63"
OD600_PANEL_BASIS = "condition_aligned_viability_relative_to_baseline"
NORMALIZATION_SCOPE = "within_reader_observation_not_cross_experiment_absolute"
NORMALIZATION_BASIS = (
    "Values are Reader SPOP normalized derepression rows. Baseline condition "
    "0 nm aTc; 0 uM IPTG is baseline=0. The observed aTc positive control at "
    "IPTG 0 is aTc positive control=1, preserving the actual aTc dose. IPTG "
    "dose tiles are condition medians and may be reconstructed from Reader "
    "normalized endpoints when raw dose RFP/OD600 rows are not carried forward."
)
Y_AXIS_LABEL = "lnRNA variants in retron Eco1 system"
STRUCTURE_THUMBNAIL_ORIENTATION = "rightward_horizontal_cap_right"
STRUCTURE_NUCLEOTIDE_TEXT_ORIENTATION = "upright"
STRUCTURE_RENDERING_MODE = "matplotlib_vector_primitives_from_viennarna_svg"
STRUCTURE_VECTOR_ASPECT_POLICY = "preserve_native_svg_aspect_ratio"
STRUCTURE_DEVIATION_REFERENCE_VARIANT = "retron26"
STRUCTURE_DEVIATION_HIGHLIGHT_MODE = "variant_text_indices_from_pairwise_alignment"
STRUCTURE_DEVIATION_LEGEND_LABEL = "amber bases differ from retron26"
STRUCTURE_DEVIATION_MARKER_COLOR = "#fde68a"
STRUCTURE_DEVIATION_MARKER_EDGE_COLOR = "#d97706"
STRUCTURE_DEVIATION_TEXT_COLOR = "#7c2d12"
STRUCTURE_DEVIATION_MARKER_SIZE = 9.0
STRUCTURE_DEVIATION_MARKER_ALPHA = 0.92
STRUCTURE_THUMBNAIL_FRAME = "none"
STRUCTURE_THUMBNAIL_CROP = "trim_white_margin"
STRUCTURE_THUMBNAIL_CROP_MARGIN_PX = 2
STRUCTURE_THUMBNAIL_INTERPOLATION = "lanczos"
STRUCTURE_THUMBNAIL_ZOOM = 0.12
STRUCTURE_THUMBNAIL_ROTATION_QUARTER_TURNS = -1
STRUCTURE_THUMBNAIL_HORIZONTAL_FLIP = True
STRUCTURE_VECTOR_ROW_HALF_HEIGHT = 0.42
STRUCTURE_VECTOR_MAX_WIDTH = 0.98
STRUCTURE_VECTOR_TEXT_FONTSIZE = 2.7
COLOR_SCALE = {"vmin": 0.0, "vmax": 1.0, "clip": True}
OD600_COLOR_SCALE = {"vmin": 0.0, "vmax": 1.2, "clip": True}
COLORBAR_ORIENTATION = "horizontal_bottom"
COLORBAR_HEIGHT_RATIO = 0.005
COLORBAR_BOTTOM = 0.095
COLORBAR_TICK_LABEL_STYLE = "compact_numeric"
LABEL_COLLISION_POLICY = "tight_cbar_row_close_to_compact_condition_ticks"
CONDITION_TICK_LABEL_STYLE = "compact_aTc_IPTG"
CONDITION_TICK_LABEL_PRESENCE = "both_heatmaps"
LAYOUT_DENSITY = "compact_adjacent_panels"
PLOT_DPI = 450


@dataclass(frozen=True, slots=True)
class SpopConditionStructurePlotManifest:
    manifest_path: str
    plot_png_path: str
    plot_svg_path: str
    variant_count: int
    condition_count: int
    missing_cell_count: int
    structure_thumbnail_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def render_spop_condition_structure_plot(
    *,
    condition_matrix: ReaderSpopConditionMatrix,
    thumbnail_rows: Sequence[RetronStructureThumbnailRow],
    output_dir: Path,
    repo_root: Path | None = None,
) -> SpopConditionStructurePlotManifest:
    """Render the study-owned SPOP condition heatmap with MSD thumbnails."""

    root = resolve_repo_root(repo_root)
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plot_png_path = resolved_output_dir / "reader_spop_condition_structure_heatmap.png"
    plot_svg_path = resolved_output_dir / "reader_spop_condition_structure_heatmap.svg"
    manifest_path = resolved_output_dir / "manifest.json"
    variants = _ordered_variants(condition_matrix.rows)
    columns = condition_matrix.condition_columns
    row_category_spans = category_spans_for_variants(variants)
    values = _matrix_values(
        condition_matrix.rows,
        variants=variants,
        condition_keys=[col.condition_key for col in columns],
    )
    od600_values = _od600_matrix_values(
        condition_matrix.rows,
        variants=variants,
        condition_keys=[col.condition_key for col in columns],
    )
    _render_heatmap(
        values=values,
        od600_values=od600_values,
        variants=variants,
        row_category_spans=row_category_spans,
        condition_labels=[_plot_condition_label(col.condition_key) for col in columns],
        thumbnail_by_variant={row.assay_subject_key: row for row in thumbnail_rows},
        root=root,
        png_path=plot_png_path,
        svg_path=plot_svg_path,
    )
    missing_cell_count = int(np.isnan(values).sum())
    payload = {
        "contract": "rt_lnrna_spop_condition_structure_plot_manifest_v1",
        "plot_premise": PLOT_PREMISE,
        "plot_title": PLOT_PREMISE,
        "typography_profile": TYPOGRAPHY_PROFILE,
        "plot_title_fontsize": PLOT_TITLE_FONTSIZE,
        "panel_title_fontsize": PANEL_TITLE_FONTSIZE,
        "condition_tick_fontsize": CONDITION_TICK_FONTSIZE,
        "variant_label_fontsize": VARIANT_LABEL_FONTSIZE,
        "primitive_text_fontsize": PRIMITIVE_TEXT_FONTSIZE,
        "panel_order": list(PANEL_ORDER),
        "variant_count": len(variants),
        "condition_count": len(columns),
        "missing_cell_count": missing_cell_count,
        "missing_cell_rendering": "white_not_zero",
        "missing_tile_color": MISSING_TILE_COLOR,
        "heatmap_tile_aspect": HEATMAP_TILE_ASPECT,
        "row_category_panel_label": ROW_CATEGORY_PANEL_LABEL,
        "row_category_band_position": ROW_CATEGORY_BAND_POSITION,
        "row_category_band_shape": ROW_CATEGORY_BAND_SHAPE,
        "row_category_count": len(row_category_spans),
        "row_category_spans": [span.to_dict() for span in row_category_spans],
        "row_category_palette": _category_palette(row_category_spans),
        "condition_tick_label_style": CONDITION_TICK_LABEL_STYLE,
        "condition_tick_label_presence": CONDITION_TICK_LABEL_PRESENCE,
        "value_palette": VALUE_PALETTE,
        "zero_value_color": ZERO_VALUE_COLOR,
        "high_value_color": HIGH_VALUE_COLOR,
        "layout_density": LAYOUT_DENSITY,
        "primitive_column_order": list(PRIMITIVE_COLUMN_ORDER),
        "primitive_column_labels": list(PRIMITIVE_COLUMN_LABELS),
        "primitive_column_source": PRIMITIVE_COLUMN_SOURCE,
        "primitive_stem_length_basis": PRIMITIVE_STEM_LENGTH_BASIS,
        "od600_panel_label": OD600_PANEL_LABEL,
        "od600_panel_basis": OD600_PANEL_BASIS,
        "od600_panel_palette": OD600_PANEL_PALETTE,
        "od600_panel_condition_count": len(columns),
        "od600_color_scale": OD600_COLOR_SCALE,
        "color_scale": COLOR_SCALE,
        "colorbar_orientation": COLORBAR_ORIENTATION,
        "colorbar_height_ratio": COLORBAR_HEIGHT_RATIO,
        "colorbar_tick_label_style": COLORBAR_TICK_LABEL_STYLE,
        "label_collision_policy": LABEL_COLLISION_POLICY,
        "plot_dpi": PLOT_DPI,
        "normalization_scope": NORMALIZATION_SCOPE,
        "normalization_basis": NORMALIZATION_BASIS,
        "x_axis_label": "",
        "y_axis_label": Y_AXIS_LABEL,
        "structure_thumbnail_orientation": STRUCTURE_THUMBNAIL_ORIENTATION,
        "structure_nucleotide_text_orientation": STRUCTURE_NUCLEOTIDE_TEXT_ORIENTATION,
        "structure_rendering_mode": STRUCTURE_RENDERING_MODE,
        "structure_vector_aspect_policy": STRUCTURE_VECTOR_ASPECT_POLICY,
        "structure_deviation_reference_variant": STRUCTURE_DEVIATION_REFERENCE_VARIANT,
        "structure_deviation_highlight_mode": STRUCTURE_DEVIATION_HIGHLIGHT_MODE,
        "structure_deviation_legend_label": STRUCTURE_DEVIATION_LEGEND_LABEL,
        "structure_deviation_marker_color": STRUCTURE_DEVIATION_MARKER_COLOR,
        "structure_deviation_marker_size": STRUCTURE_DEVIATION_MARKER_SIZE,
        "structure_deviation_marker_alpha": STRUCTURE_DEVIATION_MARKER_ALPHA,
        "structure_thumbnail_horizontal_flip": STRUCTURE_THUMBNAIL_HORIZONTAL_FLIP,
        "structure_thumbnail_frame": STRUCTURE_THUMBNAIL_FRAME,
        "structure_thumbnail_crop": STRUCTURE_THUMBNAIL_CROP,
        "structure_thumbnail_crop_margin_px": STRUCTURE_THUMBNAIL_CROP_MARGIN_PX,
        "structure_thumbnail_interpolation": STRUCTURE_THUMBNAIL_INTERPOLATION,
        "structure_thumbnail_zoom": STRUCTURE_THUMBNAIL_ZOOM,
        "structure_vector_row_half_height": STRUCTURE_VECTOR_ROW_HALF_HEIGHT,
        "structure_vector_text_fontsize": STRUCTURE_VECTOR_TEXT_FONTSIZE,
        "missing_structure_summary": _missing_structure_summary(thumbnail_rows),
        "source_reader_experiment_ids": list(condition_matrix.source_reader_experiment_ids),
        "plot_png": plot_png_path.name,
        "plot_svg": plot_svg_path.name,
        "condition_columns": [col.to_dict() for col in columns],
        "structure_thumbnail_rows": len(thumbnail_rows),
        "source_digests": _source_digests(thumbnail_rows=thumbnail_rows, root=root),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return SpopConditionStructurePlotManifest(
        manifest_path=manifest_path.as_posix(),
        plot_png_path=plot_png_path.as_posix(),
        plot_svg_path=plot_svg_path.as_posix(),
        variant_count=len(variants),
        condition_count=len(columns),
        missing_cell_count=missing_cell_count,
        structure_thumbnail_rows=len(thumbnail_rows),
    )


def _render_heatmap(
    *,
    values: np.ndarray,
    od600_values: np.ndarray,
    variants: Sequence[str],
    row_category_spans: Sequence[RetronRowCategorySpan],
    condition_labels: Sequence[str],
    thumbnail_by_variant: Mapping[str, RetronStructureThumbnailRow],
    root: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyBboxPatch

    variant_count = max(1, len(variants))
    condition_count = max(1, len(condition_labels))
    height = max(5.7, 0.38 * variant_count + 2.55)
    width = max(12.2, 0.42 * condition_count * 2 + 8.1)
    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    fig.suptitle(PLOT_PREMISE, fontsize=PLOT_TITLE_FONTSIZE, y=0.984)
    panel_bottom = 0.155
    panel_top = 0.945
    panel_height = panel_top - panel_bottom
    heatmap_width = min(0.17, panel_height * condition_count / variant_count * height / width)
    category_left = 0.032
    category_width = 0.115
    left = 0.235
    heatmap_gap = 0.012
    primitive_gap = 0.01
    primitive_width = 0.215
    structure_gap = 0.006
    primitive_left = left + 2 * heatmap_width + heatmap_gap + primitive_gap
    structure_left = primitive_left + primitive_width + structure_gap
    structure_width = min(0.23, max(0.14, 0.99 - structure_left))
    category_ax = fig.add_axes([category_left, panel_bottom, category_width, panel_height])
    od600_ax = fig.add_axes([left, panel_bottom, heatmap_width, panel_height])
    heatmap_ax = fig.add_axes([left + heatmap_width + heatmap_gap, panel_bottom, heatmap_width, panel_height])
    primitive_ax = fig.add_axes([primitive_left, panel_bottom, primitive_width, panel_height])
    thumb_ax = fig.add_axes([structure_left, panel_bottom, structure_width, panel_height])
    od600_colorbar_ax = fig.add_axes([left, COLORBAR_BOTTOM, heatmap_width, COLORBAR_HEIGHT_RATIO])
    heatmap_colorbar_ax = fig.add_axes(
        [left + heatmap_width + heatmap_gap, COLORBAR_BOTTOM, heatmap_width, COLORBAR_HEIGHT_RATIO]
    )
    fig.text(
        0.012,
        panel_bottom + panel_height / 2.0,
        Y_AXIS_LABEL,
        rotation=90,
        ha="center",
        va="center",
        fontsize=Y_AXIS_LABEL_FONTSIZE,
    )
    _draw_category_band(FancyBboxPatch, category_ax, row_category_spans=row_category_spans, variant_count=len(variants))

    cmap = LinearSegmentedColormap.from_list(VALUE_PALETTE, [ZERO_VALUE_COLOR, "#f2d6a2", HIGH_VALUE_COLOR])
    cmap.set_bad(MISSING_TILE_COLOR)
    plotted_values = np.clip(values, COLOR_SCALE["vmin"], COLOR_SCALE["vmax"])
    image = heatmap_ax.imshow(
        np.ma.masked_invalid(plotted_values),
        aspect="equal",
        cmap=cmap,
        vmin=COLOR_SCALE["vmin"],
        vmax=COLOR_SCALE["vmax"],
    )
    heatmap_ax.set_aspect("equal", adjustable="box")
    heatmap_ax.set_anchor("W")
    heatmap_ax.set_xticks(np.arange(len(condition_labels)))
    heatmap_ax.set_xticklabels(condition_labels, rotation=55, ha="right")
    heatmap_ax.set_yticks(np.arange(len(variants)))
    heatmap_ax.set_yticklabels([])
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("")
    heatmap_ax.set_title(RFP_OD600_PANEL_LABEL, fontsize=PANEL_TITLE_FONTSIZE, pad=5)
    heatmap_ax.tick_params(axis="x", labelsize=CONDITION_TICK_FONTSIZE, pad=1)
    heatmap_ax.tick_params(axis="y", length=0)
    heatmap_ax.set_xticks(np.arange(-0.5, len(condition_labels), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=0.8)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    _draw_category_separators(heatmap_ax, row_category_spans=row_category_spans)

    od600_cmap = LinearSegmentedColormap.from_list(OD600_PANEL_PALETTE, [ZERO_VALUE_COLOR, "#f2d6a2", HIGH_VALUE_COLOR])
    od600_cmap.set_bad(MISSING_TILE_COLOR)
    clipped_od600_values = np.clip(od600_values, OD600_COLOR_SCALE["vmin"], OD600_COLOR_SCALE["vmax"])
    od600_image = od600_ax.imshow(
        np.ma.masked_invalid(clipped_od600_values),
        aspect="equal",
        cmap=od600_cmap,
        vmin=OD600_COLOR_SCALE["vmin"],
        vmax=OD600_COLOR_SCALE["vmax"],
    )
    od600_ax.set_aspect("equal", adjustable="box")
    od600_ax.set_anchor("E")
    od600_ax.set_xticks(np.arange(len(condition_labels)))
    od600_ax.set_xticklabels(condition_labels, rotation=55, ha="right")
    od600_ax.set_yticks(np.arange(len(variants)))
    od600_ax.set_yticklabels(variants)
    od600_ax.set_ylabel("")
    od600_ax.set_title(OD600_PANEL_LABEL, fontsize=PANEL_TITLE_FONTSIZE, pad=5)
    od600_ax.set_xticks(np.arange(-0.5, len(condition_labels), 1), minor=True)
    od600_ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    od600_ax.grid(which="minor", color="white", linewidth=0.8)
    od600_ax.tick_params(axis="x", labelsize=CONDITION_TICK_FONTSIZE, pad=1)
    od600_ax.tick_params(axis="y", labelsize=VARIANT_LABEL_FONTSIZE, pad=2)
    od600_ax.tick_params(which="minor", bottom=False, left=False)
    _draw_category_separators(od600_ax, row_category_spans=row_category_spans)

    primitive_ax.set_xlim(0, 4.8)
    primitive_ax.set_ylim(len(variants) - 0.5, -0.5)
    primitive_ax.set_anchor("W")
    primitive_ax.set_title(PRIMITIVE_PANEL_LABEL, fontsize=PANEL_TITLE_FONTSIZE, pad=5)
    primitive_ax.set_xticks([0.45, 1.25, 2.75, 4.25])
    primitive_ax.set_xticklabels(PRIMITIVE_COLUMN_LABELS)
    primitive_ax.xaxis.tick_top()
    primitive_ax.tick_params(axis="x", labelsize=CONDITION_TICK_FONTSIZE, pad=1, length=0)
    primitive_ax.set_yticks([])
    primitive_ax.set_frame_on(False)
    primitive_ax.patch.set_alpha(0)
    for spine in primitive_ax.spines.values():
        spine.set_visible(False)
    for index, variant in enumerate(variants):
        row = thumbnail_by_variant.get(variant)
        primitive_values = _primitive_display_values(row)
        primitive_ax.text(
            0.45,
            index,
            primitive_values[0],
            ha="center",
            va="center",
            fontsize=PRIMITIVE_TEXT_FONTSIZE,
            family="monospace",
        )
        primitive_ax.text(1.25, index, primitive_values[1], ha="center", va="center", fontsize=PRIMITIVE_TEXT_FONTSIZE)
        primitive_ax.text(
            2.75,
            index,
            primitive_values[2],
            ha="center",
            va="center",
            fontsize=PRIMITIVE_FOLDBACK_TEXT_FONTSIZE,
            family="monospace",
        )
        primitive_ax.text(
            4.25,
            index,
            primitive_values[3],
            ha="center",
            va="center",
            fontsize=PRIMITIVE_TEXT_FONTSIZE,
            family="monospace",
        )
    _draw_category_separators(primitive_ax, row_category_spans=row_category_spans)

    thumb_ax.set_xlim(0, 1)
    thumb_ax.set_ylim(len(variants) - 0.5, -0.5)
    thumb_ax.set_anchor("W")
    thumb_ax.set_xticks([])
    thumb_ax.set_yticks([])
    thumb_ax.set_title("MSD structure", fontsize=PANEL_TITLE_FONTSIZE, pad=5)
    legend_y = len(variants) + 0.38
    thumb_ax.scatter(
        [0.02],
        [legend_y],
        s=STRUCTURE_DEVIATION_MARKER_SIZE,
        marker="o",
        color=STRUCTURE_DEVIATION_MARKER_COLOR,
        edgecolors=STRUCTURE_DEVIATION_MARKER_EDGE_COLOR,
        linewidths=0.22,
        alpha=STRUCTURE_DEVIATION_MARKER_ALPHA,
        clip_on=False,
        zorder=4,
    )
    thumb_ax.text(
        0.055,
        legend_y,
        STRUCTURE_DEVIATION_LEGEND_LABEL,
        ha="left",
        va="center",
        fontsize=7.4,
        color=STRUCTURE_DEVIATION_TEXT_COLOR,
        clip_on=False,
    )
    thumb_ax.set_frame_on(False)
    thumb_ax.patch.set_alpha(0)
    for spine in thumb_ax.spines.values():
        spine.set_visible(False)
    reference_sequence = _structure_reference_sequence(thumbnail_by_variant, root=root)
    for index, variant in enumerate(variants):
        row = thumbnail_by_variant.get(variant)
        structure_svg_path = _thumbnail_svg_path(row, root=root)
        if structure_svg_path is not None:
            _draw_vector_structure(
                LineCollection,
                thumb_ax,
                svg_path=structure_svg_path,
                row_index=index,
                reference_sequence=reference_sequence,
            )
            continue
        image_data = _thumbnail_image_data(row, root=root, mpimg=mpimg)
        if image_data is not None:
            thumbnail = OffsetImage(
                image_data,
                zoom=STRUCTURE_THUMBNAIL_ZOOM,
                interpolation=STRUCTURE_THUMBNAIL_INTERPOLATION,
            )
            thumb_ax.add_artist(
                AnnotationBbox(
                    thumbnail,
                    (0.0, index),
                    xycoords=("axes fraction", "data"),
                    frameon=False,
                    box_alignment=(0.0, 0.5),
                    pad=0.0,
                )
            )
            continue
        thumb_ax.text(0.02, index, "na", ha="left", va="center", fontsize=6, color="#6b7280")
    _draw_category_separators(thumb_ax, row_category_spans=row_category_spans)
    colorbar = fig.colorbar(image, cax=heatmap_colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0", ".5", "1"])
    colorbar.set_label("normalized derepression", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=0)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, pad=0)
    od600_colorbar = fig.colorbar(od600_image, cax=od600_colorbar_ax, orientation="horizontal")
    od600_colorbar.set_ticks([0.0, 1.0])
    od600_colorbar.set_ticklabels(["0", "1"])
    od600_colorbar.set_label("OD600 rel.", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=0)
    od600_colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, pad=0)
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _draw_category_band(
    fancy_bbox_patch_cls,
    ax,
    *,
    row_category_spans: Sequence[RetronRowCategorySpan],
    variant_count: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(variant_count - 0.5, -0.5)
    ax.set_title("Group", fontsize=GROUP_TITLE_FONTSIZE, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.patch.set_alpha(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for span in row_category_spans:
        row_count = max(1, span.stop_index - span.start_index)
        y0 = span.start_index - 0.45
        height = max(0.62, row_count - 0.10)
        rect = fancy_bbox_patch_cls(
            (0.02, y0),
            0.96,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            facecolor=span.color,
            edgecolor="#ffffff",
            linewidth=0.8,
            mutation_aspect=1.0,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            0.50,
            span.start_index + (row_count - 1) / 2.0,
            span.display_label,
            ha="center",
            va="center",
            fontsize=CATEGORY_LABEL_FONTSIZE_SINGLE if row_count == 1 else CATEGORY_LABEL_FONTSIZE_MULTI,
            color=span.text_color,
            linespacing=0.9,
            zorder=2,
        )


def _draw_category_separators(ax, *, row_category_spans: Sequence[RetronRowCategorySpan]) -> None:
    for span in row_category_spans[:-1]:
        ax.axhline(span.stop_index - 0.5, color="#d1d5db", linewidth=0.55, zorder=5)


def _thumbnail_image_path(row: RetronStructureThumbnailRow | None, *, root: Path) -> Path | None:
    if row is None or row.structure_status != "available":
        return None
    if not row.structure_png_path:
        raise CompositeRenderError(f"{row.assay_subject_key}: available thumbnail row has no structure_png_path")
    image_path = root / row.structure_png_path
    if not image_path.exists():
        raise CompositeRenderError(
            f"{row.assay_subject_key}: structure_status is available but thumbnail is missing: {row.structure_png_path}"
        )
    return image_path


def _thumbnail_svg_path(row: RetronStructureThumbnailRow | None, *, root: Path) -> Path | None:
    if row is None or row.structure_status != "available" or not row.structure_svg_path:
        return None
    svg_path = root / row.structure_svg_path
    if not svg_path.exists():
        raise CompositeRenderError(
            f"{row.assay_subject_key}: structure_svg_path is set but SVG is missing: {row.structure_svg_path}"
        )
    return svg_path


def _thumbnail_image_data(row: RetronStructureThumbnailRow | None, *, root: Path, mpimg) -> np.ndarray | None:
    image_path = _thumbnail_image_path(row, root=root)
    if image_path is not None:
        return _orient_thumbnail(mpimg.imread(image_path))
    return None


def _draw_vector_structure(
    line_collection_cls,
    ax,
    *,
    svg_path: Path,
    row_index: int,
    reference_sequence: str,
) -> None:
    geometry = oriented_structure_geometry(svg_path.as_posix())
    min_x, max_x, min_y, max_y = geometry.bounds
    x_span = max(1.0, max_x - min_x)
    y_span = max(1.0, max_y - min_y)
    y_center = row_index
    y_mid = (min_y + max_y) / 2.0
    x_pixels_per_data, y_pixels_per_data = _axis_pixels_per_data(ax)
    width_data, height_data = _structure_vector_data_size(
        source_width=x_span,
        source_height=y_span,
        x_pixels_per_data=x_pixels_per_data,
        y_pixels_per_data=y_pixels_per_data,
        max_width_data=STRUCTURE_VECTOR_MAX_WIDTH,
        max_height_data=STRUCTURE_VECTOR_ROW_HALF_HEIGHT * 2.0,
    )
    deviation_indices = set(
        _deviating_structure_text_indices(
            reference_sequence=reference_sequence,
            variant_sequence=_geometry_text_sequence(geometry),
        )
    )

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            (x - min_x) / x_span * width_data,
            y_center + (y - y_mid) / y_span * height_data,
        )

    segments = [tuple(map_point(point) for point in line.points) for line in geometry.lines]
    colors = [line.color for line in geometry.lines]
    widths = [max(0.16, line.width * 0.12) for line in geometry.lines]
    collection = line_collection_cls(
        segments,
        colors=colors,
        linewidths=widths,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    ax.add_collection(collection)
    for text_index, text in enumerate(geometry.texts):
        x, y = map_point(text.point)
        is_deviation = text_index in deviation_indices
        if is_deviation:
            ax.scatter(
                [x],
                [y],
                s=STRUCTURE_DEVIATION_MARKER_SIZE,
                marker="o",
                color=STRUCTURE_DEVIATION_MARKER_COLOR,
                edgecolors=STRUCTURE_DEVIATION_MARKER_EDGE_COLOR,
                linewidths=0.22,
                alpha=STRUCTURE_DEVIATION_MARKER_ALPHA,
                zorder=3.6,
            )
        ax.text(
            x,
            y,
            text.text,
            ha="center",
            va="center",
            fontsize=STRUCTURE_VECTOR_TEXT_FONTSIZE,
            fontweight="bold" if is_deviation else "normal",
            color=STRUCTURE_DEVIATION_TEXT_COLOR if is_deviation else "#111827",
            zorder=4,
        )


def _axis_pixels_per_data(ax) -> tuple[float, float]:
    fig_width, fig_height = ax.figure.get_size_inches()
    position = ax.get_position()
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    return (
        position.width * fig_width * ax.figure.dpi / max(1e-9, x_range),
        position.height * fig_height * ax.figure.dpi / max(1e-9, y_range),
    )


def _structure_vector_data_size(
    *,
    source_width: float,
    source_height: float,
    x_pixels_per_data: float,
    y_pixels_per_data: float,
    max_width_data: float,
    max_height_data: float,
) -> tuple[float, float]:
    source_width = max(1e-9, source_width)
    source_height = max(1e-9, source_height)
    scale_pixels_per_source_unit = min(
        max_width_data * x_pixels_per_data / source_width,
        max_height_data * y_pixels_per_data / source_height,
    )
    return (
        source_width * scale_pixels_per_source_unit / x_pixels_per_data,
        source_height * scale_pixels_per_source_unit / y_pixels_per_data,
    )


def _structure_reference_sequence(
    thumbnail_by_variant: Mapping[str, RetronStructureThumbnailRow],
    *,
    root: Path,
) -> str:
    reference_row = thumbnail_by_variant.get(STRUCTURE_DEVIATION_REFERENCE_VARIANT)
    reference_svg_path = _thumbnail_svg_path(reference_row, root=root)
    if reference_svg_path is None:
        return ""
    return _structure_text_sequence(reference_svg_path)


def _structure_text_sequence(svg_path: Path) -> str:
    return _geometry_text_sequence(oriented_structure_geometry(svg_path.as_posix()))


def _geometry_text_sequence(geometry) -> str:
    return "".join(text.text for text in geometry.texts)


def _deviating_structure_text_indices(*, reference_sequence: str, variant_sequence: str) -> tuple[int, ...]:
    if not reference_sequence or not variant_sequence:
        return ()
    reference = _normalize_structure_sequence(reference_sequence)
    variant = _normalize_structure_sequence(variant_sequence)
    matcher = SequenceMatcher(a=reference, b=variant, autojunk=False)
    indices: set[int] = set()
    for tag, _ref_start, _ref_stop, variant_start, variant_stop in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "insert"}:
            indices.update(range(variant_start, variant_stop))
    return tuple(sorted(indices))


def _normalize_structure_sequence(sequence: str) -> str:
    return sequence.upper().replace("U", "T")


def _orient_thumbnail(image_data: np.ndarray) -> np.ndarray:
    rotated = np.rot90(_crop_white_margin(image_data), k=STRUCTURE_THUMBNAIL_ROTATION_QUARTER_TURNS)
    if STRUCTURE_THUMBNAIL_HORIZONTAL_FLIP:
        return np.fliplr(rotated)
    return rotated


def _primitive_display_values(row: RetronStructureThumbnailRow | None) -> tuple[str, str, str, str]:
    if row is None:
        return ("na", "na", "na", "na")
    stem_length = "na" if row.stem_length_bp is None else f"{row.stem_length_bp} bp"
    return (
        row.left_base_sequence or "na",
        stem_length,
        row.foldback_sequence or "na",
        row.right_base_sequence or "na",
    )


def _crop_white_margin(image_data: np.ndarray, *, margin_px: int = STRUCTURE_THUMBNAIL_CROP_MARGIN_PX) -> np.ndarray:
    rgb = image_data[..., :3]
    non_white = np.any(rgb < 0.98, axis=2)
    if not np.any(non_white):
        return image_data
    row_indices, col_indices = np.where(non_white)
    row_start = max(0, int(row_indices.min()) - margin_px)
    row_stop = min(image_data.shape[0], int(row_indices.max()) + margin_px + 1)
    col_start = max(0, int(col_indices.min()) - margin_px)
    col_stop = min(image_data.shape[1], int(col_indices.max()) + margin_px + 1)
    return image_data[row_start:row_stop, col_start:col_stop]


def _plot_condition_label(condition_key: str) -> str:
    return short_condition_label(condition_key).replace(" nm aTc", " aTc").replace(" uM IPTG", " IPTG")


def _matrix_values(
    rows: Sequence[ReaderSpopConditionRow],
    *,
    variants: Sequence[str],
    condition_keys: Sequence[str],
) -> np.ndarray:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.assay_subject_key, row.condition_key), []).append(float(row.normalized_derepression))
    values = np.full((len(variants), len(condition_keys)), np.nan, dtype=float)
    variant_index = {variant: index for index, variant in enumerate(variants)}
    condition_index = {condition: index for index, condition in enumerate(condition_keys)}
    for (variant, condition), group_values in grouped.items():
        values[variant_index[variant], condition_index[condition]] = float(statistics.median(group_values))
    return values


def _od600_matrix_values(
    rows: Sequence[ReaderSpopConditionRow],
    *,
    variants: Sequence[str],
    condition_keys: Sequence[str],
) -> np.ndarray:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.viability_relative_to_baseline is None:
            continue
        grouped.setdefault((row.assay_subject_key, row.condition_key), []).append(
            float(row.viability_relative_to_baseline)
        )
    values = np.full((len(variants), len(condition_keys)), np.nan, dtype=float)
    variant_index = {variant: index for index, variant in enumerate(variants)}
    condition_index = {condition: index for index, condition in enumerate(condition_keys)}
    for (variant, condition), group_values in grouped.items():
        values[variant_index[variant], condition_index[condition]] = float(statistics.median(group_values))
    return values


def _ordered_variants(rows: Sequence[ReaderSpopConditionRow]) -> tuple[str, ...]:
    return tuple(sorted({row.assay_subject_key for row in rows}, key=variant_sort_key))


def _source_digests(*, thumbnail_rows: Sequence[RetronStructureThumbnailRow], root: Path) -> dict[str, str]:
    paths = {
        path
        for row in thumbnail_rows
        for path in (row.review_manifest_path, row.primitive_source_path, row.structure_svg_path)
        if path
    }
    return {path: sha256_file(root / path) for path in sorted(paths)}


def _missing_structure_summary(thumbnail_rows: Sequence[RetronStructureThumbnailRow]) -> dict[str, object]:
    by_status = Counter(row.structure_status for row in thumbnail_rows)
    missing_rows = [row for row in thumbnail_rows if row.structure_status != "available"]
    return {
        "available": by_status.get("available", 0),
        "missing": len(missing_rows),
        "by_status": dict(sorted(by_status.items())),
        "missing_assay_subject_keys": [row.assay_subject_key for row in missing_rows],
        "explanation": (
            "Rows marked missing are absent from the configured retron-hairpin "
            "materialized structure source, not silently plotted as zero."
        ),
    }


def _category_palette(row_category_spans: Sequence[RetronRowCategorySpan]) -> dict[str, str]:
    return {span.category_id: span.color for span in row_category_spans}
