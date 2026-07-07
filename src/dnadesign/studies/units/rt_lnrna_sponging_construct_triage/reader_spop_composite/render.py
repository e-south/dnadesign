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
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .condition_matrix import ReaderSpopConditionMatrix, ReaderSpopConditionRow
from .conditions import short_condition_label
from .identifiers import variant_sort_key
from .paths import resolve_repo_root, sha256_file
from .structure_manifest import RetronStructureThumbnailRow


class CompositeRenderError(ValueError):
    """Raised when the composite plot input violates a render contract."""


HEATMAP_TILE_ASPECT = "square"
VALUE_PALETTE = "white_to_darker_seagreen"
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
STRUCTURE_THUMBNAIL_FRAME = "none"
STRUCTURE_THUMBNAIL_CROP = "trim_white_margin"
STRUCTURE_THUMBNAIL_ZOOM = 0.2
STRUCTURE_THUMBNAIL_ROTATION_QUARTER_TURNS = -1
COLOR_SCALE = {"vmin": 0.0, "vmax": 1.0, "clip": True}


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
    values = _matrix_values(
        condition_matrix.rows,
        variants=variants,
        condition_keys=[col.condition_key for col in columns],
    )
    _render_heatmap(
        values=values,
        variants=variants,
        condition_labels=[short_condition_label(col.condition_key) for col in columns],
        thumbnail_by_variant={row.assay_subject_key: row for row in thumbnail_rows},
        root=root,
        png_path=plot_png_path,
        svg_path=plot_svg_path,
    )
    missing_cell_count = int(np.isnan(values).sum())
    payload = {
        "contract": "rt_lnrna_spop_condition_structure_plot_manifest_v1",
        "variant_count": len(variants),
        "condition_count": len(columns),
        "missing_cell_count": missing_cell_count,
        "missing_cell_rendering": "masked_gray_not_zero",
        "heatmap_tile_aspect": HEATMAP_TILE_ASPECT,
        "value_palette": VALUE_PALETTE,
        "color_scale": COLOR_SCALE,
        "normalization_scope": NORMALIZATION_SCOPE,
        "normalization_basis": NORMALIZATION_BASIS,
        "x_axis_label": "",
        "y_axis_label": Y_AXIS_LABEL,
        "structure_thumbnail_orientation": STRUCTURE_THUMBNAIL_ORIENTATION,
        "structure_thumbnail_frame": STRUCTURE_THUMBNAIL_FRAME,
        "structure_thumbnail_crop": STRUCTURE_THUMBNAIL_CROP,
        "structure_thumbnail_zoom": STRUCTURE_THUMBNAIL_ZOOM,
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
    variants: Sequence[str],
    condition_labels: Sequence[str],
    thumbnail_by_variant: Mapping[str, RetronStructureThumbnailRow],
    root: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    height = max(4.0, 0.34 * max(1, len(variants)) + 2.2)
    width = max(8.5, 0.44 * max(1, len(condition_labels)) + 5.4)
    fig, (heatmap_ax, thumb_ax) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [max(1.0, len(condition_labels)), 4.4]},
        constrained_layout=True,
    )
    cmap = LinearSegmentedColormap.from_list(VALUE_PALETTE, ["#ffffff", "#146c43"])
    cmap.set_bad("#e5e7eb")
    plotted_values = np.clip(values, COLOR_SCALE["vmin"], COLOR_SCALE["vmax"])
    image = heatmap_ax.imshow(
        np.ma.masked_invalid(plotted_values),
        aspect="equal",
        cmap=cmap,
        vmin=COLOR_SCALE["vmin"],
        vmax=COLOR_SCALE["vmax"],
    )
    heatmap_ax.set_aspect("equal", adjustable="box")
    heatmap_ax.set_xticks(np.arange(len(condition_labels)))
    heatmap_ax.set_xticklabels(condition_labels, rotation=35, ha="right")
    heatmap_ax.set_yticks(np.arange(len(variants)))
    heatmap_ax.set_yticklabels(variants)
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel(Y_AXIS_LABEL)
    heatmap_ax.set_xticks(np.arange(-0.5, len(condition_labels), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=0.8)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.035, pad=0.02)
    colorbar.set_label("normalized derepression\n(baseline=0, aTc positive control=1)")

    thumb_ax.set_xlim(0, 1)
    thumb_ax.set_ylim(len(variants) - 0.5, -0.5)
    thumb_ax.set_xticks([])
    thumb_ax.set_yticks([])
    thumb_ax.set_title("MSD structure", fontsize=9)
    thumb_ax.set_frame_on(False)
    thumb_ax.patch.set_alpha(0)
    for spine in thumb_ax.spines.values():
        spine.set_visible(False)
    for index, variant in enumerate(variants):
        row = thumbnail_by_variant.get(variant)
        image_path = _thumbnail_image_path(row, root=root)
        if image_path is not None:
            image_data = _orient_thumbnail(mpimg.imread(image_path))
            thumbnail = OffsetImage(image_data, zoom=STRUCTURE_THUMBNAIL_ZOOM)
            thumb_ax.add_artist(AnnotationBbox(thumbnail, (0.5, index), frameon=False))
            continue
        thumb_ax.text(0.5, index, "na", ha="center", va="center", fontsize=5, color="#6b7280")
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    plt.close(fig)


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


def _orient_thumbnail(image_data: np.ndarray) -> np.ndarray:
    return np.rot90(_crop_white_margin(image_data), k=STRUCTURE_THUMBNAIL_ROTATION_QUARTER_TURNS)


def _crop_white_margin(image_data: np.ndarray, *, margin_px: int = 6) -> np.ndarray:
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


def _ordered_variants(rows: Sequence[ReaderSpopConditionRow]) -> tuple[str, ...]:
    return tuple(sorted({row.assay_subject_key for row in rows}, key=variant_sort_key))


def _source_digests(*, thumbnail_rows: Sequence[RetronStructureThumbnailRow], root: Path) -> dict[str, str]:
    paths = {row.review_manifest_path for row in thumbnail_rows if row.review_manifest_path}
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
            "materialized review manifest, not silently plotted as zero."
        ),
    }
