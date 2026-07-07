"""
Render the Reader SPOP condition heatmap with retron MSD thumbnails.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .condition_matrix import ReaderSpopConditionMatrix, ReaderSpopConditionRow
from .identifiers import variant_sort_key
from .paths import resolve_repo_root, sha256_file
from .structure_manifest import RetronStructureThumbnailRow


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
        condition_labels=[_short_condition_label(col.condition_key) for col in columns],
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
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    height = max(4.0, 0.28 * max(1, len(variants)) + 1.8)
    width = max(8.0, 0.9 * max(1, len(condition_labels)) + 3.8)
    fig, (heatmap_ax, thumb_ax) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [max(1.0, len(condition_labels)), 1.3]},
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    image = heatmap_ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    heatmap_ax.set_xticks(np.arange(len(condition_labels)))
    heatmap_ax.set_xticklabels(condition_labels, rotation=35, ha="right")
    heatmap_ax.set_yticks(np.arange(len(variants)))
    heatmap_ax.set_yticklabels(variants)
    heatmap_ax.set_xlabel("Condition")
    heatmap_ax.set_ylabel("Variant")
    heatmap_ax.set_xticks(np.arange(-0.5, len(condition_labels), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=0.8)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.035, pad=0.02)
    colorbar.set_label("normalized derepression")

    thumb_ax.set_xlim(0, 1)
    thumb_ax.set_ylim(len(variants) - 0.5, -0.5)
    thumb_ax.set_xticks([])
    thumb_ax.set_yticks([])
    thumb_ax.set_title("MSD structure", fontsize=9)
    thumbnail_zoom = min(0.08, max(0.025, 1.2 / max(1, len(variants))))
    for index, variant in enumerate(variants):
        row = thumbnail_by_variant.get(variant)
        if row is not None and row.structure_status == "available":
            image_path = root / row.structure_png_path
            if image_path.exists():
                image_data = mpimg.imread(image_path)
                thumbnail = OffsetImage(image_data, zoom=thumbnail_zoom)
                thumb_ax.add_artist(AnnotationBbox(thumbnail, (0.5, index), frameon=False))
                continue
        thumb_ax.text(0.5, index, "na", ha="center", va="center", fontsize=5, color="#6b7280")
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    plt.close(fig)


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


def _short_condition_label(condition_key: str) -> str:
    return condition_key.replace("; ", "\n")


def _source_digests(*, thumbnail_rows: Sequence[RetronStructureThumbnailRow], root: Path) -> dict[str, str]:
    paths = {row.review_manifest_path for row in thumbnail_rows if row.review_manifest_path}
    return {path: sha256_file(root / path) for path in sorted(paths)}
