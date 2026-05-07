"""High-density scatter rasterization for notebook review surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb

RASTER_SCATTER_ROW_THRESHOLD = 50_000
RASTER_SCATTER_PANEL_PIXELS = 1_100


def should_rasterize_scatter(row_count: int) -> bool:
    return int(row_count) > RASTER_SCATTER_ROW_THRESHOLD


def _data_extent(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    pad_fraction: float = 0.04,
) -> tuple[float, float, float, float]:
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0:
        x_span = 1.0
    if y_span <= 0.0:
        y_span = 1.0
    x_pad = x_span * pad_fraction
    y_pad = y_span * pad_fraction
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def scatter_data_extent(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    pad_fraction: float = 0.04,
) -> tuple[float, float, float, float]:
    return _data_extent(x_values, y_values, pad_fraction=pad_fraction)


def _flat_pixel_indices(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = extent
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)
    columns = np.floor(((x_values - x_min) / x_span) * width).astype(np.int64)
    rows = np.floor(((y_values - y_min) / y_span) * height).astype(np.int64)
    np.clip(columns, 0, width - 1, out=columns)
    np.clip(rows, 0, height - 1, out=rows)
    return (rows * width) + columns


def _bincount(
    flat_indices: np.ndarray,
    *,
    size: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    return np.bincount(flat_indices, weights=weights, minlength=size).astype(np.float32, copy=False)


def _alpha_from_counts(counts: np.ndarray, *, point_alpha: float) -> np.ndarray:
    clipped_alpha = min(max(float(point_alpha), 0.01), 0.95)
    return np.clip(1.0 - np.power(1.0 - clipped_alpha, counts), 0.0, 1.0)


def _draw_rgba_image(
    ax,
    rgba: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
):
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )
    x_min, x_max, y_min, y_max = extent
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def draw_single_color_raster_scatter(
    ax,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    point_alpha: float,
    pixel_width: int = RASTER_SCATTER_PANEL_PIXELS,
    pixel_height: int = RASTER_SCATTER_PANEL_PIXELS,
) -> None:
    extent = _data_extent(x_values, y_values)
    width = int(pixel_width)
    height = int(pixel_height)
    size = width * height
    flat_indices = _flat_pixel_indices(x_values, y_values, extent=extent, width=width, height=height)
    counts = _bincount(flat_indices, size=size).reshape((height, width))
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = _alpha_from_counts(counts, point_alpha=point_alpha)
    _draw_rgba_image(ax, rgba, extent=extent)


def draw_categorical_raster_scatter(
    ax,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    hue_values: np.ndarray,
    category_order: Sequence[str],
    category_colors: Mapping[str, str],
    category_alpha_multipliers: Mapping[str, float] | None,
    point_alpha: float,
    pixel_width: int = RASTER_SCATTER_PANEL_PIXELS,
    pixel_height: int = RASTER_SCATTER_PANEL_PIXELS,
) -> None:
    extent = _data_extent(x_values, y_values)
    width = int(pixel_width)
    height = int(pixel_height)
    size = width * height
    flat_indices = _flat_pixel_indices(x_values, y_values, extent=extent, width=width, height=height)
    hue_text = np.asarray(hue_values, dtype=str)
    weighted_counts = np.zeros(size, dtype=np.float32)
    weighted_rgb = np.zeros((size, 3), dtype=np.float32)
    alpha_counts = np.zeros(size, dtype=np.float32)

    for category in category_order:
        category_text = str(category)
        mask = hue_text == category_text
        if not bool(np.any(mask)):
            continue
        counts = _bincount(flat_indices[mask], size=size)
        multiplier = float((category_alpha_multipliers or {}).get(category_text, 1.0))
        multiplier = min(max(multiplier, 0.0), 1.0)
        weighted = counts * multiplier
        color = np.asarray(to_rgb(category_colors.get(category_text, "#9AA5B1")), dtype=np.float32)
        weighted_counts += weighted
        alpha_counts += weighted
        weighted_rgb += weighted[:, np.newaxis] * color

    rgba = np.zeros((height * width, 4), dtype=np.float32)
    occupied = weighted_counts > 0
    rgba[occupied, :3] = weighted_rgb[occupied] / weighted_counts[occupied, np.newaxis]
    rgba[occupied, 3] = _alpha_from_counts(alpha_counts[occupied], point_alpha=point_alpha)
    _draw_rgba_image(ax, rgba.reshape((height, width, 4)), extent=extent)


def draw_continuous_raster_scatter(
    ax,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    hue_values: np.ndarray,
    cmap: str,
    norm,
    vmin: float | None,
    vmax: float | None,
    point_alpha: float,
    pixel_width: int = RASTER_SCATTER_PANEL_PIXELS,
    pixel_height: int = RASTER_SCATTER_PANEL_PIXELS,
) -> ScalarMappable:
    valid = np.isfinite(hue_values)
    extent = _data_extent(x_values[valid], y_values[valid])
    width = int(pixel_width)
    height = int(pixel_height)
    size = width * height
    flat_indices = _flat_pixel_indices(x_values[valid], y_values[valid], extent=extent, width=width, height=height)
    counts = _bincount(flat_indices, size=size)
    value_sums = _bincount(flat_indices, size=size, weights=hue_values[valid].astype(np.float64, copy=False))
    means = np.zeros(size, dtype=np.float32)
    occupied = counts > 0
    means[occupied] = value_sums[occupied] / counts[occupied]

    color_norm = norm if norm is not None else Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(str(cmap))
    rgba = np.zeros((height * width, 4), dtype=np.float32)
    rgba[occupied] = colormap(color_norm(means[occupied]))
    rgba[occupied, 3] = _alpha_from_counts(counts[occupied], point_alpha=point_alpha)
    _draw_rgba_image(ax, rgba.reshape((height, width, 4)), extent=extent)
    return ScalarMappable(norm=color_norm, cmap=colormap)
