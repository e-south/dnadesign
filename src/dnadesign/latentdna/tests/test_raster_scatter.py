from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dnadesign.latentdna.src.notebooks.raster_scatter import (
    BACKGROUND_RASTER_MIN_ALPHA,
    CATEGORICAL_RASTER_MIN_ALPHA,
    FOREGROUND_RASTER_MIN_ALPHA,
    RASTER_SCATTER_PANEL_PIXELS,
    draw_categorical_raster_scatter,
    draw_single_color_raster_scatter,
)


def _max_image_alpha(ax) -> float:
    return float(np.asarray(ax.images[0].get_array())[..., 3].max())


def test_default_raster_resolution_prioritizes_readable_full_population_points() -> None:
    assert RASTER_SCATTER_PANEL_PIXELS <= 800


def test_categorical_raster_scatter_lifts_singleton_alpha_for_readability() -> None:
    fig, ax = plt.subplots()
    try:
        draw_categorical_raster_scatter(
            ax,
            x_values=np.array([0.0]),
            y_values=np.array([0.0]),
            hue_values=np.array(["mid"]),
            category_order=["mid"],
            category_colors={"mid": "#FEE090"},
            category_alpha_multipliers=None,
            point_alpha=0.15,
            pixel_width=8,
            pixel_height=8,
        )

        assert _max_image_alpha(ax) >= CATEGORICAL_RASTER_MIN_ALPHA - 1e-6
    finally:
        plt.close(fig)


def test_single_color_raster_scatter_separates_foreground_and_background_alpha_contracts() -> None:
    fig, (foreground_ax, background_ax) = plt.subplots(ncols=2)
    try:
        draw_single_color_raster_scatter(
            foreground_ax,
            x_values=np.array([0.0]),
            y_values=np.array([0.0]),
            color="#0072B2",
            point_alpha=0.15,
            pixel_width=8,
            pixel_height=8,
        )
        draw_single_color_raster_scatter(
            background_ax,
            x_values=np.array([0.0]),
            y_values=np.array([0.0]),
            color="#CBD5E1",
            point_alpha=0.10,
            min_point_alpha=BACKGROUND_RASTER_MIN_ALPHA,
            pixel_width=8,
            pixel_height=8,
        )

        assert _max_image_alpha(foreground_ax) >= FOREGROUND_RASTER_MIN_ALPHA - 1e-6
        assert _max_image_alpha(background_ax) <= BACKGROUND_RASTER_MIN_ALPHA + 1e-6
    finally:
        plt.close(fig)
