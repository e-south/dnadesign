"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/dashboard/charts/diagnostics_style.py

Shared sizing helpers for dashboard diagnostics charts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DNAD_DIAGNOSTICS_PLOT_SIZE = 6.4


def diagnostics_figsize(*, width_scale: float = 1.0, height_scale: float = 1.0) -> tuple[float, float]:
    base = float(DNAD_DIAGNOSTICS_PLOT_SIZE)
    return base * float(width_scale), base * float(height_scale)


def apply_diagnostics_title(
    fig,
    *,
    title: str,
    subtitle: str | None = None,
    top: float = 0.84,
    title_size: float | None = None,
    subtitle_size: float = 11.0,
) -> None:
    """Apply a compact figure title with reserved subplot space."""

    fig.suptitle(str(title), y=0.985, fontsize=title_size)
    if subtitle:
        fig.text(
            0.5,
            0.935,
            str(subtitle),
            ha="center",
            va="top",
            fontsize=float(subtitle_size),
            color="#333333",
        )
    fig.subplots_adjust(top=float(top))


def finalize_single_panel_diagnostics(
    fig,
    *,
    title: str,
    subtitle: str | None = None,
    left: float = 0.14,
    right: float = 0.84,
    bottom: float = 0.14,
    top: float = 0.84,
) -> None:
    """Finalize one-axis diagnostics without tight_layout/colorbar collisions."""

    apply_diagnostics_title(fig, title=title, subtitle=subtitle, top=top)
    fig.subplots_adjust(left=float(left), right=float(right), bottom=float(bottom), top=float(top))


def diagnostics_table_figsize(
    *,
    n_cols: int,
    n_rows: int,
    width_scale: float = 1.6,
    height_scale: float = 0.8,
) -> tuple[float, float]:
    base = float(DNAD_DIAGNOSTICS_PLOT_SIZE)
    cols = max(int(n_cols), 1)
    rows = max(int(n_rows), 1)
    width = max(base * float(width_scale), base * 0.10 * cols)
    height = max(base * float(height_scale), base * 0.18 * rows)
    return width, height
