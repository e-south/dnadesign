"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/diagnostics_style.py

Shared sizing helpers for dashboard diagnostics charts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DNAD_DIAGNOSTICS_PLOT_SIZE = 3.6


def diagnostics_figsize(*, width_scale: float = 1.0, height_scale: float = 1.0) -> tuple[float, float]:
    base = float(DNAD_DIAGNOSTICS_PLOT_SIZE)
    return base * float(width_scale), base * float(height_scale)


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
