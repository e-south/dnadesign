"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_plot_trace.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel


def test_plot_health_panel_without_stats(tmp_path: Path) -> None:
    out_path = tmp_path / "plot__health__panel.png"
    plot_health_panel(None, out_path, dpi=150, png_compress_level=9)
    assert out_path.exists()
