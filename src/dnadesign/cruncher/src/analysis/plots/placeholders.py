"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/placeholders.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dnadesign.cruncher.analysis.plots._savefig import savefig


def write_placeholder_plot(
    out_path: Path,
    message: str,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    ax.set_axis_off()
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def write_placeholder_text(out_path: Path, message: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(message.rstrip() + "\n")
