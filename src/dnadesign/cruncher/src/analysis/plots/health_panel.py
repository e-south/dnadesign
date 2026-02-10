"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/health_panel.py

Render a compact optimization health panel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dnadesign.cruncher.analysis.plots._savefig import savefig


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#555555")


def _plot_move_acceptance(ax: plt.Axes, optimizer_stats: dict[str, object] | None) -> None:
    stats = optimizer_stats.get("move_stats") if isinstance(optimizer_stats, dict) else None
    if not isinstance(stats, list) or not stats:
        _empty_panel(ax, "Move acceptance unavailable")
        return
    rows = []
    for row in stats:
        if not isinstance(row, dict):
            continue
        sweep = row.get("sweep_idx")
        attempted = row.get("attempted")
        accepted = row.get("accepted")
        if not isinstance(sweep, (int, float)) or not isinstance(attempted, (int, float)):
            continue
        if not isinstance(accepted, (int, float)):
            continue
        rows.append((int(sweep), int(attempted), int(accepted)))
    if not rows:
        _empty_panel(ax, "Move acceptance unavailable")
        return
    rows.sort(key=lambda x: x[0])
    sweeps = [r[0] for r in rows]
    rates = [r[2] / r[1] if r[1] else 0.0 for r in rows]
    ax.plot(sweeps, rates, color="#f58518", linewidth=1.5)
    ax.set_title("Move acceptance over time")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0.0, 1.0)


def plot_health_panel(
    optimizer_stats: dict[str, object] | None,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    _plot_move_acceptance(ax, optimizer_stats)
    fig.text(
        0.5,
        0.01,
        "Optimization health indicators (not posterior convergence).",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
