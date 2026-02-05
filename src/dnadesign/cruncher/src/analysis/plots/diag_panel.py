"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/diag_panel.py

Render a compact diagnostics panel for trace and optimizer statistics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dnadesign.cruncher.analysis.plots._savefig import savefig


def _trace_scores(idata: Any) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not hasattr(posterior, "get"):
        return None
    scores = posterior.get("score")
    if scores is None:
        return None
    try:
        arr = np.asarray(scores)
    except Exception:
        return None
    if arr.ndim < 2:
        return None
    return arr


def _rank_histogram(ax: plt.Axes, scores: np.ndarray) -> None:
    flat = scores.reshape(-1)
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1)
    ranks = ranks.reshape(scores.shape)
    bins = np.linspace(0, len(flat), 20)
    for chain_idx in range(scores.shape[0]):
        ax.hist(
            ranks[chain_idx],
            bins=bins,
            histtype="step",
            linewidth=1.2,
            alpha=0.8,
            label=f"chain {chain_idx + 1}",
        )
    ax.set_title("Rank plot (score)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    if scores.shape[0] <= 6:
        ax.legend(frameon=False, fontsize=7)


def _plot_swap_acceptance(ax: plt.Axes, optimizer_stats: dict[str, object] | None) -> None:
    attempts = []
    accepts = []
    if isinstance(optimizer_stats, dict):
        attempts = optimizer_stats.get("swap_attempts_by_pair") or []
        accepts = optimizer_stats.get("swap_accepts_by_pair") or []
    attempts = [int(v) for v in attempts] if attempts else []
    accepts = [int(v) for v in accepts] if accepts else []
    if not attempts or len(attempts) != len(accepts):
        ax.axis("off")
        ax.text(0.5, 0.5, "Swap acceptance unavailable", ha="center", va="center", fontsize=9, color="#555555")
        return
    rates = [a / t if t else 0.0 for a, t in zip(accepts, attempts)]
    ax.bar(range(1, len(rates) + 1), rates, color="#4c78a8")
    ax.set_title("PT swap acceptance")
    ax.set_xlabel("Adjacent pair")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0.0, 1.0)


def _plot_move_acceptance(ax: plt.Axes, optimizer_stats: dict[str, object] | None) -> None:
    stats = optimizer_stats.get("move_stats") if isinstance(optimizer_stats, dict) else None
    if not isinstance(stats, list) or not stats:
        ax.axis("off")
        ax.text(0.5, 0.5, "Move acceptance unavailable", ha="center", va="center", fontsize=9, color="#555555")
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
        ax.axis("off")
        ax.text(0.5, 0.5, "Move acceptance unavailable", ha="center", va="center", fontsize=9, color="#555555")
        return
    rows.sort(key=lambda x: x[0])
    sweeps = [r[0] for r in rows]
    rates = [r[2] / r[1] if r[1] else 0.0 for r in rows]
    ax.plot(sweeps, rates, color="#f58518", linewidth=1.5)
    ax.set_title("Move acceptance over time")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0.0, 1.0)


def plot_diag_panel(
    idata: Any,
    optimizer_stats: dict[str, object] | None,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    scores = _trace_scores(idata)
    if scores is None:
        raise ValueError("Trace is missing score data.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_trace, ax_rank, ax_swap, ax_move = axes.flatten()

    for chain_idx in range(scores.shape[0]):
        ax_trace.plot(scores[chain_idx], alpha=0.6, linewidth=1.0)
    ax_trace.set_title("Score trace (per chain)")
    ax_trace.set_xlabel("Draw")
    ax_trace.set_ylabel("Score")

    _rank_histogram(ax_rank, scores)
    _plot_swap_acceptance(ax_swap, optimizer_stats)
    _plot_move_acceptance(ax_move, optimizer_stats)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
