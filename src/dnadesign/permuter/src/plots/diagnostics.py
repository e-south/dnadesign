"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/diagnostics.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def hist_mut_count(k_values, *, out_png: str | Path, title: str) -> None:
    k = np.asarray(k_values, dtype=int)
    if k.size == 0:
        return
    bins = np.arange(k.min(), k.max() + 2) - 0.5
    fig = plt.figure(figsize=(6, 3.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.hist(k, bins=bins)
    ax.set_xlabel("mut_count (k)")
    ax.set_ylabel("Number of variants")
    ax.set_title(title)
    ax.set_xticks(np.arange(k.min(), k.max() + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def hist_pairwise_hamming(distances, *, out_png: str | Path, title: str) -> None:
    d = np.asarray(distances, dtype=int)
    fig = plt.figure(figsize=(6, 3.2), dpi=150)
    ax = fig.add_subplot(111)
    if d.size:
        bins = np.arange(d.min(), d.max() + 2) - 0.5
        ax.hist(d, bins=bins)
        ax.set_xticks(np.arange(d.min(), d.max() + 1))
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough variants for pairwise distances",
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Hamming distance (AA)")
    ax.set_ylabel("Pairs")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
