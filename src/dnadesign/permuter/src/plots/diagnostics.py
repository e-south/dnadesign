"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/diagnostics.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", palette="colorblind")

# ---------------------------------------------------------------------------
# Shared visual constants (fonts + colors)
# ---------------------------------------------------------------------------

_AX_LABEL_FONTSIZE = 12
_AX_TICK_FONTSIZE = 12
_AX_TITLE_FONTSIZE = 14
_LEGEND_FONTSIZE = 10

_KDE_HUE_ORDER = ["Selected", "Random background"]
_KDE_PALETTE = dict(
    zip(_KDE_HUE_ORDER, sns.color_palette("colorblind", n_colors=len(_KDE_HUE_ORDER)))
)


def _darken_rgba(
    rgba: tuple[float, float, float, float], factor: float = 0.75
) -> tuple[float, float, float, float]:
    """
    Return a darker version of an RGBA color by linearly interpolating towards black.

    factor=1.0 → unchanged, factor=0.0 → black.
    """
    r, g, b, a = rgba
    f = float(max(0.0, min(1.0, factor)))
    return (f * r, f * g, f * b, a)


_SET_ORDER = ["Selected", "Random background"]
_SET_PALETTE = dict(
    zip(_SET_ORDER, sns.color_palette("colorblind", n_colors=len(_SET_ORDER)))
)


def _ensure_path(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Shared styling helper
# ---------------------------------------------------------------------------


def _style_axes(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    # Clean, uniform look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Uniform light gray grid on both axes, without forcing tick spacing
    ax.grid(True, axis="both", alpha=0.25, linestyle="--", linewidth=0.5)

    # Font sizes (slightly larger for easier reading)
    ax.xaxis.label.set_fontsize(_AX_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(_AX_LABEL_FONTSIZE)
    ax.title.set_fontsize(_AX_TITLE_FONTSIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(_AX_TICK_FONTSIZE)


# ---------------------------------------------------------------------------
# Mutation-count histogram (selected vs random)
# ---------------------------------------------------------------------------


def hist_mut_count(
    k_selected: Sequence[int],
    k_random: Sequence[int] | None = None,
    *,
    out_png: str | Path,
    title: str,
    figsize_in: float = 5.0,
    dpi: int = 200,
) -> None:
    """
    Histogram of mut_count (k) for Selected overlaid with a Random
    background sample (if provided). Hue encodes Selected vs Random.
    """
    k_sel = np.asarray(k_selected, dtype=float)
    if k_sel.size == 0:
        return
    out_png = _ensure_path(out_png)

    records: list[dict] = [{"k": float(v), "set": "Selected"} for v in k_sel]
    if k_random is not None:
        k_rand = np.asarray(k_random, dtype=float)
        records.extend({"k": float(v), "set": "Random background"} for v in k_rand)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return

    # Consistent hue ordering + palette across all diagnostics plots
    set_order = [h for h in _SET_ORDER if (df["set"] == h).any()]
    palette = {h: _SET_PALETTE[h] for h in set_order}

    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    ax = fig.add_subplot(111)

    k_min = int(np.floor(df["k"].min()))
    k_max = int(np.ceil(df["k"].max()))
    bins = np.arange(k_min - 0.5, k_max + 1.5, 1.0)

    for label in set_order:
        vals = df.loc[df["set"] == label, "k"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        base_color = palette[label]
        # palette entries may be RGB or RGBA; normalize to RGBA
        if len(base_color) == 3:
            rgba = (*base_color, 1.0)
        else:
            rgba = base_color
        edge = _darken_rgba(rgba, factor=0.7)

        ax.hist(
            vals,
            bins=bins,
            label=f"{label} (n={len(vals):,})",
            alpha=0.6,
            linewidth=0.8,
            edgecolor=edge,
            color=base_color,
        )

    ax.set_xticks(np.arange(k_min, k_max + 1))

    _style_axes(
        ax,
        xlabel="Mutation count k",
        ylabel="Count",
        title=title,
        x_limits=(k_min - 0.5, k_max + 0.5),
        y_limits=None,
    )

    # Only add a legend if there are labeled artists (avoids UserWarning)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title=None,
            loc="best",
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pairwise |Δk| distributions
# ---------------------------------------------------------------------------


def _pairwise_delta_k(k: np.ndarray) -> np.ndarray:
    """
    Compute |Δk| over all unordered pairs.
    """
    n = int(k.size)
    if n < 2:
        return np.asarray([], dtype=int)
    out = []
    for i in range(n):
        ki = int(k[i])
        for j in range(i + 1, n):
            out.append(abs(ki - int(k[j])))
    return np.asarray(out, dtype=int)


def plot_pairwise_delta_k_selected_vs_random(
    k_selected: Sequence[int],
    random_samples: Iterable[Sequence[int]],
    *,
    out_png: str | Path,
    title: str,
    figsize_in: float = 5.0,
    dpi: int = 200,
) -> None:
    """
    Histogram of pairwise mutation-count difference |Δk|:
      • Selected set
      • Random background (all random samples concatenated).
    """
    k_sel = np.asarray(k_selected, dtype=int)
    if k_sel.size < 2:
        return
    out_png = _ensure_path(out_png)

    d_sel = _pairwise_delta_k(k_sel)
    if d_sel.size == 0:
        return

    records: list[dict] = [{"delta_k": int(v), "set": "Selected"} for v in d_sel]

    random_dists: list[np.ndarray] = []
    for sample in random_samples:
        arr = np.asarray(sample, dtype=int)
        if arr.size >= 2:
            d = _pairwise_delta_k(arr)
            if d.size:
                random_dists.append(d)

    if random_dists:
        d_rand = np.concatenate(random_dists)
        records.extend({"delta_k": int(v), "set": "Random background"} for v in d_rand)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return

    set_order = [h for h in _SET_ORDER if (df["set"] == h).any()]
    palette = {h: _SET_PALETTE[h] for h in set_order}

    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    ax = fig.add_subplot(111)

    d_min = int(df["delta_k"].min())
    d_max = int(df["delta_k"].max())
    bins = np.arange(d_min - 0.5, d_max + 1.5, 1.0)

    for label in set_order:
        vals = df.loc[df["set"] == label, "delta_k"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        base_color = palette[label]
        if len(base_color) == 3:
            rgba = (*base_color, 1.0)
        else:
            rgba = base_color
        edge = _darken_rgba(rgba, factor=0.7)

        ax.hist(
            vals,
            bins=bins,
            label=f"{label} (n={len(vals):,} pairs)",
            alpha=0.6,
            linewidth=0.8,
            edgecolor=edge,
            color=base_color,
        )

    ax.set_xticks(np.arange(d_min, d_max + 1))

    _style_axes(
        ax,
        xlabel=r"Pairwise mutation-count difference |Δk| = |k_i - k_j|",
        ylabel="Pairwise count",
        title=title or "Pairwise mutation-count difference |Δk| (selected vs random)",
        x_limits=(d_min - 0.5, d_max + 0.5),
        y_limits=None,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title=None,
            loc="best",
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pairwise Levenshtein (AA) — selected vs random
# ---------------------------------------------------------------------------


def _pairwise_levenshtein(seq: Sequence[str]) -> np.ndarray:
    """
    Compute pairwise Levenshtein distances on AA strings.
    Falls back to Hamming if all sequences have equal length.
    """

    def _lev(a: str, b: str) -> int:
        if len(a) == len(b):
            return sum(aa != bb for aa, bb in zip(a, b))
        # classic DP Levenshtein
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                cur = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        return dp[lb]

    n = len(seq)
    out: list[int] = []
    for i in range(n):
        si = seq[i]
        for j in range(i + 1, n):
            out.append(_lev(si, seq[j]))
    return np.asarray(out, dtype=int)


def plot_pairwise_levenshtein_selected_vs_random(
    seq_selected: Sequence[str],
    random_samples: Iterable[Sequence[str]],
    *,
    out_png: str | Path,
    title: str,
    figsize_in: float = 5.0,
    dpi: int = 200,
) -> None:
    """
    Histogram of pairwise Levenshtein (AA) distances:
      • Selected set
      • One or more Random comparator samples.
    """
    if len(seq_selected) < 2:
        return
    out_png = _ensure_path(out_png)

    d_sel = _pairwise_levenshtein(seq_selected)
    if d_sel.size == 0:
        return

    records: list[dict] = [{"distance": int(v), "set": "Selected"} for v in d_sel]
    random_dists: list[np.ndarray] = []
    for sample in random_samples:
        if len(sample) < 2:
            continue
        d = _pairwise_levenshtein(list(sample))
        if d.size:
            random_dists.append(d)

    if random_dists:
        d_rand = np.concatenate(random_dists)
        records.extend({"distance": int(v), "set": "Random background"} for v in d_rand)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return

    set_order = [h for h in _SET_ORDER if (df["set"] == h).any()]
    palette = {h: _SET_PALETTE[h] for h in set_order}

    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    ax = fig.add_subplot(111)

    d_min = int(df["distance"].min())
    d_max = int(df["distance"].max())
    bins = np.arange(d_min - 0.5, d_max + 1.5, 1.0)

    for label in set_order:
        vals = df.loc[df["set"] == label, "distance"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        base_color = palette[label]
        if len(base_color) == 3:
            rgba = (*base_color, 1.0)
        else:
            rgba = base_color
        edge = _darken_rgba(rgba, factor=0.7)

        ax.hist(
            vals,
            bins=bins,
            label=f"{label} (n={len(vals):,} pairs)",
            alpha=0.6,
            linewidth=0.8,
            edgecolor=edge,
            color=base_color,
        )

    ax.set_xticks(np.arange(d_min, d_max + 1))

    _style_axes(
        ax,
        xlabel="Pairwise Levenshtein distance (AA)",
        ylabel="Pairwise count",
        title=title or "Pairwise Levenshtein distance (AA; selected vs random)",
        x_limits=(d_min - 0.5, d_max + 0.5),
        y_limits=None,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title=None,
            loc="best",
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
