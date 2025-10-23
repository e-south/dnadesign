"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/plotting.py

Plotting utilities and registry for DenseGen.

Available plots:
- compression_ratio : histogram of compression_ratio
- tf_usage          : top-K TF counts across solutions
- library_size      : histogram
- gap_fill_gc       : histogram of gap_fill_gc_actual (gap-filled sequences only)
- plan_counts       : bar chart of #records per plan

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .outputs import load_records_from_config

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
mpl.rcParams["ps.fonttype"] = 42

# Rich console for user-facing progress/info
_console = Console()

# DenseGen root (…/dnadesign/densegen)
_THIS = Path(__file__).resolve()
_DENSEGEN_ROOT = _THIS.parents[1]

# ---------------------- Utilities ----------------------


def _ensure_out_dir(cfg: dict) -> Path:
    plots_cfg = cfg.get("plots") or cfg.get("plot") or {}
    out_dir = Path(plots_cfg.get("out_dir", "outputs/plots"))
    # Resolve relative paths under DenseGen’s own tree (not the caller’s CWD)
    if not out_dir.is_absolute():
        out_dir = (_DENSEGEN_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dg(col: str) -> str:
    """Return namespaced column name for densegen metadata (JSONL already namespaced)."""
    if col.startswith("densegen__"):
        return col
    return f"densegen__{col}"


def _norm_style(style: Optional[dict]) -> dict:
    s = (style or {}).copy()
    # Defaults requested: seaborn ticks, despine, legend frame off
    s.setdefault("seaborn_style", True)
    s.setdefault("despine", True)
    s.setdefault("legend_frame", False)
    s.setdefault("figsize", (8, 4))
    s.setdefault("font_size", 13)
    return s


def _find_column(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """Return exact column if present, else a column whose .strip() matches."""
    if wanted in df.columns:
        return wanted
    for c in df.columns:
        try:
            if isinstance(c, str) and c.strip() == wanted:
                return c
        except Exception:
            pass
    return None


def _apply_mpl_style(style: dict):
    """Apply a seaborn-like ticks style without requiring seaborn."""
    if style.get("seaborn_style", True):
        try:
            plt.style.use("seaborn-v0_8-ticks")
        except Exception:
            try:
                plt.style.use("seaborn-ticks")
            except Exception:
                pass


def _apply_common_ax_style(ax, style: dict):
    if style.get("despine", True):
        if "top" in ax.spines:
            ax.spines["top"].set_visible(False)
        if "right" in ax.spines:
            ax.spines["right"].set_visible(False)
    lg = ax.get_legend()
    if lg is not None:
        lg.set_frame_on(bool(style.get("legend_frame", False)))
        # Match legend text size to ticks for readability
        for txt in lg.get_texts():
            txt.set_fontsize(style.get("tick_size", style.get("font_size", 13) * 0.9))
    # Font sizes
    fs = float(style.get("font_size", 13))
    ax.tick_params(axis="both", labelsize=float(style.get("tick_size", fs * 0.9)))
    ax.xaxis.label.set_size(float(style.get("label_size", fs)))
    ax.yaxis.label.set_size(float(style.get("label_size", fs)))
    ax.title.set_size(float(style.get("title_size", fs * 1.1)))


def _palette(style: dict, n: int):
    pal = style.get("palette")
    if isinstance(pal, str):
        cmap = plt.get_cmap(pal)
        return [cmap(i / max(1, n - 1)) for i in range(n)]
    if isinstance(pal, (list, tuple)):
        return list(pal)[:n]
    # fallback
    cmap = plt.get_cmap("tab10")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _resolve_figsize(style: dict) -> Tuple[float, float]:
    """Only honor a direct width×height tuple in inches (style.figsize)."""
    fs = style.get("figsize", (8, 4))
    return (float(fs[0]), float(fs[1]))


def _ensure_list_of_dicts(val):
    """
    Coerce a cell value into a Python list[dict] if possible.
    Handles:
      - list/tuple of dicts
      - numpy.ndarray of dicts (common from Arrow→pandas conversions)
      - JSON strings representing list[dict]
      - a single dict (promote to [dict])
      - otherwise, returns []
    """
    if val is None:
        return []
    if isinstance(val, dict):
        return [val]
    if isinstance(val, (list, tuple)):
        return [x for x in val if isinstance(x, dict)]
    # numpy array of objects -> list
    try:
        import numpy as _np  # local import to avoid global hard dep in doc builds

        if isinstance(val, _np.ndarray):
            return [x for x in val.tolist() if isinstance(x, dict)]
    except Exception:
        pass
    if isinstance(val, str):
        s = val.strip()
        if s and s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [d for d in j if isinstance(d, dict)]
            except Exception:
                return []
    return []


def _ensure_list_of_strs(val):
    """
    Best-effort: coerce a cell into list[str].
    Handles:
      - list/tuple/set of strings
      - numpy.ndarray of strings
      - Arrow ListScalar / arrays via .as_py() / .to_pylist()
      - JSON strings "[...]" → list
    Returns [] if it can’t be interpreted as a list of strings.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if str(x).strip()]
    try:
        import numpy as _np  # local-only, optional

        if isinstance(val, _np.ndarray):
            return [str(x) for x in val.tolist() if str(x).strip()]
    except Exception:
        pass
    # Duck-typed Arrow scalars/arrays
    for attr in ("as_py", "to_pylist"):
        if hasattr(val, attr):
            try:
                L = getattr(val, attr)()
                if isinstance(L, list):
                    return [str(x) for x in L if str(x).strip()]
            except Exception:
                pass
    # JSON-encoded list
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x) for x in j if str(x).strip()]
            except Exception:
                return []
    return []


def _kde_gaussian(
    x: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None
) -> np.ndarray:
    """
    Simple Gaussian KDE without SciPy.
    Uses Silverman's rule-of-thumb if bandwidth is None.
    Returns density evaluated at 'grid'.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros_like(grid, dtype=float)
    std = np.std(x)
    if std == 0:
        # jitter a hair if all the same length
        std = 1.0
        x = x + 0.01 * np.random.standard_normal(size=n)
    if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.06 * std * (n ** (-1 / 5))  # Silverman
        if bandwidth <= 0:
            bandwidth = max(0.1, std * 0.2)
    # (1 / (n*h)) sum phi((z - xi)/h)
    z = (grid[:, None] - x[None, :]) / bandwidth
    dens = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    dens = dens.sum(axis=1) / (n * bandwidth)
    return dens


# ---------------------- Plot functions ----------------------


def plot_compression_ratio(
    df: pd.DataFrame, out_path: Path, *, bins: int = 30, style: Optional[dict] = None
) -> None:
    col = _dg("compression_ratio")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in outputs.")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        raise ValueError("No numeric compression_ratio values to plot.")
    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    ax.hist(vals, bins=bins)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Count")
    ax.set_title("DenseGen: Compression Ratio")
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tf_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    top_k: Optional[int] = None,
    style: Optional[dict] = None,
) -> None:
    col = _dg("used_tf_counts")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in outputs.")
    # used_tf_counts may be dicts or JSON strings already parsed upstream
    series = df[col].dropna()
    counts: Dict[str, int] = {}
    for row in series:
        # tolerate JSON strings, dicts, and rows with None
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except Exception:
                continue
        if isinstance(row, dict):
            for tf, n in row.items():
                if n is None:
                    continue
                # coerce robustly; skip NaN / non-numeric
                if isinstance(n, (int, float)):
                    if isinstance(n, float) and (math.isnan(n) or math.isinf(n)):
                        continue
                    nn = int(n)
                else:
                    try:
                        nn = int(float(n))
                    except Exception:
                        continue
                counts[tf] = counts.get(tf, 0) + nn
    if not counts:
        raise ValueError("used_tf_counts has no data to plot.")
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = items if (top_k is None or top_k <= 0) else items[:top_k]
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    colors = _palette(style, len(labels))
    ax.bar(labels, values, color=colors)
    # rotate labels without forcing a FixedLocator (avoids warnings)
    ax.tick_params(axis="x", labelrotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel("Total placements across solutions")
    ax.set_title("DenseGen: TF usage")
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gap_fill_gc(
    df: pd.DataFrame, out_path: Path, *, style: Optional[dict] = None
) -> None:
    """Scatter: x=gap_fill_gc_actual, y=gap_fill_bases (filled sequences only)."""
    used_col = _dg("gap_fill_used")
    gc_col = _dg("gap_fill_gc_actual")
    b_col = _dg("gap_fill_bases")
    for c in (used_col, gc_col, b_col):
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not present.")
    mask = df[used_col] == True  # noqa: E712
    x = pd.to_numeric(df.loc[mask, gc_col], errors="coerce")
    y = pd.to_numeric(df.loc[mask, b_col], errors="coerce")
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    if x.empty:
        raise ValueError("No gap-fill data to plot.")
    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    ax.scatter(x.values, y.values, alpha=0.35, s=12)  # light, readable
    ax.set_xlabel("Gap-fill GC fraction")
    ax.set_ylabel("Gap-fill bases")
    ax.set_title("Gap-fill: bases vs GC fraction (filled only)")
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_plan_counts(
    df: pd.DataFrame, out_path: Path, *, style: Optional[dict] = None
) -> None:
    col = _dg("plan")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in outputs.")
    counts = df[col].value_counts()
    if counts.empty:
        raise ValueError("No plan counts to plot.")
    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    labels = counts.index.tolist()
    ax.bar(labels, counts.values.tolist())
    ax.tick_params(axis="x", labelrotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel("# sequences")
    ax.set_title("DenseGen: Plan counts")
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tf_coverage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    top_k: int = 20,
    normalize: bool = True,
    smooth_window: int = 1,
    fill: bool = False,
    fill_alpha: float = 0.35,
    stacked: bool = False,
    style: Optional[dict] = None,
) -> None:
    """
    Coverage of TF placements along the sequence.
    Uses:
      - x-axis length from 'length' (fallback to densegen__sequence_length or computed)
      - placements from densegen__used_tfbs_detail (offset + tf + tfbs length)
    """
    det_col = _dg("used_tfbs_detail")
    if det_col not in df.columns:
        raise KeyError(f"Column '{det_col}' not found in outputs.")

    # Determine length for x-axis
    if "length" in df.columns:
        L = int(pd.to_numeric(df["length"], errors="coerce").dropna().max())
    elif _dg("sequence_length") in df.columns:
        L = int(
            pd.to_numeric(df[_dg("sequence_length")], errors="coerce").dropna().max()
        )
    elif "sequence" in df.columns:
        L = int(df["sequence"].astype(str).map(len).max())
    else:
        raise ValueError("Cannot infer sequence length for x-axis.")

    details_series = df[det_col].dropna()
    coverages: dict[str, np.ndarray] = {}
    n_seqs = len(details_series)

    for row in details_series:
        placements = _ensure_list_of_dicts(row)
        for p in placements:
            tf = str(p.get("tf", "") or "").strip()
            if not tf:
                continue
            try:
                start = int(p.get("offset", 0))
            except Exception:
                try:
                    start = int(float(p.get("offset", 0)))
                except Exception:
                    start = 0
            tfbs = str(p.get("tfbs", "") or "")
            span = len(tfbs)
            if span <= 0:
                continue
            end = min(max(0, start) + span, L)
            start = max(0, min(start, L))
            if end <= start:
                continue
            arr = coverages.setdefault(tf, np.zeros(L, dtype=float))
            arr[start:end] += 1.0

    if not coverages:
        raise ValueError("No TF placements available to compute coverage.")

    # Limit to top_k TFs by total coverage
    order = sorted(coverages.items(), key=lambda kv: kv[1].sum(), reverse=True)[:top_k]

    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    xs = np.arange(L)
    colors = _palette(style, len(order))

    base = np.zeros(L, dtype=float)
    for i, (tf, arr) in enumerate(order):
        y = arr.astype(float)
        if normalize and n_seqs > 0:
            y = y / float(n_seqs)
        if smooth_window and smooth_window > 1:
            k = int(smooth_window)
            k = max(1, min(k, L))
            y = np.convolve(y, np.ones(k) / k, mode="same")
        if fill:
            if stacked:
                y_stack_top = base + y
                ax.fill_between(
                    xs,
                    base,
                    y_stack_top,
                    alpha=fill_alpha,
                    color=colors[i],
                    label=tf,
                    linewidth=1,
                )
                ax.plot(xs, y_stack_top, color=colors[i], linewidth=1.5)
                base = y_stack_top
            else:
                ax.fill_between(
                    xs, 0, y, alpha=fill_alpha, color=colors[i], label=tf, linewidth=1
                )
                ax.plot(xs, y, color=colors[i], linewidth=1.5)
        else:
            ax.plot(xs, y, label=tf, linewidth=2, color=colors[i])

    ax.set_xlabel("Nucleotide Position")
    ax.set_ylabel("Coverage" + (" (fraction of sequences)" if normalize else " Count"))
    ax.set_title("TFBS coverage along the sequence")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tfbs_length_density(
    df: pd.DataFrame,
    out_path: Path,
    *,
    bins: int | str = "auto",
    kde: bool = True,
    kde_bandwidth: Optional[float] = None,
    kde_points: int = 256,
    fill_alpha: float = 0.25,
    style: Optional[dict] = None,
) -> None:
    """
    Density (histogram) of fetched motif lengths per TF from used_tfbs_detail.
    """
    det_col = _dg("used_tfbs_detail")
    if det_col not in df.columns:
        raise KeyError(f"Column '{det_col}' not found in outputs.")
    details_series = df[det_col].dropna()
    by_tf: dict[str, list[int]] = {}
    for row in details_series:
        placements = _ensure_list_of_dicts(row)
        for p in placements:
            tf = str(p.get("tf", "") or "").strip()
            tfbs = str(p.get("tfbs", "") or "")
            if not tf or not tfbs:
                continue
            by_tf.setdefault(tf, []).append(len(tfbs))
    if not by_tf:
        raise ValueError("No TFBS lengths available to plot.")

    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    tfs = sorted(by_tf.keys())
    colors = _palette(style, len(tfs))
    for i, tf in enumerate(tfs):
        vals = np.array(by_tf[tf], dtype=float)
        if vals.size == 0:
            continue
        if kde:
            xmin, xmax = float(vals.min()), float(vals.max())
            pad = max(1.0, 0.05 * (xmax - xmin if xmax > xmin else 2.0))
            grid = np.linspace(xmin - pad, xmax + pad, max(64, int(kde_points)))
            dens = _kde_gaussian(vals, grid, bandwidth=kde_bandwidth)
            # Slightly alpha fill under the KDE curve, then draw the outline on top
            fa = max(0.0, min(1.0, float(fill_alpha)))
            if fa > 0:
                ax.fill_between(
                    grid, 0, dens, color=colors[i], alpha=fa, linewidth=0, zorder=1
                )
            ax.plot(grid, dens, linewidth=2, label=tf, color=colors[i], zorder=2)
        else:
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=tf,
                color=colors[i],
            )
    ax.set_xlabel("Motif length (nt)")
    ax.set_ylabel("Density")
    ax.set_title("Length of fetched motifs")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tfbs_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    max_sites: Optional[int] = None,
    style: Optional[dict] = None,
) -> None:
    """
    Count individual TFBS strings used across solutions (prefer densegen__used_tfbs).
    X-ticks are TFBS strings (rotated 90°). Bars sorted by frequency desc.
    """
    # Robust column resolution (tolerate accidental trailing spaces)
    used_col = _find_column(df, _dg("used_tfbs"))
    det_col = _find_column(df, _dg("used_tfbs_detail"))
    if not used_col and not det_col:
        raise KeyError(
            f"Missing both '{_dg('used_tfbs')}' and '{_dg('used_tfbs_detail')}'."
        )

    # Count by TFBS and remember which TF each TFBS belongs to (for hues)
    counts: Dict[str, int] = {}
    tf_for_tfbs: Dict[str, str] = {}  # TFBS -> TF (majority vote / first seen)
    per_tf_for_tfbs: Dict[str, Dict[str, int]] = {}

    # Preferred: densegen__used_tfbs (list of "tf:tfbs" strings)
    if used_col:
        ser = df[used_col].dropna()
        for row in ser:
            items = _ensure_list_of_strs(row)
            for s in items:
                s = str(s)
                tf = s.split(":", 1)[0].strip() if ":" in s else ""
                tfbs = s.split(":", 1)[1].strip() if ":" in s else s.strip()
                if not tfbs:
                    continue
                counts[tfbs] = counts.get(tfbs, 0) + 1
                if tf:
                    per_tf_for_tfbs.setdefault(tfbs, {}).setdefault(tf, 0)
                    per_tf_for_tfbs[tfbs][tf] += 1

    # Fallback: densegen__used_tfbs_detail (list of dicts with 'tf','tfbs')
    if not counts and det_col:
        ser = df[det_col].dropna()
        for row in ser:
            items = _ensure_list_of_dicts(row)
            for d in items:
                tf = str(d.get("tf", "") or "").strip()
                tfbs = str(d.get("tfbs", "") or "").strip()
                if not tfbs:
                    continue
                counts[tfbs] = counts.get(tfbs, 0) + 1
                if tf:
                    per_tf_for_tfbs.setdefault(tfbs, {}).setdefault(tf, 0)
                    per_tf_for_tfbs[tfbs][tf] += 1
    if not counts:
        raise ValueError("No TFBS usage data to plot.")
    # Decide a TF label for each TFBS by majority frequency (or first)
    for tfbs, tf_counts in per_tf_for_tfbs.items():
        tf_for_tfbs[tfbs] = (
            max(tf_counts.items(), key=lambda kv: kv[1])[0] if tf_counts else ""
        )

    pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if max_sites is not None and max_sites > 0:
        pairs = pairs[:max_sites]
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    style = _norm_style(style)
    _apply_mpl_style(style)
    fig, ax = plt.subplots(figsize=_resolve_figsize(style))
    # Color bars by TF (hues)
    present_tfs = sorted(
        {tf_for_tfbs.get(b, "") for b in labels if tf_for_tfbs.get(b, "")}
    )
    tf_colors = {
        tf: col for tf, col in zip(present_tfs, _palette(style, len(present_tfs)))
    }
    bar_colors = [tf_colors.get(tf_for_tfbs.get(b, ""), "#BBBBBB") for b in labels]
    ax.bar(labels, values, color=bar_colors)
    ax.tick_params(axis="x", labelrotation=90)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("center")
    ax.set_ylabel("Occurrences across solutions")
    ax.set_title("TFBS usage by TF (hues) – ranked")
    # Legend keyed by TF
    if present_tfs:
        patches = [Patch(facecolor=tf_colors[tf], label=tf) for tf in present_tfs]
        ax.legend(
            handles=patches, loc="best", frameon=bool(style.get("legend_frame", False))
        )
    _apply_common_ax_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


AVAILABLE_PLOTS: Dict[str, Dict[str, Callable]] = {
    "compression_ratio": {
        "fn": plot_compression_ratio,
        "description": "Histogram of compression ratios.",
    },
    "tf_usage": {
        "fn": plot_tf_usage,
        "description": "Total TF usage across solutions.",
    },
    "gap_fill_gc": {
        "fn": plot_gap_fill_gc,
        "description": "Gap-fill bases vs GC fraction (filled only).",
    },
    "plan_counts": {"fn": plot_plan_counts, "description": "Counts per plan item."},
    "tf_coverage": {
        "fn": plot_tf_coverage,
        "description": "Per-position TF coverage using used_tfbs_detail.",
    },
    "tfbs_length_density": {
        "fn": plot_tfbs_length_density,
        "description": "Density of fetched motif lengths (per TF).",
    },
    "tfbs_usage": {
        "fn": plot_tfbs_usage,
        "description": "Counts of individual binding sites (TFBS strings), ranked.",
    },
}


# ---------------------- Orchestration + Preflight ----------------------


def _usable_counts_for_plot(name: str, df: pd.DataFrame) -> Tuple[int, int, str]:
    """
    Return (usable_rows, total_rows, reason_if_zero)
    without throwing, so we can warn/skip gracefully.
    """
    total = len(df)
    try:
        if name == "compression_ratio":
            col = _dg("compression_ratio")
            if col not in df.columns:
                return 0, total, f"missing column '{col}'"
            usable = pd.to_numeric(df[col], errors="coerce").notna().sum()
            return int(usable), total, ""
        if name == "library_size":
            col = _dg("library_size")
            if col not in df.columns:
                return 0, total, f"missing column '{col}'"
            usable = pd.to_numeric(df[col], errors="coerce").notna().sum()
            return int(usable), total, ""
        if name == "gap_fill_gc":
            used_col = _dg("gap_fill_used")
            gc_col = _dg("gap_fill_gc_actual")
            if used_col not in df.columns or gc_col not in df.columns:
                return 0, total, f"missing '{used_col}' and/or '{gc_col}'"
            mask = (df[used_col] == True) & pd.to_numeric(  # noqa
                df[gc_col], errors="coerce"
            ).notna()  # noqa: E712
            return int(mask.sum()), total, ""
        if name == "plan_counts":
            col = _dg("plan")
            if col not in df.columns:
                return 0, total, f"missing column '{col}'"
            usable = df[col].notna().sum()
            return int(usable), total, ""
        if name == "tf_usage":
            col = _dg("used_tf_counts")
            if col not in df.columns:
                return 0, total, f"missing column '{col}'"
            ser = df[col].dropna()

            def _row_ok(v):
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        return False
                if not isinstance(v, dict):
                    return False
                for n in v.values():
                    if isinstance(n, (int, float)) and not (
                        isinstance(n, float) and (math.isnan(n) or math.isinf(n))
                    ):
                        return True
                    # allow numeric-y strings
                    try:
                        _ = float(n)
                        return True
                    except Exception:
                        pass
                return False

            return int(sum(_row_ok(x) for x in ser)), total, ""
        if name in {"tf_coverage", "tfbs_length_density"}:
            det_col = _dg("used_tfbs_detail")
            if det_col not in df.columns:
                return 0, total, f"missing column '{det_col}'"
            ser = df[det_col].dropna()

            def _row_ok(v):
                L = _ensure_list_of_dicts(v)
                if not L:
                    return False
                # require at least one placement with a non-empty tf and tfbs
                for d in L:
                    tf = str(d.get("tf", "") or "").strip()
                    tfbs = str(d.get("tfbs", "") or "")
                    if tf and tfbs:
                        return True
                return False

            return int(sum(_row_ok(x) for x in ser)), total, ""
        if name == "tfbs_usage":
            used_col = _find_column(df, _dg("used_tfbs"))
            det_col = _find_column(df, _dg("used_tfbs_detail"))
            if used_col:
                ser = df[used_col].dropna()

                def _row_ok(v):
                    return len(_ensure_list_of_strs(v)) > 0

                return int(sum(_row_ok(x) for x in ser)), total, ""
            if det_col:
                ser = df[det_col].dropna()

                def _row_ok(v):
                    L = _ensure_list_of_dicts(v)
                    return any(str(d.get("tfbs", "") or "").strip() for d in L)

                return int(sum(_row_ok(x) for x in ser)), total, ""
            return (
                0,
                total,
                f"missing '{_dg('used_tfbs')}' and '{_dg('used_tfbs_detail')}'",
            )
    except Exception as e:
        return 0, total, f"preflight error: {e}"
    return 0, total, "unhandled plot"


def run_plots_from_config(cfg: dict, *, only: Optional[str] = None) -> None:
    df, src = load_records_from_config(cfg)
    out_dir = _ensure_out_dir(cfg)

    plots_cfg = cfg.get("plots") or cfg.get("plot") or {}
    default_list = plots_cfg.get("default", list(AVAILABLE_PLOTS.keys()))
    selected = [p.strip() for p in (only.split(",") if only else default_list)]
    unknown = [p for p in selected if p not in AVAILABLE_PLOTS]
    if unknown:
        raise ValueError(f"Unknown plot(s): {', '.join(unknown)}. See `ls-plots`.")

    options = plots_cfg.get("options", {})
    global_style = plots_cfg.get("style", {})  # optional global style

    # Header panel
    _console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold]DenseGen plotting[/] • source: [cyan]{src}[/] • rows: [cyan]{len(df):,}[/]\n"
                f"Output directory: [green]{out_dir}[/]"
            ),
            border_style="blue",
        )
    )

    summary = Table("plot", "usable / total", "% usable", "saved to", "status")

    for name in selected:
        fn = AVAILABLE_PLOTS[name]["fn"]
        kwargs = (options.get(name, {}) or {}).copy()
        # Single source of truth for size: dims: [W, H] in inches
        dims = kwargs.pop("dims", None)
        style = {**(global_style or {}), **(kwargs.pop("style", {}) or {})}
        if dims:
            style["figsize"] = tuple(dims)
        out_path = out_dir / f"{name}.pdf"

        usable, total, reason = _usable_counts_for_plot(name, df)
        pct = 0.0 if total == 0 else (100.0 * usable / total)

        # No extra warnings here — just reflect status in the summary table
        if usable == 0:
            summary.add_row(name, f"0 / {total:,}", "0.0%", "—", "[red]skipped[/]")
            continue

        # Run plot, but never crash the whole batch
        try:
            fn(df, out_path, style=style, **kwargs)
            summary.add_row(
                name,
                f"{usable:,} / {total:,}",
                f"{pct:.1f}%",
                str(out_path),
                "[green]ok[/]",
            )
        except Exception:
            summary.add_row(
                name, f"{usable:,} / {total:,}", f"{pct:.1f}%", "—", "[red]failed[/]"
            )

    _console.print(summary)
