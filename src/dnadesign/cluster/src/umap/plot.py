"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/plot.py

UMAP helpers for plot cluster UMAP.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc_context

from .hues import normalize_highlight_style, resolve_hue


def _font_rc(font_scale: float) -> dict:
    # Centralized font scaling for consistency across plots
    base = 12.0 * float(font_scale)
    return {
        "font.size": base,
        "axes.titlesize": base * 2.2,
        "axes.labelsize": base * 1.4,
        "legend.fontsize": base * 1.2,
        "xtick.labelsize": base * 1.2,
        "ytick.labelsize": base * 1.2,
    }


def scatter(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_specs: list[str],
    name: str,
    highlight: dict | None,
    alpha: float,
    size: float,
    dims: tuple[int, int],
    legend: dict,
    out_path: Path | None,
    *,
    font_scale: float = 1.2,
    missing_policy: Literal["error", "drop_and_log"] = "drop_and_log",
    log_fn: Optional[Callable[[str], None]] = None,
    overlay_highlight: bool = True,
    highlight_style: Optional[dict] = None,
):
    sns.set_theme(style="ticks")
    x, y = coords[:, 0], coords[:, 1]
    hues = resolve_hue(
        df,
        color_specs,
        name,
        missing_policy=missing_policy,
        log_fn=log_fn,
        highlight=highlight,
    )
    # Prepare highlight id set (string ids) once; do not alter base hues with it.
    hi_ids: set[str] = set()
    if highlight and highlight.get("ids"):
        hi_ids = {str(i) for i in highlight["ids"]}
    # If categorical mode, precompute label mapping & categories
    hi_labels: dict[str, str] | None = None
    hi_by: str | None = None
    hi_categories: list[str] = []
    if isinstance(highlight, dict) and isinstance(highlight.get("labels"), dict):
        hi_labels = {str(k): str(v) for k, v in highlight["labels"].items()}
        hi_categories = sorted(set(hi_labels.values()))
        hi_by = str(highlight.get("by", "")) if highlight.get("by") else None

    # Build a palette for categorical highlight if needed
    def _resolve_hi_palette(categories: list[str]):
        pal_spec = hstyle.get("palette")
        if isinstance(pal_spec, dict):
            # explicit mapping wins; fill any missing keys deterministically
            mapped = {str(k): pal_spec[k] for k in pal_spec.keys() if str(k) in categories}
            remaining = [c for c in categories if c not in mapped]
            if remaining:
                cols = sns.color_palette("colorblind", n_colors=len(remaining))
                for c, col in zip(remaining, cols):
                    mapped[c] = col
            return mapped
        name = pal_spec if isinstance(pal_spec, str) else "colorblind"
        cols = sns.color_palette(name, n_colors=len(categories))
        return {cat: cols[i] for i, cat in enumerate(categories)}

    # Normalize overlay style once with the *base* size
    hstyle = normalize_highlight_style(highlight_style, base_size=size)
    # Whether we will overlay highlights on non-'highlight' hues
    do_overlay = bool(hi_ids) and bool(overlay_highlight) and bool(hstyle.get("overlay", True))
    for label, obj in hues:
        with rc_context(_font_rc(font_scale)):
            # Keep layout predictable: create a square-ish canvas and
            # reserve a right gutter for legend/colorbar.
            fig, ax = plt.subplots(figsize=dims)
            fig.subplots_adjust(right=0.82)
        _base = 12.0 * float(font_scale)

        # Always keep UMAP 1:1 in data space (and box square when possible)
        ax.set_aspect("equal", adjustable="box")
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass

        # Base keep-mask from hue-specific constraints (e.g., numeric non-finite drops)
        N = len(df)
        keep_mask = np.ones(N, dtype=bool)
        if obj.get("mask") is not None:
            keep_mask &= np.asarray(obj["mask"], dtype=bool)
        mask_pos = np.flatnonzero(keep_mask)
        if obj["categorical"]:
            # Always define; some branches intentionally skip legend
            leg = None
            vals_series = obj["values"].astype(str)
            vals_kept = vals_series.iloc[mask_pos]
            cats = pd.Categorical(vals_kept)
            if label == "highlight":
                # Dedicated highlight hue
                # Background always in light gray for context
                bg_mask = np.asarray(cats == "background", dtype=bool)
                bg_pos = mask_pos[bg_mask]
                if len(bg_pos) > 0:
                    ax.scatter(
                        x[bg_pos],
                        y[bg_pos],
                        s=max(1.0, size * 0.5),
                        c="lightgray",
                        alpha=max(0.1, alpha * 0.3),
                        label="background",
                    )
                # Two modes:
                #   1) Single‑hue (values 'highlight'/'background')
                #   2) Categorical (values are category names + background)
                if hi_labels and hi_categories:
                    pal = _resolve_hi_palette(hi_categories)
                    # draw each category
                    for cat in hi_categories:
                        cat_mask = np.asarray(cats == cat, dtype=bool)
                        cat_pos = mask_pos[cat_mask]
                        if len(cat_pos) == 0:
                            continue
                        ax.scatter(
                            x[cat_pos],
                            y[cat_pos],
                            # Respect plot.highlight.size/marker/alpha for the dedicated highlight plot.
                            s=hstyle["size"],
                            alpha=min(1.0, alpha * 1.2),
                            marker=hstyle["marker"],
                            color=pal[cat],
                            label=str(cat),
                            zorder=3,
                        )
                    # Legend for categories (respect caps)
                    max_items = int(legend.get("max_items", 40))
                    ncol = int(legend.get("ncol", 1))
                    bbox = tuple(legend.get("bbox", (1.02, 1.0)))[:2]
                    frameon = bool(legend.get("frameon", False))
                    cats_for_legend = [c for c in hi_categories if c != "background"]
                    if len(cats_for_legend) <= max_items:
                        title = f"highlight{(' by ' + hi_by) if hi_by else ''}"
                        leg = ax.legend(
                            title=title,
                            bbox_to_anchor=bbox,
                            loc="upper left",
                            ncol=ncol,
                            frameon=frameon,
                            prop={"size": max(8, int(10 * float(font_scale)))},
                        )
                        if leg and leg.get_title():
                            leg.get_title().set_fontsize(max(9, int(_base * 1.3)))
                else:
                    # single-hue highlight
                    hi_pos = mask_pos[np.asarray(cats == "highlight", dtype=bool)]
                    if len(hi_pos) > 0:
                        ax.scatter(
                            x[hi_pos],
                            y[hi_pos],
                            # Respect plot.highlight.size/marker/alpha; use edgecolor as the fill color by default.
                            s=hstyle["size"],
                            alpha=min(1.0, alpha * 1.2),
                            marker=hstyle["marker"],
                            color=hstyle.get("edgecolor", "red"),
                            label="highlight",
                            zorder=3,
                        )
            else:
                palette = sns.color_palette("colorblind", n_colors=len(cats.categories))
                for i, cat in enumerate(cats.categories):
                    cat_mask = np.asarray(cats == cat, dtype=bool)
                    cat_pos = mask_pos[cat_mask]
                    if len(cat_pos) == 0:
                        continue
                    ax.scatter(
                        x[cat_pos],
                        y[cat_pos],
                        s=size,
                        alpha=alpha,
                        label=str(cat),
                        color=palette[i % len(palette)],
                    )
            # Avoid comically long legends: allow caller to cap items
            # Normalize legend configuration once
            max_items = int(legend.get("max_items", 40))
            ncol = int(legend.get("ncol", 1))
            bbox = legend.get("bbox", (1.02, 1.0))
            if isinstance(bbox, (list, tuple)):
                bbox = tuple(bbox[:2])
            else:
                bbox = (1.02, 1.0)
            frameon = bool(legend.get("frameon", False))
            if len(cats.categories) <= max_items:
                leg = ax.legend(
                    title=label,
                    prop={"size": max(8, int(10 * float(font_scale)))},
                    bbox_to_anchor=bbox,
                    loc="upper left",
                    ncol=ncol,
                    frameon=frameon,
                )
                if leg and leg.get_title():
                    leg.get_title().set_fontsize(max(9, int(_base * 1.3)))
            # else: no legend when too many categories; color coding remains in the points

        else:
            vals = np.asarray(obj["values"].iloc[mask_pos], dtype=float)
            sc = ax.scatter(x[mask_pos], y[mask_pos], c=vals, s=size, alpha=alpha, cmap="viridis")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(label, fontsize=_base * 1.4)
            cbar.ax.tick_params(labelsize=_base * 1.2)

        # Optional overlay: emphasize highlighted ids **without** changing the hue colors.
        if do_overlay and label != "highlight":
            # Build id mask aligned to df.index
            idx_ids = df.index.astype(str)
            hi_mask = idx_ids.isin(hi_ids)
            # apply hue keep-mask
            hi_mask &= keep_mask
            if hi_labels and hi_categories:
                pal = _resolve_hi_palette(hi_categories)
                # Color‑coded rings per category
                # Build per‑category overlay masks
                for cat in hi_categories:
                    cat_mask = hi_mask & idx_ids.map(lambda z: hi_labels.get(str(z), None) == cat)
                    cat_pos = np.flatnonzero(cat_mask.values if hasattr(cat_mask, "values") else cat_mask)
                    if len(cat_pos) == 0:
                        continue
                    ax.scatter(
                        x[cat_pos],
                        y[cat_pos],
                        s=hstyle["size"],
                        alpha=hstyle["alpha"],
                        marker=hstyle["marker"],
                        facecolors=hstyle.get("facecolor", "none"),
                        edgecolors=pal[cat],
                        linewidths=hstyle["linewidth"],
                        zorder=3,
                        label=(str(cat) if hstyle.get("legend", False) else None),
                    )
            else:
                # Single‑hue overlay
                hi_pos = np.flatnonzero(hi_mask.values if hasattr(hi_mask, "values") else hi_mask)
                if len(hi_pos) > 0:
                    ax.scatter(
                        x[hi_pos],
                        y[hi_pos],
                        s=hstyle["size"],
                        alpha=hstyle["alpha"],
                        marker=hstyle["marker"],
                        facecolors=hstyle["facecolor"],
                        edgecolors=hstyle["edgecolor"],
                        linewidths=hstyle["linewidth"],
                        zorder=3,
                        label=("highlight" if hstyle.get("legend", False) else None),
                    )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP — {label}", fontsize=_base * 1.8, pad=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            base, ext = out_path.with_suffix("").as_posix(), out_path.suffix or ".png"
            # flat: write directly under <run>/umap/<name>.<label>.png
            fig.savefig(Path(f"{base}.{label}{ext}"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
