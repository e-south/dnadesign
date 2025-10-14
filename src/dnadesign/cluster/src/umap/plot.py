"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/umap/plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
from matplotlib import rc_context


def _compute_gc(seq: str) -> float:
    if not isinstance(seq, str) or not seq:
        return 0.0
    s = seq.upper()
    return float((s.count("G") + s.count("C")) / len(s))


def _ensure_numeric_series(
    df: pd.DataFrame, col: str, *, allow_non_finite: bool = False
) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Numeric hue column '{col}' not found.")
    s = df[col]
    # First: require numeric dtype or values strictly castable to numeric
    try:
        s = pd.to_numeric(s, errors="raise")
    except Exception as e:
        # Find which values were non‑numeric to aid debugging
        coerced = pd.to_numeric(s, errors="coerce")
        bad_mask = coerced.isna() & s.notna()
        # Prefer the df index if it is 'id'; else fall back to an 'id' column if present
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (
                df["id"].astype(str)
                if "id" in df.columns
                else pd.Series(["?"] * len(df))
            )
        )
        offenders = pd.DataFrame(
            {"id": ids[bad_mask], "value": s[bad_mask].astype(str)}
        )
        sample = offenders.head(15).to_dict(orient="records")
        raise TypeError(
            "Column '{col}' is not numeric. Found {n} non‑numeric value(s). "
            "Sample offenders (id→value): {sample}".format(
                col=col, n=int(bad_mask.sum()), sample=sample
            )
        ) from e
    # Second: reject NaN/±Inf with detailed context
    arr = s.to_numpy(dtype="float64", copy=False)
    non_finite_mask = ~np.isfinite(arr)
    if non_finite_mask.any() and not allow_non_finite:
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (
                df["id"].astype(str)
                if "id" in df.columns
                else pd.Series(["?"] * len(df))
            )
        )
        n_bad = int(non_finite_mask.sum())
        n_nan = int(np.isnan(arr).sum())
        n_pinf = int(np.isposinf(arr).sum())
        n_ninf = int(np.isneginf(arr).sum())
        offenders = pd.DataFrame(
            {
                "row": np.flatnonzero(non_finite_mask),
                "id": ids[non_finite_mask].values,
                "value": s[non_finite_mask].astype(object).values,
            }
        ).head(25)
        # Render a compact sample of offenders
        preview = [
            {"row": int(r), "id": str(i), "value": (None if pd.isna(v) else float(v))}
            for r, i, v in offenders.itertuples(index=False, name=None)
        ]
        raise ValueError(
            (
                "Column '{col}' contains {n_bad} non‑finite value(s) "
                "(NaN={n_nan}, +Inf={n_pinf}, -Inf={n_ninf}).\n"
                "First offenders: {preview}"
            ).format(
                col=col,
                n_bad=n_bad,
                n_nan=n_nan,
                n_pinf=n_pinf,
                n_ninf=n_ninf,
                preview=preview,
            )
        )
    return s.astype(float)


def _prepare_numeric_hue(
    df: pd.DataFrame,
    col: str,
    *,
    missing_policy: Literal["error", "drop_and_log"] = "error",
    log_fn: Optional[Callable[[str], None]] = None,
):
    """
    Return (values: Series[float], mask: np.ndarray[bool]) for a numeric hue.
    If missing_policy='drop_and_log', rows with non-finite values are excluded via the mask
    and a concise message is logged via log_fn (if provided).
    """
    # First, coerce/validate numerics (this raises on non-numeric strings etc.)
    # 1) Coerce to numeric (assertive), but allow NaN/Inf to pass through.
    #    Non‑numeric strings etc. will still raise here (as desired).
    s = _ensure_numeric_series(df, col, allow_non_finite=True)
    arr = s.to_numpy(dtype="float64", copy=False)
    non_finite_mask = ~np.isfinite(arr)
    if not non_finite_mask.any():
        # Keep all rows
        import numpy as _np

        return s, _np.ones(len(s), dtype=bool)
    if missing_policy == "error":
        # Reproduce the detailed error from _ensure_numeric_series (which we bypassed by design here),
        # but with the non-finite counts and a preview.
        import numpy as _np

        n_bad = int(non_finite_mask.sum())
        n_nan = int(_np.isnan(arr).sum())
        n_pinf = int(_np.isposinf(arr).sum())
        n_ninf = int(_np.isneginf(arr).sum())
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (
                df["id"].astype(str)
                if "id" in df.columns
                else _np.array(["?"] * len(df))
            )
        )
        offenders = [
            {
                "row": int(i),
                "id": str(ids[i]),
                "value": (None if _np.isnan(arr[i]) else float(arr[i])),
            }
            for i in _np.flatnonzero(non_finite_mask)[:25]
        ]
        raise ValueError(
            f"Column '{col}' contains {n_bad} non-finite value(s) (NaN={n_nan}, +Inf={n_pinf}, -Inf={n_ninf}). "
            f"First offenders: {offenders}"
        )
    # missing_policy == "drop_and_log": build keep-mask and log a concise note
    keep_mask = ~non_finite_mask
    if log_fn is not None:
        # Log count and a small, stable preview of ids for traceability
        bad_idx = np.flatnonzero(non_finite_mask)
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (
                df["id"].astype(str)
                if "id" in df.columns
                else pd.Series(["?"] * len(df))
            )
        )
        sample = [{"id": str(ids[i])} for i in bad_idx[:6]]
        msg = (
            f"Hue '{col}': dropping {int(non_finite_mask.sum())}/{len(df)} row(s) with NaN/Inf "
            f"(e.g., {sample})."
        )
        try:
            log_fn(msg)
        except Exception:
            pass
    return s, keep_mask


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


def _ensure_categorical_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Categorical hue column '{col}' not found.")
    s = df[col]
    # Allow ints, strings, categories – treat everything as string categories.
    if ptypes.is_float_dtype(s) and s.isna().any():
        raise ValueError(f"Column '{col}' has NaNs; fill or drop before plotting.")
    return s.astype(str)


def resolve_hue(
    df: pd.DataFrame,
    color_specs: list[str],
    name: str,
    default_norm: str = "none",
    *,
    missing_policy: Literal["error", "drop_and_log"] = "error",
    log_fn: Optional[Callable[[str], None]] = None,
    highlight: Optional[dict] = None,
) -> list[tuple[str, dict]]:
    """Return list of (label, dict with 'values' and 'is_categorical')."""
    out = []
    for spec in color_specs:
        if spec == "cluster":
            col = f"cluster__{name}"
            if col not in df.columns:
                raise ValueError(
                    f"Cluster column '{col}' not found; run 'cluster fit' first or choose a different hue."
                )
            out.append(
                ("cluster", {"values": df[col].astype(str), "categorical": True})
            )
            continue
        if spec == "gc_content":
            if "sequence" not in df.columns:
                raise KeyError("Hue 'gc_content' requires a 'sequence' column.")
            vals = df["sequence"].astype(str).apply(_compute_gc)
            out.append(("gc_content", {"values": vals, "categorical": False}))
            continue
        if spec == "seq_length" and "sequence" in df.columns:
            out.append(
                (
                    "seq_length",
                    {
                        "values": df["sequence"].astype(str).str.len(),
                        "categorical": False,
                    },
                )
            )
            continue
        if spec == "seq_length" and "sequence" not in df.columns:
            raise KeyError("Hue 'seq_length' requires a 'sequence' column.")
        if spec == "intra_sim":
            col = f"cluster__{name}__intra_sim"
            if col not in df.columns:
                raise ValueError(
                    f"Intra-sim column '{col}' missing; run 'cluster intra-sim'."
                )
            out.append(("intra_sim", {"values": df[col], "categorical": False}))
            continue
        # numeric:<col> or categorical:<col>
        if spec.startswith("numeric:"):
            col = spec.split(":", 1)[1]
            s, mask = _prepare_numeric_hue(
                df, col, missing_policy=missing_policy, log_fn=log_fn
            )
            out.append((col, {"values": s, "categorical": False, "mask": mask}))
            continue
        if spec.startswith("categorical:"):
            col = spec.split(":", 1)[1]
            s = _ensure_categorical_series(df, col)
            out.append((col, {"values": s, "categorical": True}))
            continue
        if spec == "highlight":
            # Requires highlight ids; no auto-refit/projection for missing ids.
            if not highlight or not highlight.get("ids"):
                raise ValueError(
                    "Hue 'highlight' requires --highlight <file> to supply ids."
                )
            idx_ids = (
                df.index.astype(str) if df.index.name == "id" else df["id"].astype(str)
            )
            ids_set = set(map(str, highlight["ids"]))
            # Mode A: categorical highlight if labels are provided (id -> category)
            if (
                isinstance(highlight.get("labels"), dict)
                and len(highlight["labels"]) > 0
            ):
                labels_map = {str(k): str(v) for k, v in highlight["labels"].items()}
                vals = np.where(
                    idx_ids.isin(ids_set),
                    idx_ids.map(lambda z: labels_map.get(str(z), None)),
                    "background",
                )
                out.append(
                    (
                        "highlight",
                        {
                            "values": pd.Series(vals, index=df.index),
                            "categorical": True,
                            "highlight_categories": list(
                                sorted(set(labels_map.values()))
                            ),
                            "highlight_by": str(highlight.get("by", "")),
                        },
                    )
                )
            else:
                # Mode B: single‑hue highlight (background vs highlight)
                vals = np.where(idx_ids.isin(ids_set), "highlight", "background")
                out.append(
                    (
                        "highlight",
                        {
                            "values": pd.Series(vals, index=df.index),
                            "categorical": True,
                        },
                    )
                )
            continue
        raise ValueError(f"Unknown hue spec: {spec}")
    return out


def _normalize_highlight_style(style: Optional[dict], base_size: float) -> dict:
    """
    Normalize a user-provided highlight style mapping into a concrete style dict
    with assertive defaults. We avoid hidden fallbacks in control flow — if the user
    specifies keys, they are taken verbatim; otherwise, we use explicit safe defaults.
    """
    style = dict(style or {})
    out: dict = {}
    # Size: choose explicit 'size' if provided, else apply a multiplier to base size
    if "size" in style and style["size"] is not None:
        out["size"] = float(style["size"])
    else:
        mul = float(style.get("size_multiplier", 1.6))
        out["size"] = float(base_size) * mul
    out["alpha"] = float(style.get("alpha", 0.95))
    out["facecolor"] = style.get("facecolor", "none")
    out["edgecolor"] = style.get("edgecolor", "red")
    out["linewidth"] = float(style.get("linewidth", 0.9))
    out["marker"] = style.get("marker", "o")
    out["legend"] = bool(style.get("legend", False))
    out["overlay"] = bool(style.get("overlay", True))
    if "palette" in style:
        out["palette"] = style[
            "palette"
        ]  # str (palette name) or dict {category: color}
    return out


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
            mapped = {
                str(k): pal_spec[k] for k in pal_spec.keys() if str(k) in categories
            }
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
    hstyle = _normalize_highlight_style(highlight_style, base_size=size)
    # Whether we will overlay highlights on non-'highlight' hues
    do_overlay = (
        bool(hi_ids) and bool(overlay_highlight) and bool(hstyle.get("overlay", True))
    )
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
            sc = ax.scatter(
                x[mask_pos], y[mask_pos], c=vals, s=size, alpha=alpha, cmap="viridis"
            )
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
                    cat_mask = hi_mask & idx_ids.map(
                        lambda z: hi_labels.get(str(z), None) == cat
                    )
                    cat_pos = np.flatnonzero(
                        cat_mask.values if hasattr(cat_mask, "values") else cat_mask
                    )
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
                hi_pos = np.flatnonzero(
                    hi_mask.values if hasattr(hi_mask, "values") else hi_mask
                )
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
