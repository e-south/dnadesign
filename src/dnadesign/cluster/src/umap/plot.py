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
            vals = (
                df.get("sequence").apply(_compute_gc)
                if "sequence" in df.columns
                else pd.Series([np.nan] * len(df))
            )
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
        raise ValueError(f"Unknown hue spec: {spec}")
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
):
    sns.set_theme(style="ticks")
    x, y = coords[:, 0], coords[:, 1]
    hues = resolve_hue(
        df, color_specs, name, missing_policy=missing_policy, log_fn=log_fn
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

        # Build masks as **positions** (not label-based indexing)
        if highlight and highlight.get("ids") is not None:
            bg_bool = ~df.index.isin(
                highlight["ids"]
            )  # boolean mask aligned to df.index
            bg_pos = np.flatnonzero(
                bg_bool.values if hasattr(bg_bool, "values") else bg_bool
            )
            # If this hue has a keep-mask, apply it to background points too
            if obj.get("mask") is not None:
                keep = np.asarray(obj["mask"], dtype=bool)
                bg_pos = bg_pos[keep[bg_pos]]
            ax.scatter(
                x[bg_pos],
                y[bg_pos],
                s=size * 0.5,
                c="lightgray",
                alpha=max(0.1, alpha * 0.3),
                label="background",
            )
            mask_pos = np.flatnonzero(
                ~bg_bool.values if hasattr(bg_bool, "values") else ~bg_bool
            )
            if obj.get("mask") is not None:
                keep = np.asarray(obj["mask"], dtype=bool)
                mask_pos = mask_pos[keep[mask_pos]]
        else:
            mask_pos = np.arange(len(df))
            if obj.get("mask") is not None:
                keep = np.asarray(obj["mask"], dtype=bool)
                mask_pos = mask_pos[keep[mask_pos]]
        if obj["categorical"]:
            # Always define; some branches intentionally skip legend
            leg = None
            # select the series values aligned with our mask positions
            vals_series = obj["values"].iloc[mask_pos].astype(str)
            cats = pd.Categorical(vals_series)
            palette = sns.color_palette("colorblind", n_colors=len(cats.categories))
            for i, cat in enumerate(cats.categories):
                # (cats == cat) already yields a boolean ndarray for Categorical
                cat_mask = np.asarray(cats == cat, dtype=bool)
                cat_pos = mask_pos[cat_mask]
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
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP — {label}", fontsize=_base * 1.8, pad=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            base, ext = out_path.with_suffix("").as_posix(), out_path.suffix or ".png"
            fig.savefig(Path(f"{base}.{label}{ext}"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
