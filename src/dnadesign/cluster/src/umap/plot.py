"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/umap/plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

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


def _ensure_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
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
    if non_finite_mask.any():
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


def _font_rc(font_scale: float) -> dict:
    # Centralized font scaling for consistency across plots
    base = 10.0 * float(font_scale)
    return {
        "font.size": base,
        "axes.titlesize": base * 1.4,
        "axes.labelsize": base * 1.2,
        "legend.fontsize": base * 1.0,
        "xtick.labelsize": base * 1.0,
        "ytick.labelsize": base * 1.0,
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
    df: pd.DataFrame, color_specs: list[str], name: str, default_norm: str = "none"
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
            s = _ensure_numeric_series(df, col)
            out.append((col, {"values": s, "categorical": False}))
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
):
    sns.set_theme(style="ticks")
    x, y = coords[:, 0], coords[:, 1]
    hues = resolve_hue(df, color_specs, name)
    for label, obj in hues:
        with rc_context(_font_rc(font_scale)):
            fig, ax = plt.subplots(figsize=dims)
        # Build masks as **positions** (not label-based indexing)
        if highlight and highlight.get("ids") is not None:
            bg_bool = ~df.index.isin(
                highlight["ids"]
            )  # boolean mask aligned to df.index
            bg_pos = np.flatnonzero(
                bg_bool.values if hasattr(bg_bool, "values") else bg_bool
            )
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
        else:
            mask_pos = np.arange(len(df))
        if obj["categorical"]:
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
            ax.legend(
                title=label,
                prop={"size": max(8, int(10 * float(font_scale)))},
                bbox_to_anchor=legend.get("bbox", (1.05, 1)),
                loc="upper left",
                ncol=int(legend.get("ncol", 1)),
            )
        else:
            vals = np.asarray(obj["values"].iloc[mask_pos], dtype=float)
            sc = ax.scatter(
                x[mask_pos], y[mask_pos], c=vals, s=size, alpha=alpha, cmap="viridis"
            )
            fig.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP — {label}", pad=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            base, ext = out_path.with_suffix("").as_posix(), out_path.suffix or ".png"
            fig.savefig(Path(f"{base}.{label}{ext}"), dpi=300)
        else:
            plt.show()
        plt.close(fig)
