"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/hues.py

UMAP hue-resolution and highlight-style helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def _compute_gc(seq: str) -> float:
    if not isinstance(seq, str) or not seq:
        return 0.0
    s = seq.upper()
    return float((s.count("G") + s.count("C")) / len(s))


def _ensure_numeric_series(df: pd.DataFrame, col: str, *, allow_non_finite: bool = False) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Numeric hue column '{col}' not found.")
    s = df[col]
    try:
        s = pd.to_numeric(s, errors="raise")
    except Exception as exc:
        coerced = pd.to_numeric(s, errors="coerce")
        bad_mask = coerced.isna() & s.notna()
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (df["id"].astype(str) if "id" in df.columns else pd.Series(["?"] * len(df)))
        )
        offenders = pd.DataFrame({"id": ids[bad_mask], "value": s[bad_mask].astype(str)})
        sample = offenders.head(15).to_dict(orient="records")
        raise TypeError(
            "Column '{col}' is not numeric. Found {n} non-numeric value(s). "
            "Sample offenders (id→value): {sample}".format(col=col, n=int(bad_mask.sum()), sample=sample)
        ) from exc
    arr = s.to_numpy(dtype="float64", copy=False)
    non_finite_mask = ~np.isfinite(arr)
    if non_finite_mask.any() and not allow_non_finite:
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (df["id"].astype(str) if "id" in df.columns else pd.Series(["?"] * len(df)))
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
        preview = [
            {"row": int(r), "id": str(i), "value": (None if pd.isna(v) else float(v))}
            for r, i, v in offenders.itertuples(index=False, name=None)
        ]
        raise ValueError(
            (
                "Column '{col}' contains {n_bad} non-finite value(s) "
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
    s = _ensure_numeric_series(df, col, allow_non_finite=True)
    arr = s.to_numpy(dtype="float64", copy=False)
    non_finite_mask = ~np.isfinite(arr)
    if not non_finite_mask.any():
        return s, np.ones(len(s), dtype=bool)
    if missing_policy == "error":
        n_bad = int(non_finite_mask.sum())
        n_nan = int(np.isnan(arr).sum())
        n_pinf = int(np.isposinf(arr).sum())
        n_ninf = int(np.isneginf(arr).sum())
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (df["id"].astype(str) if "id" in df.columns else np.array(["?"] * len(df)))
        )
        offenders = [
            {
                "row": int(i),
                "id": str(ids[i]),
                "value": (None if np.isnan(arr[i]) else float(arr[i])),
            }
            for i in np.flatnonzero(non_finite_mask)[:25]
        ]
        raise ValueError(
            f"Column '{col}' contains {n_bad} non-finite value(s) (NaN={n_nan}, +Inf={n_pinf}, -Inf={n_ninf}). "
            f"First offenders: {offenders}"
        )
    keep_mask = ~non_finite_mask
    if log_fn is not None:
        bad_idx = np.flatnonzero(non_finite_mask)
        ids = (
            df.index.astype(str)
            if df.index.name == "id"
            else (df["id"].astype(str) if "id" in df.columns else pd.Series(["?"] * len(df)))
        )
        sample = [{"id": str(ids[i])} for i in bad_idx[:6]]
        try:
            log_fn(
                f"Hue '{col}': dropping {int(non_finite_mask.sum())}/{len(df)} row(s) with NaN/Inf (e.g., {sample})."
            )
        except Exception:
            pass
    return s, keep_mask


def _ensure_categorical_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Categorical hue column '{col}' not found.")
    s = df[col]
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
    del default_norm
    out = []
    for spec in color_specs:
        if spec == "cluster":
            col = f"cluster__{name}"
            if col not in df.columns:
                raise ValueError(
                    f"Cluster column '{col}' not found; run 'cluster fit' first or choose a different hue."
                )
            out.append(("cluster", {"values": df[col].astype(str), "categorical": True}))
            continue
        if spec == "gc_content":
            if "sequence" not in df.columns:
                raise KeyError("Hue 'gc_content' requires a 'sequence' column.")
            vals = df["sequence"].astype(str).apply(_compute_gc)
            out.append(("gc_content", {"values": vals, "categorical": False}))
            continue
        if spec == "seq_length" and "sequence" in df.columns:
            out.append(("seq_length", {"values": df["sequence"].astype(str).str.len(), "categorical": False}))
            continue
        if spec == "seq_length" and "sequence" not in df.columns:
            raise KeyError("Hue 'seq_length' requires a 'sequence' column.")
        if spec == "intra_sim":
            col = f"cluster__{name}__intra_sim"
            if col not in df.columns:
                raise ValueError(f"Intra-sim column '{col}' missing; run 'cluster intra-sim'.")
            out.append(("intra_sim", {"values": df[col], "categorical": False}))
            continue
        if spec.startswith("numeric:"):
            col = spec.split(":", 1)[1]
            s, mask = _prepare_numeric_hue(df, col, missing_policy=missing_policy, log_fn=log_fn)
            out.append((col, {"values": s, "categorical": False, "mask": mask}))
            continue
        if spec.startswith("categorical:"):
            col = spec.split(":", 1)[1]
            s = _ensure_categorical_series(df, col)
            out.append((col, {"values": s, "categorical": True}))
            continue
        if spec == "highlight":
            if not highlight or not highlight.get("ids"):
                raise ValueError("Hue 'highlight' requires --highlight <file> to supply ids.")
            idx_ids = df.index.astype(str) if df.index.name == "id" else df["id"].astype(str)
            ids_set = set(map(str, highlight["ids"]))
            if isinstance(highlight.get("labels"), dict) and len(highlight["labels"]) > 0:
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
                            "highlight_categories": list(sorted(set(labels_map.values()))),
                            "highlight_by": str(highlight.get("by", "")),
                        },
                    )
                )
            else:
                vals = np.where(idx_ids.isin(ids_set), "highlight", "background")
                out.append(("highlight", {"values": pd.Series(vals, index=df.index), "categorical": True}))
            continue
        raise ValueError(f"Unknown hue spec: {spec}")
    return out


def normalize_highlight_style(style: Optional[dict], base_size: float) -> dict:
    style = dict(style or {})
    out: dict = {}
    if "size" in style and style["size"] is not None:
        out["size"] = float(style["size"])
    else:
        mul = float(style.get("size_multiplier", 1.6))
        out["size"] = float(base_size) * mul
    overlay_size = style.get("overlay_size", None)
    out["overlay_size"] = float(overlay_size) if overlay_size is not None else out["size"]
    out["alpha"] = float(style.get("alpha", 0.95))
    out["facecolor"] = style.get("facecolor", "none")
    if "ring" in style:
        out["facecolor"] = "none" if bool(style["ring"]) else out.get("edgecolor", "red")
    out["edgecolor"] = style.get("edgecolor", "red")
    out["linewidth"] = float(style.get("linewidth", 0.9))
    out["marker"] = style.get("marker", "o")
    out["legend"] = bool(style.get("legend", False))
    out["overlay"] = bool(style.get("overlay", True))
    if "palette" in style:
        out["palette"] = style["palette"]
    return out
