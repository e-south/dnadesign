"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/numeric_per_cluster.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc_context


def _coerce_numeric(
    df: pd.DataFrame,
    cols: list[str],
    *,
    missing_policy: str = "error",
    log_fn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Numeric column '{c}' not found.")
        try:
            out[c] = pd.to_numeric(out[c], errors="raise")
        except Exception as e:
            # Build an offender preview that includes id and sequence when available
            coerced = pd.to_numeric(out[c], errors="coerce")
            bad = coerced.isna() & out[c].notna()
            cols_to_take = ["id", c]
            if "sequence" in out.columns:
                cols_to_take.insert(1, "sequence")
            offenders_df = out.loc[bad, cols_to_take].head(15).copy()
            offenders_df.insert(0, "row", offenders_df.index.astype(int))
            # Normalize into [{'row':…, 'id':…, 'sequence':…, 'value':…}, …]
            offenders = []
            for _, r in offenders_df.iterrows():
                rec = {
                    "row": int(r["row"]),
                    "id": str(r.get("id", "?")),
                    "value": r[c],
                }
                if "sequence" in offenders_df.columns:
                    rec["sequence"] = r["sequence"]
                offenders.append(rec)
            raise TypeError(
                f"Column '{c}' is not numeric. Non‑numeric values={int(bad.sum())}. First offenders: {offenders}"
            ) from e
        arr = out[c].to_numpy(dtype="float64", copy=False)
        nf = ~np.isfinite(arr)
        if nf.any():
            import numpy as _np

            n_nan = int(_np.isnan(arr).sum())
            n_pinf = int(_np.isposinf(arr).sum())
            n_ninf = int(_np.isneginf(arr).sum())
            if missing_policy == "error":
                cols_to_take = ["id", c]
                if "sequence" in out.columns:
                    cols_to_take.insert(1, "sequence")
                offenders_df = out.loc[nf, cols_to_take].head(25).copy()
                offenders_df.insert(0, "row", offenders_df.index.astype(int))
                offenders = []
                for _, r in offenders_df.iterrows():
                    rec = {
                        "row": int(r["row"]),
                        "id": str(r.get("id", "?")),
                        "value": r[c],
                    }
                    if "sequence" in offenders_df.columns:
                        rec["sequence"] = r["sequence"]
                    offenders.append(rec)
                raise ValueError(
                    f"Column '{c}' contains {int(nf.sum())} non‑finite value(s) "
                    f"(NaN={n_nan}, +Inf={n_pinf}, -Inf={n_ninf}). "
                    f"First offenders: {offenders}"
                )
            # missing_policy == "drop_and_log": mark non‑finite as NaN; summaries/plots skip them
            if log_fn is not None:
                sample_ids = out.loc[nf, "id"].astype(str).head(6).tolist() if "id" in out.columns else []
                msg = (
                    f"Metric '{c}': dropping {int(nf.sum())}/{len(out)} row(s) with NaN/Inf "
                    f"(NaN={n_nan}, +Inf={n_pinf}, -Inf={n_ninf})" + (f"; e.g., ids={sample_ids}" if sample_ids else "")
                )
                try:
                    log_fn(msg)
                except Exception:
                    pass
            out.loc[nf, c] = np.nan
    return out


def summarize_numeric_by_cluster(
    df: pd.DataFrame,
    cluster_col: str,
    numeric_cols: Iterable[str],
    out_dir: Path,
    *,
    plots: bool = True,
    font_scale: float = 1.2,
    missing_policy: str = "error",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(numeric_cols)
    work = _coerce_numeric(df, cols, missing_policy=missing_policy, log_fn=log_fn)
    if cluster_col not in work.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found.")
    g = work.groupby(cluster_col)
    rows = []
    for cl, sub in g:
        for c in cols:
            s = sub[c]
            rows.append(
                {
                    "cluster": cl,
                    "metric": c,
                    "n_nonnull": int(s.notna().sum()),
                    "n_total": int(len(s)),
                    "frac_nonnull": float(s.notna().mean()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0,
                }
            )
    summ = pd.DataFrame(rows).sort_values(["metric", "cluster"])
    summ.to_csv(out_dir / f"numeric_summary__{cluster_col}.csv", index=False)

    if plots:
        # Default aesthetic for analysis: colorblind palette, bigger fonts, despined axes
        sns.set_theme(style="ticks", palette="colorblind")
        with rc_context(
            {
                "font.size": 10 * font_scale,
                "axes.titlesize": 14 * font_scale,
                "axes.labelsize": 12 * font_scale,
                "legend.fontsize": 10 * font_scale,
                "xtick.labelsize": 10 * font_scale,
                "ytick.labelsize": 10 * font_scale,
            }
        ):
            for c in cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                # lighter inner lines (quartiles) and lighter overlay (visual equiv. of lighter error bars)
                sns.violinplot(
                    data=work,
                    x=cluster_col,
                    y=c,
                    inner="quartile",
                    inner_kws={"color": "0.4"},
                    ax=ax,
                    cut=0,
                )
                sns.stripplot(
                    data=work,
                    x=cluster_col,
                    y=c,
                    color="0.35",
                    alpha=0.25,
                    ax=ax,
                    jitter=True,
                )
                ax.set_title(f"{c} by {cluster_col}")
                ax.set_xlabel(cluster_col)
                ax.set_ylabel(c)
                plt.xticks(rotation=90)
                sns.despine(ax=ax, top=True, right=True)
                fig.tight_layout()
                fig.savefig(out_dir / f"numeric_violin__{cluster_col}__{c}.png", dpi=300)
                plt.close(fig)
    return out_dir / f"numeric_summary__{cluster_col}.csv"
