"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/analysis.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .errors import SchemaError


@dataclass(frozen=True)
class PlotSpec:
    name: str
    description: str
    plot_fn: Callable[[pd.DataFrame, Path], Path]


def _require_any_column(df: pd.DataFrame, cols: Iterable[str], *, plot_name: str) -> None:
    cols = list(cols)
    if any(c in df.columns for c in cols):
        return
    raise SchemaError(f"Plot '{plot_name}' requires one of: {', '.join(cols)}.")


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, plot_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"Plot '{plot_name}' missing required column(s): {', '.join(missing)}.")


def _prepare_sequence_frame(df: pd.DataFrame) -> pd.DataFrame:
    _require_any_column(df, ["sequence", "length"], plot_name="sequence_features")
    out = df.copy()
    if "length" not in out.columns:
        out["length"] = out["sequence"].astype(str).str.len()
    if "sequence" in out.columns:
        seq = out["sequence"].astype(str).fillna("")
        seq_len = seq.str.len().replace(0, np.nan)
        gc = seq.str.count(r"[GgCc]")
        out["gc_content"] = (gc / seq_len).fillna(0.0)
    return out


def _save_fig(fig, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_length_hist(df: pd.DataFrame, out_path: Path) -> Path:
    plot_name = "length_hist"
    df = _prepare_sequence_frame(df)
    data = df["length"].dropna()
    if data.empty:
        raise SchemaError(f"Plot '{plot_name}' requires non-empty length data.")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.set_title("Sequence length distribution")
    ax.set_xlabel("Length (nt)")
    ax.set_ylabel("Count")
    return _save_fig(fig, out_path)


def plot_gc_hist(df: pd.DataFrame, out_path: Path) -> Path:
    plot_name = "gc_hist"
    _require_columns(df, ["sequence"], plot_name=plot_name)
    df = _prepare_sequence_frame(df)
    data = df["gc_content"].dropna()
    if data.empty:
        raise SchemaError(f"Plot '{plot_name}' requires non-empty GC content data.")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color="#55A868", edgecolor="white", alpha=0.9)
    ax.set_title("GC content distribution")
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Count")
    return _save_fig(fig, out_path)


def plot_gc_vs_length(df: pd.DataFrame, out_path: Path) -> Path:
    plot_name = "gc_vs_length"
    _require_columns(df, ["sequence"], plot_name=plot_name)
    df = _prepare_sequence_frame(df)
    if df.empty:
        raise SchemaError(f"Plot '{plot_name}' requires non-empty data.")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["length"], df["gc_content"], s=18, alpha=0.6, color="#C44E52", edgecolor="none")
    ax.set_title("GC content vs length")
    ax.set_xlabel("Length (nt)")
    ax.set_ylabel("GC fraction")
    return _save_fig(fig, out_path)


def plot_nulls_by_column(df: pd.DataFrame, out_path: Path) -> Path:
    plot_name = "nulls_by_column"
    if df.empty:
        raise SchemaError(f"Plot '{plot_name}' requires non-empty data.")
    null_pct = df.isna().mean().sort_values(ascending=False) * 100.0
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(null_pct.index.tolist(), null_pct.values, color="#8172B2", alpha=0.9)
    ax.set_title("Null percentage by column")
    ax.set_ylabel("Nulls (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    return _save_fig(fig, out_path)


def plot_namespace_coverage(df: pd.DataFrame, out_path: Path) -> Path:
    namespaces: dict[str, list[str]] = {}
    for col in df.columns:
        if "__" in col:
            ns = col.split("__", 1)[0]
            namespaces.setdefault(ns, []).append(col)
    if not namespaces:
        raise SchemaError("Plot 'namespace_coverage' requires namespaced columns (e.g., tool__field).")
    coverage = {}
    for ns, cols in namespaces.items():
        coverage[ns] = float(df[cols].notna().any(axis=1).mean() * 100.0)
    series = pd.Series(coverage).sort_values(ascending=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(series.index.tolist(), series.values, color="#CCB974", alpha=0.9)
    ax.set_title("Namespace coverage")
    ax.set_ylabel("Rows with any value (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    return _save_fig(fig, out_path)


_PLOTS: dict[str, PlotSpec] = {
    "length_hist": PlotSpec(
        name="length_hist",
        description="Histogram of sequence lengths",
        plot_fn=plot_length_hist,
    ),
    "gc_hist": PlotSpec(
        name="gc_hist",
        description="Histogram of GC content",
        plot_fn=plot_gc_hist,
    ),
    "gc_vs_length": PlotSpec(
        name="gc_vs_length",
        description="Scatter of GC content vs length",
        plot_fn=plot_gc_vs_length,
    ),
    "nulls_by_column": PlotSpec(
        name="nulls_by_column",
        description="Null percentage per column",
        plot_fn=plot_nulls_by_column,
    ),
    "namespace_coverage": PlotSpec(
        name="namespace_coverage",
        description="Percent of rows with any values per namespace",
        plot_fn=plot_namespace_coverage,
    ),
}


def list_plots() -> list[PlotSpec]:
    return list(_PLOTS.values())


def get_plot(name: str) -> PlotSpec:
    if name not in _PLOTS:
        known = ", ".join(sorted(_PLOTS.keys()))
        raise SchemaError(f"Unknown plot '{name}'. Available: {known}.")
    return _PLOTS[name]
