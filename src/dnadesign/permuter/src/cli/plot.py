"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console

from dnadesign.permuter.src.core.storage import (
    ensure_output_dir,
    read_parquet,
    read_ref_fasta,
)
from dnadesign.permuter.src.plots.metric_by_mutation_count import plot as plot_mmc

# Use the richer plot modules (ported from legacy)
from dnadesign.permuter.src.plots.position_scatter_and_heatmap import plot as plot_psh

console = Console()
_LOG = logging.getLogger("permuter.plot")


def _alias_usr_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    # rename common columns expected by plotting modules
    out = df.copy()
    renames = {
        "permuter__nt_pos": "nt_pos",
        "permuter__nt_wt": "nt_wt",
        "permuter__nt_alt": "nt_alt",
        "permuter__modifications": "modifications",
        "permuter__round": "round",
        "permuter__ref": "ref_name",
    }
    for a, b in renames.items():
        if a in out.columns and b not in out.columns:
            out[b] = out[a]
    # Build a 'metrics' dict from permuter__metric__* columns.
    metric_cols = [c for c in out.columns if c.startswith("permuter__metric__")]
    if metric_cols:

        def _row_metrics(r):
            d = {}
            for c in metric_cols:
                key = c.split("permuter__metric__", 1)[1].lstrip("_")
                val = r.get(c)
                if pd.notna(val):
                    d[key] = float(val)
            return d

        out["metrics"] = out.apply(_row_metrics, axis=1)
        # If exactly one metric present, expose it as 'score' for legacy plot helpers.
        if len(metric_cols) == 1 and "score" not in out.columns:
            out["score"] = out[metric_cols[0]].astype("float64")
    return out


def plot(
    data: Path,
    which: List[str],
    metric_id: str | None = None,
):
    df_usr = read_parquet(data)
    plots_dir = data.parent / "plots"
    ensure_output_dir(plots_dir)
    df = _alias_usr_to_legacy(df_usr)
    ref = read_ref_fasta(data.parent)
    ref_seq = ref[1] if ref else None
    job_name = str(df.get("permuter__job", pd.Series(["job"])).iloc[0])

    # Hard requirements shared by both built-in plots
    required = ["sequence", "modifications", "round"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns for plotting: {missing}")

    for name in which:
        if name in ("position_scatter_and_heatmap", "position_scatter"):
            out = plots_dir / f"{name}.png"
            _LOG.info("plot: %s → %s (metric_id=%s)", name, out, metric_id or "<auto>")
            plot_psh(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators="",
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        elif name == "metric_by_mutation_count":
            out = plots_dir / f"{name}.png"
            _LOG.info("plot: %s → %s (metric_id=%s)", name, out, metric_id or "<auto>")
            plot_mmc(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators="",
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        else:
            raise ValueError(f"Unknown plot '{name}'")
