"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_b.py

Stage-B summary plotting for library feasibility and utilization diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_common import _apply_style, _safe_filename, _style


def _require_nonempty(series: pd.Series, *, label: str, context: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{label} contains missing values for {context}.")
    arr = np.asarray(values.to_numpy(dtype=float), dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} is empty for {context}.")
    return arr


def plot_stage_b_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    library_builds_df: pd.DataFrame,
    library_members_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    cfg: dict | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if library_builds_df is None or library_builds_df.empty:
        raise ValueError("stage_b_summary requires library_builds.parquet data.")
    if library_members_df is None or library_members_df.empty:
        raise ValueError("stage_b_summary requires library_members.parquet data.")
    if composition_df is None or composition_df.empty:
        raise ValueError("stage_b_summary requires composition.parquet data.")
    required_build_cols = {"input_name", "plan_name", "library_index", "library_hash", "slack_bp"}
    missing = required_build_cols - set(library_builds_df.columns)
    if missing:
        raise ValueError(f"library_builds.parquet missing required columns: {sorted(missing)}")
    required_member_cols = {"input_name", "plan_name", "library_index", "tf", "tfbs", "library_hash"}
    missing = required_member_cols - set(library_members_df.columns)
    if missing:
        raise ValueError(f"library_members.parquet missing required columns: {sorted(missing)}")
    required_comp_cols = {"input_name", "plan_name", "solution_id", "tf", "tfbs"}
    missing = required_comp_cols - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    raw_style = style or {}
    style = _style(raw_style)
    if "figsize" not in raw_style:
        style["figsize"] = (14, 7)
    paths: list[Path] = []

    grouped_builds = library_builds_df.groupby(["input_name", "plan_name"])
    for (input_name, plan_name), builds in grouped_builds:
        context = f"{input_name}/{plan_name}"
        members = library_members_df[
            (library_members_df["input_name"].astype(str) == str(input_name))
            & (library_members_df["plan_name"].astype(str) == str(plan_name))
        ]
        if members.empty:
            raise ValueError(f"library_members.parquet missing rows for {context}.")
        comp_subset = composition_df[
            (composition_df["input_name"].astype(str) == str(input_name))
            & (composition_df["plan_name"].astype(str) == str(plan_name))
        ]
        if comp_subset.empty:
            raise ValueError(f"composition.parquet missing rows for {context}.")

        if builds.duplicated(["library_index", "library_hash"]).any():
            raise ValueError(f"library_builds.parquet has duplicate library keys for {context}.")
        if members.duplicated(["library_index", "library_hash", "tf", "tfbs"]).any():
            raise ValueError(f"library_members.parquet has duplicate rows for {context}.")

        metrics = members.groupby(["library_index", "library_hash"]).agg(
            library_size=("tfbs", "size"),
            unique_tfbs_count=("tfbs", pd.Series.nunique),
            total_bp=("tfbs", lambda x: int(sum(len(str(v)) for v in x))),
        )
        metrics = metrics.reset_index()
        if metrics.duplicated(["library_index", "library_hash"]).any():
            raise ValueError(f"library_members.parquet has duplicate library metrics for {context}.")

        merged = builds.merge(metrics, on=["library_index", "library_hash"], how="left", validate="one_to_one")
        if "library_size_x" in merged.columns and "library_size_y" in merged.columns:
            merged["library_size"] = pd.to_numeric(merged["library_size_x"], errors="coerce").fillna(
                pd.to_numeric(merged["library_size_y"], errors="coerce")
            )
        elif "library_size_x" in merged.columns:
            merged["library_size"] = merged["library_size_x"]
        elif "library_size_y" in merged.columns:
            merged["library_size"] = merged["library_size_y"]

        if merged[["library_size", "unique_tfbs_count", "total_bp"]].isna().any().any():
            raise ValueError(f"library_members.parquet missing metrics for {context}.")

        offered_counts = members.groupby(["tf", "tfbs"])["library_index"].nunique().rename("offered_count")
        if offered_counts.empty:
            raise ValueError(f"library_members.parquet missing offered TFBS counts for {context}.")
        used_counts = comp_subset.groupby(["tf", "tfbs"])["solution_id"].nunique().rename("used_count")
        if used_counts.empty:
            raise ValueError(f"composition.parquet missing used TFBS counts for {context}.")
        offered_vs_used = pd.concat([offered_counts, used_counts], axis=1).fillna(0)
        offered_vs_used["ratio"] = offered_vs_used.apply(
            lambda row: float(row["used_count"]) / float(row["offered_count"]) if row["offered_count"] > 0 else 0.0,
            axis=1,
        )

        fig, axes = plt.subplots(2, 3, figsize=style["figsize"])
        ax_slack, ax_size, ax_unique, ax_bp, ax_scatter, ax_ratio = axes.flatten()

        slack = _require_nonempty(merged["slack_bp"], label="slack_bp", context=context)
        ax_slack.hist(slack, bins="auto", color="#4c78a8", alpha=0.8)
        ax_slack.axvline(0.0, color="#e15759", linestyle="--", linewidth=1)
        ax_slack.set_title("Slack distribution")
        ax_slack.set_xlabel("Slack bp")
        ax_slack.set_ylabel("Libraries")

        sizes = _require_nonempty(merged["library_size"], label="library_size", context=context)
        ax_size.hist(sizes, bins="auto", color="#59a14f", alpha=0.8)
        ax_size.set_title("Library size")
        ax_size.set_xlabel("TFBS per library")
        ax_size.set_ylabel("Libraries")

        uniques = _require_nonempty(merged["unique_tfbs_count"], label="unique_tfbs_count", context=context)
        ax_unique.hist(uniques, bins="auto", color="#f28e2b", alpha=0.8)
        ax_unique.set_title("Unique TFBS count")
        ax_unique.set_xlabel("Unique TFBS per library")
        ax_unique.set_ylabel("Libraries")

        total_bp = _require_nonempty(merged["total_bp"], label="total_bp", context=context)
        ax_bp.hist(total_bp, bins="auto", color="#edc949", alpha=0.8)
        ax_bp.set_title("Total TFBS bp")
        ax_bp.set_xlabel("Total TFBS bp")
        ax_bp.set_ylabel("Libraries")

        if offered_vs_used.empty:
            raise ValueError(f"No offered/used TFBS data for {context}.")
        x = offered_vs_used["offered_count"].to_numpy(dtype=float)
        y = offered_vs_used["used_count"].to_numpy(dtype=float)
        if len(offered_vs_used) > 200:
            ax_scatter.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=1)
        else:
            ax_scatter.scatter(x, y, alpha=0.6, s=18, color="#4c78a8")
        ax_scatter.set_title("Offered vs used (TFBS)")
        ax_scatter.set_xlabel("Offered count")
        ax_scatter.set_ylabel("Used count")

        ratios = _require_nonempty(offered_vs_used["ratio"], label="used/offered ratio", context=context)
        ax_ratio.hist(ratios, bins="auto", color="#af7aa1", alpha=0.8)
        ax_ratio.set_title("Used / offered ratio")
        ax_ratio.set_xlabel("Used / offered")
        ax_ratio.set_ylabel("TFBS count")

        for ax in axes.flatten():
            _apply_style(ax, style)
        fig.suptitle(f"Stage-B summary -- {input_name}/{plan_name}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths
