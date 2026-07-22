"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/aggregate_plots/source_frames.py

Source-frame helpers for DenseGen axis probe aggregate plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def has_class_composition(frame: pd.DataFrame) -> bool:
    """Return true when run metrics contain selected-class distributions."""

    if "off_target_class_distribution_true" not in frame.columns:
        return False
    return any(isinstance(value, Mapping) and bool(value) for value in frame["off_target_class_distribution_true"])


def pair_label(frame: pd.DataFrame) -> pd.Series:
    """Build stable pair labels from run metric rows."""

    family = frame.get("label_family_id", pd.Series(["unknown"] * len(frame), index=frame.index)).astype(str)
    return family + "/" + frame["campaign"].astype(str) + "/" + frame["split_id"].astype(str)


def pair_label_from_mapping(row: Mapping[str, Any]) -> str:
    """Build the stable pair label used in trajectory QA rows."""

    return f"{row.get('label_family_id', 'unknown')}/{row.get('campaign')}/{row.get('split_id')}"


def vector_reference_distance_rows(configured_plots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return per-round distance rows from configured OPAL vector-summary plot CSVs."""

    rows: list[dict[str, Any]] = []
    for entry in configured_plots:
        run_key = str(entry.get("run_key") or "")
        for plot in entry.get("plots") or []:
            if not isinstance(plot, Mapping) or plot.get("kind") != "vector_summary_heatmap":
                continue
            tidy_path = next(iter(plot.get("tidy_csv_paths") or []), None)
            if not tidy_path or not Path(str(tidy_path)).exists():
                continue
            tidy = pd.read_csv(tidy_path)
            required = {"row_type", "round", "channel", "value"}
            if not required.issubset(tidy.columns):
                continue
            row_type = tidy["row_type"].astype(str)
            reference = (
                tidy.loc[row_type.isin(["reference_vector", "setpoint"]), ["channel", "value"]]
                .dropna()
                .set_index("channel")["value"]
                .astype(float)
            )
            if reference.empty:
                continue
            round_rows = tidy.loc[row_type == "round"].copy()
            for round_index, sub in round_rows.groupby("round"):
                vector = sub.set_index("channel")["value"].astype(float)
                aligned = pd.concat([reference.rename("reference"), vector.rename("value")], axis=1).dropna()
                if aligned.empty:
                    continue
                distance = float(((aligned["value"] - aligned["reference"]) ** 2).sum() ** 0.5)
                rows.append({"run_key": run_key, "round": int(round_index), "distance": distance})
    return rows


def feature_stability_rows(configured_plots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return adjacent-round feature-rank stability rows from configured OPAL plot CSVs."""

    rows: list[dict[str, Any]] = []
    for entry in configured_plots:
        run_key = str(entry.get("run_key") or "")
        for plot in entry.get("plots") or []:
            if not isinstance(plot, Mapping) or plot.get("kind") != "feature_importance_heatmap":
                continue
            tidy_path = next(iter(plot.get("tidy_csv_paths") or []), None)
            if not tidy_path or not Path(str(tidy_path)).exists():
                continue
            tidy = pd.read_csv(tidy_path)
            required = {"round", "feature_id", "importance"}
            if not required.issubset(tidy.columns):
                continue
            wide = tidy.pivot_table(index="feature_id", columns="round", values="importance", aggfunc="max").fillna(0.0)
            rounds = sorted(int(value) for value in wide.columns)
            for previous, current in zip(rounds, rounds[1:], strict=False):
                a = wide[previous].rank(method="average")
                b = wide[current].rank(method="average")
                corr = a.corr(b)
                rows.append(
                    {
                        "run_key": run_key,
                        "round": int(current),
                        "adjacent_spearman": None if pd.isna(corr) else float(corr),
                    }
                )
    return rows
