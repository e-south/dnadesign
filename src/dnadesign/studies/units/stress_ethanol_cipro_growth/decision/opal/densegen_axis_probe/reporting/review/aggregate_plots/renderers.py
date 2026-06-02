"""Matplotlib renderers for registered DenseGen axis probe aggregate plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ....core.constants import NULL_ORACLE_ID, ORACLE_ID
from ....tfbs.plot_style import (
    REVIEW_MATRIX_FIGSIZE,
    REVIEW_SQUARE_FIGSIZE,
    REVIEW_STACKED_FIGSIZE,
    REVIEW_WIDE_FIGSIZE,
    style_review_axis,
)
from .context import ProbeAggregatePlotContext
from .source_frames import pair_label, pair_label_from_mapping


def render_target_lift_and_precision(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = context.runs_frame.copy()
    df["label"] = df["run_key"].astype(str)
    x = range(len(df))
    fig, axes = plt.subplots(2, 1, figsize=REVIEW_STACKED_FIGSIZE, constrained_layout=True)
    axes[0].bar(x, pd.to_numeric(df["target_lift_at_k_true"], errors="coerce"), color="#446A8C")
    axes[0].set_ylabel("target lift@K")
    axes[0].set_title("Probe target lift")
    axes[1].bar(x, pd.to_numeric(df["selected_target_precision_at_k_true"], errors="coerce"), color="#7A6B3F")
    axes[1].set_ylabel("precision@K")
    axes[1].set_title("Probe selected target precision")
    for ax in axes:
        style_review_axis(ax)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["label"].tolist(), rotation=35, ha="right")
    _save(fig, path)


def render_round_target_lift_and_precision(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = context.round_frame.copy()
    required = {"run_key", "as_of_round", "target_lift_at_k_true", "selected_target_precision_at_k_true"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"round metric plot requires column(s): {missing}")
    df["round"] = pd.to_numeric(df["as_of_round"], errors="coerce")
    df["lift"] = pd.to_numeric(df["target_lift_at_k_true"], errors="coerce")
    df["precision"] = pd.to_numeric(df["selected_target_precision_at_k_true"], errors="coerce")
    df = df.dropna(subset=["round"])
    if df.empty:
        raise RuntimeError("round metric plot requires at least one finite round")

    fig, axes = plt.subplots(2, 1, figsize=REVIEW_STACKED_FIGSIZE, sharex=True, constrained_layout=True)
    for run_key, sub in df.sort_values(["run_key", "round"]).groupby("run_key"):
        label = str(run_key)
        axes[0].plot(sub["round"], sub["lift"], marker="o", linewidth=1.15, label=label)
        axes[1].plot(sub["round"], sub["precision"], marker="o", linewidth=1.15, label=label)
    axes[0].axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("target lift@K")
    axes[0].set_title("Round-over-round target lift")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("precision@K")
    axes[1].set_title("Round-over-round selected target precision")
    axes[1].set_ylim(bottom=0.0)
    for ax in axes:
        style_review_axis(ax)
    axes[0].legend(frameon=False, fontsize=7, ncols=2)
    _save(fig, path)


def render_selected_class_composition(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    for _, row in context.runs_frame.iterrows():
        dist = row.get("off_target_class_distribution_true") or {}
        if not isinstance(dist, Mapping) or not dist:
            continue
        out = {"run_key": str(row.get("run_key"))}
        out.update({str(key): int(value) for key, value in dist.items()})
        rows.append(out)
    if not rows:
        raise RuntimeError("class composition plot requires off_target_class_distribution_true metrics")
    wide = pd.DataFrame(rows).fillna(0)
    classes = [col for col in wide.columns if col != "run_key"]
    fig, ax = plt.subplots(figsize=REVIEW_WIDE_FIGSIZE, constrained_layout=True)
    bottom = [0] * len(wide)
    palette = ["#446A8C", "#8C4E4A", "#5D7D4F", "#7A6B3F", "#6C5F7D", "#4F7D75"]
    for index, axis_class in enumerate(classes):
        values = wide[axis_class].astype(int).to_list()
        ax.bar(wide["run_key"].tolist(), values, bottom=bottom, label=axis_class, color=palette[index % len(palette)])
        bottom = [prev + value for prev, value in zip(bottom, values, strict=True)]
    style_review_axis(ax)
    ax.set_ylabel("selected count")
    ax.set_title("Selected class composition")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    _save(fig, path)


def render_positive_null_lift_delta(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = context.runs_frame.copy()
    df["pair"] = pair_label(df)
    df["lift"] = pd.to_numeric(df["target_lift_at_k_true"], errors="coerce")
    pivot = df.pivot_table(index="pair", columns="oracle_id", values="lift", aggfunc="max")
    positive = pivot.get(ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    null = pivot.get(NULL_ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    delta = positive - null
    colors = ["#5D7D4F" if value > 0 else "#8C4E4A" for value in delta.fillna(0).tolist()]
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=True)
    x = range(len(delta))
    ax.bar(x, delta, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    style_review_axis(ax, square=True)
    ax.set_xticks(list(x))
    ax.set_xticklabels(delta.index.tolist(), rotation=30, ha="right")
    ax.set_ylabel("positive lift - null lift")
    ax.set_title("Positive/null lift separation by pair")
    _save(fig, path)


def render_evaluable_selected_count(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = context.runs_frame.copy()
    df["label"] = df["run_key"].astype(str)
    if "selected_count_in_eval" in df.columns:
        counts = pd.to_numeric(df["selected_count_in_eval"], errors="coerce")
    else:
        counts = df.get("selected_ids", pd.Series([[]] * len(df))).map(
            lambda value: len(value) if isinstance(value, list) else 0
        )
    expected = pd.to_numeric(df.get("selection_k", pd.Series([6] * len(df))), errors="coerce").fillna(6)
    colors = ["#5D7D4F" if count >= want else "#8C4E4A" for count, want in zip(counts, expected, strict=False)]
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=True)
    x = range(len(df))
    ax.bar(x, counts, color=colors)
    ax.plot(list(x), expected, color="#222222", marker="o", linewidth=1.0, label="expected K")
    style_review_axis(ax, square=True)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("evaluable selected count")
    ax.set_title("Selected IDs evaluable inside split pool")
    ax.legend(frameon=False)
    _save(fig, path)


def render_trajectory_qa_matrix(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = context.runs_frame.copy()
    df["pair"] = pair_label(df)
    df["lift"] = pd.to_numeric(df["target_lift_at_k_true"], errors="coerce")
    pivot = df.pivot_table(index="pair", columns="oracle_id", values="lift", aggfunc="max")
    positive = pivot.get(ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    null = pivot.get(NULL_ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    matrix = pd.DataFrame(
        {
            "positive_lift": positive,
            "null_lift": null,
            "positive_minus_null": positive - null,
        },
        index=pivot.index,
    )
    trajectory_pairs = context.trajectory_qa.get("pairs") if isinstance(context.trajectory_qa, Mapping) else []
    if trajectory_pairs:
        auc_delta = {
            pair_label_from_mapping(row): row.get("paired_auc_delta")
            for row in trajectory_pairs
            if isinstance(row, Mapping)
        }
        matrix["paired_auc_delta"] = pd.Series(auc_delta, dtype=float)
    values = matrix.to_numpy(dtype=float)
    height = max(REVIEW_MATRIX_FIGSIZE[1], 0.45 * len(matrix))
    fig, ax = plt.subplots(figsize=(REVIEW_MATRIX_FIGSIZE[0], height), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", interpolation="nearest")
    style_review_axis(ax, grid=False)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Trajectory QA matrix")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="value")
    _save(fig, path)


def render_vector_reference_distance(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=True)
    for run_key, sub in context.vector_reference_distance_frame.groupby("run_key"):
        ax.plot(sub["round"], sub["distance"], marker="o", linewidth=1.2, label=str(run_key))
    style_review_axis(ax, square=True)
    ax.set_xlabel("round")
    ax.set_ylabel("Euclidean distance to reference")
    ax.set_title("Selected vector distance to configured reference")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    _save(fig, path)


def render_feature_stability(context: ProbeAggregatePlotContext, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=True)
    for run_key, sub in context.feature_stability_frame.groupby("run_key"):
        ax.plot(sub["round"], sub["adjacent_spearman"], marker="o", linewidth=1.2, label=str(run_key))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    style_review_axis(ax, square=True)
    ax.set_xlabel("round")
    ax.set_ylabel("adjacent-round Spearman")
    ax.set_title("Feature-importance stability over rounds")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    _save(fig, path)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(fig)
