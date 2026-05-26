"""Aggregate review plots for DenseGen axis probe metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..artifacts import ProbeArtifactLayout
from ..constants import NULL_ORACLE_ID, ORACLE_ID


def _write_probe_plots(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
    configured_plots: list[dict[str, Any]],
) -> list[Path]:
    runs = metrics_payload.get("runs") or []
    if not runs:
        return []
    layout.review_plots_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(runs)
    paths = [
        layout.review_plots_dir / "target_lift_and_precision.png",
        layout.review_plots_dir / "positive_null_lift_delta.png",
        layout.review_plots_dir / "evaluable_selected_count.png",
        layout.review_plots_dir / "trajectory_qa_matrix.png",
    ]
    _plot_lift_and_precision(frame, paths[0])
    if _has_class_composition(frame):
        class_path = layout.review_plots_dir / "selected_class_composition.png"
        _plot_class_composition(frame, class_path)
        paths.append(class_path)
    _plot_positive_null_lift_delta(frame, paths[1])
    _plot_evaluable_selected_count(frame, paths[2])
    _plot_trajectory_qa_matrix(frame, metrics_payload.get("trajectory_qa") or {}, paths[3])
    round_rows = [row for row in metrics_payload.get("rounds") or [] if isinstance(row, Mapping)]
    if round_rows:
        round_path = layout.review_plots_dir / "round_target_lift_and_precision.png"
        _plot_round_lift_and_precision(pd.DataFrame(round_rows), round_path)
        paths.append(round_path)
    optional_paths = [
        (
            layout.review_plots_dir / "vector_distance_to_reference_over_rounds.png",
            _vector_reference_distance_rows(configured_plots),
        ),
        (layout.review_plots_dir / "feature_stability_over_rounds.png", _feature_stability_rows(configured_plots)),
    ]
    for path, rows in optional_paths:
        if not rows:
            continue
        if path.name.startswith("vector"):
            _plot_vector_reference_distance(pd.DataFrame(rows), path)
        else:
            _plot_feature_stability(pd.DataFrame(rows), path)
        paths.append(path)
    return paths


def _has_class_composition(frame: pd.DataFrame) -> bool:
    if "off_target_class_distribution_true" not in frame.columns:
        return False
    return any(isinstance(value, Mapping) and bool(value) for value in frame["off_target_class_distribution_true"])


def _plot_lift_and_precision(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
    df["label"] = df["run_key"].astype(str)
    x = range(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].bar(x, pd.to_numeric(df["target_lift_at_k_true"], errors="coerce"), color="#446A8C")
    axes[0].set_ylabel("target lift@K")
    axes[0].set_title("Probe target lift")
    axes[1].bar(x, pd.to_numeric(df["selected_target_precision_at_k_true"], errors="coerce"), color="#7A6B3F")
    axes[1].set_ylabel("precision@K")
    axes[1].set_title("Probe selected target precision")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["label"].tolist(), rotation=35, ha="right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_round_lift_and_precision(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
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

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
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
    axes[0].legend(frameon=False, fontsize=7, ncols=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_class_composition(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        dist = row.get("off_target_class_distribution_true") or {}
        if not isinstance(dist, Mapping):
            continue
        if not dist:
            continue
        out = {"run_key": str(row.get("run_key"))}
        out.update({str(key): int(value) for key, value in dist.items()})
        rows.append(out)
    if not rows:
        raise RuntimeError("class composition plot requires off_target_class_distribution_true metrics")
    wide = pd.DataFrame(rows).fillna(0)
    classes = [col for col in wide.columns if col != "run_key"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bottom = [0] * len(wide)
    palette = ["#446A8C", "#8C4E4A", "#5D7D4F", "#7A6B3F", "#6C5F7D", "#4F7D75"]
    for index, axis_class in enumerate(classes):
        values = wide[axis_class].astype(int).to_list()
        ax.bar(wide["run_key"].tolist(), values, bottom=bottom, label=axis_class, color=palette[index % len(palette)])
        bottom = [prev + value for prev, value in zip(bottom, values, strict=True)]
    ax.set_ylabel("selected count")
    ax.set_title("Selected class composition")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_positive_null_lift_delta(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
    df["pair"] = _pair_label(df)
    df["lift"] = pd.to_numeric(df["target_lift_at_k_true"], errors="coerce")
    pivot = df.pivot_table(index="pair", columns="oracle_id", values="lift", aggfunc="max")
    positive = pivot.get(ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    null = pivot.get(NULL_ORACLE_ID, pd.Series(index=pivot.index, dtype=float))
    delta = positive - null
    colors = ["#5D7D4F" if value > 0 else "#8C4E4A" for value in delta.fillna(0).tolist()]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = range(len(delta))
    ax.bar(x, delta, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(delta.index.tolist(), rotation=30, ha="right")
    ax.set_ylabel("positive lift - null lift")
    ax.set_title("Positive/null lift separation by label family, campaign, and split")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_evaluable_selected_count(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
    df["label"] = df["run_key"].astype(str)
    if "selected_count_in_eval" in df.columns:
        counts = pd.to_numeric(df["selected_count_in_eval"], errors="coerce")
    else:
        counts = df.get("selected_ids", pd.Series([[]] * len(df))).map(
            lambda value: len(value) if isinstance(value, list) else 0
        )
    expected = pd.to_numeric(df.get("selection_k", pd.Series([6] * len(df))), errors="coerce").fillna(6)
    colors = ["#5D7D4F" if count >= want else "#8C4E4A" for count, want in zip(counts, expected, strict=False)]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = range(len(df))
    ax.bar(x, counts, color=colors)
    ax.plot(list(x), expected, color="#222222", marker="o", linewidth=1.0, label="expected K")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("evaluable selected count")
    ax.set_title("Selected IDs evaluable inside split pool")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trajectory_qa_matrix(frame: pd.DataFrame, trajectory_qa: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
    df["pair"] = _pair_label(df)
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
    trajectory_pairs = trajectory_qa.get("pairs") if isinstance(trajectory_qa, Mapping) else []
    if trajectory_pairs:
        auc_delta = {
            _pair_label_from_mapping(row): row.get("paired_auc_delta")
            for row in trajectory_pairs
            if isinstance(row, Mapping)
        }
        matrix["paired_auc_delta"] = pd.Series(auc_delta, dtype=float)
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.45 * len(matrix))), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", interpolation="nearest")
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
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pair_label(frame: pd.DataFrame) -> pd.Series:
    family = frame.get("label_family_id", pd.Series(["unknown"] * len(frame), index=frame.index)).astype(str)
    return family + "/" + frame["campaign"].astype(str) + "/" + frame["split_id"].astype(str)


def _pair_label_from_mapping(row: Mapping[str, Any]) -> str:
    return f"{row.get('label_family_id', 'unknown')}/{row.get('campaign')}/{row.get('split_id')}"


def _vector_reference_distance_rows(configured_plots: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            round_rows = tidy.loc[tidy["row_type"].astype(str) == "round"].copy()
            for round_index, sub in round_rows.groupby("round"):
                vector = sub.set_index("channel")["value"].astype(float)
                aligned = pd.concat([reference.rename("reference"), vector.rename("value")], axis=1).dropna()
                if aligned.empty:
                    continue
                distance = float(((aligned["value"] - aligned["reference"]) ** 2).sum() ** 0.5)
                rows.append({"run_key": run_key, "round": int(round_index), "distance": distance})
    return rows


def _feature_stability_rows(configured_plots: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _plot_vector_reference_distance(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for run_key, sub in frame.groupby("run_key"):
        ax.plot(sub["round"], sub["distance"], marker="o", linewidth=1.2, label=str(run_key))
    ax.set_xlabel("round")
    ax.set_ylabel("Euclidean distance to reference")
    ax.set_title("Selected vector distance to configured reference")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_feature_stability(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for run_key, sub in frame.groupby("run_key"):
        ax.plot(sub["round"], sub["adjacent_spearman"], marker="o", linewidth=1.2, label=str(run_key))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("adjacent-round Spearman")
    ax.set_title("Feature-importance stability over rounds")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
