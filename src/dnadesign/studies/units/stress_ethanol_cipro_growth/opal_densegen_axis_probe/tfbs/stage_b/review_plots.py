"""Reviewer-facing plots for DenseGen TFBS Stage B realized-label review."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..plot_style import role_color, style_review_axis
from ..stage_a.manifests import file_sha256

REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_review_plots.v1"


def materialize_tfbs_stage_b_realized_review_plots(
    *,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write compact true-label plots for Stage B peer-review inspection."""

    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _read_csv(trajectory_path, label="trajectory")
    pair_summary = _read_csv(pair_path, label="pair summary")

    label_names = _shared_label_names(trajectory, pair_summary)
    plots: list[dict[str, Any]] = []
    for label_name in label_names:
        slug = _slug(label_name)
        plots.append(
            _materialize_plot(
                path=output_dir / f"{slug}__selected_true_lift_trajectory.png",
                title=f"{label_name}: selected true-label lift over rounds",
                kind="realized_label_lift_trajectory",
                label_name=label_name,
                draw=lambda path, label_name=label_name: _plot_lift_trajectory(trajectory, path, label_name=label_name),
            )
        )
        plots.append(
            _materialize_plot(
                path=output_dir / f"{slug}__positive_minus_null_lift_summary.png",
                title=f"{label_name}: positive-minus-null realized lift",
                kind="positive_null_lift_summary",
                label_name=label_name,
                draw=lambda path, label_name=label_name: _plot_positive_null_summary(
                    pair_summary,
                    path,
                    label_name=label_name,
                ),
            )
        )
    manifest = {
        "schema_version": REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_trajectory_csv_path": str(trajectory_path),
        "source_trajectory_csv_hash": file_sha256(trajectory_path),
        "source_pair_summary_csv_path": str(pair_path),
        "source_pair_summary_csv_hash": file_sha256(pair_path),
        "plot_count": len(plots),
        "plots": plots,
        "interpretation_boundary": (
            "These plots use realized oracle labels from selected rows. They are evidence surfaces for learnability "
            "review, not acquisition-score traces."
        ),
    }
    manifest_path = output_dir / "tfbs_stage_b_realized_label_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B realized review {label} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Stage B realized review {label} CSV is empty: {path}")
    return frame


def _shared_label_names(trajectory: pd.DataFrame, pair_summary: pd.DataFrame) -> list[str]:
    required = {"label_name"}
    for label, frame in {"trajectory": trajectory, "pair summary": pair_summary}.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Stage B realized review {label} missing column(s): {missing}")
    trajectory_labels = set(trajectory["label_name"].astype(str))
    pair_labels = set(pair_summary["label_name"].astype(str))
    labels = sorted(trajectory_labels & pair_labels)
    if not labels:
        raise ValueError("Stage B realized review plots require at least one shared label_name")
    missing_from_pairs = sorted(trajectory_labels - pair_labels)
    missing_from_trajectory = sorted(pair_labels - trajectory_labels)
    if missing_from_pairs or missing_from_trajectory:
        raise ValueError(
            "Stage B realized review plot label mismatch "
            f"(missing_from_pairs={missing_from_pairs}, missing_from_trajectory={missing_from_trajectory})"
        )
    return labels


def _materialize_plot(*, path: Path, title: str, kind: str, label_name: str, draw: Any) -> dict[str, Any]:
    draw(path)
    return {
        "kind": kind,
        "title": title,
        "label_name": label_name,
        "path": str(path),
        "sha256": file_sha256(path),
        "alt_text": _alt_text(kind),
    }


def _plot_lift_trajectory(frame: pd.DataFrame, path: Path, *, label_name: str) -> None:
    import matplotlib.pyplot as plt

    required = {"label_name", "oracle_role", "round", "selected_true_lift_ratio", "seed_true_lift_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B lift trajectory plot missing column(s): {missing}")

    df = frame.copy()
    df["round"] = pd.to_numeric(df["round"], errors="raise")
    df["lift"] = pd.to_numeric(df["selected_true_lift_ratio"], errors="raise")
    labels = [label_name]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9.5, max(3.2, 2.1 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label_name in zip(axes, labels, strict=True):
        sub_label = df.loc[df["label_name"].astype(str) == label_name].sort_values(["oracle_role", "round"])
        for role, sub_role in sub_label.groupby("oracle_role"):
            seed_lift = float(pd.to_numeric(sub_role["seed_true_lift_ratio"], errors="raise").iloc[0])
            ax.plot(
                sub_role["round"],
                sub_role["lift"],
                marker="o",
                linewidth=1.2,
                color=role_color(role),
                label=str(role),
            )
            ax.scatter(
                [-1],
                [seed_lift],
                marker="s",
                s=34,
                color=role_color(role),
                edgecolor="#2E3135",
                linewidth=0.4,
                zorder=4,
            )
        ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
        ax.set_xlim(left=-1.5)
        style_review_axis(ax)
        ax.set_ylabel("lift")
        ax.set_title(str(label_name), loc="left", fontsize=10)
    axes[-1].set_xlabel("round")
    axes[-1].set_xticks([-1, *sorted(df["round"].unique().tolist()[:: max(1, len(df["round"].unique()) // 6)])])
    axes[-1].set_xticklabels(["seed", *map(str, axes[-1].get_xticks()[1:].astype(int).tolist())])
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle(f"{label_name}: seed batch and selected true-label lift", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_positive_null_summary(frame: pd.DataFrame, path: Path, *, label_name: str) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "final_positive_minus_null_lift_ratio",
        "trapezoid_auc_positive_minus_null_lift_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B positive/null summary plot missing column(s): {missing}")

    df = frame.loc[frame["label_name"].astype(str) == label_name].copy().sort_values("label_name")
    if df.empty:
        raise ValueError(f"Stage B positive/null summary has no rows for label {label_name!r}")
    x = range(len(df))
    final_delta = pd.to_numeric(df["final_positive_minus_null_lift_ratio"], errors="raise")
    auc_delta = pd.to_numeric(df["trapezoid_auc_positive_minus_null_lift_ratio"], errors="raise")
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    style_review_axis(ax)
    width = 0.38
    ax.bar([index - width / 2 for index in x], final_delta, width=width, color="#446A8C", label="final delta")
    ax.bar([index + width / 2 for index in x], auc_delta, width=width, color="#8A8F98", label="normalized AUC delta")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label_name"].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylabel("positive lift - null/control lift")
    ax.set_title(f"{label_name}: positive-minus-null realized lift")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "label"


def _alt_text(kind: str) -> str:
    if kind == "realized_label_lift_trajectory":
        return (
            "Line plot of selected true-label lift over active-learning rounds for each sentinel label, with the "
            "initial seed batch shown as a square marker before round zero and positive and matched-null roles drawn "
            "separately."
        )
    if kind == "positive_null_lift_summary":
        return (
            "Bar plot comparing final and normalized trajectory AUC positive-minus-null lift for each sentinel label, "
            "using realized oracle labels from selected rows."
        )
    return "DenseGen TFBS Stage B realized-label review plot."
