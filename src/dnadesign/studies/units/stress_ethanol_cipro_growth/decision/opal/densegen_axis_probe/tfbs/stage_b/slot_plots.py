"""Slot-count confound plots for DenseGen TFBS Stage B review."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..plot_style import role_color, style_review_axis
from ..stage_a.manifests import file_sha256

SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_slot_diagnostic_plots.v1"


def materialize_tfbs_stage_b_slot_diagnostic_plots(
    *,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    count_distribution_csv_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write reviewer-facing plots for slot-label count-confound diagnostics."""

    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    distribution_path = Path(count_distribution_csv_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _read_csv(trajectory_path, label="slot trajectory")
    pair_summary = _read_csv(pair_path, label="slot pair summary")
    _read_csv(distribution_path, label="slot count distribution")

    plots = [
        _materialize_plot(
            path=output_dir / "slot_target_count_mean_trajectory.png",
            title="Selected target-family count over rounds",
            kind="slot_target_count_mean_trajectory",
            draw=lambda path: _plot_target_count_mean(trajectory, path),
        ),
        _materialize_plot(
            path=output_dir / "slot_count_stratified_lift_trajectory.png",
            title="Count-stratified slot-label lift over rounds",
            kind="slot_count_stratified_lift_trajectory",
            draw=lambda path: _plot_count_stratified_lift(trajectory, path),
        ),
        _materialize_plot(
            path=output_dir / "slot_count_stratified_lift_summary.png",
            title="Count-stratified positive-minus-null slot lift",
            kind="slot_count_stratified_lift_summary",
            draw=lambda path: _plot_count_stratified_summary(pair_summary, path),
        ),
    ]
    manifest = {
        "schema_version": SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_trajectory_csv_path": str(trajectory_path),
        "source_trajectory_csv_hash": file_sha256(trajectory_path),
        "source_pair_summary_csv_path": str(pair_path),
        "source_pair_summary_csv_hash": file_sha256(pair_path),
        "source_count_distribution_csv_path": str(distribution_path),
        "source_count_distribution_csv_hash": file_sha256(distribution_path),
        "plot_count": len(plots),
        "plots": plots,
        "interpretation_boundary": (
            "These plots diagnose whether slot-label enrichment is explained by target-family count. "
            "They are diagnostic evidence surfaces, not clean negative-control claims by themselves."
        ),
    }
    manifest_path = output_dir / "tfbs_stage_b_slot_diagnostic_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B {label} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Stage B {label} CSV is empty: {path}")
    return frame


def _materialize_plot(*, path: Path, title: str, kind: str, draw: Any) -> dict[str, Any]:
    draw(path)
    return {
        "kind": kind,
        "title": title,
        "path": str(path),
        "sha256": file_sha256(path),
        "alt_text": _alt_text(kind),
    }


def _plot_target_count_mean(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {"label_name", "oracle_role", "round", "selected_target_count_mean", "pool_target_count_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B slot count plot missing column(s): {missing}")

    df = frame.copy()
    df["round"] = pd.to_numeric(df["round"], errors="raise")
    df["selected_target_count_mean"] = pd.to_numeric(df["selected_target_count_mean"], errors="raise")
    labels = sorted(df["label_name"].astype(str).unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(9.5, max(3.1, 2.7 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label_name in zip(axes, labels, strict=True):
        sub_label = df.loc[df["label_name"].astype(str) == label_name].sort_values(["oracle_role", "round"])
        pool_mean = float(pd.to_numeric(sub_label["pool_target_count_mean"], errors="raise").iloc[0])
        for role, sub_role in sub_label.groupby("oracle_role"):
            ax.plot(
                sub_role["round"],
                sub_role["selected_target_count_mean"],
                marker="o",
                linewidth=1.2,
                color=role_color(role),
                label=str(role),
            )
        ax.axhline(pool_mean, color="#222222", linewidth=0.8, linestyle="--", label="pool mean")
        ax.set_ylabel("mean count")
        ax.set_title(str(label_name), loc="left", fontsize=10)
        style_review_axis(ax)
    axes[-1].set_xlabel("round")
    axes[0].legend(frameon=False, fontsize=8, ncols=3)
    fig.suptitle("Selected target-family count over rounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_count_stratified_lift(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {"label_name", "oracle_role", "round", "count_stratified_lift_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B slot count-stratified plot missing column(s): {missing}")

    df = frame.copy()
    df["round"] = pd.to_numeric(df["round"], errors="raise")
    df["lift"] = pd.to_numeric(df["count_stratified_lift_ratio"], errors="coerce")
    labels = sorted(df["label_name"].astype(str).unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(9.5, max(3.1, 2.7 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label_name in zip(axes, labels, strict=True):
        sub_label = df.loc[df["label_name"].astype(str) == label_name].sort_values(["oracle_role", "round"])
        for role, sub_role in sub_label.groupby("oracle_role"):
            finite = sub_role.loc[sub_role["lift"].notna()]
            invalid = sub_role.loc[sub_role["lift"].isna()]
            ax.plot(
                finite["round"],
                finite["lift"],
                marker="o",
                linewidth=1.2,
                color=role_color(role),
                label=str(role),
            )
            if not invalid.empty:
                ax.scatter(
                    invalid["round"],
                    [0.0] * len(invalid),
                    marker="x",
                    s=30,
                    color=role_color(role),
                    alpha=0.65,
                    linewidth=0.9,
                    label=f"{role} no nondeterministic selected rows",
                )
        ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="#AEB4BA", linewidth=0.7, linestyle=":")
        ax.set_ylabel("lift")
        ax.set_title(str(label_name), loc="left", fontsize=10)
        style_review_axis(ax)
    axes[-1].set_xlabel("round")
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("Slot-label lift after controlling for target-family count", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_count_stratified_summary(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "final_positive_minus_null_count_stratified_lift_ratio",
        "auc_positive_minus_null_count_stratified_lift_ratio",
        "slot_diagnostic_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B slot summary plot missing column(s): {missing}")

    df = frame.copy().sort_values("label_name")
    x = range(len(df))
    final_delta = pd.to_numeric(df["final_positive_minus_null_count_stratified_lift_ratio"], errors="coerce")
    auc_delta = pd.to_numeric(df["auc_positive_minus_null_count_stratified_lift_ratio"], errors="coerce")
    colors = [_status_color(value) for value in df["slot_diagnostic_status"].astype(str).tolist()]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    style_review_axis(ax)
    width = 0.38
    ax.bar([index - width / 2 for index in x], final_delta, width=width, color=colors, label="final delta")
    ax.bar([index + width / 2 for index in x], auc_delta, width=width, color="#8A8F98", label="normalized AUC delta")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label_name"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylabel("positive lift - null/control lift")
    ax.set_title("Count-stratified slot-position diagnostic")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _alt_text(kind: str) -> str:
    if kind == "slot_target_count_mean_trajectory":
        return (
            "Line plot of selected target-family count over active-learning rounds for slot-label campaigns, "
            "with positive and matched-null or control roles shown separately."
        )
    if kind == "slot_count_stratified_lift_trajectory":
        return (
            "Line plot of slot-label lift over rounds after deterministic count strata are excluded and selected "
            "rows are compared to count-stratum baselines. X markers at y equals zero indicate rounds with no "
            "nondeterministic selected rows rather than missing plotted data."
        )
    if kind == "slot_count_stratified_lift_summary":
        return (
            "Bar plot comparing final and normalized trajectory AUC positive-minus-null slot lift after controlling "
            "for target-family count."
        )
    return "DenseGen TFBS Stage B slot-count diagnostic plot."


def _status_color(value: str) -> str:
    if value == "position_signal_after_count_restriction":
        return "#5D7D4F"
    if value == "not_separated_after_count_restriction":
        return "#B07D3C"
    return "#8C4E4A"
