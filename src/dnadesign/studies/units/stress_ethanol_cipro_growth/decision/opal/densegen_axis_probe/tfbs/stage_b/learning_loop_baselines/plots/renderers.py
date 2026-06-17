"""Render frozen round-0 replay review plots."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill
from typing import Any

import pandas as pd

from ....label_text import tfbs_label_compact_title
from ....plot_style import (
    REVIEW_AXIS_LABEL_FONTSIZE,
    REVIEW_LEGEND_FONTSIZE,
    REVIEW_SQUARE_FIGSIZE,
    REVIEW_TICK_LABEL_FONTSIZE,
    REVIEW_TITLE_FONTSIZE,
    style_review_axis,
)
from ..contracts import LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION, LearningLoopBaselineSpec
from .contracts import FROZEN_REPLAY_STYLE_CONTRACT

_POSITIVE_ROLE = "positive"


def materialize_frozen_replay_plots(
    *,
    trajectory_csv_path: str | Path,
    endpoint_summary_csv_path: str | Path,
    claim_interpretation_csv_path: str | Path,
    out_dir: str | Path,
    spec: LearningLoopBaselineSpec,
) -> Path:
    """Write replay plots and a plot manifest."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(trajectory_csv_path)
    endpoints = pd.read_csv(endpoint_summary_csv_path)
    claims = pd.read_csv(claim_interpretation_csv_path)
    cumulative_path = output_dir / "frozen_round0_cumulative_enrichment.png"
    endpoint_path = output_dir / "frozen_round0_endpoint_adaptive_gain.png"
    ceiling_path = output_dir / "top_budget_signal_recovery.png"
    _plot_cumulative_enrichment(trajectory, cumulative_path, title=spec.comparison_set_label)
    _plot_endpoint_adaptive_gain(claims, endpoint_path)
    _plot_top_budget_signal_recovery(claims, ceiling_path)
    manifest_path = output_dir / "learning_loop_baseline_plot_manifest.json"
    manifest = {
        "schema_version": LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION,
        "source_trajectory_csv_path": str(Path(trajectory_csv_path)),
        "source_endpoint_summary_csv_path": str(Path(endpoint_summary_csv_path)),
        "source_claim_interpretation_csv_path": str(Path(claim_interpretation_csv_path)),
        "plot_count": 3,
        "style_contract": dict(FROZEN_REPLAY_STYLE_CONTRACT),
        "plots": [
            {
                "kind": "frozen_round0_cumulative_enrichment",
                "path": str(cumulative_path),
                "title": f"{spec.comparison_set_label}: active vs frozen",
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Cumulative enrichment, mean label value across all acquired sequences divided by the "
                    "candidate-pool mean, for active retraining, frozen round-0 ranking, and the best possible "
                    "same-budget ceiling."
                ),
                "alt_text": (
                    "Cumulative enrichment trajectories compare active retraining, frozen round-0 ranking, "
                    "controls, baseline, and the best possible same-budget ceiling."
                ),
            },
            {
                "kind": "frozen_round0_endpoint_adaptive_gain",
                "path": str(endpoint_path),
                "title": "Adaptive gain at the final acquired budget",
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Final cumulative active-minus-frozen lift, L_active - L_frozen. Positive values mean "
                    "retraining added enrichment beyond the initial X-based ranking."
                ),
                "alt_text": ("Bar plot of final cumulative active-minus-frozen enrichment by DenseGen label."),
            },
            {
                "kind": "top_budget_signal_recovery",
                "path": str(ceiling_path),
                "title": "Fraction of achievable enrichment recovered",
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Fraction of best possible same-budget gain recovered, (L_active - 1) / (L_top - 1), "
                    "across seed runs."
                ),
                "alt_text": (
                    "Bar plot showing how much of the best possible same-budget enrichment the active campaigns "
                    "recovered."
                ),
            },
        ],
    }
    _write_json(manifest_path, manifest)
    _validate_endpoint_source_used(endpoints)
    return manifest_path


def _plot_cumulative_enrichment(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "selection_source",
        "oracle_role",
        "cumulative_selected_count",
        "cumulative_lift_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen replay cumulative plot missing column(s): {missing}")
    df = frame.copy()
    labels = sorted(df["label_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(max(7.2, 5.4 * len(labels)), 7.2),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes[0])
    control_series_role, control_display_role = _control_roles(df)
    series_style = _series_style(control_series_role, control_display_role)
    for ax, label_name in zip(axes, labels, strict=True):
        sub_label = df.loc[df["label_name"].astype(str) == label_name].copy()
        for key, (color, linestyle, label) in series_style.items():
            selection_source, oracle_role = key
            sub = sub_label.loc[
                (sub_label["selection_source"].astype(str) == selection_source)
                & (sub_label["oracle_role"].astype(str) == oracle_role)
            ]
            if sub.empty:
                continue
            summary = _round_summary(sub)
            ax.plot(
                summary["cumulative_selected_count"],
                summary["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                label=label,
            )
            if int(summary["replicate_count"].max()) > 1:
                ax.fill_between(
                    summary["cumulative_selected_count"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        ax.axhline(1.0, color="#222222", linestyle="--", linewidth=0.9, label="Pool baseline")
        style_review_axis(ax, square=True)
        ax.set_xlabel("Acquired sequences", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
        ax.set_title(tfbs_label_compact_title(label_name), fontsize=REVIEW_AXIS_LABEL_FONTSIZE, pad=10)
    axes[0].set_ylabel(
        r"Cumulative enrichment ($\bar{y}_{acq}/\bar{y}_{pool}$)",
        fontsize=REVIEW_AXIS_LABEL_FONTSIZE,
    )
    _legend_below_figure(fig, axes[0])
    fig.suptitle(fill(f"{title}: active vs frozen", width=54), fontsize=REVIEW_TITLE_FONTSIZE, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.94, top=0.80, bottom=0.25, wspace=0.30)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _plot_endpoint_adaptive_gain(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "active_minus_frozen_final_cumulative_lift_mean",
        "active_minus_frozen_final_cumulative_lift_sample_sd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen replay endpoint plot missing column(s): {missing}")
    df = frame.copy().sort_values("label_name")
    labels = [tfbs_label_compact_title(value) for value in df["label_name"]]
    values = pd.to_numeric(df["active_minus_frozen_final_cumulative_lift_mean"], errors="raise")
    errors = pd.to_numeric(df["active_minus_frozen_final_cumulative_lift_sample_sd"], errors="raise")
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=False)
    style_review_axis(ax, square=True)
    bars = ax.bar(labels, values, color="#446A8C", width=0.58)
    if (errors > 0).any():
        ax.errorbar(
            range(len(values)),
            values,
            yerr=errors,
            fmt="none",
            ecolor="#2E3135",
            elinewidth=1.2,
            capsize=4,
            label="Sample SD across seed runs",
        )
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=REVIEW_TICK_LABEL_FONTSIZE)
    _set_bar_ylim(ax, values, errors, reference=0.0)
    ax.set_ylabel(r"Adaptive gain ($L_{active} - L_{frozen}$)", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("DenseGen label", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_title("Adaptive gain at final acquired budget", fontsize=REVIEW_TITLE_FONTSIZE, pad=14)
    ax.tick_params(axis="x", labelrotation=0)
    if (errors > 0).any():
        ax.legend(
            frameon=False,
            fontsize=REVIEW_LEGEND_FONTSIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncols=1,
        )
    fig.subplots_adjust(left=0.18, right=0.94, top=0.84, bottom=0.26)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _plot_top_budget_signal_recovery(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "active_fraction_of_top_budget_gain_recovered_mean",
        "active_fraction_of_top_budget_gain_recovered_sample_sd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Same-budget reference recovery plot missing column(s): {missing}")
    df = frame.copy().sort_values("label_name")
    labels = [tfbs_label_compact_title(value) for value in df["label_name"]]
    values = pd.to_numeric(df["active_fraction_of_top_budget_gain_recovered_mean"], errors="raise")
    errors = pd.to_numeric(df["active_fraction_of_top_budget_gain_recovered_sample_sd"], errors="raise")
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=False)
    style_review_axis(ax, square=True)
    bars = ax.bar(labels, values, color="#5F7F5F", width=0.58)
    if (errors > 0).any():
        ax.errorbar(
            range(len(values)),
            values,
            yerr=errors,
            fmt="none",
            ecolor="#2E3135",
            elinewidth=1.2,
            capsize=4,
            label="Sample SD across seed runs",
        )
    ax.axhline(1.0, color="#222222", linewidth=0.9, linestyle="--")
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=REVIEW_TICK_LABEL_FONTSIZE)
    _set_bar_ylim(ax, values, errors, reference=1.0)
    ax.set_ylabel(r"Recovered gain ($(L_{active} - 1)/(L_{top} - 1)$)", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("DenseGen label", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_title("Fraction of achievable enrichment recovered", fontsize=REVIEW_TITLE_FONTSIZE, pad=14)
    if (errors > 0).any():
        ax.legend(
            frameon=False,
            fontsize=REVIEW_LEGEND_FONTSIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncols=1,
        )
    fig.subplots_adjust(left=0.18, right=0.94, top=0.84, bottom=0.26)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _set_bar_ylim(ax: object, values: pd.Series, errors: pd.Series, *, reference: float) -> None:
    upper = max(float((values + errors).max()), float(reference), 0.0)
    lower = min(float((values - errors).min()), float(reference), 0.0)
    span = max(upper - lower, 1.0)
    ax.set_ylim(lower - 0.08 * span, upper + 0.20 * span)


def _round_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    numeric["cumulative_lift_ratio"] = pd.to_numeric(numeric["cumulative_lift_ratio"], errors="raise")
    summary = (
        numeric.groupby("cumulative_selected_count", as_index=False)["cumulative_lift_ratio"]
        .agg(mean="mean", sample_sd=lambda values: values.std(ddof=1), replicate_count="count")
        .sort_values("cumulative_selected_count")
    )
    summary["sample_sd"] = summary["sample_sd"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["sample_sd"]
    summary["upper"] = summary["mean"] + summary["sample_sd"]
    return summary


def _series_style(control_series_role: str, control_display_role: str) -> dict[tuple[str, str], tuple[str, str, str]]:
    control_label = _control_label(control_display_role)
    return {
        ("active_retraining", _POSITIVE_ROLE): ("#446A8C", "-", "Active DenseGen"),
        ("frozen_round0", _POSITIVE_ROLE): ("#446A8C", "--", "Frozen DenseGen"),
        ("top_budget_ceiling", _POSITIVE_ROLE): ("#5F7F5F", ":", "Best same-budget ceiling"),
        ("active_retraining", control_series_role): ("#8C4E4A", "-", f"Active {control_label}"),
        ("frozen_round0", control_series_role): ("#8C4E4A", "--", f"Frozen {control_label}"),
    }


def _control_roles(frame: pd.DataFrame) -> tuple[str, str]:
    roles = sorted(set(frame["oracle_role"].astype(str)) - {_POSITIVE_ROLE})
    if len(roles) != 1:
        raise ValueError(f"Learning-loop cumulative plot expected one control role; found {roles}")
    display_roles: list[str] = []
    if "scientific_control_role" in frame.columns:
        controls = frame.loc[frame["oracle_role"].astype(str) == roles[0], "scientific_control_role"]
        display_roles = sorted({role for role in controls.dropna().astype(str) if role})
        if len(display_roles) > 1:
            raise ValueError(f"Learning-loop cumulative plot found multiple scientific control roles: {display_roles}")
    return roles[0], display_roles[0] if display_roles else roles[0]


def _control_label(role: str) -> str:
    if role == "matched_null":
        return "scrambled control"
    if role == "count_fixed_shuffled_slot_negative_control":
        return "shuffled control"
    return role.replace("_", " ")


def _legend_below_figure(fig: object, ax: object) -> None:
    handles, labels_out = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_out, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncols=min(3, max(1, len(by_label))),
        frameon=False,
        fontsize=REVIEW_LEGEND_FONTSIZE - 2,
        columnspacing=0.75,
        handlelength=1.25,
        handletextpad=0.45,
    )


def _validate_endpoint_source_used(endpoints: pd.DataFrame) -> None:
    if endpoints.empty:
        raise ValueError("Frozen replay endpoint summary source is empty")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
