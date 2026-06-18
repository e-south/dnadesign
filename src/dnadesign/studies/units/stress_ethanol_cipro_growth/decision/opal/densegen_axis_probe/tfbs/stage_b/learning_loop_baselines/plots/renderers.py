"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/plots/renderers.py

Render frozen round-0 replay review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

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
from .canvas import save_review_figure
from .contracts import FROZEN_REPLAY_STYLE_CONTRACT
from .helpers import (
    POOL_AVERAGE_COLOR,
    control_roles,
    cumulative_premise_title,
    legend_below_figure,
    ordered_label_names,
    round_summary,
    series_style,
    set_bar_ylim,
    sort_frame_by_label_order,
    validate_endpoint_source_used,
    write_json,
)

_REVIEW_MPL_RC = {
    "font.family": "DejaVu Sans",
    "font.size": REVIEW_AXIS_LABEL_FONTSIZE,
    "axes.titlesize": REVIEW_TITLE_FONTSIZE,
    "axes.labelsize": REVIEW_AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": REVIEW_TICK_LABEL_FONTSIZE,
    "ytick.labelsize": REVIEW_TICK_LABEL_FONTSIZE,
    "legend.fontsize": REVIEW_LEGEND_FONTSIZE,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#444B52",
    "ytick.color": "#444B52",
}


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
    known_label_reference_path = output_dir / "same_budget_known_label_gain_recovered.png"
    import matplotlib as mpl

    with mpl.rc_context(_REVIEW_MPL_RC):
        _plot_cumulative_enrichment(trajectory, cumulative_path, title=spec.comparison_set_label)
        _plot_endpoint_adaptive_gain(claims, endpoint_path)
        _plot_known_label_gain_recovery(claims, known_label_reference_path)
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
                "title": cumulative_premise_title(spec.comparison_set_label),
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Mean cumulative enrichment across seed runs, L = acquired-label mean / pool-label mean. "
                    "Active retraining updates after each acquisition; frozen ranking keeps the initial model fixed. "
                    "The known-label reference ranks by metadata using the same budget."
                ),
                "alt_text": (
                    "Cumulative enrichment compares active retraining, frozen initial ranking, shuffled controls, "
                    "pool average, and a same-budget known-label reference."
                ),
            },
            {
                "kind": "frozen_round0_endpoint_adaptive_gain",
                "path": str(endpoint_path),
                "title": "Active retraining adds final-budget enrichment",
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Final cumulative active-minus-frozen lift, L_active - L_frozen. Positive values mean "
                    "retraining added enrichment beyond the initial X-based ranking."
                ),
                "alt_text": ("Bar plot of final cumulative active-minus-frozen enrichment by metadata target."),
            },
            {
                "kind": "known_label_gain_recovery",
                "path": str(known_label_reference_path),
                "title": "Active retraining recovers part of the known-label gain",
                "interval_kind": "sample_sd",
                "interval": {"kind": "sample_sd", "unit": "seed_replicate", "is_confidence_interval": False},
                "caption": (
                    "Fraction of same-budget known-label gain recovered, (L_active - 1) / (L_known - 1), "
                    "across seed runs."
                ),
                "alt_text": (
                    "Bar plot showing how much of the same-budget known-label enrichment the active campaigns "
                    "recovered."
                ),
            },
        ],
    }
    write_json(manifest_path, manifest)
    validate_endpoint_source_used(endpoints)
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
    labels = ordered_label_names(df["label_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(max(7.2, 5.4 * len(labels)), 7.2),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes[0])
    control_series_role, control_display_role = control_roles(df)
    series_style_map = series_style(control_series_role, control_display_role)
    for ax, label_name in zip(axes, labels, strict=True):
        sub_label = df.loc[df["label_name"].astype(str) == label_name].copy()
        for key, style in series_style_map.items():
            selection_source, oracle_role = key
            sub = sub_label.loc[
                (sub_label["selection_source"].astype(str) == selection_source)
                & (sub_label["oracle_role"].astype(str) == oracle_role)
            ]
            if sub.empty:
                continue
            summary = round_summary(sub)
            ax.plot(
                summary["cumulative_selected_count"],
                summary["mean"],
                color=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                marker=style.marker,
                markersize=4.8,
                markeredgewidth=1.0,
                markeredgecolor=style.color,
                markerfacecolor="white" if style.linestyle == "--" else style.color,
                label=style.label,
            )
            if int(summary["replicate_count"].max()) > 1:
                ax.fill_between(
                    summary["cumulative_selected_count"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=style.color,
                    alpha=0.12,
                    linewidth=0,
                )
        ax.axhline(1.0, color=POOL_AVERAGE_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, label="Pool average")
        style_review_axis(ax, square=True)
        ax.set_xlabel("Acquired sequences", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
        ax.set_title(tfbs_label_compact_title(label_name), fontsize=REVIEW_AXIS_LABEL_FONTSIZE, pad=10)
    axes[0].set_ylabel(
        r"Cumulative enrichment ($\bar{y}_{acq}/\bar{y}_{pool}$)",
        fontsize=REVIEW_AXIS_LABEL_FONTSIZE,
    )
    fig.align_ylabels(axes)
    legend_below_figure(fig, axes[0])
    fig.suptitle(fill(cumulative_premise_title(title), width=90), fontsize=REVIEW_TITLE_FONTSIZE, y=0.955)
    fig.subplots_adjust(left=0.12, right=0.94, top=0.82, bottom=0.27, wspace=0.30)
    save_review_figure(fig, path)
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
    df = sort_frame_by_label_order(frame.copy())
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
    set_bar_ylim(ax, values, errors, reference=0.0)
    ax.set_ylabel(r"Adaptive gain ($L_{active} - L_{frozen}$)", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Metadata target", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_title("Active retraining adds final-budget enrichment", fontsize=REVIEW_TITLE_FONTSIZE, pad=14)
    ax.tick_params(axis="x", labelrotation=0)
    if (errors > 0).any():
        ax.legend(
            frameon=False,
            fontsize=REVIEW_LEGEND_FONTSIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncols=1,
        )
    fig.subplots_adjust(left=0.22, right=0.94, top=0.84, bottom=0.30)
    save_review_figure(fig, path)
    plt.close(fig)


def _plot_known_label_gain_recovery(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {
        "label_name",
        "active_fraction_of_known_label_gain_recovered_mean",
        "active_fraction_of_known_label_gain_recovered_sample_sd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Known-label gain recovery plot missing column(s): {missing}")
    df = sort_frame_by_label_order(frame.copy())
    labels = [tfbs_label_compact_title(value) for value in df["label_name"]]
    values = pd.to_numeric(df["active_fraction_of_known_label_gain_recovered_mean"], errors="raise")
    errors = pd.to_numeric(df["active_fraction_of_known_label_gain_recovered_sample_sd"], errors="raise")
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
    set_bar_ylim(ax, values, errors, reference=1.0)
    ax.set_ylabel(r"Recovered gain ($(L_{active} - 1)/(L_{known} - 1)$)", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Metadata target", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Active retraining recovers part of the known-label gain",
        fontsize=REVIEW_TITLE_FONTSIZE,
        pad=14,
    )
    if (errors > 0).any():
        ax.legend(
            frameon=False,
            fontsize=REVIEW_LEGEND_FONTSIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncols=1,
        )
    fig.subplots_adjust(left=0.24, right=0.92, top=0.82, bottom=0.32)
    save_review_figure(fig, path)
    plt.close(fig)
