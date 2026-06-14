"""Renderer registry for Stage B realized-label review plots."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from ....plot_style import (
    REVIEW_AXIS_LABEL_FONTSIZE,
    REVIEW_LEGEND_FONTSIZE,
    REVIEW_SQUARE_FIGSIZE,
    REVIEW_TITLE_FONTSIZE,
    role_color,
    style_review_axis,
)
from ...notebook_visuals.specs import StageBNotebookVisualSpec
from .contracts import RealizedReviewRenderer
from .display_text import (
    INITIAL_BATCH_TICK_LABEL,
    NO_ENRICHMENT_BASELINE_LABEL,
    TRAJECTORY_X_AXIS_LABEL,
    plot_manifest_title,
    positive_null_summary_subtitle,
    role_display_label,
    seed_pair_sample_sd_label,
    trajectory_plot_subtitle,
    trajectory_y_axis_label,
)
from .statistics import (
    replicate_column,
    replicate_round_summary,
    replicate_value_summary,
    role_sort_key,
    seed_lift_summary,
    trajectory_replicate_count,
)

_TITLE_Y = 0.965
_SUBTITLE_Y = 0.855
_TRAJECTORY_LAYOUT = {"left": 0.17, "right": 0.91, "top": 0.805, "bottom": 0.23}
_SUMMARY_LAYOUT = {"left": 0.18, "right": 0.92, "top": 0.805, "bottom": 0.24}
_LEGEND_ROW_ANCHOR_Y = -0.16


def build_realized_review_renderer_registry(
    renderers: Mapping[str, RealizedReviewRenderer],
) -> Mapping[str, RealizedReviewRenderer]:
    """Build a fail-fast realized-label renderer registry keyed by visual kind."""

    registry: dict[str, RealizedReviewRenderer] = {}
    for kind, renderer in renderers.items():
        token = str(kind).strip()
        if not token:
            raise ValueError("Stage B realized review renderer kind must be nonempty")
        if token in registry:
            raise ValueError(f"Duplicate Stage B realized review renderer kind: {token}")
        registry[token] = renderer
    if not registry:
        raise ValueError("Stage B realized review renderer registry must not be empty")
    return MappingProxyType(registry)


def realized_review_renderer(spec: StageBNotebookVisualSpec) -> RealizedReviewRenderer:
    """Return the renderer registered for ``spec`` or fail fast."""

    try:
        return REALIZED_REVIEW_RENDERERS[spec.kind]
    except KeyError as exc:
        raise RuntimeError(f"registered Stage B realized review visual has no renderer: {spec.kind}") from exc


REALIZED_REVIEW_RENDERERS = build_realized_review_renderer_registry(
    {
        "realized_label_lift_trajectory": (
            lambda trajectory, pair_summary, path, label_name: _plot_lift_trajectory(
                trajectory,
                path,
                label_name=label_name,
            )
        ),
        "positive_null_lift_summary": (
            lambda trajectory, pair_summary, path, label_name: _plot_positive_null_summary(
                pair_summary,
                path,
                label_name=label_name,
            )
        ),
    }
)


def _plot_lift_trajectory(frame: pd.DataFrame, path: Path, *, label_name: str) -> None:
    import matplotlib.pyplot as plt

    required = {"label_name", "oracle_role", "round", "selected_true_lift_ratio", "seed_true_lift_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B lift trajectory plot missing column(s): {missing}")

    df = frame.copy()
    df["round"] = pd.to_numeric(df["round"], errors="raise")
    df["lift"] = pd.to_numeric(df["selected_true_lift_ratio"], errors="raise")

    sub_label = df.loc[df["label_name"].astype(str) == label_name].sort_values(["oracle_role", "round"])
    if sub_label.empty:
        raise ValueError(f"Stage B lift trajectory has no rows for label {label_name!r}")

    replicate_count = trajectory_replicate_count(sub_label)
    control_role = _control_role_for_label(sub_label)
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(**_TRAJECTORY_LAYOUT)
    for role, sub_role in sorted(sub_label.groupby("oracle_role"), key=lambda item: role_sort_key(item[0])):
        role_summary = replicate_round_summary(sub_role)
        has_spread = bool(role_summary["replicate_count"].max() > 1)
        seed_lift = seed_lift_summary(sub_role)
        role_label = role_display_label(
            role,
            label_name=label_name,
            control_role=_control_role_for_label(sub_role),
        )
        if has_spread:
            for _, sub_replicate in sub_role.groupby(replicate_column(sub_role), dropna=False):
                ax.plot(
                    sub_replicate["round"],
                    sub_replicate["lift"],
                    linewidth=0.9,
                    color=role_color(role),
                    alpha=0.20,
                    zorder=1,
                    label="_nolegend_",
                )
        ax.plot(
            role_summary["round"],
            role_summary["lift_mean"],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=role_color(role),
            label=role_label,
            zorder=3,
        )
        if has_spread:
            ax.fill_between(
                role_summary["round"].to_numpy(dtype=float),
                role_summary["lift_lower"].to_numpy(dtype=float),
                role_summary["lift_upper"].to_numpy(dtype=float),
                color=role_color(role),
                alpha=0.16,
                linewidth=0,
                label="_nolegend_",
                zorder=2,
            )
        ax.scatter(
            [-1],
            [seed_lift["mean"]],
            marker="s",
            s=46,
            color=role_color(role),
            edgecolor="#2E3135",
            linewidth=0.6,
            zorder=4,
        )
        if int(seed_lift["replicate_count"]) > 1:
            ax.vlines(
                [-1],
                [seed_lift["mean"] - seed_lift["sample_sd"]],
                [seed_lift["mean"] + seed_lift["sample_sd"]],
                color=role_color(role),
                linewidth=1.4,
                alpha=0.72,
                zorder=3,
            )
    ax.axhline(1.0, color="#222222", linewidth=0.9, linestyle="--", label=NO_ENRICHMENT_BASELINE_LABEL)
    ax.set_xlim(left=-1.5)
    ax.margins(y=0.08)
    style_review_axis(ax, square=True)
    ax.set_xlabel(TRAJECTORY_X_AXIS_LABEL, fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(trajectory_y_axis_label(label_name), fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    fig.text(
        0.50,
        _TITLE_Y,
        _plot_title(
            plot_manifest_title(
                "realized_label_lift_trajectory", label_name=label_name, replicate_count=replicate_count
            )
        ),
        ha="center",
        va="top",
        fontsize=REVIEW_TITLE_FONTSIZE,
    )
    fig.text(
        0.50,
        _SUBTITLE_Y,
        trajectory_plot_subtitle(label_name, replicate_count=replicate_count, control_role=control_role),
        ha="center",
        va="top",
        fontsize=REVIEW_LEGEND_FONTSIZE,
        color="#4D555C",
    )
    _set_trajectory_round_ticks(ax, df)
    ax.legend(
        frameon=False,
        fontsize=REVIEW_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, _LEGEND_ROW_ANCHOR_Y),
        ncols=3,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.5,
    )
    fig.savefig(path, dpi=160, facecolor="white")
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
    final_delta = pd.to_numeric(df["final_positive_minus_null_lift_ratio"], errors="raise")
    auc_delta = pd.to_numeric(df["trapezoid_auc_positive_minus_null_lift_ratio"], errors="raise")
    final_summary = replicate_value_summary(final_delta)
    auc_summary = replicate_value_summary(auc_delta)
    replicate_count = max(int(final_summary["replicate_count"]), int(auc_summary["replicate_count"]))
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(**_SUMMARY_LAYOUT)
    style_review_axis(ax, square=True)
    values = [final_summary["mean"], auc_summary["mean"]]
    bars = ax.bar([0, 1], values, width=0.54, color=["#446A8C", "#8A8F98"])
    if final_summary["replicate_count"] > 1 or auc_summary["replicate_count"] > 1:
        ax.errorbar(
            [0, 1],
            values,
            yerr=[
                [final_summary["sample_sd"], auc_summary["sample_sd"]],
                [final_summary["sample_sd"], auc_summary["sample_sd"]],
            ],
            fmt="none",
            ecolor="#2E3135",
            elinewidth=1.2,
            capsize=4,
            label=seed_pair_sample_sd_label(replicate_count=replicate_count),
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.margins(y=0.16)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=REVIEW_LEGEND_FONTSIZE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Final round", "Trajectory AUC"], rotation=0, ha="center")
    ax.set_ylabel(
        "DenseGen - control enrichment",
        fontsize=REVIEW_AXIS_LABEL_FONTSIZE,
    )
    fig.text(
        0.50,
        _TITLE_Y,
        _plot_title(
            plot_manifest_title("positive_null_lift_summary", label_name=label_name, replicate_count=replicate_count)
        ),
        ha="center",
        va="top",
        fontsize=REVIEW_TITLE_FONTSIZE,
    )
    fig.text(
        0.50,
        _SUBTITLE_Y,
        positive_null_summary_subtitle(replicate_count=replicate_count),
        ha="center",
        va="top",
        fontsize=REVIEW_LEGEND_FONTSIZE,
        color="#4D555C",
    )
    if final_summary["replicate_count"] > 1 or auc_summary["replicate_count"] > 1:
        ax.legend(
            frameon=False,
            fontsize=REVIEW_LEGEND_FONTSIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, _LEGEND_ROW_ANCHOR_Y),
            ncols=1,
            handlelength=1.5,
            handletextpad=0.5,
        )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _set_trajectory_round_ticks(ax: object, df: pd.DataFrame) -> None:
    round_values = sorted(int(value) for value in df["round"].dropna().unique().tolist())
    if not round_values:
        ax.set_xticks([-1])
        ax.set_xticklabels([INITIAL_BATCH_TICK_LABEL])
        return
    step = max(1, len(round_values) // 6)
    tick_rounds = [value for value in round_values[::step] if value > 1]
    if round_values[-1] > 1 and (not tick_rounds or tick_rounds[-1] != round_values[-1]):
        tick_rounds.append(round_values[-1])
    if not tick_rounds and round_values[-1] != 0:
        tick_rounds.append(round_values[-1])
    ax.set_xticks([-1, *tick_rounds])
    ax.set_xticklabels([INITIAL_BATCH_TICK_LABEL, *(str(value) for value in tick_rounds)])


def _plot_title(text: str) -> str:
    return fill(str(text), width=42, break_long_words=False)


def _single_nonempty(values: object) -> str:
    series = pd.Series(values, dtype="object")
    clean = sorted({str(value) for value in series.tolist() if str(value) not in {"", "nan", "None"}})
    return clean[0] if len(clean) == 1 else ""


def _control_role_for_label(frame: pd.DataFrame) -> str:
    if "null_control_role" not in frame.columns:
        return ""
    matched_null = frame.loc[frame["oracle_role"].astype(str) == "matched_null", "null_control_role"]
    role = _single_nonempty(matched_null)
    if role:
        return role
    return _single_nonempty(frame["null_control_role"])
