"""Renderer registry for Stage B slot-diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from ....label_text import tfbs_label_title
from ....plot_style import (
    REVIEW_AXIS_LABEL_FONTSIZE,
    REVIEW_LEGEND_FONTSIZE,
    REVIEW_SQUARE_FIGSIZE,
    REVIEW_TITLE_FONTSIZE,
    role_color,
    style_review_axis,
)
from ...notebook_visuals.specs import StageBNotebookVisualSpec
from .contracts import SlotDiagnosticRenderer

_ROW_PANEL_HEIGHT = 6.8
_ROW_PANEL_WIDTH = 4.8


def build_slot_diagnostic_renderer_registry(
    renderers: Mapping[str, SlotDiagnosticRenderer],
) -> Mapping[str, SlotDiagnosticRenderer]:
    """Build a fail-fast slot-diagnostic renderer registry keyed by visual kind."""

    registry: dict[str, SlotDiagnosticRenderer] = {}
    for kind, renderer in renderers.items():
        token = str(kind).strip()
        if not token:
            raise ValueError("Stage B slot-diagnostic renderer kind must be nonempty")
        if token in registry:
            raise ValueError(f"Duplicate Stage B slot-diagnostic renderer kind: {token}")
        registry[token] = renderer
    if not registry:
        raise ValueError("Stage B slot-diagnostic renderer registry must not be empty")
    return MappingProxyType(registry)


def slot_diagnostic_renderer(spec: StageBNotebookVisualSpec) -> SlotDiagnosticRenderer:
    """Return the renderer registered for ``spec`` or fail fast."""

    try:
        return SLOT_DIAGNOSTIC_RENDERERS[spec.kind]
    except KeyError as exc:
        raise RuntimeError(f"registered Stage B slot diagnostic visual has no renderer: {spec.kind}") from exc


SLOT_DIAGNOSTIC_RENDERERS = build_slot_diagnostic_renderer_registry(
    {
        "slot_target_count_mean_trajectory": (
            lambda trajectory, pair_summary, count_distribution, path: _plot_target_count_mean(trajectory, path)
        ),
        "slot_count_stratified_lift_trajectory": (
            lambda trajectory, pair_summary, count_distribution, path: _plot_count_stratified_lift(trajectory, path)
        ),
        "slot_count_stratified_lift_summary": (
            lambda trajectory, pair_summary, count_distribution, path: _plot_count_stratified_summary(
                pair_summary,
                path,
            )
        ),
    }
)


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
    fig, axes = _row_square_axes(label_count=len(labels), sharey=False)
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
                label=_role_label(role),
            )
        ax.axhline(pool_mean, color="#222222", linewidth=0.8, linestyle="--", label="pool mean")
        ax.set_xlabel("Round", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
        ax.set_title(tfbs_label_title(label_name), fontsize=REVIEW_AXIS_LABEL_FONTSIZE, pad=10)
        style_review_axis(ax, square=True)
    axes[0].set_ylabel("Selected target-family mean count", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    _legend_below(fig, axes)
    fig.suptitle("Selected target-family count over rounds", fontsize=REVIEW_TITLE_FONTSIZE, y=0.975)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.24, wspace=0.34)
    fig.savefig(path, dpi=160, facecolor="white")
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
    fig, axes = _row_square_axes(label_count=len(labels), sharey=True)
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
                label=_role_label(role),
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
                    label=f"{_role_label(role)} no nondeterministic selected rows",
                )
        ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="#AEB4BA", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Round", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
        ax.set_title(tfbs_label_title(label_name), fontsize=REVIEW_AXIS_LABEL_FONTSIZE, pad=10)
        style_review_axis(ax, square=True)
    axes[0].set_ylabel("Count-stratified lift ratio", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    _legend_below(fig, axes)
    fig.suptitle(
        fill("Slot-label lift after controlling for target-family count", width=44),
        fontsize=REVIEW_TITLE_FONTSIZE,
        y=0.975,
    )
    fig.subplots_adjust(left=0.16, right=0.94, top=0.78, bottom=0.25, wspace=0.34)
    fig.savefig(path, dpi=160, facecolor="white")
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
    fig, ax = plt.subplots(figsize=REVIEW_SQUARE_FIGSIZE, constrained_layout=True)
    style_review_axis(ax, square=True)
    width = 0.38
    ax.bar([index - width / 2 for index in x], final_delta, width=width, color=colors, label="final delta")
    ax.bar([index + width / 2 for index in x], auc_delta, width=width, color="#8A8F98", label="normalized AUC delta")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [tfbs_label_title(value) for value in df["label_name"].astype(str).tolist()], rotation=20, ha="right"
    )
    ax.set_ylabel("DenseGen lift minus scrambled-control lift", fontsize=REVIEW_AXIS_LABEL_FONTSIZE)
    ax.set_title("Count-stratified slot-position diagnostic", fontsize=REVIEW_TITLE_FONTSIZE)
    ax.legend(frameon=False, fontsize=REVIEW_LEGEND_FONTSIZE)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _status_color(value: str) -> str:
    if value == "position_signal_after_count_restriction":
        return "#5D7D4F"
    if value == "not_separated_after_count_restriction":
        return "#B07D3C"
    return "#8C4E4A"


def _role_label(role: object) -> str:
    role_text = str(role)
    if role_text == "positive":
        return "DenseGen label"
    if role_text == "matched_null":
        return "scrambled control"
    return role_text.replace("_", " ")


def _row_square_axes(*, label_count: int, sharey: bool) -> tuple[object, list[object]]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        int(label_count),
        figsize=(max(7.2, _ROW_PANEL_WIDTH * int(label_count)), _ROW_PANEL_HEIGHT),
        sharey=sharey,
        squeeze=False,
    )
    return fig, list(axes[0])


def _legend_below(fig: object, axes: list[object]) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        fontsize=REVIEW_LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncols=min(3, max(1, len(by_label))),
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.5,
    )
