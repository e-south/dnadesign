"""Shared helpers for frozen replay plot renderers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ....plot_style import REVIEW_LEGEND_FONTSIZE

POSITIVE_ROLE = "positive"
POOL_AVERAGE_COLOR = "#4D4D4D"
TARGET_METADATA_COLOR = "#0072B2"
SHUFFLED_CONTROL_COLOR = "#D55E00"
KNOWN_LABEL_COLOR = "#009E73"
LABEL_ORDER = {
    "lexA_count_fraction": 10,
    "cpxR_count_fraction": 20,
    "baeR_count_fraction": 30,
    "lexA_in_slot0": 110,
    "baeR_in_slot1": 120,
    "cpxR_or_baeR_in_slot2": 130,
}


@dataclass(frozen=True)
class FrozenReplaySeriesStyle:
    """Visual encoding for one cumulative learning-loop series."""

    color: str
    linestyle: str
    marker: str
    label: str
    linewidth: float = 2.0


def set_bar_ylim(ax: object, values: pd.Series, errors: pd.Series, *, reference: float) -> None:
    upper = max(float((values + errors).max()), float(reference), 0.0)
    lower = min(float((values - errors).min()), float(reference), 0.0)
    span = max(upper - lower, 1.0)
    ax.set_ylim(lower - 0.08 * span, upper + 0.20 * span)


def round_summary(frame: pd.DataFrame) -> pd.DataFrame:
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


def series_style(
    control_series_role: str,
    control_display_role: str,
) -> dict[tuple[str, str], FrozenReplaySeriesStyle]:
    control_label = control_role_label(control_display_role)
    return {
        ("active_retraining", POSITIVE_ROLE): FrozenReplaySeriesStyle(
            color=TARGET_METADATA_COLOR,
            linestyle="-",
            marker="o",
            label="Active target metadata",
            linewidth=2.3,
        ),
        ("frozen_round0", POSITIVE_ROLE): FrozenReplaySeriesStyle(
            color=TARGET_METADATA_COLOR,
            linestyle="--",
            marker="^",
            label="Frozen target metadata",
            linewidth=1.8,
        ),
        ("known_label_ranking", POSITIVE_ROLE): FrozenReplaySeriesStyle(
            color=KNOWN_LABEL_COLOR,
            linestyle=":",
            marker="D",
            label="Known-label best ranking",
            linewidth=1.8,
        ),
        ("active_retraining", control_series_role): FrozenReplaySeriesStyle(
            color=SHUFFLED_CONTROL_COLOR,
            linestyle="-",
            marker="s",
            label=f"Active {control_label}",
            linewidth=2.0,
        ),
        ("frozen_round0", control_series_role): FrozenReplaySeriesStyle(
            color=SHUFFLED_CONTROL_COLOR,
            linestyle="--",
            marker="v",
            label=f"Frozen {control_label}",
            linewidth=1.6,
        ),
    }


def control_roles(frame: pd.DataFrame) -> tuple[str, str]:
    roles = sorted(set(frame["oracle_role"].astype(str)) - {POSITIVE_ROLE})
    if len(roles) != 1:
        raise ValueError(f"Learning-loop cumulative plot expected one control role; found {roles}")
    display_roles: list[str] = []
    if "scientific_control_role" in frame.columns:
        controls = frame.loc[frame["oracle_role"].astype(str) == roles[0], "scientific_control_role"]
        display_roles = sorted({role for role in controls.dropna().astype(str) if role})
        if len(display_roles) > 1:
            raise ValueError(f"Learning-loop cumulative plot found multiple scientific control roles: {display_roles}")
    return roles[0], display_roles[0] if display_roles else roles[0]


def control_role_label(role: str) -> str:
    if role == "matched_null":
        return "row-shuffled control"
    if role == "count_fixed_shuffled_slot_negative_control":
        return "slot-shuffled control"
    return role.replace("_", " ")


def cumulative_premise_title(comparison_set_label: object) -> str:
    label = str(comparison_set_label or "").strip().lower()
    if "composition" in label:
        return "Active retraining adds count-fraction enrichment beyond the initial ranking"
    if "placement" in label:
        return "Active retraining improves placement enrichment, but not uniformly"
    return "Active retraining is compared with a frozen initial ranking"


def ordered_label_names(values: list[str]) -> list[str]:
    return sorted(values, key=lambda value: (LABEL_ORDER.get(str(value), 10_000), str(value)))


def sort_frame_by_label_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_label_order"] = [LABEL_ORDER.get(str(label_name), 10_000) for label_name in ordered["label_name"]]
    ordered["_label_name_sort"] = ordered["label_name"].astype(str)
    return ordered.sort_values(["_label_order", "_label_name_sort"]).drop(columns=["_label_order", "_label_name_sort"])


def legend_below_figure(fig: object, ax: object) -> None:
    handles, labels_out = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_out, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncols=min(3, max(1, len(by_label))),
        frameon=False,
        fontsize=REVIEW_LEGEND_FONTSIZE,
        columnspacing=0.85,
        handlelength=1.55,
        handletextpad=0.45,
    )


def validate_endpoint_source_used(endpoints: pd.DataFrame) -> None:
    if endpoints.empty:
        raise ValueError("Frozen replay endpoint summary source is empty")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
