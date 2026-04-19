"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_b_summary.py

Stage-B summary plots that bridge retained Stage-A pools to deployed output use.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.artifacts.pool import TFBSPoolArtifact
from .plot_common import _add_anchored_box, _apply_style, _palette, _rename_artifact_path, _save_figure, _style
from .plot_run_helpers import compact_plan_label, compact_regulator_label, order_regulators_for_display
from .plot_stage_a_common import (
    _is_background_regulator,
    _pastelize_color,
    _stage_a_regulator_colors,
)
from .plot_stage_b_summary_data import (
    deployed_tfbs_frame as _deployed_tfbs_frame,
)
from .plot_stage_b_summary_data import (
    normalize_output_records as _normalize_output_records,
)
from .plot_stage_b_summary_data import (
    retained_pool_frame as _retained_pool_frame,
)
from .plot_stage_b_summary_data import (
    sampling_summary_frame as _sampling_summary_frame,
)
from .plot_stage_b_summary_data import (
    summary_output_dir as _summary_output_dir,
)


def _display_regulator_label(regulator: str) -> str:
    if _is_background_regulator(regulator):
        return "Background"
    return compact_regulator_label(regulator)


def _shared_regulator_order(
    retained_frame: pd.DataFrame | None = None,
    deployed_frame: pd.DataFrame | None = None,
    sampling_frame: pd.DataFrame | None = None,
) -> list[str]:
    regulators: list[str] = []
    counts_by_regulator: dict[str, int] = {}
    for frame in (retained_frame, deployed_frame, sampling_frame):
        if frame is None or frame.empty or "regulator" not in frame.columns:
            continue
        regulators.extend(frame["regulator"].astype(str).tolist())
    if deployed_frame is not None and not deployed_frame.empty:
        counts_by_regulator.update(
            deployed_frame["regulator"].astype(str).value_counts(dropna=True).astype(int).to_dict()
        )
    return order_regulators_for_display(regulators, counts_by_regulator=counts_by_regulator)


def _summary_left_margin(regulators: list[str]) -> float:
    max_chars = max((len(_display_regulator_label(regulator)) for regulator in regulators), default=12)
    return max(0.24, min(0.42, 0.16 + 0.0045 * float(max_chars)))


def _blend_rgba(color: object, *, toward: object = "#ffffff", amount: float = 0.5) -> tuple[float, float, float, float]:
    base = np.asarray(mpl.colors.to_rgba(color), dtype=float)
    target = np.asarray(mpl.colors.to_rgba(toward), dtype=float)
    mixed = base * (1.0 - float(amount)) + target * float(amount)
    mixed[3] = base[3]
    return tuple(float(value) for value in mixed)


def _regulator_length_shade(
    base_color: object,
    length_index: int,
    total_lengths: int,
) -> tuple[float, float, float, float]:
    if int(total_lengths) <= 1:
        return _blend_rgba(base_color, toward="#ffffff", amount=0.28)
    fraction = float(length_index) / float(max(1, int(total_lengths) - 1))
    lighten_amount = 0.6 - 0.55 * fraction
    return _blend_rgba(base_color, toward="#ffffff", amount=max(0.04, lighten_amount))


def _single_hue_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "densegen_seagreen",
        ["#ffffff", "#d7f0e5", "#7cc7aa", "#2e8b75", "#0b5d4f"],
    )


def _score_histogram_by_regulator(pool_manifest: TFBSPoolArtifact) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for entry in pool_manifest.inputs.values():
        sampling = entry.stage_a_sampling
        if not isinstance(sampling, dict):
            continue
        eligible_hist = sampling.get("eligible_score_hist") or []
        if not isinstance(eligible_hist, list):
            continue
        for row in eligible_hist:
            if not isinstance(row, dict):
                continue
            regulator = str(row.get("regulator") or "").strip()
            if not regulator or _is_background_regulator(regulator):
                continue
            rows[regulator] = dict(row)
    if not rows:
        raise ValueError("Stage-A pool manifest does not contain eligible score histograms for bridge plotting.")
    return rows


def _mean_pairwise_hamming(sequences: list[str]) -> float | None:
    unique = sorted({str(sequence or "").strip().upper() for sequence in sequences if str(sequence or "").strip()})
    if len(unique) < 2:
        return None
    lengths = {len(sequence) for sequence in unique}
    if len(lengths) != 1:
        return None
    distances: list[int] = []
    for left_idx, left in enumerate(unique[:-1]):
        for right in unique[left_idx + 1 :]:
            distances.append(sum(ch_left != ch_right for ch_left, ch_right in zip(left, right, strict=True)))
    if not distances:
        return None
    return float(np.mean(distances))


def _deployed_core_pairwise_hamming_by_regulator(
    *,
    retained: pd.DataFrame,
    deployed: pd.DataFrame,
) -> dict[str, float | None]:
    if retained.empty or deployed.empty:
        return {}
    retained_lookup = (
        retained.dropna(subset=["regulator", "sequence", "core_sequence"])
        .groupby(["regulator", "sequence"])["core_sequence"]
        .agg(lambda series: str(series.iloc[0]))
        .to_dict()
    )
    deployed = deployed.copy()
    deployed["core_sequence"] = [
        retained_lookup.get((str(regulator), str(sequence)))
        for regulator, sequence in deployed[["regulator", "sequence"]].itertuples(index=False, name=None)
    ]
    summary: dict[str, float | None] = {}
    for regulator, subset in deployed.groupby("regulator"):
        core_sequences = [
            str(value).strip().upper()
            for value in subset["core_sequence"].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        summary[str(regulator)] = _mean_pairwise_hamming(core_sequences)
    return summary


def _unique_deployed_length_summary(
    deployed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[tuple[str, int], int], list[int]]:
    if deployed.empty:
        return deployed.copy(), {}, {}, []
    unique_deployed = deployed.drop_duplicates(subset=["regulator", "sequence"]).copy()
    counts_by_regulator = {
        str(regulator): int(count)
        for regulator, count in (
            unique_deployed.groupby("regulator")["sequence"].nunique(dropna=True).astype(int).to_dict().items()
        )
    }
    counts_by_regulator_and_length = {
        (str(regulator), int(length)): int(count)
        for (regulator, length), count in (
            unique_deployed.groupby(["regulator", "length"])["sequence"]
            .nunique(dropna=True)
            .astype(int)
            .to_dict()
            .items()
        )
    }
    length_values = sorted(
        {int(value) for value in unique_deployed["length"].dropna().astype(int).tolist()},
        reverse=True,
    )
    return unique_deployed, counts_by_regulator, counts_by_regulator_and_length, length_values


def _stacked_share_bars(
    *,
    ax: plt.Axes,
    share_table: pd.DataFrame,
    regulators: list[str],
    categories: list[object],
    label_lookup: dict[object, str],
    colors: list[tuple[float, float, float, float]],
    source_labels: tuple[str, str],
    style: dict,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.15),
    legend_ncol: int | None = None,
    legend_title: str | None = None,
    source_label_font_scale: float = 0.82,
    source_label_x: float = -0.045,
    regulator_tick_pad: float = 14.0,
) -> None:
    y_positions = np.arange(len(regulators), dtype=float)
    offsets = {source_labels[0]: 0.18, source_labels[1]: -0.18}
    bar_height = 0.28
    font_size = float(style.get("label_size", style.get("font_size", 13)))
    source_font_size = font_size * float(source_label_font_scale)
    for idx, regulator in enumerate(regulators):
        for source_name in source_labels:
            subset = share_table[
                (share_table["regulator"].astype(str) == str(regulator))
                & (share_table["source"].astype(str) == str(source_name))
            ]
            left = 0.0
            for category_idx, category in enumerate(categories):
                value = 0.0
                if not subset.empty:
                    value = float(subset.loc[subset["category"].astype(object) == category, "share"].sum())
                if value <= 0:
                    continue
                ax.barh(
                    y_positions[idx] + offsets[source_name],
                    value,
                    left=left,
                    height=bar_height,
                    color=colors[category_idx],
                    edgecolor="white",
                    linewidth=0.7,
                )
                left += value
            ax.text(
                source_label_x,
                y_positions[idx] + offsets[source_name],
                source_name,
                ha="right",
                va="center",
                fontsize=source_font_size,
                color="#475467",
                clip_on=False,
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_display_regulator_label(regulator) for regulator in regulators])
    ax.tick_params(axis="y", pad=regulator_tick_pad)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share within regulator")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticklabels([f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    legend_handles = [
        mpl.patches.Patch(facecolor=colors[idx], edgecolor="white", label=label_lookup[category])
        for idx, category in enumerate(categories)
    ]
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=int(legend_ncol or min(8, max(1, len(legend_handles)))),
        frameon=False,
        fontsize=font_size,
        title=legend_title,
        title_fontsize=font_size,
    )


def plot_accepted_arrays_by_plan(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> list[Path]:
    normalized = _normalize_output_records(df)
    counts = normalized["densegen__plan"].astype(str).value_counts(dropna=True).sort_values(ascending=False)
    if counts.empty:
        raise ValueError("accepted_arrays_by_plan requires DenseGen output records with densegen__plan.")
    style = _style(style)
    fig_height = max(3.2, 0.65 * float(len(counts)) + 1.6)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), constrained_layout=False)
    palette = _palette(style, max(len(counts), 4), no_repeat=False)
    ax.barh(np.arange(len(counts)), counts.to_numpy(dtype=float), color=palette[: len(counts)], edgecolor="white")
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels([compact_plan_label(plan_name) for plan_name in counts.index.astype(str)])
    ax.invert_yaxis()
    ax.set_xlabel("Accepted arrays")
    ax.set_title("Accepted arrays by plan", pad=10)
    for idx, count in enumerate(counts.to_list()):
        ax.text(float(count) + max(1.0, float(max(counts)) * 0.01), idx, f"{int(count):,}", va="center", fontsize=9.5)
    _apply_style(ax, style)
    ax.grid(False)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"accepted_arrays_by_plan{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_plan_by_regulator_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> list[Path]:
    deployed = _deployed_tfbs_frame(df)
    matrix = (
        deployed.groupby(["plan_name", "regulator"]).size().unstack(fill_value=0).sort_index(axis=0).sort_index(axis=1)
    )
    if matrix.empty:
        raise ValueError("plan_by_regulator_heatmap requires deployed TFBS annotations.")
    regulators = _shared_regulator_order(deployed_frame=deployed)
    matrix = matrix.reindex(columns=[reg for reg in regulators if reg in matrix.columns], fill_value=0)
    plans = list(matrix.index.astype(str))
    style = _style(style)
    plot_font_size = max(18.0, float(style.get("font_size", 13)) * 1.28)
    local_style = dict(style)
    local_style["tick_size"] = plot_font_size
    local_style["label_size"] = plot_font_size
    local_style["title_size"] = plot_font_size
    fig_width = max(4.6, 0.52 * float(len(plans)) + 1.95)
    fig_height = max(2.4, 0.22 * float(len(regulators)) + 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    matrix_values = matrix.to_numpy(dtype=float).T
    image = ax.imshow(
        matrix_values,
        aspect="auto",
        cmap=_single_hue_cmap(),
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(matrix_values))),
    )
    ax.set_xticks(np.arange(len(plans)))
    ax.set_xticklabels([compact_plan_label(plan_name) for plan_name in plans], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(regulators)))
    ax.set_yticklabels([_display_regulator_label(regulator) for regulator in regulators])
    ax.set_title("Each DenseGen plan combines a different regulator mix", pad=9)
    if int(matrix.size) <= 36:
        max_value = max(1, int(matrix.to_numpy(dtype=int).max()))
        for row_idx, regulator in enumerate(regulators):
            for col_idx, plan_name in enumerate(plans):
                value = int(matrix.loc[plan_name, regulator])
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= max_value * 0.55 else "#0b1f17",
                    fontsize=plot_font_size,
                )
    label_size = float(local_style.get("label_size", local_style.get("font_size", 13)))
    tick_size = float(local_style.get("tick_size", label_size))
    cbar = fig.colorbar(image, ax=ax, fraction=0.11, pad=0.03)
    cbar.set_label("Deployed TFBS count", size=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    _apply_style(ax, local_style)
    ax.grid(False)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"plan_by_regulator_heatmap{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_retained_vs_deployed_length_shift(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del pool_manifest
    retained = _retained_pool_frame(pools)
    deployed = _deployed_tfbs_frame(df)
    regulators = _shared_regulator_order(retained_frame=retained, deployed_frame=deployed)
    lengths = sorted({int(value) for value in retained["length"].tolist() + deployed["length"].tolist()})
    share_rows: list[dict[str, object]] = []
    for source_name, frame in (("Retained", retained), ("Deployed", deployed)):
        grouped = frame.groupby("regulator")
        for regulator, subset in grouped:
            counts = subset["length"].value_counts(dropna=True)
            total = float(max(1, int(counts.sum())))
            for length in lengths:
                share_rows.append(
                    {
                        "regulator": str(regulator),
                        "source": source_name,
                        "category": int(length),
                        "share": float(counts.get(length, 0)) / total,
                    }
                )
    share_table = pd.DataFrame(share_rows)
    style = _style(style)
    plot_font_size = max(18.0, float(style.get("font_size", 13)) * 1.28)
    local_style = dict(style)
    local_style["tick_size"] = plot_font_size
    local_style["label_size"] = plot_font_size
    local_style["title_size"] = plot_font_size
    fig_side = max(7.8, 0.72 * float(len(regulators)) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side), constrained_layout=False)
    cmap = plt.cm.Blues(np.linspace(0.38, 0.9, len(lengths)))
    _stacked_share_bars(
        ax=ax,
        share_table=share_table,
        regulators=regulators,
        categories=lengths,
        label_lookup={length: f"{int(length)} bp" for length in lengths},
        colors=[tuple(color) for color in cmap],
        source_labels=("Retained", "Deployed"),
        style=local_style,
        legend_loc="center left",
        legend_bbox_to_anchor=(1.02, 0.5),
        legend_ncol=1,
        legend_title="Length",
        source_label_font_scale=0.78,
        source_label_x=-0.055,
        regulator_tick_pad=18.0,
    )
    ax.set_title("DenseGen arrays preferentially include shorter TFBS", pad=10)
    ax.set_box_aspect(1.0)
    ax.xaxis.labelpad = 14.0
    _apply_style(ax, local_style)
    ax.grid(False)
    fig.subplots_adjust(left=_summary_left_margin(regulators) + 0.05, right=0.8, bottom=0.14, top=0.9)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"retained_vs_deployed_length_shift{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_used_unique_vs_retained(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del pool_manifest
    retained = _retained_pool_frame(pools)
    deployed = _deployed_tfbs_frame(df)
    retained_counts = retained.groupby("regulator")["sequence"].nunique()
    deployed_counts = deployed.groupby("regulator")["sequence"].nunique()
    regulators = _shared_regulator_order(retained_frame=retained, deployed_frame=deployed)
    style = _style(style)
    plot_font_size = max(18.0, float(style.get("font_size", 13)) * 1.28)
    local_style = dict(style)
    local_style["tick_size"] = plot_font_size
    local_style["label_size"] = plot_font_size
    local_style["title_size"] = plot_font_size
    fig_side = max(6.8, 0.82 * float(len(regulators)) + 4.8)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side), constrained_layout=False)
    y_positions = np.arange(len(regulators), dtype=float)
    retained_values = [int(retained_counts.get(regulator, 0)) for regulator in regulators]
    deployed_values = [int(deployed_counts.get(regulator, 0)) for regulator in regulators]
    ax.barh(y_positions + 0.18, retained_values, height=0.3, color="#475467", edgecolor="#344054", label="Retained")
    ax.barh(
        y_positions - 0.18,
        deployed_values,
        height=0.3,
        color="#98a2b3",
        edgecolor="#667085",
        label="Unique deployed",
    )
    deployed_dx = max(1.0, max(deployed_values or [0]) * 0.02)
    for idx, value in enumerate(deployed_values):
        ax.text(
            float(value) + deployed_dx,
            y_positions[idx] - 0.18,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=plot_font_size,
            color="#475467",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_display_regulator_label(regulator) for regulator in regulators])
    ax.invert_yaxis()
    ax.set_xlabel("Unique TFBS sequences")
    ax.set_title("Only part of each retained regulator pool is deployed into accepted arrays", pad=10)
    ax.set_box_aspect(1.0)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=plot_font_size)
    _apply_style(ax, local_style)
    ax.grid(False)
    fig.subplots_adjust(left=_summary_left_margin(regulators), right=0.8, bottom=0.14, top=0.9)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"used_unique_vs_retained{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_retained_vs_deployed_tier_mix(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del pool_manifest
    retained = _retained_pool_frame(pools)
    if "tier" not in retained.columns:
        raise ValueError("retained_vs_deployed_tier_mix requires Stage-A pools with tier assignments.")
    retained = retained.dropna(subset=["tier"]).copy()
    if retained.empty:
        raise ValueError("retained_vs_deployed_tier_mix requires retained TFBS tier metadata.")
    deployed = _deployed_tfbs_frame(df)
    tier_lookup = retained.dropna(subset=["tier"]).groupby(["regulator", "sequence"])["tier"].min().to_dict()
    deployed["tier"] = [
        tier_lookup.get((str(regulator), str(sequence)), np.nan)
        for regulator, sequence in deployed[["regulator", "sequence"]].itertuples(index=False, name=None)
    ]
    matched_deployed = deployed.dropna(subset=["tier"]).copy()
    if matched_deployed.empty:
        raise ValueError("retained_vs_deployed_tier_mix could not map deployed TFBS back to retained Stage-A tiers.")
    matched_deployed["tier"] = matched_deployed["tier"].astype(int)
    retained["tier"] = retained["tier"].astype(int)
    regulators = _shared_regulator_order(retained_frame=retained, deployed_frame=matched_deployed)
    tiers = sorted({int(value) for value in retained["tier"].tolist() + matched_deployed["tier"].tolist()})
    share_rows: list[dict[str, object]] = []
    for source_name, frame in (("Retained", retained), ("Deployed", matched_deployed)):
        grouped = frame.groupby("regulator")
        for regulator, subset in grouped:
            counts = subset["tier"].value_counts(dropna=True)
            total = float(max(1, int(counts.sum())))
            for tier in tiers:
                share_rows.append(
                    {
                        "regulator": str(regulator),
                        "source": source_name,
                        "category": int(tier),
                        "share": float(counts.get(tier, 0)) / total,
                    }
                )
    share_table = pd.DataFrame(share_rows)
    style = _style(style)
    plot_font_size = max(18.0, float(style.get("font_size", 13)) * 1.28)
    local_style = dict(style)
    local_style["tick_size"] = plot_font_size
    local_style["label_size"] = plot_font_size
    local_style["title_size"] = plot_font_size
    fig_side = max(7.8, 0.72 * float(len(regulators)) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side), constrained_layout=False)
    cmap = plt.cm.Oranges(np.linspace(0.34, 0.9, len(tiers)))
    _stacked_share_bars(
        ax=ax,
        share_table=share_table,
        regulators=regulators,
        categories=tiers,
        label_lookup={tier: f"Tier {int(tier)}" for tier in tiers},
        colors=[tuple(color) for color in cmap],
        source_labels=("Retained", "Deployed"),
        style=local_style,
        legend_loc="center left",
        legend_bbox_to_anchor=(1.02, 0.5),
        legend_ncol=1,
        legend_title="Tier",
        source_label_font_scale=0.78,
        source_label_x=-0.055,
        regulator_tick_pad=18.0,
    )
    unmatched = int(len(deployed) - len(matched_deployed))
    if unmatched > 0:
        _add_anchored_box(
            ax,
            [f"{unmatched:,} deployed TFBS entries could not be mapped back to retained Stage-A tiers."],
            loc="lower right",
            fontsize=8.7,
            alpha=0.94,
        )
    ax.set_title("DenseGen arrays preferentially include a narrower TFBS subset", pad=10)
    ax.set_box_aspect(1.0)
    ax.xaxis.labelpad = 14.0
    _apply_style(ax, local_style)
    ax.grid(False)
    fig.subplots_adjust(left=_summary_left_margin(regulators) + 0.05, right=0.8, bottom=0.14, top=0.9)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"retained_vs_deployed_tier_mix{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_upstream_evidence_quality_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    del df, pools
    summary = _sampling_summary_frame(pool_manifest)
    regulators = _shared_regulator_order(sampling_frame=summary)
    summary = summary.set_index("regulator").reindex(regulators).reset_index()
    style = _style(style)
    fig_height = max(3.8, 0.68 * float(len(regulators)) + 2.1)
    fig, (ax_counts, ax_score) = plt.subplots(
        1,
        2,
        figsize=(11.8, fig_height),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    y_positions = np.arange(len(regulators), dtype=float)
    counts_palette = ["#98a2b3", "#FDB022", "#12B76A"]
    ax_counts.barh(
        y_positions + 0.22,
        summary["candidates_with_hit"].astype(float),
        height=0.2,
        color=counts_palette[0],
        label="Source hits",
    )
    ax_counts.barh(
        y_positions,
        summary["eligible_unique"].astype(float),
        height=0.2,
        color=counts_palette[1],
        label="Eligible unique",
    )
    ax_counts.barh(
        y_positions - 0.22, summary["retained"].astype(float), height=0.2, color=counts_palette[2], label="Retained"
    )
    ax_counts.set_yticks(y_positions)
    ax_counts.set_yticklabels([_display_regulator_label(regulator) for regulator in regulators])
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("Count")
    ax_counts.set_title("Upstream motif evidence counts by regulator", pad=10)
    ax_counts.legend(frameon=False, loc="lower right")
    max_count = float(
        max(
            1.0,
            summary["candidates_with_hit"].max(),
            summary["eligible_unique"].max(),
            summary["retained"].max(),
        )
    )
    count_label_dx = max(120.0, max_count * 0.012)
    for y_offset, column_name in (
        (0.22, "candidates_with_hit"),
        (0.0, "eligible_unique"),
        (-0.22, "retained"),
    ):
        for y_position, value in zip(y_positions, summary[column_name].astype(float).tolist()):
            if float(value) <= 0.0:
                continue
            ax_counts.text(
                float(value) + count_label_dx,
                float(y_position) + float(y_offset),
                f"{int(value):,}",
                va="center",
                ha="left",
                fontsize=8.8,
                color="#475467",
            )
    ax_counts.set_xlim(0.0, max_count + count_label_dx * 6.0)
    if (
        float(summary["retained"].max())
        < max(
            float(summary["candidates_with_hit"].max()),
            float(summary["eligible_unique"].max()),
        )
        * 0.02
    ):
        _add_anchored_box(
            ax_counts,
            ["Retained counts are present but visually compressed on this shared linear axis."],
            loc="lower right",
            fontsize=8.4,
            alpha=0.94,
        )

    ratios = pd.to_numeric(summary["consensus_ratio"], errors="coerce").fillna(0.0)
    ax_score.scatter(ratios.to_numpy(dtype=float), y_positions, color="#155EEF", s=44, zorder=3)
    for idx, ratio in enumerate(ratios.to_list()):
        ax_score.text(float(ratio) + 0.02, y_positions[idx], f"{float(ratio):.2f}", va="center", fontsize=9.0)
    ax_score.set_xlim(0.0, max(1.0, float(ratios.max()) + 0.14))
    ax_score.set_yticks(y_positions)
    ax_score.set_yticklabels([])
    ax_score.set_xlabel("Consensus score / theoretical max")
    ax_score.set_title("Source-PWM consensus strength proxy", pad=10)

    for ax in (ax_counts, ax_score):
        _apply_style(ax, style)
        ax.grid(False)
    ax_score.grid(axis="x", alpha=0.18)
    fig.suptitle(
        "Motif supply narrows sharply before motifs reach the retained Stage-A pool",
        y=0.98,
        fontsize=float(style.get("title_size", style.get("font_size", 13) * 1.1)),
    )
    fig.subplots_adjust(left=_summary_left_margin(regulators), right=0.98, bottom=0.14, top=0.88, wspace=0.28)
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"upstream_evidence_quality_summary{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def plot_score_strata_and_deployed_length_by_regulator(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if pool_manifest is None:
        raise ValueError("score_strata_and_deployed_length_by_regulator requires a Stage-A pool manifest.")
    retained = _retained_pool_frame(pools)
    deployed = _deployed_tfbs_frame(df)
    hist_by_regulator = _score_histogram_by_regulator(pool_manifest)
    retained_non_background = retained[~retained["regulator"].map(_is_background_regulator)].copy()
    if retained_non_background.empty:
        raise ValueError("score_strata_and_deployed_length_by_regulator requires non-background Stage-A pools.")
    regulators = [
        regulator
        for regulator in _shared_regulator_order(
            retained_frame=retained_non_background,
            deployed_frame=deployed[~deployed["regulator"].map(_is_background_regulator)].copy(),
            sampling_frame=pd.DataFrame({"regulator": list(hist_by_regulator)}),
        )
        if regulator in hist_by_regulator and not _is_background_regulator(regulator)
    ]
    if not regulators:
        raise ValueError("score_strata_and_deployed_length_by_regulator could not resolve regulator order.")

    non_background_deployed = deployed[~deployed["regulator"].map(_is_background_regulator)].copy()
    deployed_core_pairwise = _deployed_core_pairwise_hamming_by_regulator(
        retained=retained_non_background,
        deployed=non_background_deployed,
    )
    style = _style(style)
    plot_font_size = max(16.0, float(style.get("font_size", 13)) * 1.18)
    local_style = dict(style)
    local_style["tick_size"] = plot_font_size
    local_style["label_size"] = plot_font_size
    local_style["title_size"] = plot_font_size
    fig_side = max(6.6, min(7.4, 1.12 * float(len(regulators)) + 3.0))
    fig = plt.figure(figsize=(fig_side * 2.14, fig_side), constrained_layout=False)
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.34)
    ax_score = fig.add_subplot(outer_grid[0])
    ax_length = fig.add_subplot(outer_grid[1])
    base_reg_colors = _stage_a_regulator_colors(regulators, style)
    reg_colors = {regulator: _pastelize_color(color, amount=0.46) for regulator, color in base_reg_colors.items()}
    global_scores: list[float] = []
    retained_lookup = (
        retained_non_background.dropna(subset=["regulator", "sequence", "core_sequence"])
        .groupby(["regulator", "sequence"])["core_sequence"]
        .agg(lambda series: str(series.iloc[0]))
        .to_dict()
    )
    mapped_deployed = deployed.copy()
    mapped_deployed["core_sequence"] = [
        retained_lookup.get((str(regulator), str(sequence)))
        for regulator, sequence in mapped_deployed[["regulator", "sequence"]].itertuples(index=False, name=None)
    ]
    ridge_positions = np.arange(len(regulators) - 1, -1, -1, dtype=float)
    unique_deployed, deployed_unique_counts, deployed_counts_by_regulator_and_length, length_values = (
        _unique_deployed_length_summary(mapped_deployed)
    )
    for regulator_idx, regulator in enumerate(regulators):
        row = hist_by_regulator.get(regulator)
        if not isinstance(row, dict):
            raise ValueError(
                f"score_strata_and_deployed_length_by_regulator is missing eligible score histograms for {regulator}."
            )
        edges = [float(value) for value in row.get("edges") or []]
        counts = [int(value) for value in row.get("counts") or []]
        if len(edges) < 2 or len(counts) != len(edges) - 1:
            raise ValueError(
                f"score_strata_and_deployed_length_by_regulator found invalid score histograms for {regulator}."
            )
        centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
        global_scores.extend(edges)
        eligible_density = np.asarray(counts, dtype=float)
        eligible_density = eligible_density / max(1.0, float(eligible_density.max()))

        reg_retained = retained_non_background[
            retained_non_background["regulator"].astype(str) == str(regulator)
        ].copy()
        retained_scores = pd.to_numeric(reg_retained["best_hit_score"], errors="coerce").dropna().to_numpy(dtype=float)
        if retained_scores.size <= 0:
            raise ValueError(
                f"score_strata_and_deployed_length_by_regulator requires retained best-hit scores for {regulator}."
            )
        global_scores.extend(retained_scores.tolist())
        color = reg_colors.get(regulator, "#9dcfc3")
        line_color = base_reg_colors.get(regulator, "#2e8b75")
        ridge_base = ridge_positions[regulator_idx]
        ax_score.fill_between(
            centers,
            ridge_base,
            ridge_base + eligible_density * 0.86,
            color=color,
            alpha=0.36,
            linewidth=0.0,
            zorder=1,
        )
        ax_score.plot(centers, ridge_base + eligible_density * 0.86, color=line_color, linewidth=1.8, zorder=2)
        ax_score.hlines(
            ridge_base,
            float(min(edges)),
            float(max(edges)),
            color="#d9dee7",
            linewidth=0.85,
            zorder=0,
        )

        min_retained = float(np.min(retained_scores))
        lollipop_top = ridge_base + 0.98
        ax_score.vlines(min_retained, ridge_base, lollipop_top, color="#111827", linewidth=1.7, zorder=3)
        ax_score.scatter([min_retained], [lollipop_top], s=22.0, color="#111827", edgecolors="none", zorder=4)

        core_hamming = deployed_core_pairwise.get(str(regulator))
        deployed_count = int(deployed_unique_counts.get(str(regulator), 0))
        annotation_lines = [f"Used: {deployed_count:,}", "AVG. pairwise"]
        if core_hamming is None:
            annotation_lines.append("hamming n/a")
        else:
            annotation_lines.append(f"hamming {core_hamming:.1f}")
        annotation = "\n".join(annotation_lines)
        ax_score.text(
            min_retained + max(0.08, (max(edges) - min(edges)) * 0.015),
            ridge_base + (lollipop_top - ridge_base) / 2.0,
            annotation,
            ha="left",
            va="center",
            fontsize=plot_font_size * 0.8,
            color="#344054",
            linespacing=1.08,
        )

    score_min = float(min(global_scores))
    score_max = float(max(global_scores))
    score_pad = max(0.25, (score_max - score_min) * 0.04) if score_max > score_min else 0.25
    ax_score.set_xlim(score_min - score_pad, score_max + score_pad)
    ax_score.set_ylim(-0.15, ridge_positions[0] + 1.14)
    ax_score.set_yticks(ridge_positions)
    ax_score.set_yticklabels([_display_regulator_label(regulator) for regulator in regulators])
    for tick, regulator in zip(ax_score.get_yticklabels(), regulators, strict=True):
        tick.set_color(base_reg_colors.get(regulator, "#111111"))
    ax_score.tick_params(axis="y", labelsize=plot_font_size * 0.96, pad=8.0)
    ax_score.set_xlabel("FIMO log-odds score")
    ax_score.set_title("Mine high-scoring PWM matches", pad=10)
    ax_score.set_ylabel("Normalized density", labelpad=36)
    ax_score.yaxis.set_label_coords(-0.34, 0.5)
    ax_score.set_box_aspect(1.0)
    ax_score.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.24)
    ax_score.grid(axis="y", visible=False)
    _apply_style(ax_score, local_style)
    for spine_name in ("top", "right"):
        if spine_name in ax_score.spines:
            ax_score.spines[spine_name].set_visible(False)

    right_regulators = list(regulators)
    deployed_regulators = unique_deployed["regulator"].astype(str).tolist()
    if any(_is_background_regulator(regulator) for regulator in deployed_regulators):
        right_regulators.append("background")
    if not length_values:
        raise ValueError("score_strata_and_deployed_length_by_regulator requires deployed TFBS lengths.")
    right_regulator_colors: dict[str, object] = {
        regulator: reg_colors.get(regulator, "#cbd5e1") for regulator in right_regulators
    }
    if "background" in right_regulator_colors:
        right_regulator_colors["background"] = "#d0d5dd"
    y_base = np.arange(len(right_regulators), dtype=float)
    bar_height = min(0.22, 0.76 / max(1, len(length_values)))
    offsets = (np.arange(len(length_values), dtype=float) - (float(len(length_values) - 1) / 2.0)) * (bar_height * 1.18)
    for length_idx, length in enumerate(length_values):
        values = [
            int(deployed_counts_by_regulator_and_length.get((str(regulator), int(length)), 0))
            for regulator in right_regulators
        ]
        ax_length.barh(
            y_base + offsets[length_idx],
            values,
            height=bar_height,
            color=[
                _regulator_length_shade(
                    right_regulator_colors.get(regulator, "#cbd5e1"),
                    length_idx,
                    len(length_values),
                )
                for regulator in right_regulators
            ],
            edgecolor="white",
            linewidth=0.7,
            label=f"{int(length)} bp",
        )
    ax_length.set_yticks(y_base)
    ax_length.set_yticklabels([_display_regulator_label(regulator) for regulator in right_regulators])
    ax_length.invert_yaxis()
    ax_length.set_xlabel("Unique deployed TFBS count")
    ax_length.set_title("TFBS length counts in DenseGen arrays", pad=10)
    ax_length.set_box_aspect(1.0)
    ax_length.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        fontsize=plot_font_size,
        title="Length",
        title_fontsize=plot_font_size * 0.92,
    )
    _apply_style(ax_length, local_style)
    ax_length.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.24)
    ax_length.grid(axis="y", visible=False)
    fig.subplots_adjust(
        left=max(_summary_left_margin(regulators) + 0.1, 0.32),
        right=0.84,
        bottom=0.16,
        top=0.9,
    )
    target_dir = _summary_output_dir(out_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"score_strata_and_deployed_length_by_regulator{out_path.suffix}"
    _save_figure(fig, path, style=style)
    plt.close(fig)
    return [path]


def _rename_summary_outputs(paths: list[Path], *, stem: str) -> list[Path]:
    renamed: list[Path] = []
    for path in paths:
        target = path.with_name(f"{stem}{path.suffix}")
        renamed.append(_rename_artifact_path(path, target))
    return renamed


def plot_plan_regulator_deployment_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_plan_by_regulator_heatmap(df, out_path, style=style),
        stem="plan_regulator_deployment_heatmap",
    )


def plot_retained_vs_deployed_length_mix_by_regulator(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_retained_vs_deployed_length_shift(
            df,
            out_path,
            pools=pools,
            pool_manifest=pool_manifest,
            style=style,
        ),
        stem="retained_vs_deployed_length_mix_by_regulator",
    )


def plot_retained_pool_coverage_by_regulator(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_used_unique_vs_retained(
            df,
            out_path,
            pools=pools,
            pool_manifest=pool_manifest,
            style=style,
        ),
        stem="retained_pool_coverage_by_regulator",
    )


def plot_retained_vs_deployed_tier_mix_by_regulator(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_retained_vs_deployed_tier_mix(
            df,
            out_path,
            pools=pools,
            pool_manifest=pool_manifest,
            style=style,
        ),
        stem="retained_vs_deployed_tier_mix_by_regulator",
    )


def plot_upstream_motif_supply_and_pwm_strength(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_upstream_evidence_quality_summary(
            df,
            out_path,
            pools=pools,
            pool_manifest=pool_manifest,
            style=style,
        ),
        stem="upstream_motif_supply_and_pwm_strength",
    )


def plot_score_strata_and_deployed_length_bridge(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    return _rename_summary_outputs(
        plot_score_strata_and_deployed_length_by_regulator(
            df,
            out_path,
            pools=pools,
            pool_manifest=pool_manifest,
            style=style,
        ),
        stem="score_strata_and_deployed_length_bridge",
    )


__all__ = [
    "plot_accepted_arrays_by_plan",
    "plot_plan_regulator_deployment_heatmap",
    "plot_plan_by_regulator_heatmap",
    "plot_retained_pool_coverage_by_regulator",
    "plot_retained_vs_deployed_length_mix_by_regulator",
    "plot_retained_vs_deployed_tier_mix_by_regulator",
    "plot_retained_vs_deployed_length_shift",
    "plot_retained_vs_deployed_tier_mix",
    "plot_score_strata_and_deployed_length_bridge",
    "plot_score_strata_and_deployed_length_by_regulator",
    "plot_upstream_motif_supply_and_pwm_strength",
    "plot_upstream_evidence_quality_summary",
    "plot_used_unique_vs_retained",
]
