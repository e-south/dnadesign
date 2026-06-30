"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_masked_marginal/plots.py

Lightweight plots for ESMC masked-marginal DMS artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import rc_context
from matplotlib.patches import Patch

from dnadesign.permuter.src.scoring.esmc_masked_marginal.contracts import CANONICAL_AMINO_ACIDS

_TITLE_SIZE = 16
_LABEL_SIZE = 14
_TICK_SIZE = 12
_SMALL_TICK_SIZE = 10
_HEATMAP_TITLE_SIZE = 24
_HEATMAP_LABEL_SIZE = 18
_HEATMAP_TICK_SIZE = 14


@dataclass(frozen=True)
class MaskedMarginalPlotArtifacts:
    """Paths for rendered masked-marginal review plots."""

    entropy_by_position_path: Path
    fraction_negative_alternate_llr_path: Path
    substitution_llr_heatmap_path: Path


def render_masked_marginal_plots(
    *,
    position_entropy_path: Path,
    substitution_llr_path: Path,
    output_root: Path,
    file_prefix: str = "",
    position_context_spans: Sequence[Mapping[str, object]] | None = None,
) -> MaskedMarginalPlotArtifacts:
    """Render compact SVG plots from masked-marginal Parquet tables."""

    output_root.mkdir(parents=True, exist_ok=True)
    positions = pq.read_table(position_entropy_path).to_pandas()
    substitutions = pq.read_table(substitution_llr_path).to_pandas()
    accepted_positions = positions.loc[positions["status"] == "accepted"].sort_values("canonical_position")
    accepted_substitutions = substitutions.loc[substitutions["status"] == "accepted"].copy()
    context_spans = _normalize_context_spans(position_context_spans or [])
    artifacts = MaskedMarginalPlotArtifacts(
        entropy_by_position_path=output_root / f"{file_prefix}entropy_by_position.svg",
        fraction_negative_alternate_llr_path=output_root
        / f"{file_prefix}fraction_negative_alternate_llr_by_position.svg",
        substitution_llr_heatmap_path=output_root / f"{file_prefix}substitution_llr_heatmap.svg",
    )
    _render_position_bars(
        accepted_positions,
        y="canonical_entropy_bits",
        ylabel="Canonical AA entropy (bits)",
        title="ESMC uncertainty varies across the reference sequence",
        path=artifacts.entropy_by_position_path,
        annotate_lowest_count=50,
        context_spans=context_spans,
    )
    _render_position_scatter(
        accepted_positions,
        y="fraction_negative_alternate_llr",
        ylabel="Fraction negative alternate LLR",
        title="Most reference positions reject many alternate residues",
        path=artifacts.fraction_negative_alternate_llr_path,
        context_spans=context_spans,
    )
    _render_llr_heatmap(accepted_substitutions, path=artifacts.substitution_llr_heatmap_path)
    return artifacts


def _render_position_bars(
    data: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    title: str,
    path: Path,
    context_spans: Sequence[dict[str, object]],
    annotate_lowest_count: int = 0,
) -> None:
    description = (
        f"Bar plot of {ylabel.lower()} by reference protein position. "
        f"The {annotate_lowest_count} lowest-entropy positions are annotated with the reference amino acid."
    )
    data = data.sort_values("canonical_position")
    fig_width = _position_plot_width(data)
    fig, ax = plt.subplots(figsize=(fig_width, 4.0))
    if data.empty:
        ax.text(0.5, 0.5, "No accepted positions", ha="center", va="center", transform=ax.transAxes)
    else:
        _draw_position_context_spans(ax, context_spans)
        ax.bar(
            data["canonical_position"],
            data[y],
            color="#56B4E9",
            width=0.86,
            linewidth=0,
            alpha=0.62,
            zorder=3,
        )
        _annotate_lowest_positions(ax, data, y=y, count=annotate_lowest_count)
        _add_reference_residue_axis(ax, data)
        _set_position_ticks(ax, data)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel("Ec86 position", fontsize=_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    _style_position_axis(ax)
    _add_context_legend(fig, context_spans)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bottom = 0.24 if context_spans else 0.16
    fig.subplots_adjust(left=0.045, right=0.965, bottom=bottom, top=0.86, hspace=0.03, wspace=0.025)
    _save_accessible_svg(fig, path, title=title, description=description)
    plt.close(fig)


def _render_position_scatter(
    data: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    title: str,
    path: Path,
    context_spans: Sequence[dict[str, object]],
) -> None:
    description = f"Scatter plot of {ylabel.lower()} by reference protein position."
    data = data.sort_values("canonical_position")
    fig_width = _position_plot_width(data)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    if data.empty:
        ax.text(0.5, 0.5, "No accepted positions", ha="center", va="center", transform=ax.transAxes)
    else:
        _draw_position_context_spans(ax, context_spans)
        ax.scatter(
            data["canonical_position"],
            data[y],
            color="#0072B2",
            edgecolors="#ffffff",
            linewidths=0.35,
            s=28,
            alpha=0.9,
            zorder=3,
        )
        _add_reference_residue_axis(ax, data)
        _set_position_ticks(ax, data)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel("Ec86 position", fontsize=_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    _style_position_axis(ax)
    _add_context_legend(fig, context_spans)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bottom = 0.24 if context_spans else 0.16
    fig.subplots_adjust(left=0.045, right=0.965, bottom=bottom, top=0.86, hspace=0.03, wspace=0.025)
    _save_accessible_svg(fig, path, title=title, description=description)
    plt.close(fig)


def _render_llr_heatmap(data: pd.DataFrame, *, path: Path) -> None:
    title = "ESMC masked-marginal scores form a WT substitution matrix"
    description = (
        "Heatmap of single amino-acid substitution log-likelihood ratios computed "
        "from masked-position ESMC sequence logits."
    )
    if data.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.text(0.5, 0.5, "No accepted substitutions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save_accessible_svg(fig, path, title=title, description=description)
        plt.close(fig)
        return
    data = data.sort_values(["canonical_position", "alt_aa"])
    matrix = data.pivot(index="alt_aa", columns="canonical_position", values="llr")
    matrix = matrix.reindex(list(CANONICAL_AMINO_ACIDS))
    wt_by_position = (
        data[["canonical_position", "wt_aa"]]
        .drop_duplicates(subset=["canonical_position"])
        .set_index("canonical_position")["wt_aa"]
        .to_dict()
    )
    for position, wt_aa in wt_by_position.items():
        if wt_aa in matrix.index and position in matrix.columns:
            matrix.loc[wt_aa, position] = 0.0
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = max(limit, 0.1)
    cell_size = 0.24
    heatmap_width = max(8.0, matrix.shape[1] * cell_size)
    heatmap_height = max(2.4, matrix.shape[0] * cell_size)
    fig, ax = plt.subplots(figsize=(heatmap_width + 1.35, heatmap_height + 1.65))
    image = ax.imshow(values, aspect="equal", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_title(title, fontsize=_HEATMAP_TITLE_SIZE, pad=14)
    ax.set_xlabel("Ec86 position", fontsize=_HEATMAP_LABEL_SIZE)
    ax.set_ylabel("Alternate residue", fontsize=_HEATMAP_LABEL_SIZE)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=_HEATMAP_TICK_SIZE)
    x_positions = list(matrix.columns)
    tick_indices = _position_tick_indices(len(x_positions), target_count=34)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(x_positions[index]) for index in tick_indices], fontsize=_HEATMAP_TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_heatmap_reference_residue_axis(ax, x_positions, wt_by_position)
    _draw_reference_diagonal(ax, matrix, wt_by_position)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.012, pad=0.012)
    colorbar.set_label("LLR vs WT", fontsize=_HEATMAP_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=_HEATMAP_TICK_SIZE)
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.15, top=0.84)
    _save_accessible_svg(fig, path, title=title, description=description)
    plt.close(fig)


def _position_plot_width(data: pd.DataFrame) -> float:
    count = int(len(data))
    return max(10.0, min(28.0, count / 12.0))


def _normalize_context_spans(spans: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        label = str(span.get("label") or span.get("legend_label") or "Context span")
        legend_label = str(span.get("legend_label") or label)
        annotate_label = str(span.get("annotate_label") or "")
        color = str(span.get("color") or "#8c959f")
        alpha = float(span.get("alpha") or 0.08)
        zorder = float(span.get("zorder") or 0.2)
        label_y = float(span.get("label_y") or 0.97)
        normalized.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "legend_label": legend_label,
                "annotate_label": annotate_label,
                "color": color,
                "alpha": max(0.0, min(alpha, 1.0)),
                "zorder": zorder,
                "label_y": max(0.0, min(label_y, 1.05)),
            }
        )
    return normalized


def _draw_position_context_spans(ax: object, spans: Sequence[dict[str, object]]) -> None:
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=str(span["color"]),
            alpha=float(span["alpha"]),
            linewidth=0,
            zorder=float(span["zorder"]),
        )
        annotate_label = str(span.get("annotate_label") or "")
        if annotate_label:
            ax.text(
                (start + end) / 2.0,
                float(span.get("label_y") or 0.97),
                annotate_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=_SMALL_TICK_SIZE,
                color="#2f363d",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                clip_on=False,
                zorder=5,
            )


def _add_context_legend(fig: object, spans: Sequence[dict[str, object]]) -> None:
    handles = _context_handles(spans)
    if not handles:
        return
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=_SMALL_TICK_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(4, len(handles)),
    )


def _context_handles(spans: Sequence[dict[str, object]]) -> list[Patch]:
    seen: set[str] = set()
    handles: list[Patch] = []
    for span in spans:
        label = str(span["legend_label"])
        if label in seen:
            continue
        seen.add(label)
        handles.append(Patch(facecolor=str(span["color"]), alpha=float(span["alpha"]), label=label))
    return handles


def _style_position_axis(ax: object) -> None:
    ax.set_axisbelow(True)
    ax.grid(color="#d0d7de", alpha=0.35, linewidth=0.7, zorder=0)


def _set_position_ticks(ax: object, data: pd.DataFrame) -> None:
    positions = [int(position) for position in data["canonical_position"]]
    if not positions:
        return
    tick_indices = _position_tick_indices(len(positions), target_count=40)
    ticks = [positions[index] for index in tick_indices]
    ax.set_xticks(ticks)
    tick_font_size = 8.5 if len(positions) > 160 else _TICK_SIZE
    ax.set_xticklabels([str(tick) for tick in ticks], fontsize=tick_font_size)
    ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)


def _position_tick_indices(count: int, *, target_count: int = 30) -> list[int]:
    if count <= 0:
        return []
    step = max(1, int(np.ceil(count / target_count)))
    step = _nice_step(step)
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def _nice_step(step: int) -> int:
    if step <= 1:
        return 1
    magnitude = 10 ** int(np.floor(np.log10(step)))
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier * magnitude
        if candidate >= step:
            return int(candidate)
    return int(step)


def _add_reference_residue_axis(ax: object, data: pd.DataFrame) -> None:
    positions = [int(position) for position in data["canonical_position"]]
    wt_by_position = {
        int(row["canonical_position"]): str(row["wt_aa"])
        for row in data[["canonical_position", "wt_aa"]].to_dict("records")
    }
    if not positions:
        return
    ticks = positions
    labels = [wt_by_position[position] for position in ticks]
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels(labels, fontsize=8.5 if len(positions) > 160 else _SMALL_TICK_SIZE, family="monospace")
    top.tick_params(length=0, pad=2)


def _annotate_lowest_positions(ax: object, data: pd.DataFrame, *, y: str, count: int) -> None:
    if count <= 0 or data.empty:
        return
    selected = data.nsmallest(min(count, len(data)), y)
    y_max = float(data[y].max()) if np.isfinite(data[y]).any() else 1.0
    y_offset = max(0.04 * y_max, 0.05)
    for row in selected.to_dict("records"):
        ax.text(
            int(row["canonical_position"]),
            float(row[y]) + y_offset,
            str(row["wt_aa"]),
            ha="center",
            va="bottom",
            fontsize=_SMALL_TICK_SIZE,
            family="monospace",
            color="#333333",
            clip_on=False,
        )


def _add_heatmap_reference_residue_axis(ax: object, positions: list[int], wt_by_position: dict[int, str]) -> None:
    if not positions:
        return
    tick_indices = list(range(0, len(positions)))
    labels = [str(wt_by_position.get(positions[index], "")) for index in tick_indices]
    top = ax.secondary_xaxis("top")
    top.set_xticks(tick_indices)
    top.set_xticklabels(labels, fontsize=8.0 if len(positions) > 160 else 9.0, family="monospace")
    top.tick_params(length=0, pad=2)


def _draw_reference_diagonal(ax: object, matrix: pd.DataFrame, wt_by_position: dict[int, str]) -> None:
    row_by_aa = {aa: index for index, aa in enumerate(matrix.index)}
    columns = list(matrix.columns)
    for column_index, position in enumerate(columns):
        row_index = row_by_aa.get(str(wt_by_position.get(position, "")))
        if row_index is None:
            continue
        ax.plot(
            [column_index - 0.5, column_index + 0.5],
            [row_index - 0.5, row_index + 0.5],
            linewidth=0.45,
            color="black",
            alpha=0.25,
            zorder=4,
        )
        ax.plot(
            [column_index - 0.5, column_index + 0.5],
            [row_index + 0.5, row_index - 0.5],
            linewidth=0.45,
            color="black",
            alpha=0.25,
            zorder=4,
        )


def _save_accessible_svg(fig: object, path: Path, *, title: str, description: str) -> None:
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, format="svg", bbox_inches="tight")
    _inject_svg_accessibility(path, title=title, description=description)


def _inject_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    title_id = f"{path.stem}-title"
    desc_id = f"{path.stem}-desc"
    if "<svg " in text and 'role="img"' not in text:
        text = text.replace("<svg ", f'<svg role="img" aria-labelledby="{title_id} {desc_id}" ', 1)
    if "<title" not in text and "<svg" in text:
        insert_at = text.find(">", text.find("<svg")) + 1
        text = (
            text[:insert_at]
            + f'\n<title id="{escape(title_id)}">{escape(title)}</title>'
            + f'\n<desc id="{escape(desc_id)}">{escape(description)}</desc>'
            + text[insert_at:]
        )
    path.write_text(text, encoding="utf-8")
