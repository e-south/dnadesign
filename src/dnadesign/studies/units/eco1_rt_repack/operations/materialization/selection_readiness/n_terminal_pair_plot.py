"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/n_terminal_pair_plot.py

Focused N-terminal comparison for WT Eco1 and the selected distal pair.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.patches import Rectangle

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    TITLE_SIZE,
    save_accessible_svg,
)

from .plot_support import plot_row
from .sequence_export import CANONICAL_RT_LENGTH
from .visual_inventory import SELECTION_PLOT_PLAIN_TITLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_WT_FASTA_ID = "wild_type"
_DISTAL_VARIANT_IDS = ("Eco1RT-G3-D01", "Eco1RT-G3-D02")
_DISPLAY_END = 32
_ALPHA1_END = 14
_CHARGE_WINDOW_START = 4
_CHARGE_WINDOW_END = 16
_BASIC_RESIDUES = frozenset("KRH")
_ACIDIC_RESIDUES = frozenset("DE")
_FONT_FAMILY = "Arial"
_WHITE = "#FFFFFF"
_GRAPHITE = "#34495E"
_D01_TEAL = "#008C95"
_D02_ROSE = "#B96B72"
_D01_FILL = "#E6F3F4"
_D02_FILL = "#F6EAEC"
_ALPHA1_FILL = "#EAF3F5"
_GRID = "#CDD8DE"


def build_n_terminal_pair_comparison(
    *,
    panel_rows: Sequence[Mapping[str, object]],
    canonical_sequences_by_id: Mapping[str, str],
) -> list[dict[str, object]]:
    """Return source-derived WT, D01, and D02 sequence-comparison rows."""

    wt_sequence = _canonical_sequence(canonical_sequences_by_id, _WT_FASTA_ID)
    candidate_id_by_variant: dict[str, str] = {}
    for row in panel_rows:
        variant_id = str(row.get("variant_id") or "")
        if variant_id not in _DISTAL_VARIANT_IDS:
            continue
        if variant_id in candidate_id_by_variant:
            raise ValueError(f"Selection panel contains duplicate distal variant id: {variant_id}")
        candidate_id_by_variant[variant_id] = str(row.get("candidate_id") or "")

    missing = [variant_id for variant_id in _DISTAL_VARIANT_IDS if not candidate_id_by_variant.get(variant_id)]
    if missing:
        raise ValueError(f"Selection panel is missing distal variant ids: {', '.join(missing)}")

    comparison_rows = [_comparison_row(label="WT", candidate_id=_WT_FASTA_ID, sequence=wt_sequence, wt=wt_sequence)]
    for variant_id in _DISTAL_VARIANT_IDS:
        candidate_id = candidate_id_by_variant[variant_id]
        sequence = _canonical_sequence(canonical_sequences_by_id, candidate_id)
        comparison_rows.append(
            _comparison_row(
                label=variant_id.rsplit("-", maxsplit=1)[-1],
                candidate_id=candidate_id,
                sequence=sequence,
                wt=wt_sequence,
            )
        )
    return comparison_rows


def write_n_terminal_pair_comparison_plot(
    plot_root: Path,
    *,
    panel_rows: Sequence[Mapping[str, object]],
    canonical_sequences_by_id: Mapping[str, str],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the focused WT/D01/D02 N-terminal comparison SVG."""

    rows = build_n_terminal_pair_comparison(
        panel_rows=panel_rows,
        canonical_sequences_by_id=canonical_sequences_by_id,
    )
    title = SELECTION_PLOT_PLAIN_TITLES["selection_distal_pair_n_terminal_comparison"]
    fig = plt.figure(figsize=(15.2, 5.7), facecolor=_WHITE)
    sequence_ax = fig.add_axes([0.065, 0.35, 0.91, 0.49])
    table_ax = fig.add_axes([0.655, 0.045, 0.30, 0.22])

    _draw_sequence_rows(sequence_ax, rows)
    _draw_charge_table(table_ax, rows)
    fig.suptitle(
        title,
        fontsize=TITLE_SIZE + 1.5,
        fontweight="bold",
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        (
            f"Canonical residues 1–32  ·  D01: {rows[1]['total_substitution_count']} substitutions  ·  "
            f"D02: {rows[2]['total_substitution_count']} substitutions"
        ),
        ha="center",
        va="center",
        fontsize=11.0,
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
    )
    fig.text(
        0.065,
        0.215,
        "Outlined cells = substitutions from WT",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
    )
    outside_view = [
        str(value) for value in rows[1]["substitutions"] if _substitution_position(str(value)) > _DISPLAY_END
    ]
    if outside_view:
        fig.text(
            0.065,
            0.157,
            f"D01  ·  {', '.join(outside_view)} outside view",
            ha="left",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            fontfamily=_FONT_FAMILY,
            color=_D01_TEAL,
        )
    fig.text(
        0.065,
        0.095,
        "K/R/H = +1; D/E = −1  ·  descriptive, not causal",
        ha="left",
        va="center",
        fontsize=10.2,
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
    )

    path = plot_root / "selection_distal_pair_n_terminal_comparison.svg"
    charge_counts = ", ".join(_charge_count_description(row) for row in rows[:-1])
    final_row = rows[-1]
    alt = (
        "WT, D01, and D02 Eco1 RT residues 1 through 32, with the alpha-1 helix at residues 1 through 14 shaded, "
        f"F10 and R13 marked, and substitutions highlighted. In residues 4 through 16, {charge_counts}, and "
        f"{_charge_count_description(final_row)}."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_distal_pair_n_terminal_comparison",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Compares canonical WT, D01, and D02 sequence identities across residues 1-32 and reports a simple "
            "basic-minus-acidic residue count for positions 4-16."
        ),
        interpretation_limit=(
            "The charge proxy and sequence differences are descriptive, not causal. This comparison does not "
            "identify residues responsible for activity or show tolerance to an altered retron scaffold."
        ),
        render_mode="wide_visual",
        role="review_only",
        data_sources=[
            "foldcheck_request/input_sequences.fasta",
            "selection/candidate_selection_panel.parquet",
        ],
    )


def _comparison_row(*, label: str, candidate_id: str, sequence: str, wt: str) -> dict[str, object]:
    substitutions = tuple(
        f"{wt_residue}{position}{residue}"
        for position, (wt_residue, residue) in enumerate(zip(wt, sequence, strict=True), start=1)
        if wt_residue != residue
    )
    charge_window = sequence[_CHARGE_WINDOW_START - 1 : _CHARGE_WINDOW_END]
    basic_count = sum(residue in _BASIC_RESIDUES for residue in charge_window)
    acidic_count = sum(residue in _ACIDIC_RESIDUES for residue in charge_window)
    return {
        "label": label,
        "candidate_id": candidate_id,
        "sequence_window": sequence[:_DISPLAY_END],
        "substitutions": substitutions,
        "changed_positions": tuple(
            position for position in range(1, _DISPLAY_END + 1) if sequence[position - 1] != wt[position - 1]
        ),
        "total_substitution_count": len(substitutions),
        "basic_count_4_16": basic_count,
        "acidic_count_4_16": acidic_count,
        "net_charge_proxy_4_16": basic_count - acidic_count,
    }


def _canonical_sequence(canonical_sequences_by_id: Mapping[str, str], candidate_id: str) -> str:
    sequence = str(canonical_sequences_by_id.get(candidate_id) or "").strip().upper()
    if len(sequence) != CANONICAL_RT_LENGTH:
        raise ValueError(
            f"N-terminal comparison requires a {CANONICAL_RT_LENGTH}-aa canonical sequence for {candidate_id!r}"
        )
    return sequence


def _draw_sequence_rows(ax: Any, rows: list[dict[str, object]]) -> None:
    ax.set_xlim(-5.0, _DISPLAY_END + 0.7)
    ax.set_ylim(-0.65, 3.25)
    ax.set_facecolor(_WHITE)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.5, -0.52),
            _ALPHA1_END,
            3.4,
            facecolor=_ALPHA1_FILL,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(_ALPHA1_END + 0.5, ymin=0.07, ymax=0.89, color=_GRAPHITE, linewidth=0.9)
    ax.text(
        (_ALPHA1_END + 1) / 2,
        3.05,
        "α1 helix · residues 1–14",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
    )
    for position, label in ((10, "F10"), (13, "R13")):
        ax.text(
            position,
            2.68,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            fontfamily=_FONT_FAMILY,
            color=_GRAPHITE,
        )
        ax.plot([position, position], [2.52, 2.42], color=_GRAPHITE, linewidth=1.0, solid_capstyle="butt")

    row_y = {"WT": 2.05, "D01": 1.15, "D02": 0.25}
    row_colors = {"WT": _GRAPHITE, "D01": _D01_TEAL, "D02": _D02_ROSE}
    mutation_fills = {"D01": _D01_FILL, "D02": _D02_FILL}
    for row in rows:
        label = str(row["label"])
        y = row_y[label]
        changed = set(int(value) for value in row["changed_positions"])
        total = int(row["total_substitution_count"])
        row_color = row_colors[label]
        ax.text(
            -4.85,
            y,
            label,
            ha="left",
            va="center",
            fontsize=13.0,
            fontweight="bold",
            fontfamily=_FONT_FAMILY,
            color=row_color,
        )
        if label != "WT":
            ax.text(
                -3.35,
                y,
                f"{total} substitutions",
                ha="left",
                va="center",
                fontsize=9.5,
                fontfamily=_FONT_FAMILY,
                color=_GRAPHITE,
            )
        for position, residue in enumerate(str(row["sequence_window"]), start=1):
            if position in changed:
                ax.add_patch(
                    Rectangle(
                        (position - 0.42, y - 0.35),
                        0.84,
                        0.70,
                        facecolor=mutation_fills[label],
                        edgecolor=row_color,
                        linewidth=0.9,
                        zorder=1,
                    )
                )
            ax.text(
                position,
                y,
                residue,
                ha="center",
                va="center",
                fontsize=11.5,
                fontfamily=_FONT_FAMILY,
                fontweight="bold" if position in changed else "normal",
                color=row_color if position in changed else _GRAPHITE,
                zorder=2,
            )
    for position in range(1, _DISPLAY_END + 1):
        ax.text(
            position,
            -0.41,
            str(position),
            ha="center",
            va="center",
            fontsize=8.3,
            fontfamily=_FONT_FAMILY,
            color=_GRAPHITE,
        )


def _draw_charge_table(ax: Any, rows: list[dict[str, object]]) -> None:
    ax.set_facecolor(_WHITE)
    ax.axis("off")
    ax.set_title(
        "Charge proxy · residues 4–16",
        fontsize=11.5,
        fontweight="bold",
        fontfamily=_FONT_FAMILY,
        color=_GRAPHITE,
        pad=7,
    )
    table = ax.table(
        cellText=[
            [
                str(row["label"]),
                str(row["basic_count_4_16"]),
                str(row["acidic_count_4_16"]),
                _signed_count(int(row["net_charge_proxy_4_16"])),
            ]
            for row in rows
        ],
        colLabels=["", "Basic", "Acidic", "Net"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.42)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor(_GRAPHITE)
            cell.set_text_props(fontweight="bold", fontfamily=_FONT_FAMILY, color=_WHITE)
        else:
            row_label = str(rows[row_index - 1]["label"])
            row_color = {"WT": _GRAPHITE, "D01": _D01_TEAL, "D02": _D02_ROSE}[row_label]
            row_fill = {"WT": _WHITE, "D01": _D01_FILL, "D02": _D02_FILL}[row_label]
            cell.set_facecolor(row_fill)
            cell.set_text_props(fontfamily=_FONT_FAMILY, color=_GRAPHITE)
            if col_index == 0:
                cell.set_text_props(fontweight="bold", fontfamily=_FONT_FAMILY, color=row_color)
            elif col_index == 3:
                cell.set_text_props(fontweight="bold", fontfamily=_FONT_FAMILY, color=row_color)


def _substitution_position(substitution: str) -> int:
    return int(substitution[1:-1])


def _charge_count_description(row: Mapping[str, object]) -> str:
    basic_count = int(row["basic_count_4_16"])
    acidic_count = int(row["acidic_count_4_16"])
    return (
        f"{row['label']} has {_residue_count(basic_count, kind='basic')} and "
        f"{_residue_count(acidic_count, kind='acidic')}"
    )


def _residue_count(count: int, *, kind: str) -> str:
    noun = "residue" if count == 1 else "residues"
    return f"{count} {kind} {noun}"


def _signed_count(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


__all__ = ["build_n_terminal_pair_comparison", "write_n_terminal_pair_comparison_plot"]
