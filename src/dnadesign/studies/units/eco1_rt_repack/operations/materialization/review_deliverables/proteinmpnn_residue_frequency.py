"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_residue_frequency.py

ProteinMPNN residue-frequency heatmap for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import ALL_SPECS

from .constants import CONSERVATION_CLADE9_PROFILE_ID, CONSERVATION_SUBTYPE_PROFILE_ID, SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import file_hashes, make_deliverable_row
from .mask_rows import read_mask_residues
from .rendering import LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, save_accessible_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
TITLE = "ProteinMPNN samples amino acids within each fixed mask"
_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z*])(?P<position>[0-9]+)(?P<alt>[A-Z*])$")
_AXIS_FONT_SIZE = 11.0
_Y_TICK_FONT_SIZE = 9.5
_TOP_AMINO_ACID_FONT_SIZE = 3.0
_BOTTOM_POSITION_LABEL_STEP = 20
_POSITION_MINOR_TICK_STEP = 1


def write_residue_frequency_heatmap(
    *,
    panel_root: Path,
    candidate_pool_path: Path,
    baseline_mask_set_path: Path,
    design_classes_root: Path,
) -> dict[str, Any]:
    """Render a design-class small-multiple heatmap of sampled amino-acid choices."""

    class_rows = _load_design_class_rows(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    positions = sorted({position for row in class_rows for position in row["wt_by_position"]})
    if not positions:
        raise ValueError("ProteinMPNN residue-frequency heatmap requires mapped mask positions")
    wt_by_position = _validated_wt_by_position(class_rows)
    candidate_rows = _accepted_candidate_rows(candidate_pool_path)
    counts_by_class = _mutation_counts_by_class(candidate_rows)
    max_count = max(
        (count for class_counts in counts_by_class.values() for count in class_counts.values()),
        default=0,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    view_rows = []
    class_candidate_counts = Counter(str(row["design_class_id"]) for row in candidate_rows)
    for index, class_row in enumerate(class_rows):
        path = _class_view_path(panel_root=panel_root, class_row=class_row, index=index)
        class_id = str(class_row["design_class_id"])
        class_count = int(class_candidate_counts[class_id])
        _render_class_frequency_heatmap(
            path=path,
            class_row=class_row,
            positions=positions,
            wt_by_position=wt_by_position,
            counts=counts_by_class.get(class_id, Counter()),
            max_count=max_count,
            candidate_count=class_count,
        )
        view_rows.append(
            {
                "label": _class_title(class_row),
                "design_class_id": class_id,
                "path": _manifest_relative_path(path=path, panel_root=panel_root),
                "candidate_count": class_count,
                "fixed_position_count": len(class_row["fixed_positions"]),
            }
        )
    path = _class_view_path(panel_root=panel_root, class_row=class_rows[0], index=0)
    alt = (
        f"Selectable amino-acid frequency heatmap for {len(candidate_rows)} accepted ProteinMPNN candidate-pool "
        f"rows across {len(class_rows)} fixed-mask design classes. Each view shows one design class: columns are "
        "Ec86 residue positions, rows are amino acids, color is the count of variants that changed to that amino "
        "acid, x markers show the WT residue, and pale vertical bands mark fixed positions."
    )
    source_paths = _source_paths(
        candidate_pool_path=candidate_pool_path,
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    return make_deliverable_row(
        deliverable_id="proteinmpnn_residue_frequency_heatmap",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="proteinmpnn_residue_frequency_bundle",
        status="rendered",
        path=path,
        source_tables=[
            "design_classes/candidate_pool.parquet",
            "mask_set.yaml",
            *(
                f"design_classes/{spec.design_class_id}/mask_set.yaml"
                for spec in ALL_SPECS
                if spec.design_class_id != BASELINE_CLASS_ID
            ),
        ],
        input_hashes=file_hashes(source_paths),
        alt_text=alt,
        description=(
            "Shows one fixed-mask design class at a time so residue choices can be inspected without stacking "
            "six repeated heatmaps. The notebook selector changes the design class while keeping the color "
            "scale and residue axes constant."
        ),
        interpretation_limit=(
            "Residue frequencies describe ProteinMPNN sampling under the masks. They do not measure fold "
            "quality, expression, RT activity, processivity, or strand displacement."
        ),
        title=TITLE,
        role="manuscript_facing",
        render_mode="wide_visual",
        evidence_summary={
            "candidate_count": len(candidate_rows),
            "design_class_count": len(class_rows),
            "position_count": len(positions),
            "amino_acid_rows": len(AMINO_ACIDS),
            "design_class_views": view_rows,
        },
    )


def _render_class_frequency_heatmap(
    *,
    path: Path,
    class_row: dict[str, Any],
    positions: list[int],
    wt_by_position: dict[int, str],
    counts: Counter[tuple[int, str]],
    max_count: int,
    candidate_count: int,
) -> None:
    fig_width = max(18.0, min(26.0, len(positions) * 0.075))
    fig_height = 5.8
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    position_index = {position: index for index, position in enumerate(positions)}
    residue_index = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    ax.set_facecolor("#ffffff")
    for position in set(class_row["fixed_positions"]):
        index = position_index.get(position)
        if index is not None:
            ax.axvspan(index - 0.5, index + 0.5, color="#f0f1f2", alpha=0.95, linewidth=0, zorder=0)
    matrix = _class_count_matrix(counts, positions=positions, residue_index=residue_index)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    rendered_image = ax.imshow(
        masked_matrix,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=max(1, max_count),
        zorder=2,
    )
    _mark_wt_residues(ax, positions=positions, wt_by_position=wt_by_position, residue_index=residue_index)
    _configure_frequency_axes(ax, positions=positions, wt_by_position=wt_by_position)
    ax.set_title(TITLE, fontsize=TITLE_SIZE, loc="left", pad=32)
    ax.text(
        0.0,
        1.055,
        f"{_class_title(class_row)} | {candidate_count} accepted candidates",
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        ha="left",
        va="bottom",
    )
    colorbar = fig.colorbar(rendered_image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Variants with this amino acid", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=LEGEND_SIZE - 1.0)
    _add_frequency_legend(fig)
    fig.subplots_adjust(left=0.055, right=0.942, bottom=0.19, top=0.76)
    save_accessible_svg(
        fig,
        path,
        title=f"{TITLE} | {_class_title(class_row)}",
        description=(
            f"ProteinMPNN amino-acid frequency heatmap for {_class_title(class_row)}. Columns are Ec86 "
            "residue positions, top ticks are wild-type amino acids, bottom ticks are residue numbers, "
            "rows are sampled amino acids, color is candidate count, x marks show WT amino acids, and pale "
            "vertical bands show positions fixed by the mask."
        ),
        dpi=260,
    )


def _accepted_candidate_rows(candidate_pool_path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(
        candidate_pool_path,
        columns=["candidate_id", "design_class_id", "canonical_mutations", "status"],
    ).to_pylist()
    accepted = [row for row in rows if str(row.get("status") or "") == "accepted"]
    if not accepted:
        raise ValueError(f"No accepted candidate-pool rows found in {candidate_pool_path}")
    return accepted


def _mutation_counts_by_class(rows: list[dict[str, Any]]) -> dict[str, Counter[tuple[int, str]]]:
    counts: dict[str, Counter[tuple[int, str]]] = {}
    for row in rows:
        class_id = str(row.get("design_class_id") or "")
        if not class_id:
            raise ValueError(f"candidate {row.get('candidate_id')!r} is missing design_class_id")
        class_counts = counts.setdefault(class_id, Counter())
        for mutation in row.get("canonical_mutations") or []:
            match = _MUTATION_RE.match(str(mutation))
            if match is None:
                raise ValueError(f"Unrecognized canonical mutation for {row.get('candidate_id')}: {mutation!r}")
            alt = str(match.group("alt"))
            if alt in AMINO_ACIDS:
                class_counts[(int(match.group("position")), alt)] += 1
    return counts


def _class_count_matrix(
    counts: Counter[tuple[int, str]],
    *,
    positions: list[int],
    residue_index: dict[str, int],
) -> np.ndarray:
    matrix = np.zeros((len(AMINO_ACIDS), len(positions)), dtype=float)
    position_index = {position: index for index, position in enumerate(positions)}
    for (position, residue), count in counts.items():
        x_index = position_index.get(position)
        y_index = residue_index.get(residue)
        if x_index is None or y_index is None:
            continue
        matrix[y_index, x_index] = float(count)
    return matrix


def _mark_wt_residues(
    ax: Any,
    *,
    positions: list[int],
    wt_by_position: dict[int, str],
    residue_index: dict[str, int],
) -> None:
    xs: list[int] = []
    ys: list[int] = []
    for x_index, position in enumerate(positions):
        residue = wt_by_position[position]
        if residue not in residue_index:
            continue
        xs.append(x_index)
        ys.append(residue_index[residue])
    ax.scatter(xs, ys, marker="x", s=7.5, linewidths=0.45, color="#2f2f2f", zorder=3)


def _configure_frequency_axes(ax: Any, *, positions: list[int], wt_by_position: dict[int, str]) -> None:
    ax.set_yticks(range(len(AMINO_ACIDS)), AMINO_ACIDS, fontsize=_Y_TICK_FONT_SIZE)
    ax.set_ylabel("Sampled amino acid", fontsize=_AXIS_FONT_SIZE)
    tick_indexes = _position_tick_indexes(positions)
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=_AXIS_FONT_SIZE)
    ax.set_xticks(_minor_position_tick_indexes(positions), minor=True)
    ax.set_xlabel("Residue position", fontsize=_AXIS_FONT_SIZE, labelpad=7)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(len(AMINO_ACIDS) - 0.5, -0.5)
    ax.tick_params(axis="x", which="major", length=3.2, labelsize=_AXIS_FONT_SIZE, pad=3)
    ax.tick_params(axis="x", which="minor", length=1.3, color="#737373")
    ax.tick_params(axis="y", which="major", length=2.4, labelsize=_Y_TICK_FONT_SIZE, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737373")
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    _add_top_amino_acid_axis(ax, positions, wt_by_position=wt_by_position)


def _add_top_amino_acid_axis(ax: Any, positions: list[int], *, wt_by_position: dict[int, str]) -> None:
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_indexes = list(range(len(positions)))
    top_ax.set_xticks(
        top_indexes,
        [wt_by_position[positions[index]] for index in top_indexes],
        fontsize=_TOP_AMINO_ACID_FONT_SIZE,
    )
    for tick_label in top_ax.get_xticklabels():
        tick_label.set_fontfamily("DejaVu Sans Mono")
    top_ax.tick_params(axis="x", which="major", length=1.2, labelsize=_TOP_AMINO_ACID_FONT_SIZE, pad=1)
    top_ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    top_ax.spines[["right", "left", "bottom"]].set_visible(False)
    top_ax.spines["top"].set_color("#737373")
    top_ax.spines["top"].set_linewidth(0.55)


def _add_frequency_legend(fig: Any) -> None:
    wt_handle = Line2D(
        [0],
        [0],
        marker="x",
        color="#2f2f2f",
        linestyle="none",
        markersize=5.5,
        markeredgewidth=0.8,
        label="WT residue",
    )
    fixed_handle = Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        linestyle="none",
        markerfacecolor="#f0f1f2",
        markeredgecolor="#f0f1f2",
        markersize=7,
        label="Fixed position",
    )
    fig.legend(
        handles=[wt_handle, fixed_handle],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
    )


def _class_view_path(*, panel_root: Path, class_row: dict[str, Any], index: int) -> Path:
    if index == 0:
        return panel_root / "proteinmpnn_residue_frequency_heatmap.svg"
    slug = re.sub(r"[^a-z0-9]+", "_", str(class_row["label"]).lower()).strip("_")
    return panel_root / f"proteinmpnn_residue_frequency_heatmap_{slug}.svg"


def _manifest_relative_path(*, path: Path, panel_root: Path) -> str:
    try:
        return str(path.relative_to(panel_root.parent))
    except ValueError:
        return str(path)


def _load_design_class_rows(*, baseline_mask_set_path: Path, design_classes_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ALL_SPECS:
        mask_set_path = (
            baseline_mask_set_path
            if spec.design_class_id == BASELINE_CLASS_ID
            else design_classes_root / spec.design_class_id / "mask_set.yaml"
        )
        if not mask_set_path.exists():
            raise FileNotFoundError(mask_set_path)
        residues = read_mask_residues(mask_set_path)
        rows.append(
            {
                "design_class_id": spec.design_class_id,
                "label": _spec_label(
                    conservation_profile_id=spec.conservation_profile_id,
                    conservation_threshold=spec.conservation_threshold,
                    contact_threshold_angstrom=spec.contact_threshold_angstrom,
                ),
                "fixed_positions": {int(row["canonical_position"]) for row in residues if bool(row.get("protected"))},
                "wt_by_position": {
                    int(row["canonical_position"]): str(row.get("wt_aa") or "").strip().upper() for row in residues
                },
            }
        )
    return rows


def _validated_wt_by_position(class_rows: list[dict[str, Any]]) -> dict[int, str]:
    wt_by_position: dict[int, str] = {}
    for class_row in class_rows:
        for position, wt_aa in dict(class_row["wt_by_position"]).items():
            if len(wt_aa) != 1:
                raise ValueError(f"mask_set row {position} must include a one-letter WT amino acid")
            existing = wt_by_position.setdefault(int(position), str(wt_aa))
            if existing != wt_aa:
                raise ValueError(f"mask sets disagree on WT amino acid at position {position}")
    return wt_by_position


def _class_title(class_row: dict[str, Any]) -> str:
    return str(class_row["label"])


def _spec_label(
    *,
    conservation_profile_id: str,
    conservation_threshold: float,
    contact_threshold_angstrom: float,
) -> str:
    threshold = int(round(float(conservation_threshold) * 100))
    contact = int(round(float(contact_threshold_angstrom)))
    if conservation_profile_id == CONSERVATION_CLADE9_PROFILE_ID:
        denominator = "Clade 9"
    elif conservation_profile_id == CONSERVATION_SUBTYPE_PROFILE_ID:
        denominator = "II-A3/42_1"
    else:
        denominator = conservation_profile_id
    return f"{denominator} {threshold}% + {contact} A"


def _position_tick_indexes(positions: list[int]) -> list[int]:
    if len(positions) <= _BOTTOM_POSITION_LABEL_STEP * 2:
        return list(range(len(positions)))
    indexes = [0]
    indexes.extend(index for index, position in enumerate(positions) if position % _BOTTOM_POSITION_LABEL_STEP == 0)
    if indexes[-1] != len(positions) - 1:
        indexes.append(len(positions) - 1)
    return sorted(set(indexes))


def _minor_position_tick_indexes(positions: list[int]) -> list[int]:
    if not positions:
        return []
    return list(range(0, len(positions), _POSITION_MINOR_TICK_STEP))


def _source_paths(
    *, candidate_pool_path: Path, baseline_mask_set_path: Path, design_classes_root: Path
) -> dict[str, Path]:
    paths = {"candidate_pool": candidate_pool_path, "baseline_mask_set": baseline_mask_set_path}
    for spec in ALL_SPECS:
        if spec.design_class_id == BASELINE_CLASS_ID:
            continue
        paths[f"{spec.design_class_id}_mask_set"] = design_classes_root / spec.design_class_id / "mask_set.yaml"
    return paths
