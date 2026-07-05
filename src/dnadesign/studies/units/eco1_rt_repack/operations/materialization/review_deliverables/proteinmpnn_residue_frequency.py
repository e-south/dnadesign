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
from .rendering import LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, save_accessible_svg, style_open_axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
TITLE = "ProteinMPNN samples alternate amino acids within each fixed-mask design class"
_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z*])(?P<position>[0-9]+)(?P<alt>[A-Z*])$")


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

    fig_width = max(14.8, min(22.0, len(positions) * 0.056))
    fig_height = max(9.2, len(class_rows) * 1.35 + 2.4)
    fig, axes = plt.subplots(
        len(class_rows),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=False,
    )
    if len(class_rows) == 1:
        axes = [axes]

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    position_index = {position: index for index, position in enumerate(positions)}
    residue_index = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    rendered_image = None
    for ax, class_row in zip(axes, class_rows, strict=True):
        fixed_positions = set(class_row["fixed_positions"])
        for position in fixed_positions:
            index = position_index.get(position)
            if index is not None:
                ax.axvspan(index - 0.5, index + 0.5, color="#f0f1f2", linewidth=0, zorder=0)
        matrix = _class_count_matrix(
            counts_by_class.get(class_row["design_class_id"], Counter()),
            positions=positions,
            residue_index=residue_index,
        )
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        rendered_image = ax.imshow(
            masked_matrix,
            aspect="equal",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=max(1, max_count),
            zorder=2,
        )
        _mark_wt_residues(ax, positions=positions, wt_by_position=wt_by_position, residue_index=residue_index)
        ax.set_yticks(range(len(AMINO_ACIDS)), AMINO_ACIDS, fontsize=LEGEND_SIZE - 1.0)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel("Amino acid", fontsize=LABEL_SIZE)
        ax.set_title(_class_title(class_row), fontsize=LABEL_SIZE, loc="left", pad=4)
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(len(AMINO_ACIDS) - 0.5, -0.5)
        style_open_axes(ax)
        ax.spines[["top", "right"]].set_visible(False)
    tick_indexes = _position_tick_indexes(positions)
    axes[-1].set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=LEGEND_SIZE)
    axes[-1].set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    if rendered_image is not None:
        colorbar_ax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
        colorbar = fig.colorbar(rendered_image, cax=colorbar_ax)
        colorbar.set_label("Changed residue count", fontsize=LEGEND_SIZE)
        colorbar.ax.tick_params(labelsize=LEGEND_SIZE - 1.0)
    wt_handle = Line2D(
        [0],
        [0],
        marker="x",
        color="#2f2f2f",
        linestyle="none",
        markersize=5.5,
        markeredgewidth=0.8,
        label="WT residue mark",
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
        label="Fixed by class mask",
    )
    fig.legend(
        handles=[wt_handle, fixed_handle],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
    )
    fig.suptitle(TITLE, fontsize=TITLE_SIZE, y=0.975)
    fig.subplots_adjust(left=0.085, right=0.905, bottom=0.09, top=0.915, hspace=0.34)

    panel_root.mkdir(parents=True, exist_ok=True)
    path = panel_root / "proteinmpnn_residue_frequency_heatmap.svg"
    alt = (
        f"Small-multiple amino-acid frequency heatmap for {len(candidate_rows)} accepted ProteinMPNN "
        f"candidate-pool rows across {len(class_rows)} fixed-mask design classes. Columns are Ec86 residue "
        "positions, rows are amino acids, color is the count of variants that changed to that amino acid, "
        "and x markers show the WT residue."
    )
    source_paths = _source_paths(
        candidate_pool_path=candidate_pool_path,
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    save_accessible_svg(fig, path, title=TITLE, description=alt, dpi=260)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_residue_frequency_heatmap",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
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
            "Shows which alternate amino acids ProteinMPNN sampled at each Ec86 residue position after each "
            "fixed-mask design class was applied. Fixed columns are shown as context, and WT residue marks "
            "orient the heatmap without listing every variant as a separate row."
        ),
        interpretation_limit=(
            "Residue frequencies describe ProteinMPNN sampling under the masks. They do not measure fold "
            "quality, expression, RT activity, processivity, or strand displacement."
        ),
        title=TITLE,
        role="review_only",
        render_mode="wide_visual",
        evidence_summary={
            "candidate_count": len(candidate_rows),
            "design_class_count": len(class_rows),
            "position_count": len(positions),
            "amino_acid_rows": len(AMINO_ACIDS),
        },
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
    ax.scatter(xs, ys, marker="x", s=4.2, linewidths=0.3, color="#2f2f2f", zorder=3)


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
    if len(positions) <= 40:
        return list(range(len(positions)))
    indexes = [0]
    indexes.extend(index for index, position in enumerate(positions) if position % 25 == 0)
    if indexes[-1] != len(positions) - 1:
        indexes.append(len(positions) - 1)
    return sorted(set(indexes))


def _source_paths(
    *, candidate_pool_path: Path, baseline_mask_set_path: Path, design_classes_root: Path
) -> dict[str, Path]:
    paths = {"candidate_pool": candidate_pool_path, "baseline_mask_set": baseline_mask_set_path}
    for spec in ALL_SPECS:
        if spec.design_class_id == BASELINE_CLASS_ID:
            continue
        paths[f"{spec.design_class_id}_mask_set"] = design_classes_root / spec.design_class_id / "mask_set.yaml"
    return paths
