"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/design_class_masks.py

Design-class mask overview deliverables for Eco1 review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STATE_MISSING = 0
_STATE_PROTECTED = 1
_STATE_DESIGNABLE = 2
_STATE_MISSING_BACKBONE = 3
_STATE_COLORS = ("#f8f7f2", "#4d4d4d", OKABE_ITO["green"], "#b8b8b8")


def write_design_class_mask_overview(
    *,
    panel_root: Path,
    baseline_mask_set_path: Path,
    design_classes_root: Path,
) -> dict[str, Any]:
    """Render a design-class-aware mask matrix over canonical positions."""

    class_rows = _load_design_class_rows(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    positions = sorted({position for row in class_rows for position in row["residues_by_position"]})
    if not positions:
        raise ValueError("No canonical positions found in design-class mask sets")

    matrix = [[_state_value(row["residues_by_position"].get(position)) for position in positions] for row in class_rows]
    title = "Design-class mask rules show which residues were fixed or designable"
    fig_width = max(12.8, min(19.0, len(positions) * 0.055))
    fig_height = 5.8
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(_STATE_COLORS),
        vmin=0,
        vmax=3,
    )
    ax.set_yticks(
        range(len(class_rows)),
        [_class_axis_label(row) for row in class_rows],
        fontsize=TICK_SIZE,
    )
    tick_indexes = _position_tick_indexes(positions)
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=TICK_SIZE)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("ProteinMPNN mask design class", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    ax.tick_params(axis="both", length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    legend_handles = [
        Patch(facecolor="#4d4d4d", label="Protected union"),
        Patch(facecolor=OKABE_ITO["green"], label="ProteinMPNN-designable residues"),
        Patch(facecolor="#b8b8b8", label="Missing backbone context"),
        Patch(facecolor="none", edgecolor=OKABE_ITO["blue"], label="25% WT plurality"),
        Patch(facecolor="none", edgecolor=OKABE_ITO["purple"], label="50% WT plurality"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.018),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.185, right=0.995, bottom=0.25, top=0.82)

    path = panel_root / "design_class_mask_overview.svg"
    source_paths = _source_paths_by_label(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    alt = (
        "Matrix comparing the six Eco1 RT ProteinMPNN design-class masks. Rows are conservation and "
        "nucleic-acid proximity policies; cells show protected residues, ProteinMPNN-designable residues, "
        "or missing-backbone context."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=300)
    return make_deliverable_row(
        deliverable_id="design_class_mask_overview",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=_source_table_labels(),
        input_hashes=file_hashes(source_paths),
        alt_text=alt,
        description=(
            "Compares the root 5 A clade-9 baseline mask with the five expanded design-class masks. "
            "Each row names the conservation denominator, WT-plurality threshold, and retained DNA/RNA "
            "distance threshold used to decide which residues were fixed or left designable."
        ),
        interpretation_limit=(
            "Mask membership explains the ProteinMPNN design surface. It does not rank sequences, "
            "measure fold quality, or imply strand-displacement activity."
        ),
        title=title,
        render_mode="compact_wide_visual",
        evidence_summary={
            "design_class_count": len(class_rows),
            "total_positions": len(positions),
            "design_classes": [
                {
                    "design_class_id": row["design_class_id"],
                    "protected_position_count": row["protected_count"],
                    "non_fixed_mapped_position_count": row["designable_count"],
                    "conservation_threshold": row["conservation_threshold"],
                    "contact_threshold_angstrom": row["contact_threshold_angstrom"],
                }
                for row in class_rows
            ],
        },
    )


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
        protected_count = sum(1 for row in residues if bool(row.get("protected")))
        designable_count = sum(1 for row in residues if bool(row.get("non_fixed")))
        rows.append(
            {
                "design_class_id": spec.design_class_id,
                "label": class_label(spec.design_class_id),
                "conservation_threshold": spec.conservation_threshold,
                "contact_threshold_angstrom": spec.contact_threshold_angstrom,
                "protected_count": protected_count,
                "designable_count": designable_count,
                "residues_by_position": {int(row["canonical_position"]): row for row in residues},
            }
        )
    return rows


def _class_axis_label(row: dict[str, Any]) -> str:
    threshold = int(round(float(row["conservation_threshold"]) * 100))
    designable = int(row["designable_count"])
    protected = int(row["protected_count"])
    return f"{row['label']} | p{threshold} WT | {designable} designable, {protected} fixed"


def _state_value(residue: dict[str, Any] | None) -> int:
    if residue is None:
        return _STATE_MISSING
    if bool(residue.get("non_fixed_missing_backbone")):
        return _STATE_MISSING_BACKBONE
    if bool(residue.get("non_fixed")):
        return _STATE_DESIGNABLE
    if bool(residue.get("protected")):
        return _STATE_PROTECTED
    return _STATE_MISSING


def _position_tick_indexes(positions: list[int]) -> list[int]:
    if len(positions) <= 80:
        return list(range(len(positions)))
    indexes = [0]
    indexes.extend(index for index, position in enumerate(positions) if position % 40 == 0)
    if indexes[-1] != len(positions) - 1:
        indexes.append(len(positions) - 1)
    return sorted(set(indexes))


def _source_paths_by_label(*, baseline_mask_set_path: Path, design_classes_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {"baseline_mask_set": baseline_mask_set_path}
    for spec in ALL_SPECS:
        if spec.design_class_id == BASELINE_CLASS_ID:
            continue
        paths[f"{spec.design_class_id}_mask_set"] = design_classes_root / spec.design_class_id / "mask_set.yaml"
    return paths


def _source_table_labels() -> list[str]:
    labels = ["mask_set.yaml"]
    labels.extend(
        f"design_classes/{spec.design_class_id}/mask_set.yaml"
        for spec in ALL_SPECS
        if spec.design_class_id != BASELINE_CLASS_ID
    )
    return labels
