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
    CONSERVATION_CLADE9_PROFILE_ID,
    CONSERVATION_SUBTYPE_PROFILE_ID,
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

_STATE_EMPTY = 0
_STATE_MOTIF = 1
_STATE_WANG = 2
_STATE_CONSERVATION = 3
_STATE_CONTACT = 4
_STATE_FIXED = 5
_STATE_DESIGNABLE = 6
_STATE_MISSING_BACKBONE = 7
_STATE_COLORS = (
    "#f8f7f2",
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    "#4d4d4d",
    OKABE_ITO["green"],
    "#b8b8b8",
)


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

    matrix_rows = _mask_matrix_rows(class_rows)
    matrix = [[_matrix_state(row, position) for position in positions] for row in matrix_rows]
    title = "Design-class residue mask evidence across Ec86 RT"
    fig_width = max(13.8, min(20.0, len(positions) * 0.06))
    fig_height = max(8.2, len(matrix_rows) * 0.35 + 3.3)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(_STATE_COLORS),
        vmin=0,
        vmax=len(_STATE_COLORS) - 1,
    )
    ax.set_yticks(
        range(len(matrix_rows)),
        [str(row["label"]) for row in matrix_rows],
        fontsize=TICK_SIZE,
    )
    tick_indexes = _position_tick_indexes(positions)
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=TICK_SIZE)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mask evidence and design-class policy", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
    ax.tick_params(axis="both", length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    legend_handles = [
        Patch(facecolor=OKABE_ITO["vermillion"], label="Catalytic motif anchor"),
        Patch(facecolor=OKABE_ITO["purple"], label="Wang/Ec86 prior"),
        Patch(facecolor=OKABE_ITO["blue"], label="Conservation threshold"),
        Patch(facecolor=OKABE_ITO["orange"], label="DNA/RNA contact threshold"),
        Patch(facecolor="#4d4d4d", label="Fixed by row mask policy"),
        Patch(facecolor=OKABE_ITO["green"], label="Designable by row mask policy"),
        Patch(facecolor="#b8b8b8", label="Missing backbone context"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE - 0.9,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.22, right=0.995, bottom=0.3, top=0.83)

    path = panel_root / "design_class_mask_overview.svg"
    source_paths = _source_paths_by_label(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    alt = (
        "Matrix comparing Eco1 RT mask evidence and the six ProteinMPNN design-class policies. Evidence rows "
        "separate motif anchors, Wang/Ec86 priors, WT-plurality thresholds, and retained DNA/RNA contact "
        "thresholds. Policy rows show which residues each design class fixed or left designable."
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
            "Provides one residue-coordinate source of truth for mask review. Evidence rows show motif, "
            "prior, conservation, and retained DNA/RNA proximity rules; policy rows show the resulting "
            "fixed and designable residues for the six design classes."
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
                "conservation_profile_id": spec.conservation_profile_id,
                "conservation_threshold": spec.conservation_threshold,
                "contact_threshold_angstrom": spec.contact_threshold_angstrom,
                "protected_count": protected_count,
                "designable_count": designable_count,
                "residues_by_position": {int(row["canonical_position"]): row for row in residues},
            }
        )
    return rows


def _mask_matrix_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        _binary_row("Catalytic motif anchors", _STATE_MOTIF, _positions_with_field(class_rows, "motif_protected")),
        _binary_row(
            "Wang/Ec86 substrate-contact priors",
            _STATE_WANG,
            _positions_with_field(class_rows, "wang_ec86_direct_contact_prior"),
        ),
    ]
    rows.extend(_conservation_rows(class_rows))
    rows.extend(_contact_rows(class_rows))
    rows.extend(
        {
            "kind": "policy",
            "label": _class_axis_label(row),
            "residues_by_position": row["residues_by_position"],
        }
        for row in class_rows
    )
    return rows


def _binary_row(label: str, state: int, positions: set[int]) -> dict[str, Any]:
    return {"kind": "binary", "label": label, "state": state, "positions": positions}


def _conservation_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_id, threshold in _ordered_conservation_keys(class_rows):
        matching_rows = [
            row
            for row in class_rows
            if row["conservation_profile_id"] == profile_id
            and int(round(float(row["conservation_threshold"]) * 100)) == threshold
        ]
        positions: set[int] = set()
        for row in matching_rows:
            positions.update(
                position
                for position, residue in row["residues_by_position"].items()
                if bool(residue.get("selected_conservation_rule_passed"))
            )
        rows.append(
            _binary_row(
                f"{_profile_label(profile_id)} p{threshold} conservation",
                _STATE_CONSERVATION,
                positions,
            )
        )
    return rows


def _ordered_conservation_keys(class_rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    observed = {
        (str(row["conservation_profile_id"]), int(round(float(row["conservation_threshold"]) * 100)))
        for row in class_rows
    }
    preferred = [
        (CONSERVATION_CLADE9_PROFILE_ID, 25),
        (CONSERVATION_CLADE9_PROFILE_ID, 50),
        (CONSERVATION_SUBTYPE_PROFILE_ID, 50),
    ]
    ordered = [key for key in preferred if key in observed]
    ordered.extend(sorted(observed - set(ordered), key=lambda key: (_profile_label(key[0]), key[1])))
    return ordered


def _contact_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = sorted({int(round(float(row["contact_threshold_angstrom"]))) for row in class_rows})
    for threshold in thresholds:
        positions: set[int] = set()
        for row in class_rows:
            if int(round(float(row["contact_threshold_angstrom"]))) != threshold:
                continue
            positions.update(
                position
                for position, residue in row["residues_by_position"].items()
                if bool(residue.get("selected_retained_dna_rna_contact"))
            )
        rows.append(_binary_row(f"DNA/RNA <={threshold} A contact", _STATE_CONTACT, positions))
    return rows


def _positions_with_field(class_rows: list[dict[str, Any]], field: str) -> set[int]:
    return {
        position
        for row in class_rows
        for position, residue in row["residues_by_position"].items()
        if bool(residue.get(field))
    }


def _profile_label(profile_id: str) -> str:
    if profile_id == CONSERVATION_CLADE9_PROFILE_ID:
        return "clade 9"
    if profile_id == CONSERVATION_SUBTYPE_PROFILE_ID:
        return "II-A3/42_1"
    return profile_id


def _class_axis_label(row: dict[str, Any]) -> str:
    designable = int(row["designable_count"])
    protected = int(row["protected_count"])
    return f"{row['label']} policy | {designable} designable, {protected} fixed"


def _matrix_state(row: dict[str, Any], position: int) -> int:
    if row["kind"] == "binary":
        return int(row["state"]) if position in row["positions"] else _STATE_EMPTY
    return _policy_state_value(row["residues_by_position"].get(position))


def _policy_state_value(residue: dict[str, Any] | None) -> int:
    if residue is None:
        return _STATE_EMPTY
    if bool(residue.get("non_fixed_missing_backbone")):
        return _STATE_MISSING_BACKBONE
    if bool(residue.get("non_fixed")):
        return _STATE_DESIGNABLE
    if bool(residue.get("protected")):
        return _STATE_FIXED
    return _STATE_EMPTY


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
