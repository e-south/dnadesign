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
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
)

from .design_class_mask_annotations import add_rt_annotation_context
from .rt_annotation_context import RTAnnotationContext

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STATE_EMPTY = 0
_STATE_MOTIF = 1
_STATE_WANG = 2
_STATE_CONSERVATION = 3
_STATE_CONTACT = 4
_STATE_FIXED = 5
_MASK_AXIS_FONT_SIZE = 11.0
_TOP_AMINO_ACID_FONT_SIZE = 3.2
_POSITION_MINOR_TICK_STEP = 1
_BOTTOM_POSITION_LABEL_STEP = 20
_TOP_AMINO_ACID_LABEL_STEP = 1
_MASK_EMPTY_STATE_ALPHA = 0.0
_MASK_MATRIX_ZORDER = 2.0
_STATE_COLORS = (
    (248 / 255.0, 247 / 255.0, 242 / 255.0, _MASK_EMPTY_STATE_ALPHA),
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    "#222222",
)


def write_design_class_mask_overview(
    *,
    panel_root: Path,
    baseline_mask_set_path: Path,
    design_classes_root: Path,
    rt_annotation_context: RTAnnotationContext,
) -> dict[str, Any]:
    """Render a design-class-aware mask matrix over canonical positions."""

    class_rows = _load_design_class_rows(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    positions = sorted({position for row in class_rows for position in row["residues_by_position"]})
    if not positions:
        raise ValueError("No canonical positions found in design-class mask sets")
    wt_aa_by_position = _wt_aa_by_position(class_rows)

    matrix_rows = _mask_matrix_rows(class_rows)
    matrix = [[_matrix_state(row, position) for position in positions] for row in matrix_rows]
    title = "Design-class residue mask evidence across EC86 RT"
    fig_width = max(13.8, min(20.0, len(positions) * 0.06))
    fig_height = max(7.6, len(matrix_rows) * 0.35 + 2.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_facecolor("#f8f7f2")
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(_STATE_COLORS),
        vmin=0,
        vmax=len(_STATE_COLORS) - 1,
        zorder=_MASK_MATRIX_ZORDER,
    )
    add_rt_annotation_context(ax, positions, row_count=len(matrix_rows), context=rt_annotation_context)
    ax.set_yticks(
        range(len(matrix_rows)),
        [str(row["label"]) for row in matrix_rows],
        fontsize=_MASK_AXIS_FONT_SIZE,
    )
    tick_indexes = _position_tick_indexes(positions)
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=_MASK_AXIS_FONT_SIZE)
    ax.set_xticks(_minor_position_tick_indexes(positions), minor=True)
    ax.set_xlabel("Residue position", fontsize=_MASK_AXIS_FONT_SIZE, labelpad=7)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=58)
    ax.tick_params(axis="x", which="major", length=3.2, labelsize=_MASK_AXIS_FONT_SIZE, pad=3)
    ax.tick_params(axis="x", which="minor", length=1.3, color="#737373")
    ax.tick_params(axis="y", which="both", length=0, labelsize=_MASK_AXIS_FONT_SIZE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#737373")
    ax.spines["bottom"].set_linewidth(0.55)
    _add_row_group_separator(ax, matrix_rows)
    _add_top_amino_acid_axis(ax, positions, wt_aa_by_position=wt_aa_by_position)
    fig.subplots_adjust(left=0.23, right=0.995, bottom=0.115, top=0.72)

    path = panel_root / "design_class_mask_overview.svg"
    source_paths = _source_paths_by_label(
        baseline_mask_set_path=baseline_mask_set_path,
        design_classes_root=design_classes_root,
    )
    source_paths.update(rt_annotation_context.source_paths)
    alt = (
        "Matrix comparing Eco1 RT mask evidence and the six ProteinMPNN design-class masks. Evidence rows "
        "separate motif anchors, Wang/EC86 priors, WT-plurality cutoffs, and retained DNA/RNA proximity rows. "
        "Class rows show which residues are fixed. Display bands mark audited RT context spans, "
        "RT1-RT7 intervals, and motif-anchor neighborhoods."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=300)
    return make_deliverable_row(
        deliverable_id="design_class_mask_overview",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=_source_table_labels() + rt_annotation_context.source_table_labels,
        input_hashes=file_hashes(source_paths),
        alt_text=alt,
        description=(
            "Provides one residue-coordinate source of truth for mask review. Evidence rows show motif, "
            "prior, conservation, and retained DNA/RNA proximity rules; class rows show fixed residues for "
            "the six design classes. Display-only RT spans provide motif and domain context without changing "
            "mask membership."
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
            "rt_annotation_feature_count": len(rt_annotation_context.features),
            "rt_annotation_target_sequence_hash": rt_annotation_context.target_sequence_hash,
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
            "Wang/EC86 substrate-contact priors",
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
                _conservation_axis_label(profile_id=profile_id, threshold=threshold),
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
        rows.append(_binary_row(f"DNA/RNA within {threshold} A", _STATE_CONTACT, positions))
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
        return "Clade 9"
    if profile_id == CONSERVATION_SUBTYPE_PROFILE_ID:
        return "II-A3/42_1"
    return profile_id


def _conservation_axis_label(*, profile_id: str, threshold: int) -> str:
    return f"{_profile_label(profile_id)}: >={threshold}% WT plurality"


def _class_axis_label(row: dict[str, Any]) -> str:
    protected = int(row["protected_count"])
    threshold = int(round(float(row["conservation_threshold"]) * 100))
    contact = int(round(float(row["contact_threshold_angstrom"])))
    class_basis = f"{_profile_label(str(row['conservation_profile_id']))} {threshold}% + {contact} A"
    return f"{class_basis} | {protected} fixed"


def _matrix_state(row: dict[str, Any], position: int) -> int:
    if row["kind"] == "binary":
        return int(row["state"]) if position in row["positions"] else _STATE_EMPTY
    return _policy_state_value(row["residues_by_position"].get(position))


def _policy_state_value(residue: dict[str, Any] | None) -> int:
    if residue is None:
        return _STATE_EMPTY
    if bool(residue.get("protected")):
        return _STATE_FIXED
    return _STATE_EMPTY


def _position_tick_indexes(positions: list[int]) -> list[int]:
    return _labeled_position_tick_indexes(positions, step=_BOTTOM_POSITION_LABEL_STEP)


def _top_amino_acid_tick_indexes(positions: list[int]) -> list[int]:
    return _labeled_position_tick_indexes(positions, step=_TOP_AMINO_ACID_LABEL_STEP)


def _labeled_position_tick_indexes(positions: list[int], *, step: int) -> list[int]:
    if len(positions) <= step * 2:
        return list(range(len(positions)))
    indexes = [0]
    indexes.extend(index for index, position in enumerate(positions) if position % step == 0)
    if indexes[-1] != len(positions) - 1:
        indexes.append(len(positions) - 1)
    return sorted(set(indexes))


def _minor_position_tick_indexes(positions: list[int]) -> list[int]:
    if not positions:
        return []
    return list(range(0, len(positions), _POSITION_MINOR_TICK_STEP))


def _add_top_amino_acid_axis(ax: Any, positions: list[int], *, wt_aa_by_position: dict[int, str]) -> None:
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_tick_indexes = _top_amino_acid_tick_indexes(positions)
    top_ax.set_xticks(
        top_tick_indexes,
        [wt_aa_by_position[positions[index]] for index in top_tick_indexes],
        fontsize=_TOP_AMINO_ACID_FONT_SIZE,
    )
    for tick_label in top_ax.get_xticklabels():
        tick_label.set_fontfamily("DejaVu Sans Mono")
    top_ax.tick_params(axis="x", which="major", length=1.2, labelsize=_TOP_AMINO_ACID_FONT_SIZE, pad=1)
    top_ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    top_ax.spines[["right", "left", "bottom"]].set_visible(False)
    top_ax.spines["top"].set_color("#737373")
    top_ax.spines["top"].set_linewidth(0.55)


def _add_row_group_separator(ax: Any, matrix_rows: list[dict[str, Any]]) -> None:
    evidence_row_count = sum(1 for row in matrix_rows if row["kind"] == "binary")
    if evidence_row_count <= 0 or evidence_row_count >= len(matrix_rows):
        return
    ax.axhline(evidence_row_count - 0.5, color="#5f6368", linewidth=0.7)


def _wt_aa_by_position(class_rows: list[dict[str, Any]]) -> dict[int, str]:
    wt_by_position: dict[int, str] = {}
    for class_row in class_rows:
        for position, residue in class_row["residues_by_position"].items():
            wt_aa = str(residue.get("wt_aa") or "").strip().upper()
            if len(wt_aa) != 1:
                raise ValueError(f"mask_set row {position} must include one-letter wt_aa for the EC86 sequence axis")
            existing = wt_by_position.setdefault(position, wt_aa)
            if existing != wt_aa:
                raise ValueError(f"mask_set rows disagree on wt_aa for canonical position {position}")
    return wt_by_position


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
