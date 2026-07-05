"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/mask_structure_highlights.py

Design-class and mask-input highlight sets for Eco1 structure browsers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    CONSERVATION_CLADE9_PROFILE_ID,
    CONSERVATION_SUBTYPE_PROFILE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)

from .structure_browser_common import RESIDUE_CATEGORY_HIGHLIGHT_COLOR, relative_path

GROUP_DESIGN_CLASS_FIXED_MASKS = "Design-class fixed masks"
GROUP_MASK_INPUT_EVIDENCE = "Mask input evidence"
GROUP_RT_ANNOTATION_SPANS = "RT annotation spans"
MASK_HIGHLIGHT_COLOR = RESIDUE_CATEGORY_HIGHLIGHT_COLOR
MASK_INPUT_CONSERVATION_COLOR = "#0072B2"
MASK_INPUT_CONTACT_COLOR = "#E69F00"
MASK_INPUT_PRIOR_COLOR = "#6A3D9A"


def load_design_class_mask_rows(*, baseline_mask_set_path: Path, design_classes_root: Path) -> list[dict[str, Any]]:
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
                "source_path": mask_set_path,
                "profile_label": _profile_label(spec.conservation_profile_id),
                "conservation_profile_id": spec.conservation_profile_id,
                "conservation_threshold_percent": int(round(float(spec.conservation_threshold) * 100)),
                "contact_threshold_angstrom": int(round(float(spec.contact_threshold_angstrom))),
                "protected_count": sum(1 for row in residues if bool(row.get("protected"))),
                "residues_by_position": {int(row["canonical_position"]): row for row in residues},
            }
        )
    return rows


def design_class_fixed_mask_views(
    *,
    design_class_rows: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for row in design_class_rows:
        canonical_residue_numbers = sorted(
            position for position, residue in row["residues_by_position"].items() if bool(residue.get("protected"))
        )
        label = f"Fixed mask: {_design_class_short_label(row)}"
        views.extend(
            reference_selection_view(
                view_id=str(row["design_class_id"]),
                label=label,
                group=GROUP_DESIGN_CLASS_FIXED_MASKS,
                description=f"Residues fixed under the {_design_class_short_label(row)} design-class mask.",
                canonical_residue_numbers=canonical_residue_numbers,
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=MASK_HIGHLIGHT_COLOR,
            )
        )
    return views


def mask_input_evidence_views(
    *,
    design_class_rows: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    views.extend(
        reference_selection_view(
            view_id="mask_input_wang_ec86_direct_contact_prior",
            label="Wang/Ec86 substrate-contact priors",
            group=GROUP_MASK_INPUT_EVIDENCE,
            description="Residues from the Ec86 structural prior that directly contact substrate.",
            canonical_residue_numbers=_positions_with_field(design_class_rows, "wang_ec86_direct_contact_prior"),
            reference_path=reference_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
            color=MASK_INPUT_PRIOR_COLOR,
        )
    )
    for profile_id, threshold in _ordered_conservation_keys(design_class_rows):
        label = f"{_profile_label(profile_id)} p{threshold} WT plurality"
        views.extend(
            reference_selection_view(
                view_id=f"mask_input_conservation_{_safe_id(profile_id)}_p{threshold}",
                label=label,
                group=GROUP_MASK_INPUT_EVIDENCE,
                description=f"Residues passing the {label} conservation input.",
                canonical_residue_numbers=_conservation_positions(
                    design_class_rows,
                    profile_id=profile_id,
                    threshold=threshold,
                ),
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=MASK_INPUT_CONSERVATION_COLOR,
            )
        )
    for threshold in _ordered_contact_thresholds(design_class_rows):
        views.extend(
            reference_selection_view(
                view_id=f"mask_input_retained_dna_rna_contact_{threshold}a",
                label=f"DNA/RNA within {threshold} A",
                group=GROUP_MASK_INPUT_EVIDENCE,
                description=f"Residues within {threshold} A of retained DNA/RNA atoms.",
                canonical_residue_numbers=_contact_positions(design_class_rows, threshold=threshold),
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=MASK_INPUT_CONTACT_COLOR,
            )
        )
    return views


def reference_selection_view(
    *,
    view_id: str,
    label: str,
    group: str,
    description: str,
    canonical_residue_numbers: list[int] | set[int],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
    color: str,
) -> list[dict[str, Any]]:
    canonical_numbers = sorted(set(int(position) for position in canonical_residue_numbers))
    residue_numbers = [
        reference_number_by_canonical[position]
        for position in canonical_numbers
        if position in reference_number_by_canonical
    ]
    if not residue_numbers:
        return []
    return [
        {
            "candidate_id": view_id,
            "display_label": label,
            "group": group,
            "local_path": relative_path(reference_path, manifest_root),
            "structure_format": reference_structure_format,
            "color": color,
            "structure_view_mode": "reference_selection",
            "description": description,
            "selection_styles": [
                {
                    "selection_id": view_id,
                    "model_id": "ec86kit_7v9u_reference",
                    "label": label,
                    "source_coordinate_basis": "canonical_position",
                    "selection_coordinate_basis": selection_coordinate_basis,
                    "canonical_residue_numbers": canonical_numbers,
                    "residue_numbers": residue_numbers,
                    "residue_scope": "protein",
                    "color": color,
                }
            ],
            "selection_residue_count": len(residue_numbers),
        }
    ]


def design_class_source_paths(design_class_rows: list[dict[str, Any]]) -> dict[str, Path]:
    return {
        f"design_class_mask_{row['design_class_id']}": Path(row["source_path"])
        for row in design_class_rows
        if str(row["design_class_id"]) != BASELINE_CLASS_ID
    }


def design_class_source_table_labels(design_class_rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in design_class_rows:
        source_path = Path(row["source_path"])
        if str(row["design_class_id"]) == BASELINE_CLASS_ID:
            labels.append("mask_set.yaml")
        else:
            labels.append(str(Path("design_classes") / str(row["design_class_id"]) / source_path.name))
    return labels


def _positions_with_field(design_class_rows: list[dict[str, Any]], field: str) -> set[int]:
    return {
        position
        for row in design_class_rows
        for position, residue in row["residues_by_position"].items()
        if bool(residue.get(field))
    }


def _ordered_conservation_keys(design_class_rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    observed = {
        (str(row["conservation_profile_id"]), int(row["conservation_threshold_percent"])) for row in design_class_rows
    }
    preferred = [
        (CONSERVATION_CLADE9_PROFILE_ID, 25),
        (CONSERVATION_CLADE9_PROFILE_ID, 50),
        (CONSERVATION_SUBTYPE_PROFILE_ID, 50),
    ]
    ordered = [key for key in preferred if key in observed]
    ordered.extend(sorted(observed - set(ordered), key=lambda key: (_profile_label(key[0]), key[1])))
    return ordered


def _conservation_positions(
    design_class_rows: list[dict[str, Any]],
    *,
    profile_id: str,
    threshold: int,
) -> set[int]:
    positions: set[int] = set()
    for row in design_class_rows:
        if str(row["conservation_profile_id"]) != profile_id:
            continue
        if int(row["conservation_threshold_percent"]) != threshold:
            continue
        positions.update(
            position
            for position, residue in row["residues_by_position"].items()
            if bool(residue.get("selected_conservation_rule_passed"))
        )
    return positions


def _ordered_contact_thresholds(design_class_rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["contact_threshold_angstrom"]) for row in design_class_rows})


def _contact_positions(design_class_rows: list[dict[str, Any]], *, threshold: int) -> set[int]:
    positions: set[int] = set()
    for row in design_class_rows:
        if int(row["contact_threshold_angstrom"]) != threshold:
            continue
        positions.update(
            position
            for position, residue in row["residues_by_position"].items()
            if bool(residue.get("selected_retained_dna_rna_contact"))
        )
    return positions


def _profile_label(profile_id: str) -> str:
    if profile_id == CONSERVATION_CLADE9_PROFILE_ID:
        return "Clade 9"
    if profile_id == CONSERVATION_SUBTYPE_PROFILE_ID:
        return "II-A3/42_1"
    return profile_id


def _design_class_short_label(row: dict[str, Any]) -> str:
    profile_label = str(row["profile_label"])
    threshold = int(row["conservation_threshold_percent"])
    contact = int(row["contact_threshold_angstrom"])
    return f"{profile_label} p{threshold} + {contact} A"


def _safe_id(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
