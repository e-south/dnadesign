"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/mask_structure_browser.py

Interactive mask-evidence structure-browser manifest for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .structure_browser_common import (
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    reference_residue_number_by_canonical,
    reference_selection_coordinate_basis,
    relative_path,
    repo_relative_hint,
)

MASK_STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "mask_structure_browser_manifest.yaml"
_MASK_HIGHLIGHT_COLOR = RESIDUE_CATEGORY_HIGHLIGHT_COLOR
_MASK_SELECTIONS = (
    (
        "motif_protected",
        "Catalytic motif anchors",
        "motif_protected",
        "NAxxH, YADD, and VTG motif residues fixed before ProteinMPNN design.",
    ),
    (
        "wang_ec86_direct_contact_prior",
        "Wang/Ec86 substrate-contact priors",
        "wang_ec86_direct_contact_prior",
        "Residues from the Ec86 structural prior that directly contact substrate.",
    ),
    (
        "direct_retained_dna_rna_contact_5a",
        "Retained DNA/RNA-contact residues within 5 A",
        "direct_retained_dna_rna_contact_5a",
        "Residues near retained nucleic-acid atoms in the reference structure.",
    ),
    (
        "evolutionarily_conserved_clade9_25pct_plurality",
        "Clade 9 plurality-protected residues",
        "evolutionarily_conserved_clade9_25pct_plurality",
        "Residues fixed by the clade 9 WT-plurality rule.",
    ),
    (
        "protected",
        "Protected union",
        "protected",
        "All residues fixed by at least one active mask rule.",
    ),
    (
        "non_fixed",
        "ProteinMPNN-designable residues",
        "non_fixed",
        "Residues exposed to ProteinMPNN redesign in the current campaign.",
    ),
)


def write_mask_structure_browser_manifest(
    *,
    panel_root: Path,
    mask_set_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write a manifest for interactive mask-category highlighting on the reference backbone."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / MASK_STRUCTURE_BROWSER_MANIFEST_FILE_NAME
    if not reference_structure_path.exists():
        return _missing_mask_row(manifest_path, reference_structure_path)
    views = _mask_structure_views(
        mask_residues=mask_residues,
        reference_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        manifest_root=manifest_path.parent,
    )
    payload = {
        "schema_id": "eco1_rt.interactive_structure_browser_manifest",
        "schema_version": 1,
        "status": "materialized",
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": [
            repo_relative_hint(mask_set_path),
            repo_relative_hint(reference_structure_path),
        ],
        "reference": {
            "model_id": "ec86kit_7v9u_reference",
            "display_label": _reference_display_label(reference_structure_path, reference_structure_format),
            "local_path": relative_path(reference_structure_path, manifest_path.parent),
            "structure_format": reference_structure_format,
            "color": REFERENCE_COLOR,
        },
        "alignment": {"status": "disabled", "method": "reference_selection"},
        "control_label": "Highlight",
        "structures": views,
        "structure_count": len(views),
        "interpretation_limit": (
            "This browser view maps mask evidence onto the reference structure. It does "
            "not evaluate candidate fold quality or RT activity."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="mask_structure_browser_manifest",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=["mask_set.yaml", repo_relative_hint(reference_structure_path)],
        input_hashes=file_hashes({"mask_set": mask_set_path, "reference_structure": reference_structure_path}),
        alt_text="Interactive Ec86 reference structure viewer with selectable fixed-residue mask highlights.",
        description=(
            "Shows the Ec86/7V9U reference structure with one mask or motif category highlighted at a time. "
            "The base structure remains off-white so the selected evidence category is visually separable."
        ),
        interpretation_limit=payload["interpretation_limit"],
        title="Ec86 reference structure maps fixed-residue evidence interactively",
        role="interactive_review",
    )


def _reference_display_label(reference_structure_path: Path, reference_structure_format: str) -> str:
    if reference_structure_format == "mmcif" or "all_atom" in reference_structure_path.stem:
        return "Ec86/7V9U all-atom reference"
    return "ec86kit/7V9U reference"


def _missing_mask_row(manifest_path: Path, missing_path: Path) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id="mask_structure_browser_manifest",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="structure_browser_manifest",
        status="skipped_missing_input",
        path=manifest_path,
        source_tables=["mask_set.yaml", "proteinmpnn_request/chain_a_backbone.pdb"],
        input_hashes={},
        alt_text="Interactive mask structure browser was not generated.",
        description="Interactive mask highlighting is skipped until the reference backbone PDB exists.",
        interpretation_limit="Missing structure paths cannot support interactive mask review.",
        title="Ec86 reference mask browser is skipped until the backbone PDB is available",
        role="interactive_review",
        skip_reason=f"Missing input reference structure: {missing_path}",
    )


def _mask_structure_views(
    *,
    mask_residues: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    reference_number_by_canonical = reference_residue_number_by_canonical(
        mask_residues,
        reference_structure_format=reference_structure_format,
    )
    selection_coordinate_basis = reference_selection_coordinate_basis(
        reference_structure_format=reference_structure_format,
    )
    for view_id, label, field, description in _MASK_SELECTIONS:
        canonical_residue_numbers = sorted(
            int(row["canonical_position"]) for row in mask_residues if bool(row.get(field))
        )
        residue_numbers = [
            reference_number_by_canonical[position]
            for position in canonical_residue_numbers
            if position in reference_number_by_canonical
        ]
        if not residue_numbers:
            continue
        views.append(
            {
                "candidate_id": view_id,
                "display_label": label,
                "group": "Reference mask evidence",
                "local_path": relative_path(reference_path, manifest_root),
                "structure_format": reference_structure_format,
                "color": _MASK_HIGHLIGHT_COLOR,
                "structure_view_mode": "reference_selection",
                "description": description,
                "selection_styles": [
                    {
                        "selection_id": view_id,
                        "model_id": "ec86kit_7v9u_reference",
                        "label": label,
                        "source_coordinate_basis": "canonical_position",
                        "selection_coordinate_basis": selection_coordinate_basis,
                        "canonical_residue_numbers": canonical_residue_numbers,
                        "residue_numbers": residue_numbers,
                        "residue_scope": "protein",
                        "color": _MASK_HIGHLIGHT_COLOR,
                    }
                ],
                "selection_residue_count": len(residue_numbers),
            }
        )
    return views
