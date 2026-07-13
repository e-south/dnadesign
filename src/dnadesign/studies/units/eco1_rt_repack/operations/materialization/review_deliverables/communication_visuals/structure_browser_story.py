"""Browser-manifest materialization for the Eco1 structure story."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
)

from ..molecular_scene_contract import (
    REFERENCE_MODEL_ID,
    molecular_visual_contract,
    reference_complex_molecule_styles,
)
from ..structure_browser_common import relative_path
from .style import PROTEIN_SURFACE_COLOR


def build_structure_browser_payload(
    *,
    reference_structure_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
    scene_specs: tuple[dict[str, Any], ...],
    mask_set_path: Path,
    policy_positions_path: Path,
) -> dict[str, Any]:
    """Build the renderer-neutral structure story manifest."""

    structures = [
        _browser_scene(
            scene=scene,
            reference_structure_path=reference_structure_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
        )
        for scene in scene_specs
    ]
    return {
        "schema_id": "eco1_rt.communication_structure_story",
        "schema_version": 1,
        "status": "materialized",
        "title": "The retained complex separates protected evidence from design space",
        "alt_text": (
            "Interactive Ec86 reference structure with surface scenes for protected evidence and design spaces."
        ),
        "description": (
            "Each scene uses the same all-atom RT-DNA-RNA reference and exposes one residue-set premise at a time."
        ),
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "visual_contract": molecular_visual_contract(),
        "protein_surface_default": False,
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": [mask_set_path.name, policy_positions_path.name, reference_structure_path.name],
        "source_hashes": file_hashes(
            {
                "mask_set": mask_set_path,
                "generation_policy_positions": policy_positions_path,
                "reference_structure": reference_structure_path,
            }
        ),
        "reference": {
            "model_id": REFERENCE_MODEL_ID,
            "display_label": "Ec86/7V9U all-atom reference",
            "local_path": relative_path(reference_structure_path, manifest_root),
            "structure_format": reference_structure_format,
            "color": PROTEIN_SURFACE_COLOR,
        },
        "alignment": {"status": "disabled", "method": "reference_selection"},
        "control_label": "Structure premise",
        "structures": structures,
        "structure_count": len(structures),
        "interpretation_limit": (
            "These views map declared residue sets onto the reference complex and do not predict RT activity."
        ),
    }


def _browser_scene(
    *,
    scene: dict[str, Any],
    reference_structure_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> dict[str, Any]:
    canonical_numbers = sorted(set(int(value) for value in scene["positions"]))
    residue_numbers = [
        reference_number_by_canonical[position]
        for position in canonical_numbers
        if position in reference_number_by_canonical
    ]
    selection_styles = []
    if residue_numbers:
        selection_styles.append(
            {
                "selection_id": scene["scene_id"],
                "model_id": REFERENCE_MODEL_ID,
                "label": scene["label"],
                "source_coordinate_basis": "canonical_position",
                "selection_coordinate_basis": selection_coordinate_basis,
                "canonical_residue_numbers": canonical_numbers,
                "residue_numbers": residue_numbers,
                "residue_scope": "protein",
                "color": scene["color"],
            }
        )
    row = {
        "candidate_id": scene["scene_id"],
        "display_label": scene["label"],
        "group": scene["group"],
        "local_path": relative_path(reference_structure_path, manifest_root),
        "structure_format": reference_structure_format,
        "color": PROTEIN_SURFACE_COLOR,
        "structure_view_mode": "reference_selection",
        "description": scene["description"],
        "molecule_styles": reference_complex_molecule_styles(include_protein_surface=True),
        "selection_styles": selection_styles,
    }
    if residue_numbers:
        row["selection_residue_count"] = len(residue_numbers)
    return row
