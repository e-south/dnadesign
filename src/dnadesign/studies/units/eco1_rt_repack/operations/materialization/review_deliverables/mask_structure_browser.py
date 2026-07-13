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

from ..shared.rt_annotation_context import RTAnnotationContext, RTAnnotationFeature
from .communication_visuals.structure_scenes import structure_scene_specs
from .molecular_scene_contract import (
    REFERENCE_MODEL_ID,
    molecular_visual_contract,
    reference_complex_molecule_styles,
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
GROUP_MASK_INPUT_EVIDENCE = "Current mask evidence"
GROUP_RT_ANNOTATION_SPANS = "RT annotation spans"
GROUP_DESIGN_SPACES = "Design spaces"
_RT_CONTEXT_HIGHLIGHT_COLOR = "#6f4c7d"
_RT_CORE_INTERVAL_HIGHLIGHT_COLOR = "#28566a"
_RT_MOTIF_HIGHLIGHT_COLOR = "#8a4a11"
_MASK_PROTECTED_COLOR = RESIDUE_CATEGORY_HIGHLIGHT_COLOR
_MASK_INPUT_CONSERVATION_COLOR = "#0072B2"
_MASK_INPUT_CONTACT_COLOR = "#E69F00"
_MASK_INPUT_PRIOR_COLOR = "#6A3D9A"
_MASK_INPUT_MOTIF_COLOR = "#8A4A11"
_TRACK_CONTEXT = "retron_rt_context_spans"
_TRACK_CORE_INTERVALS = "retron_rt_core_intervals"
_TRACK_MOTIF_ANCHORS = "retron_rt_motif_anchors"


def write_mask_structure_browser_manifest(
    *,
    panel_root: Path,
    mask_set_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
    rt_annotation_context: RTAnnotationContext,
    policy_position_rows: list[dict[str, Any]],
    policy_positions_path: Path,
) -> dict[str, Any]:
    """Write a manifest for interactive mask-category highlighting on the reference backbone."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / MASK_STRUCTURE_BROWSER_MANIFEST_FILE_NAME
    title = "The Ec86 structure maps the active mask evidence"
    alt_text = (
        "Interactive Ec86 reference structure viewer with selectable active mask evidence and RT annotation spans."
    )
    description = (
        "Shows the Ec86/7V9U reference structure with one active mask-evidence or RT annotation choice "
        "highlighted at a time. The base structure remains off-white so the selected residue set is "
        "visually separable."
    )
    if not reference_structure_path.exists():
        return _missing_mask_row(manifest_path, reference_structure_path)
    views = _mask_structure_views(
        mask_residues=mask_residues,
        reference_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        manifest_root=manifest_path.parent,
        rt_annotation_context=rt_annotation_context,
        policy_position_rows=policy_position_rows,
    )
    source_paths = {
        "mask_set": mask_set_path,
        "generation_policy_positions": policy_positions_path,
        "reference_structure": reference_structure_path,
    }
    source_paths.update(rt_annotation_context.source_paths)
    payload = {
        "schema_id": "eco1_rt.interactive_structure_browser_manifest",
        "schema_version": 1,
        "status": "materialized",
        "title": title,
        "alt_text": alt_text,
        "description": description,
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "visual_contract": molecular_visual_contract(),
        "protein_surface_default": False,
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": [
            repo_relative_hint(mask_set_path),
            repo_relative_hint(policy_positions_path),
            repo_relative_hint(reference_structure_path),
            *rt_annotation_context.source_table_labels,
        ],
        "source_hashes": file_hashes(source_paths),
        "reference": {
            "model_id": REFERENCE_MODEL_ID,
            "display_label": _reference_display_label(reference_structure_path, reference_structure_format),
            "local_path": relative_path(reference_structure_path, manifest_path.parent),
            "structure_format": reference_structure_format,
            "color": REFERENCE_COLOR,
        },
        "alignment": {"status": "disabled", "method": "reference_selection"},
        "control_label": "Structure scene",
        "structures": views,
        "structure_count": len(views),
        "interpretation_limit": (
            "This browser view maps active mask evidence and RT annotations onto the reference structure. "
            "It does not evaluate candidate fold quality or RT activity."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="mask_structure_browser_manifest",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=[repo_relative_hint(mask_set_path), repo_relative_hint(reference_structure_path)]
        + [repo_relative_hint(policy_positions_path)]
        + rt_annotation_context.source_table_labels,
        input_hashes=file_hashes(source_paths),
        alt_text=alt_text,
        description=description,
        interpretation_limit=payload["interpretation_limit"],
        title=title,
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
        alt_text="Mask structure browsing is unavailable because the reference backbone PDB is missing.",
        description="Interactive mask highlighting is skipped until the reference backbone PDB exists.",
        interpretation_limit="Missing structure paths cannot support interactive mask review.",
        title="The Ec86 mask browser waits for the reference backbone PDB",
        role="interactive_review",
        skip_reason=f"Missing input reference structure: {missing_path}",
    )


def _mask_structure_views(
    *,
    mask_residues: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    rt_annotation_context: RTAnnotationContext,
    policy_position_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    reference_number_by_canonical = reference_residue_number_by_canonical(
        mask_residues,
        reference_structure_format=reference_structure_format,
    )
    selection_coordinate_basis = reference_selection_coordinate_basis(
        reference_structure_format=reference_structure_format,
    )
    views.extend(
        _mask_input_evidence_views(
            mask_residues=mask_residues,
            reference_path=reference_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
        )
    )
    views.extend(
        _design_space_views(
            policy_position_rows=policy_position_rows,
            reference_path=reference_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
        )
    )
    views.extend(
        _rt_annotation_structure_views(
            rt_annotation_context=rt_annotation_context,
            reference_path=reference_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
        )
    )
    return views


def _design_space_views(
    *,
    policy_position_rows: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for scene in structure_scene_specs(policy_position_rows):
        if str(scene.get("group") or "") != "3 Design spaces":
            continue
        views.extend(
            reference_selection_view(
                view_id=str(scene["scene_id"]),
                label=str(scene["label"]),
                group=GROUP_DESIGN_SPACES,
                description=str(scene["description"]),
                canonical_residue_numbers=set(scene["positions"]),
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=str(scene["color"]),
            )
        )
    return views


def _mask_input_evidence_views(
    *,
    mask_residues: list[dict[str, Any]],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> list[dict[str, Any]]:
    rows_by_position = {int(row["canonical_position"]): row for row in mask_residues}
    view_specs = (
        (
            "active_mask_protected_positions",
            "Protected residues",
            "Residues fixed by the current Eco1 RT mask rule.",
            _positions_with_field(rows_by_position, "protected"),
            _MASK_PROTECTED_COLOR,
        ),
        (
            "active_mask_motif_anchors",
            "Catalytic and retron motif anchors",
            "NAxxH, YADD, VTG, and other motif-anchor residues fixed by the current mask rule.",
            _positions_with_field(rows_by_position, "motif_protected"),
            _MASK_INPUT_MOTIF_COLOR,
        ),
        (
            "active_mask_wang_ec86_direct_contact_prior",
            "Wang/Ec86 substrate-contact priors",
            "Residues from the Ec86 structural prior that directly contact substrate.",
            _positions_with_field(rows_by_position, "wang_ec86_direct_contact_prior"),
            _MASK_INPUT_PRIOR_COLOR,
        ),
        (
            "active_mask_clade9_25pct_wt_plurality",
            "Clade 9 >=25% WT plurality",
            "Residues fixed because the Eco1 amino acid is the clade 9 plurality residue at the mask threshold.",
            _positions_with_any_field(
                rows_by_position,
                ("selected_conservation_rule_passed", "evolutionarily_conserved_clade9_25pct_plurality"),
            ),
            _MASK_INPUT_CONSERVATION_COLOR,
        ),
        (
            "active_mask_direct_retained_dna_rna_contact_5a",
            "Retained DNA/RNA <=5 A",
            "Residues fixed because they are within 5 A of retained DNA/RNA atoms.",
            _positions_with_any_field(
                rows_by_position,
                ("selected_retained_dna_rna_contact", "direct_retained_dna_rna_contact_5a"),
            ),
            _MASK_INPUT_CONTACT_COLOR,
        ),
    )
    views: list[dict[str, Any]] = []
    for view_id, label, description, positions, color in view_specs:
        views.extend(
            reference_selection_view(
                view_id=view_id,
                label=label,
                group=GROUP_MASK_INPUT_EVIDENCE,
                description=description,
                canonical_residue_numbers=positions,
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=color,
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
            "molecule_styles": reference_complex_molecule_styles(include_protein_surface=True),
            "selection_styles": [
                {
                    "selection_id": view_id,
                    "model_id": REFERENCE_MODEL_ID,
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


def _positions_with_field(rows_by_position: dict[int, dict[str, Any]], field: str) -> set[int]:
    return {position for position, row in rows_by_position.items() if bool(row.get(field))}


def _positions_with_any_field(rows_by_position: dict[int, dict[str, Any]], fields: tuple[str, ...]) -> set[int]:
    return {position for position, row in rows_by_position.items() if any(bool(row.get(field)) for field in fields)}


def _rt_annotation_structure_views(
    *,
    rt_annotation_context: RTAnnotationContext,
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    reference_number_by_canonical: dict[int, int],
    selection_coordinate_basis: str,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for feature in rt_annotation_context.features:
        canonical_residue_numbers = [
            position for position in range(feature.start, feature.end + 1) if position in reference_number_by_canonical
        ]
        residue_numbers = [reference_number_by_canonical[position] for position in canonical_residue_numbers]
        if not residue_numbers:
            continue
        color = _rt_annotation_color(feature)
        views.extend(
            reference_selection_view(
                view_id=feature.feature_id,
                label=feature.label,
                group=GROUP_RT_ANNOTATION_SPANS,
                description=(
                    f"Display-only RT annotation span from {feature.start}-{feature.end}; "
                    "included for structural orientation, not as a mask rule."
                ),
                canonical_residue_numbers=canonical_residue_numbers,
                reference_path=reference_path,
                reference_structure_format=reference_structure_format,
                manifest_root=manifest_root,
                reference_number_by_canonical=reference_number_by_canonical,
                selection_coordinate_basis=selection_coordinate_basis,
                color=color,
            )
        )
    return views


def _rt_annotation_color(feature: RTAnnotationFeature) -> str:
    if feature.track_id == _TRACK_CONTEXT:
        return _RT_CONTEXT_HIGHLIGHT_COLOR
    if feature.track_id == _TRACK_CORE_INTERVALS:
        return _RT_CORE_INTERVAL_HIGHLIGHT_COLOR
    if feature.track_id == _TRACK_MOTIF_ANCHORS:
        return _RT_MOTIF_HIGHLIGHT_COLOR
    return RESIDUE_CATEGORY_HIGHLIGHT_COLOR
