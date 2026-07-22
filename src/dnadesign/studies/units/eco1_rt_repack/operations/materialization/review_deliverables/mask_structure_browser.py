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

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from ..shared.rt_annotation_context import RTAnnotationContext, RTAnnotationFeature
from .communication_visuals.structure_scenes import (
    SCENE_KIND_DESIGN,
    SCENE_KIND_FIXED,
    structure_scene_specs,
)
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
GROUP_FIXED_POSITIONS = "Fixed positions"
GROUP_RT_ANNOTATION_SPANS = "RT annotations"
GROUP_DESIGN_SPACES = "Design spaces"
_RT_CONTEXT_HIGHLIGHT_COLOR = "#6f4c7d"
_RT_CORE_INTERVAL_HIGHLIGHT_COLOR = "#28566a"
_RT_MOTIF_HIGHLIGHT_COLOR = "#8a4a11"
_TRACK_CONTEXT = "retron_rt_context_spans"
_TRACK_CORE_INTERVALS = "retron_rt_core_intervals"
_TRACK_MOTIF_ANCHORS = "retron_rt_motif_anchors"


def read_policy_position_rows(path: Path) -> list[dict[str, Any]]:
    """Load the generation-policy rows used to construct mask-browser views."""

    return [dict(row) for row in pq.read_table(path).to_pylist()]


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
    title = "The Ec86 structure maps fixed and open residue sets"
    alt_text = (
        "Interactive Ec86 reference structure viewer with selectable fixed positions, open design spaces, and RT "
        "annotations."
    )
    description = (
        "Shows the Ec86/7V9U reference structure with one generation-policy residue set or RT annotation "
        "highlighted at a time. Fixed and open positions come from the active generation-policy manifest."
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
            "This browser maps the declared generation contract and RT annotations onto the reference structure. "
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
        _generation_policy_views(
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


def _generation_policy_views(
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
        scene_kind = str(scene.get("scene_kind") or "")
        if scene_kind not in {SCENE_KIND_FIXED, SCENE_KIND_DESIGN}:
            continue
        group = GROUP_FIXED_POSITIONS if scene_kind == SCENE_KIND_FIXED else GROUP_DESIGN_SPACES
        views.extend(
            reference_selection_view(
                view_id=str(scene["scene_id"]),
                label=str(scene["label"]),
                group=group,
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
    canonical_numbers = sorted(
        {int(position) for position in canonical_residue_numbers if int(position) in reference_number_by_canonical}
    )
    residue_numbers = [reference_number_by_canonical[position] for position in canonical_numbers]
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
                description=_rt_annotation_description(feature),
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


def _rt_annotation_description(feature: RTAnnotationFeature) -> str:
    if feature.track_id == _TRACK_CONTEXT:
        return (
            f"Fixed context window from {feature.start}-{feature.end} around an exact motif anchor; "
            "the flanking residues are a precautionary study choice."
        )
    if feature.track_id == _TRACK_MOTIF_ANCHORS:
        return (
            f"Exact literature-annotated motif anchor from {feature.start}-{feature.end}; "
            "it remains fixed within the wider declared context window."
        )
    return (
        f"Display-only RT annotation span from {feature.start}-{feature.end}; "
        "included for structural orientation, not as a protection rule."
    )


def _rt_annotation_color(feature: RTAnnotationFeature) -> str:
    if feature.track_id == _TRACK_CONTEXT:
        return _RT_CONTEXT_HIGHLIGHT_COLOR
    if feature.track_id == _TRACK_CORE_INTERVALS:
        return _RT_CORE_INTERVAL_HIGHLIGHT_COLOR
    if feature.track_id == _TRACK_MOTIF_ANCHORS:
        return _RT_MOTIF_HIGHLIGHT_COLOR
    return RESIDUE_CATEGORY_HIGHLIGHT_COLOR
