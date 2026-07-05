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

from .mask_structure_highlights import (
    GROUP_RT_ANNOTATION_SPANS,
    design_class_fixed_mask_views,
    design_class_source_paths,
    design_class_source_table_labels,
    load_design_class_mask_rows,
    mask_input_evidence_views,
    reference_selection_view,
)
from .rt_annotation_context import RTAnnotationContext, RTAnnotationFeature
from .structure_browser_common import (
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    reference_residue_number_by_canonical,
    reference_selection_coordinate_basis,
    relative_path,
    repo_relative_hint,
)

MASK_STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "mask_structure_browser_manifest.yaml"
_RT_CONTEXT_HIGHLIGHT_COLOR = "#6f4c7d"
_RT_CORE_INTERVAL_HIGHLIGHT_COLOR = "#28566a"
_RT_MOTIF_HIGHLIGHT_COLOR = "#8a4a11"
_TRACK_CONTEXT = "retron_rt_context_spans"
_TRACK_CORE_INTERVALS = "retron_rt_core_intervals"
_TRACK_MOTIF_ANCHORS = "retron_rt_motif_anchors"


def write_mask_structure_browser_manifest(
    *,
    panel_root: Path,
    mask_set_path: Path,
    design_classes_root: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
    rt_annotation_context: RTAnnotationContext,
) -> dict[str, Any]:
    """Write a manifest for interactive mask-category highlighting on the reference backbone."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / MASK_STRUCTURE_BROWSER_MANIFEST_FILE_NAME
    if not reference_structure_path.exists():
        return _missing_mask_row(manifest_path, reference_structure_path)
    design_class_rows = load_design_class_mask_rows(
        baseline_mask_set_path=mask_set_path,
        design_classes_root=design_classes_root,
    )
    views = _mask_structure_views(
        mask_residues=mask_residues,
        reference_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        manifest_root=manifest_path.parent,
        design_class_rows=design_class_rows,
        rt_annotation_context=rt_annotation_context,
    )
    source_paths = {
        "mask_set": mask_set_path,
        "reference_structure": reference_structure_path,
    }
    source_paths.update(design_class_source_paths(design_class_rows))
    source_paths.update(rt_annotation_context.source_paths)
    payload = {
        "schema_id": "eco1_rt.interactive_structure_browser_manifest",
        "schema_version": 1,
        "status": "materialized",
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": [
            *design_class_source_table_labels(design_class_rows),
            repo_relative_hint(reference_structure_path),
            *rt_annotation_context.source_table_labels,
        ],
        "source_hashes": file_hashes(source_paths),
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
            "This browser view maps fixed-mask choices, mask inputs, and RT annotations onto the "
            "reference structure. It does not evaluate candidate fold quality or RT activity."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="mask_structure_browser_manifest",
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=design_class_source_table_labels(design_class_rows)
        + [repo_relative_hint(reference_structure_path)]
        + rt_annotation_context.source_table_labels,
        input_hashes=file_hashes(source_paths),
        alt_text=(
            "Interactive Ec86 reference structure viewer with selectable design-class fixed masks, "
            "mask inputs, and RT annotation spans."
        ),
        description=(
            "Shows the Ec86/7V9U reference structure with one fixed-mask, mask-input, or RT annotation "
            "choice highlighted at a time. The base structure remains off-white so the selected residue "
            "set is visually separable."
        ),
        interpretation_limit=payload["interpretation_limit"],
        title="The Ec86 structure shows which residues each fixed-mask rule protects",
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
    design_class_rows: list[dict[str, Any]],
    rt_annotation_context: RTAnnotationContext,
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
        design_class_fixed_mask_views(
            design_class_rows=design_class_rows,
            reference_path=reference_path,
            reference_structure_format=reference_structure_format,
            manifest_root=manifest_root,
            reference_number_by_canonical=reference_number_by_canonical,
            selection_coordinate_basis=selection_coordinate_basis,
        )
    )
    views.extend(
        mask_input_evidence_views(
            design_class_rows=design_class_rows,
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
