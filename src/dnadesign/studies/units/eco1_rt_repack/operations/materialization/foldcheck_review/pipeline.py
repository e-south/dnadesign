"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/pipeline.py

Materialize Eco1 fold-check review artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.chimerax import (
    write_chimerax_script,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    ATLAS_SUBSET_FILE_NAME,
    ATLAS_SUBSET_SCHEMA_ID,
    CANDIDATE_TABLE_FILE_NAME,
    CHIMERAX_DIR_NAME,
    CHIMERAX_SCRIPT_NAME,
    DEFAULT_OUTPUT_ROOT,
    FOLDCHECK_REPORT_FILE_NAME,
    FOLDCHECK_REQUEST_MANIFEST_RELATIVE_PATH,
    FULL_CHIMERAX_SCRIPT_NAME,
    FULL_STRUCTURE_SET_DIR_NAME,
    FULL_STRUCTURE_SET_FILE_NAME,
    RANKING_FILE_NAME,
    REFERENCE_BACKBONE_RELATIVE_PATH,
    RESIDUE_MAP_FILE_NAME,
    REVIEW_DIR_NAME,
    STRUCTURE_PANEL_FILE_NAME,
    STRUCTURES_DIR_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.models import (
    MaterializedFoldCheckReviewArtifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.ranking import (
    build_foldcheck_ranking_rows,
    wild_type_reference_row,
    write_foldcheck_ranking,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.selection import (
    build_atlas_subset_rows,
    select_structure_panel_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.structures import (
    stage_full_structure_set,
    stage_structure_panel,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.visuals import (
    write_review_visuals,
)


def materialize_foldcheck_review(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    render_chimerax_overlay: bool = False,
) -> MaterializedFoldCheckReviewArtifacts:
    """Materialize ranking, structure-panel, ChimeraX, and Atlas-subset review artifacts."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    review_root = out_root / REVIEW_DIR_NAME
    review_root.mkdir(parents=True, exist_ok=True)

    request_manifest_path = out_root / FOLDCHECK_REQUEST_MANIFEST_RELATIVE_PATH
    candidate_table_path = out_root / CANDIDATE_TABLE_FILE_NAME
    foldcheck_report_path = out_root / FOLDCHECK_REPORT_FILE_NAME
    residue_map_path = out_root / RESIDUE_MAP_FILE_NAME
    reference_backbone_path = out_root / REFERENCE_BACKBONE_RELATIVE_PATH
    for required_path in (
        request_manifest_path,
        candidate_table_path,
        foldcheck_report_path,
        residue_map_path,
        reference_backbone_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    request_manifest = _load_yaml(request_manifest_path)
    source_request_hash = str(request_manifest["request_hash"])
    structures_root = review_root / STRUCTURES_DIR_NAME
    full_structure_root = structures_root / FULL_STRUCTURE_SET_DIR_NAME
    ranking_rows = build_foldcheck_ranking_rows(
        candidate_table_path=candidate_table_path,
        foldcheck_report_path=foldcheck_report_path,
        residue_map_path=residue_map_path,
        reference_backbone_path=reference_backbone_path,
        local_model_root=full_structure_root,
    )
    ranking_path = review_root / RANKING_FILE_NAME
    write_foldcheck_ranking(ranking_path, ranking_rows, source_request_hash=source_request_hash)

    wt_fold_row = wild_type_reference_row(foldcheck_report_path)
    full_structure_set_path = review_root / FULL_STRUCTURE_SET_FILE_NAME
    full_entries = stage_full_structure_set(
        structures_root=full_structure_root,
        reference_backbone_path=reference_backbone_path,
        wt_fold_row=wt_fold_row,
        ranking_rows=ranking_rows,
        full_structure_set_path=full_structure_set_path,
        source_request_hash=source_request_hash,
    )
    full_chimerax_script_path = review_root / CHIMERAX_DIR_NAME / FULL_CHIMERAX_SCRIPT_NAME
    write_chimerax_script(
        path=full_chimerax_script_path,
        reference_local_path=structures_root / "ec86kit_chain_a_backbone_reference.pdb",
        entries=full_entries,
    )

    selected_rows = select_structure_panel_rows(ranking_rows)
    structure_panel_path = review_root / STRUCTURE_PANEL_FILE_NAME
    entries = stage_structure_panel(
        structures_root=structures_root,
        reference_backbone_path=reference_backbone_path,
        wt_fold_row=wt_fold_row,
        selected_rows=selected_rows,
        structure_panel_path=structure_panel_path,
        source_request_hash=source_request_hash,
        fallback_model_root=full_structure_root,
    )

    atlas_subset_manifest_path = review_root / ATLAS_SUBSET_FILE_NAME
    _write_atlas_subset_manifest(
        atlas_subset_manifest_path,
        source_request_hash=source_request_hash,
        subset_rows=build_atlas_subset_rows(selected_rows),
    )

    chimerax_script_path = review_root / CHIMERAX_DIR_NAME / CHIMERAX_SCRIPT_NAME
    write_chimerax_script(
        path=chimerax_script_path,
        reference_local_path=structures_root / "ec86kit_chain_a_backbone_reference.pdb",
        entries=entries,
    )
    visual_manifest_path, notebook_path, plot_count = write_review_visuals(
        review_root=review_root,
        output_root=out_root,
        ranking_rows=ranking_rows,
        reference_local_path=structures_root / "ec86kit_chain_a_backbone_reference.pdb",
        panel_entries=entries,
        source_request_hash=source_request_hash,
        render_chimerax_overlay=render_chimerax_overlay,
    )
    return MaterializedFoldCheckReviewArtifacts(
        ranking_path=ranking_path,
        structure_panel_path=structure_panel_path,
        full_structure_set_path=full_structure_set_path,
        atlas_subset_manifest_path=atlas_subset_manifest_path,
        chimerax_script_path=chimerax_script_path,
        full_chimerax_script_path=full_chimerax_script_path,
        visual_manifest_path=visual_manifest_path,
        notebook_path=notebook_path,
        selected_structure_count=len(entries),
        full_structure_count=len(full_entries),
        plot_count=plot_count,
    )


def _write_atlas_subset_manifest(
    path: Path,
    *,
    source_request_hash: str,
    subset_rows: list[dict[str, str]],
) -> None:
    manifest: dict[str, Any] = {
        "schema_id": ATLAS_SUBSET_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "source_request_hash": source_request_hash,
        "selected_sequence_ids": [row["sequence_id"] for row in subset_rows],
        "selection_rows": subset_rows,
        "atlas_policy": {
            "candidate_acceptance_gate": False,
            "default_fold_on_miss": False,
            "allowed_use": "semantic audit and assay-panel stratification only",
            "forbidden_use": "processivity, strand-displacement, or hairpin-readthrough acceptance",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
