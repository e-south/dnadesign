"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_interpretation.py

Biohub ESMC SAE feature-review panels for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .biohub_esmc_sae_activation_pattern import write_wt_activation_pattern_panel
from .biohub_esmc_sae_audit import sae_provenance_audit
from .biohub_esmc_sae_heatmap_manifest import write_feature_heatmap_manifest
from .biohub_esmc_sae_interpretation_shared import (
    INTERPRETATION_LIMIT,
    METHOD_SUMMARY,
    SECTION,
    SOURCE_NOTEBOOK,
    SOURCE_TABLES,
    TOP_FEATURE_COUNT,
    missing_row,
    remove_retired_outputs,
    top_wt_feature_rows,
)
from .biohub_esmc_sae_tables import (
    make_protein_top_feature_table_row,
    write_protein_top_feature_table,
)
from .biohub_esmc_sae_umap import write_sae_delta_umap_panel

_SOURCE_TABLES = SOURCE_TABLES


def write_biohub_esmc_sae_interpretation_panels(
    *,
    panel_root: Path,
    heatmap_root: Path,
    profile_path: Path,
    protein_features_path: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    foldcheck_ranking_path: Path,
    candidate_preference_table_path: Path,
    mask_residues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render lightweight SAE interpretation panels from existing sparse Biohub rows."""

    required_inputs = (
        profile_path,
        protein_features_path,
        residue_features_path,
        feature_catalog_path,
        request_manifest_path,
    )
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        return [missing_row(panel_root, missing)]
    feature_rows = top_wt_feature_rows(
        protein_features_path,
        residue_features_path=residue_features_path,
        top_n=TOP_FEATURE_COUNT,
    )
    if not feature_rows:
        return [missing_row(panel_root, [protein_features_path], reason="WT SAE protein feature rows are absent")]
    selected_features = [int(row["feature_index"]) for row in feature_rows]
    provenance_audit = sae_provenance_audit(
        profile_path=profile_path,
        protein_features_path=protein_features_path,
        residue_features_path=residue_features_path,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    remove_retired_outputs(panel_root)
    top_feature_table_path = panel_root / "protein_top_sae_features.parquet"
    write_protein_top_feature_table(
        path=top_feature_table_path,
        protein_features_path=protein_features_path,
        residue_features_path=residue_features_path,
        feature_catalog_path=feature_catalog_path,
    )
    return [
        make_protein_top_feature_table_row(
            table_path=top_feature_table_path,
            protein_features_path=protein_features_path,
            residue_features_path=residue_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            section=SECTION,
            source_tables=_SOURCE_TABLES,
            interpretation_limit=INTERPRETATION_LIMIT,
            method_summary=METHOD_SUMMARY,
            source_notebook=SOURCE_NOTEBOOK,
        ),
        write_wt_activation_pattern_panel(
            panel_root=panel_root,
            residue_features_path=residue_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            feature_rows=feature_rows,
            sae_provenance_audit=provenance_audit,
        ),
        write_sae_delta_umap_panel(
            panel_root=panel_root,
            profile_path=profile_path,
            protein_features_path=protein_features_path,
            request_manifest_path=request_manifest_path,
            candidate_preference_table_path=candidate_preference_table_path,
        ),
        write_feature_heatmap_manifest(
            heatmap_root=heatmap_root,
            protein_features_path=protein_features_path,
            residue_features_path=residue_features_path,
            feature_catalog_path=feature_catalog_path,
            request_manifest_path=request_manifest_path,
            foldcheck_ranking_path=foldcheck_ranking_path,
            selected_features=selected_features,
            wt_feature_rows=feature_rows,
            mask_residues=mask_residues,
            sae_provenance_audit=provenance_audit,
        ),
    ]
