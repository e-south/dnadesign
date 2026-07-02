"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_heatmap_manifest.py

Selected-feature Biohub ESMC SAE heatmap manifest materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .biohub_esmc_sae_interpretation_shared import (
    FEATURE_PREVALENCE_THRESHOLD,
    INTERPRETATION_LIMIT,
    METHOD_SUMMARY,
    SECTION,
    SOURCE_TABLES,
    evidence_summary,
    feature_axis_labels,
    relative_to,
    wt_sequence_from_mask,
)


def write_feature_heatmap_manifest(
    *,
    heatmap_root: Path,
    protein_features_path: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    foldcheck_ranking_path: Path,
    selected_features: list[int],
    wt_feature_rows: list[dict[str, Any]],
    mask_residues: list[dict[str, Any]],
    sae_provenance_audit: dict[str, Any],
) -> dict[str, Any]:
    title = "Selected SAE feature activation across Eco1 RT variants"
    heatmap_root.mkdir(parents=True, exist_ok=True)
    path = heatmap_root / "sae_feature_heatmap_manifest.yaml"
    candidate_order = _candidate_order(protein_features_path, foldcheck_ranking_path)
    wt_sequence = wt_sequence_from_mask(mask_residues)
    feature_labels = feature_axis_labels(feature_catalog_path, selected_features)
    feature_activation_rows = {
        int(row["feature_index"]): {
            "feature_index": int(row["feature_index"]),
            "label": label,
            "wt_activation_max": float(row["activation_max"]),
            "wt_activation_sum": float(row["activation_sum"]),
            "wt_nonzero_residue_count": int(row["nonzero_residue_count"]),
            "wt_prevalence_activation_threshold": FEATURE_PREVALENCE_THRESHOLD,
            "wt_prevalent_residue_count": int(row.get("prevalent_residue_count", 0)),
            "wt_mean_prevalent_activation": float(row.get("mean_prevalent_activation", 0.0)),
        }
        for row, label in zip(wt_feature_rows, feature_labels, strict=True)
    }
    payload = {
        "schema_id": "eco1_rt.biohub_esmc_sae_feature_heatmap",
        "schema_version": 1,
        "status": "materialized",
        "path_policy": "paths_relative_to_this_manifest",
        "candidate_order": candidate_order,
        "candidate_count": len(candidate_order),
        "sequence_length": len(wt_sequence),
        "wt_sequence": wt_sequence,
        "features": [feature_activation_rows[feature] for feature in selected_features],
        "feature_count": len(selected_features),
        "feature_selection_policy": (
            "top WT features by activation_max, tie-broken by >0.01 activation prevalence and activation_sum"
        ),
        "residue_features_path": relative_to(path.parent, residue_features_path),
        "protein_features_path": relative_to(path.parent, protein_features_path),
        "feature_catalog_path": relative_to(path.parent, feature_catalog_path),
        "request_manifest_path": relative_to(path.parent, request_manifest_path),
        "sae_provenance_audit": sae_provenance_audit,
        "source_tables": SOURCE_TABLES,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    alt = (
        f"Interactive heatmap for {len(selected_features)} WT-active SAE features across "
        f"{len(candidate_order)} Biohub ESMC query sequences and {len(wt_sequence)} Ec86 residue positions."
    )
    return make_deliverable_row(
        deliverable_id="biohub_esmc_sae_feature_activation_heatmap",
        section=SECTION,
        artifact_kind="sae_feature_heatmap_manifest",
        status="rendered",
        path=path,
        source_tables=[*SOURCE_TABLES, "mask_set.yaml", "foldcheck_review/foldcheck_candidate_ranking.parquet"],
        input_hashes=file_hashes(
            {
                "residue_features": residue_features_path,
                "protein_features": protein_features_path,
                "feature_catalog": feature_catalog_path,
                "request_manifest": request_manifest_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
            }
        ),
        alt_text=alt,
        description=(
            "Lets the notebook render one selected WT-active SAE feature at a time. Rows are WT plus "
            "ProteinMPNN variants ordered by fold-review ranking, columns are Ec86 canonical positions, "
            "top tick labels are WT residue letters, and color is the per-residue SAE activation value."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary=evidence_summary(
            wt_feature_rows,
            request_manifest_path=request_manifest_path,
            sae_provenance_audit=sae_provenance_audit,
        )
        | {"sequence_rows": len(candidate_order), "sequence_length": len(wt_sequence)},
        role="manuscript_facing",
    )


def _candidate_order(protein_features_path: Path, foldcheck_ranking_path: Path) -> list[str]:
    candidates = sorted(
        {
            str(row["candidate_id"])
            for row in pq.read_table(protein_features_path, columns=["candidate_id"]).to_pylist()
            if str(row["candidate_id"])
        }
    )
    ordered = ["wild_type"] if "wild_type" in candidates else []
    if foldcheck_ranking_path.exists():
        ranking_rows = pq.read_table(
            foldcheck_ranking_path,
            columns=["candidate_id", "wt_runtime_ca_rmsd", "plddt"],
        ).to_pylist()
        ranked = sorted(
            [row for row in ranking_rows if str(row.get("candidate_id")) in candidates],
            key=lambda row: (float(row.get("wt_runtime_ca_rmsd") or 1e12), -float(row.get("plddt") or 0.0)),
        )
        ordered.extend(str(row["candidate_id"]) for row in ranked if str(row["candidate_id"]) not in ordered)
    ordered.extend(candidate for candidate in candidates if candidate not in ordered)
    return ordered
