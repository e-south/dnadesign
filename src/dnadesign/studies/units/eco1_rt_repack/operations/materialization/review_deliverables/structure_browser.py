"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_browser.py

Interactive candidate-fold structure-browser manifest for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .structure_browser_common import (
    CANDIDATE_LOW_CONFIDENCE_COLOR,
    CANDIDATE_PASS_COLOR,
    PROTEIN_CLASS_COLOR,
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    display_label,
    nullable_float,
    nullable_int,
    relative_path,
    repo_relative_hint,
)

STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "interactive_structure_browser_manifest.yaml"
_MUTATION_PATTERN = re.compile(r"^[A-Z](?P<position>\d+)[A-Z]$")


def write_interactive_structure_browser_manifest(
    *,
    panel_root: Path,
    full_structure_set_path: Path,
    foldcheck_ranking_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    alignment_reference_path: Path,
    candidate_table_path: Path | None = None,
    candidate_preference_table_path: Path | None = None,
) -> dict[str, Any]:
    """Write a compact manifest that lets marimo browse local fold structures."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / STRUCTURE_BROWSER_MANIFEST_FILE_NAME
    if not full_structure_set_path.exists():
        return _missing_row(manifest_path, full_structure_set_path)

    source = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_id") != "eco1_rt.foldcheck_full_structure_set":
        raise ValueError(f"Expected eco1_rt.foldcheck_full_structure_set at {full_structure_set_path}")

    ranking = _ranking_by_candidate(foldcheck_ranking_path)
    if not reference_structure_path.exists():
        raise ValueError(f"interactive structure browser reference path is missing: {reference_structure_path}")
    if not alignment_reference_path.exists():
        raise ValueError(
            f"interactive structure browser alignment reference path is missing: {alignment_reference_path}"
        )
    query_start_residue = 3
    reference_start_residue = 1
    structures = _structure_rows(
        source_rows=list(source.get("structures") or []),
        source_root=full_structure_set_path.parent,
        manifest_root=manifest_path.parent,
        ranking=ranking,
        mutations_by_candidate=_mutation_rows_by_candidate(
            candidate_table_path,
            query_start_residue=query_start_residue,
            reference_start_residue=reference_start_residue,
        ),
        esmc_scores_by_candidate=_esmc_scores_by_candidate(candidate_preference_table_path),
    )
    source_tables = [
        repo_relative_hint(full_structure_set_path),
        repo_relative_hint(foldcheck_ranking_path),
    ]
    input_hash_paths = {
        "full_structure_set": full_structure_set_path,
        "foldcheck_ranking": foldcheck_ranking_path,
        "reference_structure": reference_structure_path,
        "alignment_reference": alignment_reference_path,
    }
    if candidate_table_path is not None:
        source_tables.append(repo_relative_hint(candidate_table_path))
        input_hash_paths["candidate_table"] = candidate_table_path
    if candidate_preference_table_path is not None and candidate_preference_table_path.exists():
        source_tables.append(repo_relative_hint(candidate_preference_table_path))
        input_hash_paths["candidate_preference_table"] = candidate_preference_table_path
    payload = {
        "schema_id": "eco1_rt.interactive_structure_browser_manifest",
        "schema_version": 1,
        "status": "materialized",
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": source_tables,
        "reference": {
            "model_id": "ec86kit_7v9u_reference",
            "display_label": _reference_display_label(reference_structure_path, reference_structure_format),
            "local_path": relative_path(reference_structure_path, manifest_path.parent),
            "structure_format": reference_structure_format,
            "color": REFERENCE_COLOR,
        },
        "alignment": {
            "status": "enabled",
            "method": "mapped_ca_kabsch",
            "query_start_residue": query_start_residue,
            "reference_start_residue": reference_start_residue,
            "residue_count": 309,
            "reference_local_path": relative_path(alignment_reference_path, manifest_path.parent),
            "reference_structure_format": "pdb",
            "output_policy": "query coordinates are aligned in memory for browser viewing; local PDB files stay raw",
        },
        "structures": structures,
        "structure_count": len(structures),
        "interpretation_limit": (
            "The browser view is for interactive review. ChimeraX remains the publication-still and pose-capture path."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="interactive_structure_browser_manifest",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=[
            "foldcheck_review/foldcheck_full_structure_set.yaml",
            "foldcheck_review/foldcheck_candidate_ranking.parquet",
            repo_relative_hint(reference_structure_path),
        ],
        input_hashes=file_hashes(input_hash_paths),
        alt_text="Manifest for interactive browser review of reference-fitted local ColabFold structure models.",
        description=(
            "Lists the local reference and fold-check PDB paths used by the browser-native "
            "structure view. Query coordinates are aligned in memory "
            "over mapped C-alpha atoms before rendering; the local raw PDB files "
            "remain unchanged."
        ),
        interpretation_limit=payload["interpretation_limit"],
        title="Reference-fitted ColabFold structures can be inspected one at a time",
        role="interactive_review",
    )


def _reference_display_label(reference_structure_path: Path, reference_structure_format: str) -> str:
    if reference_structure_format == "mmcif" or "all_atom" in reference_structure_path.stem:
        return "Ec86/7V9U all-atom reference"
    return "ec86kit/7V9U reference"


def _missing_row(manifest_path: Path, missing_path: Path) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id="interactive_structure_browser_manifest",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="structure_browser_manifest",
        status="skipped_missing_input",
        path=manifest_path,
        source_tables=["foldcheck_review/foldcheck_full_structure_set.yaml"],
        input_hashes={},
        alt_text="Interactive structure browser manifest was not generated.",
        description="Interactive structure browsing is skipped until the local fold structure set exists.",
        interpretation_limit="Missing structure paths cannot support interactive structure review.",
        title="Interactive structure browser is skipped until local PDBs are available",
        role="interactive_review",
        skip_reason=f"Missing input manifest: {missing_path}",
    )


def _ranking_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "candidate_id",
            "review_rank",
            "review_class",
            "plddt",
            "wt_runtime_ca_rmsd",
            "cryoem_mapped_ca_rmsd",
            "seq_recovery",
            "mutation_count",
        ],
    )
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _mutation_rows_by_candidate(
    path: Path | None,
    *,
    query_start_residue: int,
    reference_start_residue: int,
) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    offset = int(query_start_residue) - int(reference_start_residue)
    rows = pq.read_table(path, columns=["candidate_id", "canonical_mutations"]).to_pylist()
    mutations_by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        canonical_mutations = [str(mutation) for mutation in row.get("canonical_mutations") or []]
        canonical_positions: list[int] = []
        for mutation in canonical_mutations:
            match = _MUTATION_PATTERN.match(mutation)
            if not match:
                raise ValueError(f"Malformed canonical mutation for {candidate_id}: {mutation!r}")
            canonical_positions.append(int(match.group("position")))
        mutations_by_candidate[candidate_id] = {
            "canonical_mutations": canonical_mutations,
            "canonical_positions": sorted(set(canonical_positions)),
            "query_residue_numbers": sorted({position + offset for position in canonical_positions}),
        }
    return mutations_by_candidate


def _structure_rows(
    *,
    source_rows: list[Any],
    source_root: Path,
    manifest_root: Path,
    ranking: dict[str, dict[str, Any]],
    mutations_by_candidate: dict[str, dict[str, Any]],
    esmc_scores_by_candidate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(source_rows):
        if not isinstance(row, dict):
            raise ValueError(f"full structure-set row {row_index} is not a mapping")
        candidate_id = str(row.get("candidate_id") or "")
        local_value = str(row.get("local_model_artifact_path") or "")
        local_path = source_root / local_value
        if not candidate_id:
            raise ValueError(f"full structure-set row {row_index} is missing candidate_id")
        if not local_value:
            raise ValueError(f"full structure-set row {row_index} is missing local_model_artifact_path")
        if not local_path.exists():
            raise ValueError(f"declared structure path is missing for {candidate_id}: {local_path}")
        rank_row = ranking.get(candidate_id, {})
        mutation_payload = mutations_by_candidate.get(candidate_id, {})
        esmc_payload = esmc_scores_by_candidate.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "display_label": display_label(candidate_id, row),
                "group": _structure_group(candidate_id, rank_row),
                "local_path": relative_path(local_path, manifest_root),
                "structure_format": "pdb",
                "color": _structure_color(candidate_id, rank_row),
                "review_rank": nullable_int(rank_row.get("review_rank")),
                "review_class": str(rank_row.get("review_class") or ""),
                "plddt": nullable_float(rank_row.get("plddt")),
                "wt_runtime_ca_rmsd": nullable_float(
                    row.get("wt_runtime_ca_rmsd") or rank_row.get("wt_runtime_ca_rmsd")
                ),
                "cryoem_mapped_ca_rmsd": nullable_float(rank_row.get("cryoem_mapped_ca_rmsd")),
                "sequence_identity_percent": nullable_float(row.get("sequence_identity_percent")),
                "mutation_count": nullable_int(rank_row.get("mutation_count"))
                or len(mutation_payload.get("canonical_positions") or []),
                "canonical_mutations": mutation_payload.get("canonical_mutations", []),
                "mutation_residue_numbers": mutation_payload.get("query_residue_numbers", []),
                "esmc_llr_total": nullable_float(esmc_payload.get("llr_total")),
                "esmc_llr_per_mutation": nullable_float(esmc_payload.get("llr_per_mutation")),
                "esmc_mutations_scored_count": nullable_int(esmc_payload.get("mutations_scored_count")),
                "esmc_scoring_method_id": str(esmc_payload.get("scoring_method_id") or ""),
            }
        )
    return sorted(rows, key=_structure_sort_key)


def _esmc_scores_by_candidate(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "candidate_id",
            "scoring_method_id",
            "mutation_count",
            "mutations_scored_count",
            "llr_total",
            "llr_per_mutation",
            "status",
        ],
    )
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _structure_group(candidate_id: str, rank_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "0 WT ColabFold baseline"
    review_class = str(rank_row.get("review_class") or "")
    if review_class in {"strong_fold_preserved", "good_fold_preserved"}:
        return "1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)"
    if review_class in {"structural_outlier", "low_confidence"}:
        return "3 Hold for review (CA RMSD > 5.0 A or pLDDT < 90)"
    return "2 Intermediate fold review band"


def _structure_color(candidate_id: str, rank_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return PROTEIN_CLASS_COLOR
    review_class = str(rank_row.get("review_class") or "")
    if review_class == "structural_outlier":
        return RESIDUE_CATEGORY_HIGHLIGHT_COLOR
    if review_class == "low_confidence":
        return CANDIDATE_LOW_CONFIDENCE_COLOR
    return CANDIDATE_PASS_COLOR


def _structure_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    if row["candidate_id"] == "wild_type":
        return (0, 0, str(row["candidate_id"]))
    rank = row.get("review_rank")
    return (1, int(rank) if rank is not None else 10_000, str(row["candidate_id"]))
