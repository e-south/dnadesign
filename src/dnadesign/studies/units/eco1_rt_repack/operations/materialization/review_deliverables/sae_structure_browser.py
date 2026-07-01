"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/sae_structure_browser.py

Interactive SAE activation structure-browser manifest for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_ESMC_FEATURE_REVIEW,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .structure_browser_common import (
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

SAE_STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "biohub_esmc_sae_structure_browser_manifest.yaml"
_SAE_HIGHLIGHT_COLOR = RESIDUE_CATEGORY_HIGHLIGHT_COLOR
_SAE_STRUCTURE_FEATURES_PER_PROTEIN = 10


def write_sae_structure_browser_manifest(
    *,
    panel_root: Path,
    top_feature_table_path: Path,
    residue_features_path: Path,
    full_structure_set_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    alignment_reference_path: Path,
) -> dict[str, Any]:
    """Write an interactive structure-browser manifest for SAE activation regions."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / SAE_STRUCTURE_BROWSER_MANIFEST_FILE_NAME
    required_paths = (top_feature_table_path, residue_features_path, full_structure_set_path)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        return _missing_sae_structure_row(manifest_path, missing)

    source = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_id") != "eco1_rt.foldcheck_full_structure_set":
        raise ValueError(f"Expected eco1_rt.foldcheck_full_structure_set at {full_structure_set_path}")

    if not reference_structure_path.exists():
        raise ValueError(f"SAE structure browser reference path is missing: {reference_structure_path}")
    if not alignment_reference_path.exists():
        raise ValueError(f"SAE structure browser alignment reference path is missing: {alignment_reference_path}")
    structure_paths = _structure_path_by_candidate(
        source_rows=list(source.get("structures") or []),
        source_root=full_structure_set_path.parent,
    )
    top_feature_rows = _sae_structure_feature_rows(
        pq.read_table(top_feature_table_path).to_pylist(),
        per_protein=_SAE_STRUCTURE_FEATURES_PER_PROTEIN,
    )
    activation_rows = _sae_activation_rows(
        top_feature_rows=top_feature_rows,
        residue_features_path=residue_features_path,
        structure_paths=structure_paths,
        reference_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        manifest_root=manifest_path.parent,
        query_start_residue=3,
        reference_start_residue=1,
    )
    if not activation_rows:
        return _missing_sae_structure_row(
            manifest_path,
            [top_feature_table_path, residue_features_path],
            reason="No SAE feature rows with mapped residue activations and local structures were available.",
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
            repo_relative_hint(top_feature_table_path),
            repo_relative_hint(residue_features_path),
            repo_relative_hint(full_structure_set_path),
            repo_relative_hint(reference_structure_path),
        ],
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
            "query_start_residue": 3,
            "reference_start_residue": 1,
            "residue_count": 309,
            "reference_local_path": relative_path(alignment_reference_path, manifest_path.parent),
            "reference_structure_format": "pdb",
            "output_policy": "query coordinates are aligned in memory for browser viewing; local PDB files stay raw",
        },
        "control_label": "SAE activation feature",
        "feature_selection_policy": {
            "rank_basis": "per_protein_rank_by_max_activation",
            "features_per_protein": _SAE_STRUCTURE_FEATURES_PER_PROTEIN,
            "scope": "interactive_structure_browser",
        },
        "structures": activation_rows,
        "structure_count": len(activation_rows),
        "interpretation_limit": (
            "This browser maps Biohub ESMC SAE activation regions onto available structures. It supports "
            "semantic review, not activity, processivity, strand-displacement, or acceptance claims."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="biohub_esmc_sae_structure_browser_manifest",
        section=SECTION_ESMC_FEATURE_REVIEW,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=[
            "review_deliverables/biohub_esmc_sae_interpretation/protein_top_sae_features.parquet",
            "biohub_esmc_residue_features.parquet",
            "foldcheck_review/foldcheck_full_structure_set.yaml",
        ],
        input_hashes=file_hashes(
            {
                "top_feature_table": top_feature_table_path,
                "residue_features": residue_features_path,
                "full_structure_set": full_structure_set_path,
                "reference_structure": reference_structure_path,
                "alignment_reference": alignment_reference_path,
            }
        ),
        alt_text="Interactive structure browser for Biohub ESMC SAE feature activation regions.",
        description=(
            "Maps the highest peak-ranked per-protein SAE activation regions onto the Ec86 reference or fitted "
            "ProteinMPNN candidate structures. The viewer keeps activation highlights separate from "
            "fold-quality and activity interpretation."
        ),
        interpretation_limit=payload["interpretation_limit"],
        title="Biohub ESMC SAE activation regions can be inspected on structure",
        role="interactive_review",
    )


def _reference_display_label(reference_structure_path: Path, reference_structure_format: str) -> str:
    if reference_structure_format == "mmcif" or "all_atom" in reference_structure_path.stem:
        return "Ec86/7V9U all-atom reference"
    return "ec86kit/7V9U reference"


def _missing_sae_structure_row(
    manifest_path: Path,
    missing: list[Path],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    message = reason or "Missing SAE structure-browser input: " + ", ".join(str(path) for path in missing)
    return make_deliverable_row(
        deliverable_id="biohub_esmc_sae_structure_browser_manifest",
        section=SECTION_ESMC_FEATURE_REVIEW,
        artifact_kind="structure_browser_manifest",
        status="skipped_missing_input",
        path=manifest_path,
        source_tables=[
            "review_deliverables/biohub_esmc_sae_interpretation/protein_top_sae_features.parquet",
            "biohub_esmc_residue_features.parquet",
            "foldcheck_review/foldcheck_full_structure_set.yaml",
        ],
        input_hashes=file_hashes({f"input_{index}": path for index, path in enumerate(missing)}),
        alt_text="Interactive SAE activation structure browser was not generated.",
        description="SAE activation structure browsing is skipped until feature rows and local structures exist.",
        interpretation_limit="Missing structure or activation inputs cannot support SAE structure review.",
        title="Biohub ESMC SAE activation structure browser is skipped until inputs are available",
        role="interactive_review",
        skip_reason=message,
    )


def _structure_path_by_candidate(*, source_rows: list[Any], source_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
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
        paths[candidate_id] = local_path
    return paths


def _sae_activation_rows(
    *,
    top_feature_rows: list[dict[str, Any]],
    residue_features_path: Path,
    structure_paths: dict[str, Path],
    reference_path: Path,
    reference_structure_format: str,
    manifest_root: Path,
    query_start_residue: int,
    reference_start_residue: int,
) -> list[dict[str, Any]]:
    selected_rows = [
        dict(row)
        for row in top_feature_rows
        if str(row.get("candidate_id") or "") in structure_paths and row.get("feature_index") is not None
    ]
    if not selected_rows:
        return []
    candidate_ids = sorted({str(row["candidate_id"]) for row in selected_rows})
    feature_indices = sorted({int(row["feature_index"]) for row in selected_rows})
    residue_rows = pq.read_table(
        residue_features_path,
        filters=[
            ("candidate_id", "in", candidate_ids),
            ("feature_index", "in", feature_indices),
        ],
        columns=["candidate_id", "sequence_position_one_based", "feature_index", "value"],
    ).to_pylist()
    positions_by_pair: dict[tuple[str, int], list[int]] = {}
    for row in residue_rows:
        if float(row.get("value") or 0.0) <= 0.0:
            continue
        key = (str(row["candidate_id"]), int(row["feature_index"]))
        positions_by_pair.setdefault(key, []).append(int(row["sequence_position_one_based"]))
    query_offset = int(query_start_residue) - int(reference_start_residue)
    rows: list[dict[str, Any]] = []
    for top_row in sorted(selected_rows, key=_sae_top_row_sort_key):
        candidate_id = str(top_row["candidate_id"])
        feature_index = int(top_row["feature_index"])
        sequence_positions = sorted(set(positions_by_pair.get((candidate_id, feature_index), [])))
        if not sequence_positions:
            continue
        is_reference = candidate_id == "wild_type"
        model_id = "ec86kit_7v9u_reference" if is_reference else candidate_id
        residue_numbers = (
            sequence_positions if is_reference else [position + query_offset for position in sequence_positions]
        )
        local_path = reference_path if is_reference else structure_paths[candidate_id]
        rows.append(
            {
                "candidate_id": f"{candidate_id}__sae_feature_{feature_index}",
                "source_candidate_id": candidate_id,
                "display_label": _sae_structure_label(top_row),
                "group": "WT/reference SAE activations" if is_reference else "ProteinMPNN variant SAE activations",
                "local_path": relative_path(local_path, manifest_root),
                "structure_format": reference_structure_format if is_reference else "pdb",
                "color": PROTEIN_CLASS_COLOR if is_reference else CANDIDATE_PASS_COLOR,
                "structure_view_mode": "reference_selection" if is_reference else "sae_activation",
                "description": _sae_structure_description(top_row),
                "selection_styles": [
                    {
                        "selection_id": f"sae_feature_{feature_index}",
                        "model_id": model_id,
                        "label": "SAE activation region",
                        "source_coordinate_basis": "sequence_position_one_based",
                        "selection_coordinate_basis": (
                            "reference_residue_number" if is_reference else "query_pdb_residue_number"
                        ),
                        "sequence_positions": sequence_positions,
                        "residue_numbers": residue_numbers,
                        "color": _SAE_HIGHLIGHT_COLOR,
                    }
                ],
                "selection_residue_count": len(residue_numbers),
                "feature_index": feature_index,
                "activation_max": nullable_float(top_row.get("activation_max")),
                "activation_sum": nullable_float(top_row.get("activation_sum")),
                "nonzero_residue_count": nullable_int(top_row.get("nonzero_residue_count")),
            }
        )
    return rows


def _sae_structure_feature_rows(rows: list[dict[str, Any]], *, per_protein: int) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    for candidate_id in sorted(by_candidate, key=lambda value: (value != "wild_type", value)):
        selected.extend(sorted(by_candidate[candidate_id], key=_sae_feature_rank_key)[:per_protein])
    return selected


def _sae_feature_rank_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
    max_rank = row.get("rank_by_max_activation")
    prevalence_rank = row.get("rank_by_prevalence")
    return (
        int(max_rank) if max_rank is not None else 10_000,
        int(prevalence_rank) if prevalence_rank is not None else 10_000,
        -float(row.get("activation_max") or 0.0),
        int(row.get("feature_index") or 10_000_000),
    )


def _sae_top_row_sort_key(row: dict[str, Any]) -> tuple[int, str, int, int]:
    candidate_id = str(row.get("candidate_id") or "")
    rank = row.get("rank_by_max_activation")
    return (
        0 if candidate_id == "wild_type" else 1,
        candidate_id,
        int(rank) if rank is not None else 10_000,
        int(row.get("feature_index") or 10_000_000),
    )


def _sae_structure_label(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    prefix = "WT Ec86" if candidate_id == "wild_type" else display_label(candidate_id, {})
    feature_index = int(row["feature_index"])
    rank = row.get("rank_by_max_activation") or "-"
    return f"{prefix} F{feature_index} | peak rank {rank}"


def _sae_structure_description(row: dict[str, Any]) -> str:
    feature_index = int(row["feature_index"])
    description = str(row.get("description") or "").strip()
    if not description:
        description = "No source-backed description is available for this exact SAE dictionary."
    else:
        description = _concise_sae_description(description)
    return (
        f"Candidate SAE activation view for feature F{feature_index}. {description} "
        "Residues are highlighted where the sparse activation is nonzero for the selected protein."
    )


def _concise_sae_description(description: str, *, max_chars: int = 260) -> str:
    source = " ".join(description.split())
    if source.lower().startswith("summary:"):
        source = source.split(":", 1)[1].strip()
    for delimiter in (" Activation pattern:", " Exemplars:", " Caveats:", " Strongest examples:"):
        if delimiter in source:
            source = source.split(delimiter, 1)[0].strip()
    first_sentence = source.split(". ", 1)[0].strip().rstrip(".")
    if len(first_sentence) <= max_chars:
        return first_sentence + "."
    return first_sentence[: max_chars - 1].rstrip() + "."
