"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_browser.py

Interactive structure-browser manifest for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "interactive_structure_browser_manifest.yaml"
REFERENCE_STRUCTURE_RELATIVE_PATH = "structures/ec86kit_chain_a_backbone_reference.pdb"


def write_interactive_structure_browser_manifest(
    *,
    panel_root: Path,
    full_structure_set_path: Path,
    foldcheck_ranking_path: Path,
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
    reference_path = full_structure_set_path.parent / REFERENCE_STRUCTURE_RELATIVE_PATH
    structures = _structure_rows(
        source_rows=list(source.get("structures") or []),
        source_root=full_structure_set_path.parent,
        manifest_root=manifest_path.parent,
        ranking=ranking,
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
            _repo_relative_hint(full_structure_set_path),
            _repo_relative_hint(foldcheck_ranking_path),
        ],
        "reference": {
            "model_id": "ec86kit_7v9u_reference",
            "display_label": "ec86kit/7V9U reference",
            "local_path": _relative_path(reference_path, manifest_path.parent),
            "color": "#d8d8d8",
        },
        "structures": structures,
        "structure_count": len(structures),
        "interpretation_limit": (
            "The browser viewer is for interactive review. ChimeraX remains the "
            "publication-still and pose-capture path."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id="interactive_structure_browser_manifest",
        section="fold_review",
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=[
            "foldcheck_review/foldcheck_full_structure_set.yaml",
            "foldcheck_review/foldcheck_candidate_ranking.parquet",
        ],
        input_hashes=file_hashes(
            {
                "full_structure_set": full_structure_set_path,
                "foldcheck_ranking": foldcheck_ranking_path,
            }
        ),
        alt_text="Manifest for interactive browser review of local ColabFold structure models.",
        description=(
            "Lists the local reference and fold-check PDB paths used by the marimo "
            "interactive structure viewer without duplicating structure files."
        ),
        interpretation_limit=payload["interpretation_limit"],
        title="Interactive structure browser uses local ColabFold PDBs without copying them",
        role="interactive_review",
    )


def _missing_row(manifest_path: Path, missing_path: Path) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id="interactive_structure_browser_manifest",
        section="fold_review",
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


def _structure_rows(
    *,
    source_rows: list[Any],
    source_root: Path,
    manifest_root: Path,
    ranking: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        local_value = str(row.get("local_model_artifact_path") or "")
        local_path = source_root / local_value
        if not candidate_id or not local_value or not local_path.exists():
            continue
        rank_row = ranking.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "display_label": _display_label(candidate_id, row),
                "group": _structure_group(candidate_id, rank_row),
                "local_path": _relative_path(local_path, manifest_root),
                "color": _structure_color(candidate_id, rank_row),
                "review_rank": _nullable_int(rank_row.get("review_rank")),
                "review_class": str(rank_row.get("review_class") or ""),
                "plddt": _nullable_float(rank_row.get("plddt")),
                "wt_runtime_ca_rmsd": _nullable_float(
                    row.get("wt_runtime_ca_rmsd") or rank_row.get("wt_runtime_ca_rmsd")
                ),
                "cryoem_mapped_ca_rmsd": _nullable_float(rank_row.get("cryoem_mapped_ca_rmsd")),
                "sequence_identity_percent": _nullable_float(row.get("sequence_identity_percent")),
                "mutation_count": _nullable_int(rank_row.get("mutation_count")),
            }
        )
    return sorted(rows, key=_structure_sort_key)


def _display_label(candidate_id: str, row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "WT ColabFold baseline"
    label = str(row.get("display_label") or "")
    if label:
        return label
    return f"ProteinMPNN variant {candidate_id.removeprefix('thread_candidate_')[:12]}"


def _structure_group(candidate_id: str, rank_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "WT baseline"
    review_class = str(rank_row.get("review_class") or "")
    if review_class in {"strong_fold_preserved", "good_fold_preserved"}:
        return "Fold-preserved candidates"
    if review_class in {"structural_outlier", "low_confidence"}:
        return "Review outliers"
    return "Other fold-accepted candidates"


def _structure_color(candidate_id: str, rank_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "#0072B2"
    review_class = str(rank_row.get("review_class") or "")
    if review_class == "structural_outlier":
        return "#D55E00"
    if review_class == "low_confidence":
        return "#CC79A7"
    return "#009E73"


def _structure_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    if row["candidate_id"] == "wild_type":
        return (0, 0, str(row["candidate_id"]))
    rank = row.get("review_rank")
    return (1, int(rank) if rank is not None else 10_000, str(row["candidate_id"]))


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _repo_relative_hint(path: Path) -> str:
    if path.parent.name == "foldcheck_review":
        return str(Path("foldcheck_review") / path.name)
    return path.name


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
