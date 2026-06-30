"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/ranking.py

Candidate ranking for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    FOLDCHECK_RANKING_SCHEMA_ID,
    WT_SEQUENCE_ID,
)
from dnadesign.thread.adapters.colabfold.metrics import ca_coordinates, ca_rmsd

_RANKING_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("source_sample_id", pa.string()),
        ("proteinmpnn_rank", pa.int64()),
        ("proteinmpnn_score", pa.float64()),
        ("proteinmpnn_global_score", pa.float64()),
        ("seq_recovery", pa.float64()),
        ("seed", pa.int64()),
        ("temperature", pa.float64()),
        ("sample_index", pa.int64()),
        ("mutation_count", pa.int64()),
        ("mutable_mutation_count", pa.int64()),
        ("foldcheck_status", pa.string()),
        ("plddt", pa.float64()),
        ("pae_mean", pa.float64()),
        ("pae_max", pa.float64()),
        ("wt_runtime_ca_rmsd", pa.float64()),
        ("cryoem_mapped_ca_rmsd", pa.float64()),
        ("cryoem_mapped_ca_rmsd_status", pa.string()),
        ("review_class", pa.string()),
        ("review_rank", pa.int64()),
        ("model_artifact_path", pa.string()),
        ("score_artifact_path", pa.string()),
    ]
)


def build_foldcheck_ranking_rows(
    *,
    candidate_table_path: Path,
    foldcheck_report_path: Path,
    residue_map_path: Path,
    reference_backbone_path: Path,
    local_model_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Join candidate and fold-check metrics into review-ranking rows."""

    candidate_rows = [
        row for row in pq.read_table(candidate_table_path).to_pylist() if str(row.get("status")) == "accepted"
    ]
    fold_rows = _fold_rows_by_candidate_id(foldcheck_report_path)
    mapped_positions = _mapped_canonical_positions(residue_map_path)
    reference_coords = _reference_coordinates(reference_backbone_path, mapped_position_count=len(mapped_positions))

    ranking_rows: list[dict[str, Any]] = []
    for candidate in sorted(candidate_rows, key=lambda row: (int(row["rank"]), str(row["candidate_id"]))):
        candidate_id = str(candidate["candidate_id"])
        fold = fold_rows.get(candidate_id)
        if fold is None:
            raise ValueError(f"foldcheck_report.parquet is missing candidate {candidate_id!r}")
        cryoem_rmsd, cryoem_status = _cryoem_mapped_rmsd(
            candidate_id=candidate_id,
            model_artifact_path=Path(str(fold.get("model_artifact_path", ""))),
            mapped_positions=mapped_positions,
            reference_coords=reference_coords,
            local_model_root=local_model_root,
        )
        row = {
            "candidate_id": candidate_id,
            "source_sample_id": str(candidate.get("source_sample_id", "")),
            "proteinmpnn_rank": int(candidate["rank"]),
            "proteinmpnn_score": float(candidate["score"]),
            "proteinmpnn_global_score": float(candidate["global_score"]),
            "seq_recovery": float(candidate["seq_recovery"]),
            "seed": int(candidate["seed"]),
            "temperature": float(candidate["temperature"]),
            "sample_index": int(candidate["sample_index"]),
            "mutation_count": int(candidate["mutation_count"]),
            "mutable_mutation_count": int(candidate["mutable_mutation_count"]),
            "foldcheck_status": str(fold.get("status", "")),
            "plddt": _optional_float(fold.get("plddt")),
            "pae_mean": _pae_value(fold.get("pae_summary"), "mean"),
            "pae_max": _pae_value(fold.get("pae_summary"), "max"),
            "wt_runtime_ca_rmsd": _optional_float(fold.get("backbone_rmsd_to_reference")),
            "cryoem_mapped_ca_rmsd": cryoem_rmsd,
            "cryoem_mapped_ca_rmsd_status": cryoem_status,
            "review_class": _review_class(
                plddt=_optional_float(fold.get("plddt")),
                wt_runtime_ca_rmsd=_optional_float(fold.get("backbone_rmsd_to_reference")),
            ),
            "review_rank": 0,
            "model_artifact_path": str(fold.get("model_artifact_path", "")),
            "score_artifact_path": str(fold.get("score_artifact_path", "")),
        }
        ranking_rows.append(row)
    ranking_rows.sort(
        key=lambda row: (
            str(row["foldcheck_status"]) != "accepted",
            float(row["wt_runtime_ca_rmsd"]) if row["wt_runtime_ca_rmsd"] is not None else 9999.0,
            -float(row["plddt"]) if row["plddt"] is not None else 9999.0,
            int(row["proteinmpnn_rank"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(ranking_rows, start=1):
        row["review_rank"] = index
    return ranking_rows


def write_foldcheck_ranking(path: Path, rows: list[dict[str, Any]], *, source_request_hash: str) -> None:
    """Write review-ranking rows to Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"schema_id": FOLDCHECK_RANKING_SCHEMA_ID.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"source_request_hash": source_request_hash.encode("utf-8"),
        b"rmsd_semantics": (
            b"wt_runtime_ca_rmsd is from foldcheck_report; "
            b"cryoem_mapped_ca_rmsd is explicit ec86kit-reference comparison when available"
        ),
    }
    table = pa.Table.from_pylist(rows, schema=_RANKING_SCHEMA)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _mapped_canonical_positions(residue_map_path: Path) -> list[int]:
    rows = pq.read_table(residue_map_path).to_pylist()
    mapped = [int(row["canonical_position"]) for row in rows if str(row.get("mapping_status")) == "mapped"]
    if not mapped:
        raise ValueError("residue_map.parquet must contain mapped canonical positions")
    return sorted(mapped)


def _fold_rows_by_candidate_id(foldcheck_report_path: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(foldcheck_report_path).to_pylist()
    by_candidate_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_candidate_id:
            duplicate_ids.add(candidate_id)
            continue
        by_candidate_id[candidate_id] = row
    if duplicate_ids:
        formatted = ", ".join(sorted(duplicate_ids))
        raise ValueError(f"foldcheck_report.parquet contains duplicate candidate_id rows: {formatted}")
    return by_candidate_id


def _reference_coordinates(path: Path, *, mapped_position_count: int) -> Any | None:
    if not path.exists():
        return None
    coords = ca_coordinates(path)
    if len(coords) != mapped_position_count:
        raise ValueError(
            "reference backbone CA count does not match mapped residue count; "
            f"expected {mapped_position_count}, observed {len(coords)} at {path}"
        )
    return coords


def _cryoem_mapped_rmsd(
    *,
    candidate_id: str,
    model_artifact_path: Path,
    mapped_positions: list[int],
    reference_coords: Any | None,
    local_model_root: Path | None,
) -> tuple[float | None, str]:
    if reference_coords is None:
        return None, "reference_backbone_unavailable"
    if not str(model_artifact_path):
        return None, "model_artifact_missing"
    resolved_model_artifact_path = _resolve_local_model_artifact_path(
        candidate_id=candidate_id,
        model_artifact_path=model_artifact_path,
        local_model_root=local_model_root,
    )
    if resolved_model_artifact_path is None:
        return None, "model_artifact_not_local"
    mobile_coords = ca_coordinates(resolved_model_artifact_path)
    if len(mobile_coords) >= max(mapped_positions):
        selected = mobile_coords[[position - 1 for position in mapped_positions]]
    elif len(mobile_coords) == len(mapped_positions):
        selected = mobile_coords
    else:
        raise ValueError(
            f"unverified staged model or candidate {candidate_id} ColabFold model CA count does not match "
            "full sequence or mapped-residue basis; "
            f"observed {len(mobile_coords)}, mapped residues {len(mapped_positions)}, "
            f"max mapped position {max(mapped_positions)}"
        )
    rmsd = ca_rmsd(selected, reference_coords)
    if rmsd is None:
        return None, "rmsd_calculation_failed"
    return rmsd, "available"


def _resolve_local_model_artifact_path(
    *,
    candidate_id: str,
    model_artifact_path: Path,
    local_model_root: Path | None,
) -> Path | None:
    if model_artifact_path.exists():
        return model_artifact_path
    if local_model_root is None:
        return None
    candidate_named = local_model_root / f"{candidate_id}.pdb"
    if candidate_named.exists():
        return candidate_named
    source_named = local_model_root / model_artifact_path.name
    if source_named.exists():
        return source_named
    return None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _pae_value(value: Any, key: str) -> float | None:
    if isinstance(value, dict) and value.get(key) is not None:
        return float(value[key])
    return None


def _review_class(*, plddt: float | None, wt_runtime_ca_rmsd: float | None) -> str:
    if plddt is None or wt_runtime_ca_rmsd is None:
        return "metric_missing"
    if wt_runtime_ca_rmsd > 5.0:
        return "structural_outlier"
    if plddt < 90.0:
        return "low_confidence"
    if wt_runtime_ca_rmsd <= 1.25 and plddt >= 91.5:
        return "strong_fold_preserved"
    if wt_runtime_ca_rmsd <= 2.0:
        return "good_fold_preserved"
    return "review_band"


def wild_type_reference_row(foldcheck_report_path: Path) -> dict[str, Any]:
    """Return the accepted WT fold-check row."""

    row = _fold_rows_by_candidate_id(foldcheck_report_path).get(WT_SEQUENCE_ID)
    if row is not None and str(row.get("status")) == "accepted":
        return row
    raise ValueError("foldcheck_report.parquet must contain an accepted wild_type row")
