"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/mask_join.py

Join WT ESMC mutation scoring rows to Eco1 mask context.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

MASK_JOIN_SCHEMA_ID = "eco1_rt_repack.biohub_esmc_wt_mutation_scoring.mask_join"
_MASK_JOIN_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("canonical_position", pa.int32()),
        ("wt_aa", pa.string()),
        ("protected", pa.bool_()),
        ("non_fixed", pa.bool_()),
        ("non_fixed_missing_backbone", pa.bool_()),
        ("protection_reasons_json", pa.string()),
        ("motif_protected", pa.bool_()),
        ("wang_ec86_direct_contact_prior", pa.bool_()),
        ("direct_retained_dna_rna_contact_5a", pa.bool_()),
        ("evolutionarily_conserved_clade9_25pct_plurality", pa.bool_()),
        ("wt_plurality_frequency", pa.float64()),
        ("min_distance_to_retained_dna_rna_angstrom", pa.float64()),
        ("entropy_bits", pa.float64()),
        ("canonical_entropy_bits", pa.float64()),
        ("fraction_negative_alternate_llr", pa.float64()),
        ("best_alt_aa", pa.string()),
        ("best_alt_llr", pa.float64()),
        ("worst_alt_aa", pa.string()),
        ("worst_alt_llr", pa.float64()),
        ("status", pa.string()),
        ("mask_context_status", pa.string()),
    ]
)


def write_mask_join(
    *,
    position_entropy_path: Path,
    mask_set_path: Path,
    output_path: Path,
    request_hash: str,
) -> Path:
    """Write a study-owned position table joining ESMC scores to mask context."""

    position_rows = pq.read_table(position_entropy_path).to_pylist()
    mask_rows = _load_mask_rows(mask_set_path)
    joined: list[dict[str, object]] = []
    for row in position_rows:
        canonical_position = int(row["canonical_position"])
        mask_row = mask_rows.get(canonical_position)
        if mask_row is None:
            raise ValueError(f"mask_set.yaml is missing canonical_position {canonical_position}")
        joined.append(_join_row(row, mask_row))
    table = pa.Table.from_pylist(joined, schema=_MASK_JOIN_SCHEMA)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_id"] = MASK_JOIN_SCHEMA_ID.encode("utf-8")
    metadata[b"request_hash"] = request_hash.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), output_path)
    return output_path


def _load_mask_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"mask_set.yaml must be a mapping: {path}")
    if payload.get("schema_id") != "thread.mask_set":
        raise ValueError("mask_set.yaml must use schema_id thread.mask_set")
    residues = payload.get("residues")
    if not isinstance(residues, list):
        raise ValueError("mask_set.yaml requires residues")
    rows: dict[int, Mapping[str, Any]] = {}
    for residue in residues:
        if not isinstance(residue, Mapping):
            raise ValueError("mask_set.yaml residues must be mappings")
        rows[int(residue["canonical_position"])] = residue
    return rows


def _join_row(score_row: Mapping[str, Any], mask_row: Mapping[str, Any]) -> dict[str, object]:
    return {
        "sequence_id": str(score_row["sequence_id"]),
        "sequence_hash": str(score_row["sequence_hash"]),
        "canonical_position": int(score_row["canonical_position"]),
        "wt_aa": str(score_row["wt_aa"]),
        "protected": bool(mask_row.get("protected")),
        "non_fixed": bool(mask_row.get("non_fixed")),
        "non_fixed_missing_backbone": bool(mask_row.get("non_fixed_missing_backbone")),
        "protection_reasons_json": json.dumps(mask_row.get("protection_reasons") or [], sort_keys=True),
        "motif_protected": bool(mask_row.get("motif_protected")),
        "wang_ec86_direct_contact_prior": bool(mask_row.get("wang_ec86_direct_contact_prior")),
        "direct_retained_dna_rna_contact_5a": bool(mask_row.get("direct_retained_dna_rna_contact_5a")),
        "evolutionarily_conserved_clade9_25pct_plurality": bool(
            mask_row.get("evolutionarily_conserved_clade9_25pct_plurality")
        ),
        "wt_plurality_frequency": _optional_float(mask_row.get("wt_plurality_frequency")),
        "min_distance_to_retained_dna_rna_angstrom": _optional_float(
            mask_row.get("min_distance_to_retained_dna_rna_angstrom")
        ),
        "entropy_bits": _optional_float(score_row.get("entropy_bits")),
        "canonical_entropy_bits": _optional_float(score_row.get("canonical_entropy_bits")),
        "fraction_negative_alternate_llr": _optional_float(score_row.get("fraction_negative_alternate_llr")),
        "best_alt_aa": str(score_row.get("best_alt_aa") or ""),
        "best_alt_llr": _optional_float(score_row.get("best_alt_llr")),
        "worst_alt_aa": str(score_row.get("worst_alt_aa") or ""),
        "worst_alt_llr": _optional_float(score_row.get("worst_alt_llr")),
        "status": str(score_row.get("status") or ""),
        "mask_context_status": "joined",
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
