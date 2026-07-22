"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/resume.py

Resume helpers for Eco1 Biohub ESMC SAE-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ExistingBiohubEsmcRows:
    """Reusable Biohub ESMC rows keyed by candidate id."""

    profile_rows_by_candidate: dict[str, dict[str, object]]
    protein_features_path: Path | None
    residue_features_path: Path | None
    feature_catalog_path: Path | None
    _feature_catalog_rows: list[dict[str, object]] | None = field(default=None, init=False, repr=False)

    def protein_feature_rows(self, candidate_id: str) -> list[dict[str, object]]:
        """Load reusable protein-level feature rows for one candidate."""

        return _read_candidate_rows(self.protein_features_path, candidate_id)

    def residue_feature_rows(self, candidate_id: str) -> list[dict[str, object]]:
        """Load reusable residue-level feature rows for one candidate."""

        return _read_candidate_rows(self.residue_features_path, candidate_id)

    @property
    def feature_catalog_rows(self) -> list[dict[str, object]]:
        """Load reusable feature-catalog rows once when needed."""

        if self._feature_catalog_rows is None:
            rows = pq.read_table(self.feature_catalog_path).to_pylist() if self.feature_catalog_path else []
            object.__setattr__(self, "_feature_catalog_rows", [dict(row) for row in rows])
        return self._feature_catalog_rows


def load_existing_rows(output_root: Path) -> ExistingBiohubEsmcRows | None:
    """Load existing Biohub ESMC rows for exact-query resume."""

    profile_path = output_root / "biohub_esmc_sae_profile.parquet"
    protein_path = output_root / "biohub_esmc_protein_features.parquet"
    residue_path = output_root / "biohub_esmc_residue_features.parquet"
    feature_path = output_root / "biohub_esmc_feature_catalog.parquet"
    existing_paths = [path for path in (profile_path, protein_path, residue_path, feature_path) if path.exists()]
    if not profile_path.exists():
        if existing_paths:
            raise ValueError("stale Biohub ESMC SAE cache: feature tables exist without a profile table")
        return None
    profile_table = pq.read_table(profile_path)
    if "biohub_query_hash" not in profile_table.column_names:
        raise ValueError("stale Biohub ESMC SAE cache: profile table is missing biohub_query_hash")
    return ExistingBiohubEsmcRows(
        profile_rows_by_candidate={str(row["candidate_id"]): dict(row) for row in profile_table.to_pylist()},
        protein_features_path=protein_path if protein_path.exists() else None,
        residue_features_path=residue_path if residue_path.exists() else None,
        feature_catalog_path=feature_path if feature_path.exists() else None,
    )


def cached_profile_row(
    *,
    existing_rows: ExistingBiohubEsmcRows | None,
    candidate_id: str,
    sequence_hash: str,
    biohub_query_hash: str,
    biohub_request_hash: str,
    source_request_hash: str,
) -> dict[str, object] | None:
    """Return a reusable accepted profile row when the per-sequence query matches."""

    if existing_rows is None:
        return None
    row = existing_rows.profile_rows_by_candidate.get(candidate_id)
    if row is None:
        return None
    if str(row.get("status", "")) != "accepted":
        return None
    if str(row.get("sequence_hash", "")) != sequence_hash:
        return None
    if str(row.get("biohub_query_hash", "")) != biohub_query_hash:
        return None
    copied = dict(row)
    copied["biohub_request_hash"] = biohub_request_hash
    copied["source_request_hash"] = source_request_hash
    return copied


def _read_candidate_rows(path: Path | None, candidate_id: str) -> list[dict[str, object]]:
    if path is None:
        return []
    rows = pq.read_table(path, filters=[("candidate_id", "==", candidate_id)]).to_pylist()
    return [dict(row) for row in rows]
