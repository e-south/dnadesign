"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/structure_set.py

Validated foldcheck structure-set input for Eco1 communication visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    sha256,
)

_SCHEMA_ID = "eco1_rt.foldcheck_full_structure_set"
_SCHEMA_VERSION = 1
_WT_CANDIDATE_ID = "wild_type"


@dataclass(frozen=True)
class FoldcheckStructure:
    """One locally materialized foldcheck structure."""

    candidate_id: str
    path: Path
    display_label: str
    content_digest: str
    full_sequence_identity_percent: float


@dataclass(frozen=True)
class FoldcheckStructureSet:
    """Parsed structure-set manifest with one WT row and unique candidates."""

    manifest_path: Path
    wild_type: FoldcheckStructure
    candidates: tuple[FoldcheckStructure, ...]

    @property
    def candidate_by_id(self) -> dict[str, FoldcheckStructure]:
        """Return candidate structures keyed by stable candidate ID."""

        return {structure.candidate_id: structure for structure in self.candidates}


def read_foldcheck_structure_set(path: Path) -> FoldcheckStructureSet:
    """Parse and validate one manifest-relative foldcheck structure set."""

    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    if payload.get("schema_id") != _SCHEMA_ID or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported foldcheck structure-set schema at {path}: "
            f"{payload.get('schema_id')!r} version {payload.get('schema_version')!r}"
        )
    if payload.get("path_policy") != "local_paths_manifest_relative":
        raise ValueError(f"Foldcheck structure-set paths must be manifest-relative: {path}")
    raw_structures = payload.get("structures")
    if not isinstance(raw_structures, list):
        raise ValueError(f"Foldcheck structure-set structures must be a list: {path}")
    declared_count = payload.get("structure_count")
    if not isinstance(declared_count, int) or declared_count != len(raw_structures):
        raise ValueError(
            f"Foldcheck structure_count does not match structures at {path}: "
            f"declared {declared_count!r}, observed {len(raw_structures)}"
        )

    structures: list[FoldcheckStructure] = []
    observed_ids: set[str] = set()
    for index, raw in enumerate(raw_structures):
        structure = _parse_structure_row(raw, index=index, manifest_path=path)
        if structure.candidate_id in observed_ids:
            raise ValueError(f"Foldcheck structure set has duplicate candidate_id: {structure.candidate_id}")
        observed_ids.add(structure.candidate_id)
        if not structure.path.is_file():
            raise FileNotFoundError(f"Foldcheck structure is missing for {structure.candidate_id}: {structure.path}")
        observed_digest = "sha256:" + sha256(structure.path)
        if observed_digest != structure.content_digest:
            raise ValueError(
                f"Foldcheck structure digest mismatch for {structure.candidate_id}: "
                f"expected {structure.content_digest}, observed {observed_digest}"
            )
        structures.append(structure)

    wild_type_rows = [structure for structure in structures if structure.candidate_id == _WT_CANDIDATE_ID]
    if len(wild_type_rows) != 1:
        raise ValueError(f"Foldcheck structure set must contain exactly one {_WT_CANDIDATE_ID!r} row: {path}")
    candidates = tuple(structure for structure in structures if structure.candidate_id != _WT_CANDIDATE_ID)
    return FoldcheckStructureSet(
        manifest_path=path.resolve(),
        wild_type=wild_type_rows[0],
        candidates=candidates,
    )


def _parse_structure_row(raw: Any, *, index: int, manifest_path: Path) -> FoldcheckStructure:
    if not isinstance(raw, dict):
        raise ValueError(f"Foldcheck structure row {index} must be a mapping: {manifest_path}")
    candidate_id = str(raw.get("candidate_id") or "").strip()
    relative_path = str(raw.get("local_model_artifact_path") or "").strip()
    content_digest = str(raw.get("source_model_artifact_hash") or "").strip()
    if not candidate_id or not relative_path or not content_digest:
        raise ValueError(
            "Foldcheck structure row "
            f"{index} requires candidate_id, local_model_artifact_path, and source_model_artifact_hash: "
            f"{manifest_path}"
        )
    if not _is_sha256_digest(content_digest):
        raise ValueError(
            f"Foldcheck structure row {index} has an invalid source_model_artifact_hash: {content_digest!r}"
        )
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ValueError(f"Foldcheck structure path must be manifest-relative for {candidate_id}: {candidate_path}")
    resolved_path = (manifest_path.parent / candidate_path).resolve()
    try:
        resolved_path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError(
            f"Foldcheck structure path escapes its manifest root for {candidate_id}: {candidate_path}"
        ) from error
    return FoldcheckStructure(
        candidate_id=candidate_id,
        path=resolved_path,
        display_label=str(raw.get("display_label") or candidate_id),
        content_digest=content_digest,
        full_sequence_identity_percent=_required_percent(
            raw.get("full_sequence_identity_percent"),
            field="full_sequence_identity_percent",
            candidate_id=candidate_id,
        ),
    )


def _is_sha256_digest(value: str) -> bool:
    prefix = "sha256:"
    if not value.startswith(prefix):
        return False
    hexadecimal = value.removeprefix(prefix)
    return len(hexadecimal) == 64 and all(character in "0123456789abcdef" for character in hexadecimal)


def _required_percent(value: Any, *, field: str, candidate_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Foldcheck structure {candidate_id} requires numeric {field}")
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"Foldcheck structure {candidate_id} has out-of-range {field}: {number}")
    return number


__all__ = ["FoldcheckStructure", "FoldcheckStructureSet", "read_foldcheck_structure_set"]
