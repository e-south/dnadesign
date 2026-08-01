"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/selected_lineage/loading.py

Repository-confined loading for selected materialized MSD variant lineage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ...catalog.strict_mapping_io import DuplicateMappingKeyError, load_unique_yaml
from .contracts import MaterializedVariantLineageError, MaterializedVariantLineageV1


def load_lineage_document(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[MaterializedVariantLineageV1, Path]:
    """Load one typed lineage document and return its resolved repository root."""

    root = Path(repo_root).expanduser().resolve()
    lineage_path = repo_file(root, path, field="materialized-variant lineage")
    try:
        payload = load_unique_yaml(lineage_path)
        lineage = MaterializedVariantLineageV1.model_validate(payload)
    except (DuplicateMappingKeyError, OSError, ValidationError) as exc:
        raise MaterializedVariantLineageError(f"Invalid materialized-variant lineage {lineage_path}: {exc}") from exc
    return lineage, root


def cached_mapping(cache: dict[Path, dict[str, Any]], path: Path, *, label: str) -> dict[str, Any]:
    if path not in cache:
        cache[path] = load_mapping(path, label=label)
    return cache[path]


def load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = load_unique_yaml(path)
    except (DuplicateMappingKeyError, OSError) as exc:
        raise MaterializedVariantLineageError(f"Could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaterializedVariantLineageError(f"{label} must be a mapping: {path}")
    return payload


def repo_file(repo_root: Path, raw: object, *, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise MaterializedVariantLineageError(f"Missing {field} path.")
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except OSError as exc:
            raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    else:
        try:
            resolved = (repo_root / candidate).resolve(strict=True)
        except OSError as exc:
            raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MaterializedVariantLineageError(f"{field} path escapes the repository: {candidate}") from exc
    if not resolved.is_file():
        raise MaterializedVariantLineageError(f"{field} path is not a file: {candidate}")
    return resolved


def linked_file(base: Path, raw: object, *, repo_root: Path, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise MaterializedVariantLineageError(f"Missing {field} path.")
    candidate = Path(value)
    if candidate.is_absolute():
        raise MaterializedVariantLineageError(f"{field} must be relative: {candidate}")
    try:
        resolved = (base / candidate).resolve(strict=True)
    except OSError as exc:
        raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MaterializedVariantLineageError(f"{field} path escapes the repository: {candidate}") from exc
    if not resolved.is_file():
        raise MaterializedVariantLineageError(f"{field} path is not a file: {candidate}")
    return resolved
