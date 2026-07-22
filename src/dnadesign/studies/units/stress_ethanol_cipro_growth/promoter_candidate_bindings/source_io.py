"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/source_io.py

Repository I/O and digest helpers for promoter candidate-binding sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import BindingSourceArtifact, PromoterCandidateBindingsError


def read_parquet(
    path: Path,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise PromoterCandidateBindingsError(f"Required promoter binding source table not found: {path}")
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except Exception as exc:
        raise PromoterCandidateBindingsError(f"Could not read promoter binding source table {path}: {exc}") from exc


def source_artifact(repo_root: Path, artifact_id: str, path: Path) -> BindingSourceArtifact:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PromoterCandidateBindingsError(f"Binding source escapes repository root: {resolved}") from exc
    return BindingSourceArtifact(artifact_id=artifact_id, path=str(relative), sha256=file_sha256(resolved))


def candidate_selection_sha256(candidates: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in candidates.sort_values("id", kind="stable").to_dict(orient="records"):
        payload = {key: _jsonable(value) for key, value in row.items()}
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in list(value)]
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
