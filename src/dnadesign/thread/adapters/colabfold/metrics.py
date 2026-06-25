"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/metrics.py

Coordinate and metric helpers for ColabFold output normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def ca_coordinates(path: Path) -> np.ndarray:
    """Extract C-alpha coordinates from a PDB file."""

    coords: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coords, dtype=float)


def mean_ca_plddt(path: Path) -> float | None:
    """Return mean C-alpha pLDDT from the PDB B-factor column."""

    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        values.append(float(line[60:66]))
    if not values:
        return None
    return round(float(sum(values) / len(values)), 3)


def ca_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float | None:
    """Return Kabsch-aligned C-alpha RMSD for equal-length coordinate arrays."""

    if mobile.shape != reference.shape or len(mobile) == 0:
        return None
    mobile_centered = mobile - mobile.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    covariance = mobile_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    determinant = np.linalg.det(right_t.T @ left.T)
    correction = np.diag([1.0, 1.0, determinant])
    rotation = right_t.T @ correction @ left.T
    aligned = mobile_centered @ rotation
    squared = np.sum((aligned - reference_centered) ** 2) / len(mobile)
    return round(float(math.sqrt(squared)), 3)


def pae_summary(json_path: Path | None) -> dict[str, Any]:
    """Return a compact PAE summary from a ColabFold JSON file when present."""

    if json_path is None:
        return {"status": "not_found"}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    matrix = _find_pae_matrix(payload)
    if matrix is None:
        return {"status": "not_found", "source_path": str(json_path)}
    values = [float(value) for row in matrix for value in row]
    if not values:
        return {"status": "empty", "source_path": str(json_path)}
    return {
        "status": "parsed",
        "mean": round(sum(values) / len(values), 3),
        "max": round(max(values), 3),
        "source_path": str(json_path),
    }


def _find_pae_matrix(payload: Any) -> list[list[float]] | None:
    if isinstance(payload, dict):
        for key in ("pae", "predicted_aligned_error"):
            matrix = payload.get(key)
            if _looks_like_matrix(matrix):
                return matrix
        for value in payload.values():
            matrix = _find_pae_matrix(value)
            if matrix is not None:
                return matrix
    if isinstance(payload, list):
        if _looks_like_matrix(payload):
            return payload
        for value in payload:
            matrix = _find_pae_matrix(value)
            if matrix is not None:
                return matrix
    return None


def _looks_like_matrix(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row for row in value)
        and all(isinstance(item, int | float) for row in value for item in row)
    )
