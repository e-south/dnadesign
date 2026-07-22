"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/pdb_alignment.py

Small PDB alignment helpers for Eco1 fold-check review renders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_XYZ_START = 30
_XYZ_END = 54


def write_aligned_pdb_to_reference_ca(
    *,
    query_path: Path,
    reference_path: Path,
    output_path: Path,
    query_start_residue: int = 3,
    reference_start_residue: int = 1,
    residue_count: int = 309,
) -> float:
    """Align a query PDB to the Eco1 reference over mapped C-alpha atoms."""

    aligned_text, mapped_ca_rmsd = align_pdb_text_to_reference_ca(
        query_text=query_path.read_text(encoding="utf-8"),
        reference_text=reference_path.read_text(encoding="utf-8"),
        query_start_residue=query_start_residue,
        reference_start_residue=reference_start_residue,
        residue_count=residue_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(aligned_text, encoding="utf-8")
    return mapped_ca_rmsd


def align_pdb_text_to_reference_ca(
    *,
    query_text: str,
    reference_text: str,
    query_start_residue: int = 3,
    reference_start_residue: int = 1,
    residue_count: int = 309,
) -> tuple[str, float]:
    """Return query PDB text aligned to reference over mapped C-alpha atoms."""

    query_atoms = _atom_lines_from_text(query_text)
    reference_atoms = _atom_lines_from_text(reference_text)
    query_ca = _ca_coordinates_by_residue(query_atoms)
    reference_ca = _ca_coordinates_by_residue(reference_atoms)
    query_coords = []
    reference_coords = []
    for offset in range(residue_count):
        query_key = query_start_residue + offset
        reference_key = reference_start_residue + offset
        if query_key not in query_ca or reference_key not in reference_ca:
            raise ValueError(
                f"Missing C-alpha pair for query residue {query_key} and reference residue {reference_key}"
            )
        query_coords.append(query_ca[query_key])
        reference_coords.append(reference_ca[reference_key])

    rotation, query_centroid, reference_centroid = _kabsch(
        np.asarray(query_coords, dtype=float),
        np.asarray(reference_coords, dtype=float),
    )
    aligned_text = "".join(_transform_pdb_text(query_text, rotation, query_centroid, reference_centroid))
    aligned_query = (_as_array(query_coords) - query_centroid) @ rotation + reference_centroid
    return aligned_text, float(np.sqrt(np.mean(np.sum((aligned_query - _as_array(reference_coords)) ** 2, axis=1))))


def _read_atom_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if line.startswith("ATOM")]


def _atom_lines_from_text(text: str) -> list[str]:
    return [line for line in text.splitlines(keepends=True) if line.startswith("ATOM")]


def _ca_coordinates_by_residue(lines: list[str]) -> dict[int, NDArray[np.float64]]:
    coordinates: dict[int, NDArray[np.float64]] = {}
    for line in lines:
        if line[12:16].strip() != "CA" or line[21].strip() != "A":
            continue
        residue_index = int(line[22:26])
        coordinates[residue_index] = _line_xyz(line)
    return coordinates


def _line_xyz(line: str) -> NDArray[np.float64]:
    return np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)


def _kabsch(
    query_coords: NDArray[np.float64],
    reference_coords: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    query_centroid = query_coords.mean(axis=0)
    reference_centroid = reference_coords.mean(axis=0)
    query_centered = query_coords - query_centroid
    reference_centered = reference_coords - reference_centroid
    covariance = query_centered.T @ reference_centered
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_matrix @ vt_matrix))
    rotation = u_matrix @ correction @ vt_matrix
    return rotation, query_centroid, reference_centroid


def _transform_pdb_text(
    text: str,
    rotation: NDArray[np.float64],
    query_centroid: NDArray[np.float64],
    reference_centroid: NDArray[np.float64],
) -> list[str]:
    transformed: list[str] = []
    for line in text.splitlines(keepends=True):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            transformed.append(line)
            continue
        aligned = (_line_xyz(line) - query_centroid) @ rotation + reference_centroid
        transformed.append(f"{line[:_XYZ_START]}{aligned[0]:8.3f}{aligned[1]:8.3f}{aligned[2]:8.3f}{line[_XYZ_END:]}")
    return transformed


def _as_array(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float)
