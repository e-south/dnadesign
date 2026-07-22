"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/io/parsers/ddg_table.py

Parse delta-delta-G binding-energy tables into probability PWMs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.io.parsers.backend import register

_POSITION_HEADERS = {"PO", "POS", "POSITION"}
_BASES = ("A", "C", "G", "T")
_R_KCAL_PER_MOL_K = 1.98720425864083e-3
_DEFAULT_TEMPERATURE_K = 298.15


def _nonempty_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"ddG table is empty: {path}")
    return lines


def _parse_header(line: str, path: Path) -> tuple[str, list[str]]:
    tokens = line.split()
    if len(tokens) != 5:
        raise ValueError(f"Invalid ddG header in {path}: expected 5 columns, got {len(tokens)}")
    pos_header = tokens[0].upper()
    if pos_header not in _POSITION_HEADERS:
        raise ValueError(
            f"Invalid ddG header in {path}: first column must be one of {sorted(_POSITION_HEADERS)}, got {tokens[0]!r}"
        )
    base_headers = [token.upper() for token in tokens[1:]]
    if set(base_headers) != set(_BASES):
        raise ValueError(
            f"Invalid ddG header in {path}: expected nucleotide columns {list(_BASES)}, got {tokens[1:]!r}"
        )
    if len(set(base_headers)) != len(base_headers):
        raise ValueError(f"Invalid ddG header in {path}: nucleotide columns must be unique")
    return pos_header, base_headers


def _parse_rows(lines: list[str], path: Path) -> np.ndarray:
    _, base_headers = _parse_header(lines[0], path)
    base_index = {base: idx for idx, base in enumerate(base_headers, start=1)}
    expected_position = 1
    rows: list[list[float]] = []
    for raw in lines[1:]:
        tokens = raw.split()
        if len(tokens) != 5:
            raise ValueError(f"Invalid ddG row in {path}: expected 5 columns, got {len(tokens)}: {raw!r}")
        try:
            position = int(tokens[0])
        except ValueError as exc:
            raise ValueError(f"Invalid ddG position in {path}: {tokens[0]!r}") from exc
        if position != expected_position:
            raise ValueError(f"Invalid ddG position order in {path}: expected {expected_position}, got {position}")
        expected_position += 1
        ordered: list[float] = []
        for base in _BASES:
            token = tokens[base_index[base]]
            try:
                ordered.append(float(token))
            except ValueError as exc:
                raise ValueError(f"Invalid ddG value for base {base} in {path}: {token!r}") from exc
        rows.append(ordered)
    if not rows:
        raise ValueError(f"ddG table has no data rows: {path}")
    return np.asarray(rows, dtype=float)


def ddg_to_probability_matrix(
    ddg_matrix: np.ndarray,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
) -> np.ndarray:
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be > 0, got {temperature_k!r}")
    matrix = np.asarray(ddg_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"ddG matrix must have shape (L, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("ddG matrix values must be finite")
    beta = 1.0 / (_R_KCAL_PER_MOL_K * float(temperature_k))
    probabilities: list[list[float]] = []
    for row_idx, row in enumerate(matrix):
        shifted = row - float(np.min(row))
        weights = np.exp(-shifted * beta)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"ddG row {row_idx} produced an invalid Boltzmann normalization constant")
        probabilities.append((weights / total).tolist())
    return np.asarray(probabilities, dtype=float)


@register("DDG_TABLE")
def parse_ddg_table(path: Path) -> PWM:
    ddg_matrix = _parse_rows(_nonempty_lines(path), path)
    prob_matrix = ddg_to_probability_matrix(ddg_matrix)
    return PWM(name=path.stem, matrix=prob_matrix, alphabet=_BASES)
