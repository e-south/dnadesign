"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/ingest/adapters/regulondb_alignment.py

Alignment matrix parsing helpers for RegulonDB motif intake.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _tokenize_line(line: str) -> list[str]:
    return [tok for tok in re.split(r"[,\s]+", line.strip()) if tok]


def _parse_float(token: str) -> float:
    if not _FLOAT_RE.match(token):
        raise ValueError(f"invalid numeric token: {token}")
    return float(token)


def _parse_alignment_matrix(text: str) -> List[List[float]]:
    if not text or not text.strip():
        raise ValueError("alignment matrix payload is empty")
    raw = text.strip()
    # JSON payload
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        if all(isinstance(row, list) for row in parsed):
            if all(len(row) == 4 for row in parsed):
                return [[float(v) for v in row] for row in parsed]
            if len(parsed) == 4 and all(isinstance(row, list) for row in parsed):
                lengths = {len(row) for row in parsed}
                if len(lengths) != 1:
                    raise ValueError("alignment matrix JSON rows must have equal length")
                length = lengths.pop()
                return [
                    [
                        float(parsed[0][i]),
                        float(parsed[1][i]),
                        float(parsed[2][i]),
                        float(parsed[3][i]),
                    ]
                    for i in range(length)
                ]
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("alignment matrix has no data rows")
    tokens = [_tokenize_line(line) for line in lines]
    # header row A C G T
    if len(tokens) > 1 and len(tokens[0]) == 4 and all(tok.upper() in "ACGT" for tok in tokens[0]):
        rows = []
        for row in tokens[1:]:
            if len(row) != 4:
                raise ValueError("alignment matrix rows must have 4 numeric columns")
            rows.append([_parse_float(v) for v in row])
        return rows
    # base-labeled rows
    if all(row and row[0].upper() in "ACGT" for row in tokens):
        base_rows: Dict[str, list[float]] = {}
        for row in tokens:
            base = row[0].upper()
            nums = [_parse_float(v) for v in row[1:]]
            base_rows[base] = nums
        if set(base_rows.keys()) != {"A", "C", "G", "T"}:
            raise ValueError("alignment matrix must include A/C/G/T rows")
        lengths = {len(vals) for vals in base_rows.values()}
        if len(lengths) != 1:
            raise ValueError("alignment matrix base rows must have equal length")
        length = lengths.pop()
        return [[base_rows["A"][i], base_rows["C"][i], base_rows["G"][i], base_rows["T"][i]] for i in range(length)]
    # position rows with 4 numeric columns
    if all(len(row) == 4 for row in tokens):
        return [[_parse_float(v) for v in row] for row in tokens]
    raise ValueError("unrecognized alignment matrix format")


def _parse_alignment_sequences(text: str) -> List[str]:
    if not text or not text.strip():
        raise ValueError("alignment payload is empty")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("alignment payload has no sequences")
    sequences: List[str] = []
    if any(line.startswith(">") for line in lines):
        current: list[str] = []
        for line in lines:
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            sequences.append("".join(current))
    else:
        sequences = lines
    cleaned: List[str] = []
    for seq in sequences:
        seq = seq.strip().upper()
        if not seq:
            continue
        if any(ch not in "ACGT-" for ch in seq):
            raise ValueError("alignment sequences must contain only A/C/G/T/- characters")
        cleaned.append(seq)
    if not cleaned:
        raise ValueError("alignment sequences are empty after cleaning")
    return cleaned


def _compute_pwm_from_alignment(sequences: List[str]) -> List[List[float]]:
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError("alignment sequences must be the same length")
    length = lengths.pop()
    matrix: List[List[float]] = []
    for i in range(length):
        counts = Counter()
        for seq in sequences:
            base = seq[i]
            if base in "ACGT":
                counts[base] += 1
        total = sum(counts.get(b, 0) for b in "ACGT")
        if total == 0:
            raise ValueError("alignment column has no A/C/G/T bases")
        matrix.append(
            [
                counts.get("A", 0) / total,
                counts.get("C", 0) / total,
                counts.get("G", 0) / total,
                counts.get("T", 0) / total,
            ]
        )
    return matrix
