"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/motif_artifacts.py

Helpers for exporting DenseGen motif artifacts.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def load_motif_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Motif payload must be a JSON object.")
    return payload


def _normalize_prob_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not matrix:
        raise ValueError("matrix is empty")
    norm: list[list[float]] = []
    for i, row in enumerate(matrix):
        if len(row) != 4:
            raise ValueError(f"matrix row {i} length must be 4")
        if any(v < 0 for v in row):
            raise ValueError("matrix rows must be non-negative")
        s = float(sum(row))
        if s <= 0:
            raise ValueError(f"matrix row {i} must sum to > 0")
        norm.append([float(v) / s for v in row])
    return norm


def _normalize_background(background: Sequence[float]) -> list[float]:
    if len(background) != 4:
        raise ValueError("background must have 4 values (A,C,G,T)")
    total = float(sum(background))
    if total <= 0:
        raise ValueError("background must sum to > 0")
    return [float(v) / total for v in background]


def _compute_log_odds(
    matrix: Sequence[Sequence[float]],
    background: Sequence[float],
    *,
    pseudocount: float | None = None,
) -> list[list[float]]:
    if pseudocount is not None and pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")
    bg = _normalize_background(background)
    out: list[list[float]] = []
    for row in matrix:
        row_vals: list[float] = []
        for p, b in zip(row, bg):
            p_val = float(p)
            b_val = float(b)
            if pseudocount is not None and pseudocount > 0:
                p_val = (p_val + float(pseudocount) * b_val) / (1.0 + float(pseudocount))
            if p_val <= 0 or b_val <= 0:
                raise ValueError(
                    "log-odds requires positive probabilities/background. "
                    "Provide pseudocounts or adjust the background policy."
                )
            row_vals.append(math.log(p_val / b_val))
        out.append(row_vals)
    return out


def _normalize_log_odds_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not matrix:
        raise ValueError("log_odds_matrix is empty")
    out: list[list[float]] = []
    for i, row in enumerate(matrix):
        if len(row) != 4:
            raise ValueError(f"log_odds_matrix row {i} length must be 4")
        vals = [float(v) for v in row]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("log_odds_matrix rows must be finite")
        out.append(vals)
    return out


def _background_from_matrix(matrix: Sequence[Sequence[float]]) -> list[float]:
    sums = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for row in matrix:
        if len(row) != 4:
            raise ValueError("matrix rows must have length 4 to compute background")
        for idx, val in enumerate(row):
            sums[idx] += float(val)
        count += 1
    if count == 0:
        raise ValueError("matrix is empty")
    return [val / float(count) for val in sums]


def resolve_background(payload: dict, *, policy: str) -> list[float]:
    if policy == "record":
        bg = payload.get("background")
        if bg is None:
            raise ValueError("Motif record is missing background; choose --background uniform or matrix.")
        if isinstance(bg, dict):
            return [bg["A"], bg["C"], bg["G"], bg["T"]]
        return list(bg)
    if policy == "uniform":
        return [0.25, 0.25, 0.25, 0.25]
    if policy == "matrix":
        matrix = payload.get("matrix")
        if matrix is None:
            raise ValueError("Motif record is missing matrix; cannot compute background.")
        return _background_from_matrix(matrix)
    raise ValueError(f"Unknown background policy: {policy}")


def build_densegen_artifact(
    payload: dict,
    *,
    producer: str,
    background_policy: str,
    pseudocount: float | None,
) -> dict:
    descriptor = payload.get("descriptor") or {}
    if descriptor.get("alphabet") not in (None, "ACGT"):
        raise ValueError("Only ACGT motifs are supported for DenseGen artifacts.")
    motif_id = descriptor.get("motif_id")
    if not motif_id:
        raise ValueError("Motif record is missing motif_id.")
    matrix_semantics = payload.get("matrix_semantics")
    if matrix_semantics != "probabilities":
        raise ValueError("DenseGen motif artifacts require probability matrices.")
    matrix = payload.get("matrix")
    if matrix is None:
        raise ValueError("Motif record is missing matrix.")
    probs = _normalize_prob_matrix(matrix)
    background = resolve_background(payload, policy=background_policy)
    background = _normalize_background(background)
    log_odds = None
    raw_log_odds = payload.get("log_odds_matrix")
    if pseudocount in (None, 0) and raw_log_odds is not None:
        log_odds = _normalize_log_odds_matrix(raw_log_odds)
        if len(log_odds) != len(probs):
            raise ValueError("log_odds_matrix length must match matrix length")
    if log_odds is None:
        log_odds = _compute_log_odds(probs, background, pseudocount=pseudocount)

    def _as_row_dict(row: Sequence[float]) -> dict:
        return {"A": float(row[0]), "C": float(row[1]), "G": float(row[2]), "T": float(row[3])}

    artifact = {
        "schema_version": "1.0",
        "producer": producer,
        "motif_id": motif_id,
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": _as_row_dict(background),
        "probabilities": [_as_row_dict(row) for row in probs],
        "log_odds": [_as_row_dict(row) for row in log_odds],
    }
    optional_fields = {
        "tf_name": descriptor.get("tf_name"),
        "source": descriptor.get("source"),
        "organism": descriptor.get("organism"),
        "provenance": payload.get("provenance"),
        "checksums": payload.get("checksums"),
        "tags": descriptor.get("tags"),
        "length": descriptor.get("length"),
    }
    for key, val in optional_fields.items():
        if val is not None:
            artifact[key] = val
    return artifact


def artifact_filename(*, tf_name: str | None, source: str | None, motif_id: str) -> str:
    parts = [p for p in (tf_name, source, motif_id) if p]
    stem = "__".join(_safe_stem(p) for p in parts if p)
    return f"{stem}.json" if stem else "motif.json"


def _safe_stem(label: str) -> str:
    out = []
    for ch in label:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "motif"


def build_manifest(
    *,
    producer: str,
    entries: Iterable[dict],
    config_path: Path,
    catalog_root: Path,
    background_policy: str,
    pseudocount: float | None,
) -> dict:
    return {
        "schema_version": "1.0",
        "producer": producer,
        "created_at": None,
        "config_path": str(config_path),
        "catalog_root": str(catalog_root),
        "background_policy": background_policy,
        "pseudocount": pseudocount,
        "artifacts": list(entries),
    }
