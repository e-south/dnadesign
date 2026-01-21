"""
Stable identifier helpers for DenseGen artifacts.

These hashes are intended to be deterministic and join-friendly across runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

_BASES = ("A", "C", "G", "T")
_FLOAT_DIGITS = 10


def _fmt_float(value: float) -> str:
    return format(float(value), f".{_FLOAT_DIGITS}g")


def _stable_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def hash_pwm_motif(
    *,
    motif_label: str,
    matrix: Sequence[Mapping[str, float]],
    background: Mapping[str, float],
    source_kind: str,
    source_label: str | None = None,
) -> str:
    rows = []
    for row in matrix:
        rows.append([_fmt_float(row.get(base, 0.0)) for base in _BASES])
    payload = {
        "source_kind": source_kind,
        "source_label": source_label or "",
        "motif_label": str(motif_label),
        "matrix": rows,
        "background": {base: _fmt_float(background.get(base, 0.0)) for base in _BASES},
    }
    return _hash_payload(payload)


def hash_label_motif(*, label: str | None, source_kind: str, source_label: str | None = None) -> str:
    payload = {
        "source_kind": source_kind,
        "source_label": source_label or "",
        "label": str(label or ""),
    }
    return _hash_payload(payload)


def hash_tfbs_id(
    *,
    motif_id: str | None,
    sequence: str,
    scoring_backend: str,
    matched_start: int | None = None,
    matched_stop: int | None = None,
    matched_strand: str | None = None,
) -> str:
    payload = {
        "motif_id": str(motif_id or ""),
        "sequence": str(sequence),
        "scoring_backend": str(scoring_backend),
        "matched_start": matched_start,
        "matched_stop": matched_stop,
        "matched_strand": matched_strand or "",
    }
    return _hash_payload(payload)


def hash_attempt_id(
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    library_hash: str,
    attempt_index: int,
) -> str:
    payload = {
        "run_id": str(run_id),
        "input_name": str(input_name),
        "plan_name": str(plan_name),
        "library_hash": str(library_hash),
        "attempt_index": int(attempt_index),
    }
    return _hash_payload(payload)


def hash_solution_id(
    *,
    run_id: str,
    input_name: str,
    plan_name: str,
    sequence_hash: str,
    library_hash: str,
) -> str:
    payload = {
        "run_id": str(run_id),
        "input_name": str(input_name),
        "plan_name": str(plan_name),
        "sequence_hash": str(sequence_hash),
        "library_hash": str(library_hash),
    }
    return _hash_payload(payload)


def hash_candidate_id(
    *,
    input_name: str | None = None,
    motif_id: str,
    sequence: str,
    scoring_backend: str,
) -> str:
    payload = {
        "input_name": str(input_name) if input_name is not None else None,
        "motif_id": str(motif_id),
        "sequence": str(sequence),
        "scoring_backend": str(scoring_backend),
    }
    return _hash_payload(payload)
