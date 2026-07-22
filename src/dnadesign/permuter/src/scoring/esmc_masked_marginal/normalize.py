"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_masked_marginal/normalize.py

Normalize ESMC sequence logits into masked-marginal DMS rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from dnadesign.permuter.src.scoring.esmc_masked_marginal.contracts import (
    CANONICAL_AMINO_ACIDS,
    MASKED_MARGINAL_SCORING_METHOD_ID,
    MaskedMarginalJob,
    MaskedMarginalRows,
)


def normalize_masked_marginal_response(
    *,
    job: MaskedMarginalJob,
    logits_response: Mapping[str, Any],
    aa_token_indices: Mapping[str, int],
    model: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    retrieved_at: str,
) -> MaskedMarginalRows:
    """Convert one masked-position sequence-logits response into DMS-shaped rows."""

    _validate_token_map(aa_token_indices)
    logits = extract_sequence_logits(logits_response)
    logit_offset = _infer_sequence_logit_offset(token_count=len(logits), sequence_length=len(job.sequence))
    vector = np.asarray(logits[job.residue_index_zero_based + logit_offset], dtype=np.float64)
    if vector.ndim != 1 or vector.size <= max(aa_token_indices.values()):
        raise ValueError("sequence logits vector does not cover all canonical amino-acid token ids")
    log_probs = _log_softmax(vector)
    probabilities = np.exp(log_probs)
    wt_token_index = int(aa_token_indices[job.wt_aa])
    wt_log_probability = float(log_probs[wt_token_index])
    substitution_rows = _substitution_rows(
        job=job,
        aa_token_indices=aa_token_indices,
        log_probs=log_probs,
        wt_log_probability=wt_log_probability,
        model=model,
        biohub_request_hash=biohub_request_hash,
        biohub_query_hash=biohub_query_hash,
        retrieved_at=retrieved_at,
    )
    llrs = [float(row["llr"]) for row in substitution_rows]
    best = max(substitution_rows, key=lambda row: float(row["llr"]))
    worst = min(substitution_rows, key=lambda row: float(row["llr"]))
    logits_hash = _raw_response_hash(logits_response)
    position_row = {
        "sequence_id": job.sequence_id,
        "sequence_hash": job.sequence_hash,
        "model": model,
        "scoring_method_id": MASKED_MARGINAL_SCORING_METHOD_ID,
        "biohub_request_hash": biohub_request_hash,
        "biohub_query_hash": biohub_query_hash,
        "canonical_position": job.canonical_position,
        "residue_index_zero_based": job.residue_index_zero_based,
        "wt_aa": job.wt_aa,
        "masked_sequence_hash": job.masked_sequence_hash,
        "token_count": len(logits),
        "vocab_size": int(vector.size),
        "logit_residue_offset": logit_offset,
        "entropy_bits": float(-np.sum(probabilities * log_probs) / math.log(2.0)),
        "canonical_entropy_bits": _canonical_entropy_bits(log_probs, aa_token_indices),
        "wt_log_probability": wt_log_probability,
        "fraction_negative_alternate_llr": sum(1 for value in llrs if value < 0.0) / len(llrs),
        "best_alt_aa": str(best["alt_aa"]),
        "best_alt_llr": float(best["llr"]),
        "worst_alt_aa": str(worst["alt_aa"]),
        "worst_alt_llr": float(worst["llr"]),
        "raw_logits_response_hash": logits_hash,
        "retrieved_at": retrieved_at,
        "status": "accepted",
        "failure_reason": "",
    }
    return MaskedMarginalRows(position_row=position_row, substitution_rows=substitution_rows)


def extract_sequence_logits(response: Mapping[str, Any]) -> list[list[float]]:
    """Extract a token-by-vocabulary logits matrix from likely Biohub response shapes."""

    value: Any = response.get("logits")
    outputs = response.get("outputs")
    if value is None and isinstance(outputs, Mapping):
        value = outputs.get("logits") or outputs.get("sequence_logits")
    if isinstance(value, Mapping):
        value = value.get("sequence") or value.get("logits")
    if isinstance(value, Mapping):
        value = value.get("data") or value.get("values")
    if not isinstance(value, list):
        raise ValueError("Biohub logits response did not include JSON sequence logits")
    if _looks_like_batched_logits(value):
        value = value[0]
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("sequence logits must be a non-empty token-by-vocabulary matrix")
        matrix.append([float(entry) for entry in row])
    if not matrix:
        raise ValueError("sequence logits matrix must be non-empty")
    vocab_sizes = {len(row) for row in matrix}
    if len(vocab_sizes) != 1:
        raise ValueError("sequence logits rows must have a consistent vocabulary size")
    return matrix


def build_error_position_row(
    *,
    job: MaskedMarginalJob,
    model: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    retrieved_at: str,
    failure_reason: str,
) -> dict[str, object]:
    """Build an explicit position-level failure row."""

    return {
        "sequence_id": job.sequence_id,
        "sequence_hash": job.sequence_hash,
        "model": model,
        "scoring_method_id": MASKED_MARGINAL_SCORING_METHOD_ID,
        "biohub_request_hash": biohub_request_hash,
        "biohub_query_hash": biohub_query_hash,
        "canonical_position": job.canonical_position,
        "residue_index_zero_based": job.residue_index_zero_based,
        "wt_aa": job.wt_aa,
        "masked_sequence_hash": job.masked_sequence_hash,
        "token_count": None,
        "vocab_size": None,
        "logit_residue_offset": None,
        "entropy_bits": None,
        "canonical_entropy_bits": None,
        "wt_log_probability": None,
        "fraction_negative_alternate_llr": None,
        "best_alt_aa": "",
        "best_alt_llr": None,
        "worst_alt_aa": "",
        "worst_alt_llr": None,
        "raw_logits_response_hash": "",
        "retrieved_at": retrieved_at,
        "status": "errored",
        "failure_reason": _compact_failure_reason(failure_reason),
    }


def _substitution_rows(
    *,
    job: MaskedMarginalJob,
    aa_token_indices: Mapping[str, int],
    log_probs: np.ndarray,
    wt_log_probability: float,
    model: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    retrieved_at: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alt_aa in CANONICAL_AMINO_ACIDS:
        if alt_aa == job.wt_aa:
            continue
        alt_log_probability = float(log_probs[int(aa_token_indices[alt_aa])])
        rows.append(
            {
                "sequence_id": job.sequence_id,
                "sequence_hash": job.sequence_hash,
                "model": model,
                "scoring_method_id": MASKED_MARGINAL_SCORING_METHOD_ID,
                "biohub_request_hash": biohub_request_hash,
                "biohub_query_hash": biohub_query_hash,
                "canonical_position": job.canonical_position,
                "residue_index_zero_based": job.residue_index_zero_based,
                "wt_aa": job.wt_aa,
                "alt_aa": alt_aa,
                "masked_sequence_hash": job.masked_sequence_hash,
                "wt_log_probability": wt_log_probability,
                "alt_log_probability": alt_log_probability,
                "llr": alt_log_probability - wt_log_probability,
                "retrieved_at": retrieved_at,
                "status": "accepted",
            }
        )
    return rows


def _validate_token_map(aa_token_indices: Mapping[str, int]) -> None:
    missing = sorted(set(CANONICAL_AMINO_ACIDS) - set(aa_token_indices))
    if missing:
        raise ValueError(f"amino-acid token map is missing residue(s): {missing}")


def _looks_like_batched_logits(value: list[Any]) -> bool:
    return len(value) == 1 and isinstance(value[0], list) and bool(value[0]) and isinstance(value[0][0], list)


def _infer_sequence_logit_offset(*, token_count: int, sequence_length: int) -> int:
    if token_count == sequence_length + 2:
        return 1
    if token_count == sequence_length:
        return 0
    raise ValueError(f"sequence logits token count {token_count} does not match sequence length {sequence_length}")


def _log_softmax(vector: np.ndarray) -> np.ndarray:
    shifted = vector - np.max(vector)
    return shifted - np.log(np.sum(np.exp(shifted)))


def _canonical_entropy_bits(log_probs: np.ndarray, aa_token_indices: Mapping[str, int]) -> float:
    canonical_log_probs = np.array([log_probs[int(aa_token_indices[aa])] for aa in CANONICAL_AMINO_ACIDS])
    normalized = _log_softmax(canonical_log_probs)
    probabilities = np.exp(normalized)
    return float(-np.sum(probabilities * normalized) / math.log(2.0))


def _compact_failure_reason(reason: str, *, max_length: int = 500) -> str:
    compact = " ".join(str(reason).split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _raw_response_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
