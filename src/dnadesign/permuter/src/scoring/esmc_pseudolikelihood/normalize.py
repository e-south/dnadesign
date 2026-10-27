"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_pseudolikelihood/normalize.py

Normalize ESMC sequence logits into leave-one-out pseudo-likelihood rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dnadesign.permuter.src.scoring.esmc_masked_marginal.normalize import (
    _infer_sequence_logit_offset,
    _log_softmax,
    _raw_response_hash,
    extract_sequence_logits,
)
from dnadesign.permuter.src.scoring.esmc_pseudolikelihood.contracts import (
    ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
    EsmcPseudolikelihoodJob,
    EsmcPseudolikelihoodRows,
)


def normalize_pseudolikelihood_response(
    *,
    job: EsmcPseudolikelihoodJob,
    logits_response: Mapping[str, Any],
    aa_token_indices: Mapping[str, int],
    model: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    retrieved_at: str,
) -> EsmcPseudolikelihoodRows:
    """Convert one masked-position sequence-logits response into a PLL position row."""

    _validate_token_map(job.residue, aa_token_indices)
    logits = extract_sequence_logits(logits_response)
    logit_offset = _infer_sequence_logit_offset(token_count=len(logits), sequence_length=len(job.sequence))
    vector = np.asarray(logits[job.residue_index_zero_based + logit_offset], dtype=np.float64)
    residue_token_index = int(aa_token_indices[job.residue])
    if vector.ndim != 1 or vector.size <= residue_token_index:
        raise ValueError("sequence logits vector does not cover the residue token id")
    log_probs = _log_softmax(vector)
    residue_log_probability = float(log_probs[residue_token_index])
    return EsmcPseudolikelihoodRows(
        position_row={
            "sequence_id": job.sequence_id,
            "sequence_hash": job.sequence_hash,
            "model": model,
            "scoring_method_id": ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
            "biohub_request_hash": biohub_request_hash,
            "biohub_query_hash": biohub_query_hash,
            "canonical_position": job.canonical_position,
            "residue_index_zero_based": job.residue_index_zero_based,
            "residue": job.residue,
            "masked_sequence_hash": job.masked_sequence_hash,
            "token_count": len(logits),
            "vocab_size": int(vector.size),
            "logit_residue_offset": logit_offset,
            "residue_log_probability": residue_log_probability,
            "raw_logits_response_hash": _raw_response_hash(logits_response),
            "retrieved_at": retrieved_at,
            "status": "accepted",
            "failure_reason": "",
        }
    )


def build_error_pseudolikelihood_position_row(
    *,
    job: EsmcPseudolikelihoodJob,
    model: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    retrieved_at: str,
    failure_reason: str,
) -> dict[str, object]:
    """Build an explicit pseudo-likelihood position failure row."""

    return {
        "sequence_id": job.sequence_id,
        "sequence_hash": job.sequence_hash,
        "model": model,
        "scoring_method_id": ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
        "biohub_request_hash": biohub_request_hash,
        "biohub_query_hash": biohub_query_hash,
        "canonical_position": job.canonical_position,
        "residue_index_zero_based": job.residue_index_zero_based,
        "residue": job.residue,
        "masked_sequence_hash": job.masked_sequence_hash,
        "token_count": None,
        "vocab_size": None,
        "logit_residue_offset": None,
        "residue_log_probability": None,
        "raw_logits_response_hash": "",
        "retrieved_at": retrieved_at,
        "status": "errored",
        "failure_reason": _compact_failure_reason(failure_reason),
    }


def _validate_token_map(residue: str, aa_token_indices: Mapping[str, int]) -> None:
    if residue not in aa_token_indices:
        raise ValueError(f"amino-acid token map is missing residue {residue!r}")


def _compact_failure_reason(reason: str, *, max_length: int = 500) -> str:
    compact = " ".join(str(reason).split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."
