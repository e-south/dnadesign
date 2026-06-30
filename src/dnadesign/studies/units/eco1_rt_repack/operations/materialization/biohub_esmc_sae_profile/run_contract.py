"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/run_contract.py

Run manifest and completion checks for Eco1 Biohub ESMC SAE-profile outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.constants import (
    DEFAULT_BIOHUB_API_VERSION,
)


def build_request_manifest(
    *,
    request_hash: str,
    source_request_hash: str,
    sequence_ids: list[str],
    biohub_api_base_url: str,
    model: str,
    sae_model: str,
    normalize_features: bool,
    key_label: str,
    selected_sequence_count: int,
    accepted_sequence_count: int,
    errored_sequence_count: int,
    max_new_requests: int | None,
    request_timeout_seconds: float,
    feature_description_summary: dict[str, object],
    retrieved_at: str,
) -> dict[str, object]:
    """Build the redacted run manifest for one Biohub ESMC SAE materialization."""

    return {
        "schema_id": "thread.biohub_esmc.request",
        "schema_version": 1,
        "biohub_request_hash": request_hash,
        "source_request_hash": source_request_hash,
        "biohub_api_base_url": biohub_api_base_url,
        "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
        "endpoint_flow": ["POST /api/v1/encode", "POST /api/v1/logits"],
        "model": model,
        "sae_model": sae_model,
        "normalize_features": normalize_features,
        "key_label": key_label,
        "authorization": "<redacted>",
        "method_references": [
            {
                "title": "Biohub ESMC SAE feature interpretation notebook",
                "url": (
                    "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
                    "esmc_sae_feature_interpretation.ipynb"
                ),
                "role": "SAE feature ranking, residue activation, and interpretation workflow reference",
            },
            {
                "title": "Biohub /api/v1/logits documentation",
                "url": "https://www.biohub.ai/api-reference/logits",
                "role": "Authenticated logits endpoint used for ESMC SAE outputs",
            },
            {
                "title": "Biohub ESMC SAE model card",
                "url": "https://huggingface.co/biohub/ESMC-6B-sae-layer60-k64-codebook16384",
                "role": (
                    "SAE model-family provenance, codebook-size semantics, top-k sparsity, "
                    "and source-backed feature-description availability for this dictionary"
                ),
            },
        ],
        "request_timeout_seconds": float(request_timeout_seconds),
        "feature_descriptions": feature_description_summary,
        "selected_sequence_count": selected_sequence_count,
        "accepted_sequence_count": accepted_sequence_count,
        "errored_sequence_count": errored_sequence_count,
        "materialization_mode": "complete_required" if max_new_requests is None else "resumable_capped",
        "materialization_status": "complete" if errored_sequence_count == 0 else "partial",
        "max_new_requests": max_new_requests,
        "selected_sequence_ids": sequence_ids,
        "retrieved_at": retrieved_at,
    }


def profile_status_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    """Count accepted and non-accepted Biohub profile rows."""

    accepted = sum(1 for row in rows if str(row.get("status") or "") == "accepted")
    errored = sum(1 for row in rows if str(row.get("status") or "") != "accepted")
    return {"accepted": accepted, "errored": errored}


def require_complete_final_run(
    *,
    profile_rows: list[dict[str, object]],
    selected_sequence_ids: list[str],
    max_new_requests: int | None,
) -> None:
    """Reject uncapped final materializations that left any selected sequence unaccepted."""

    if max_new_requests is not None or not selected_sequence_ids:
        return
    accepted_ids = {
        str(row.get("candidate_id") or "") for row in profile_rows if str(row.get("status") or "") == "accepted"
    }
    missing_or_errored = [sequence_id for sequence_id in selected_sequence_ids if sequence_id not in accepted_ids]
    if missing_or_errored:
        preview = ", ".join(missing_or_errored[:8])
        if len(missing_or_errored) > 8:
            preview += f", ... (+{len(missing_or_errored) - 8} more)"
        raise ValueError(
            "Complete Biohub ESMC SAE-profile materialization requires every selected sequence "
            f"to be accepted; missing_or_errored_sequence_ids: {preview}"
        )
