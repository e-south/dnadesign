"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sequence_pseudolikelihood/pipeline.py

Materialize Eco1 Biohub ESMC leave-one-out sequence pseudo-likelihood artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from dnadesign.permuter import (
    ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
    build_error_pseudolikelihood_position_row,
    build_pseudolikelihood_jobs,
    build_sequence_pseudolikelihood_rows,
    normalize_pseudolikelihood_response,
    validate_pseudolikelihood_artifacts,
    write_pseudolikelihood_artifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.selection import (
    BiohubEsmcSequenceRecord,
    select_fold_accepted_biohub_esmc_sequences,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_wt_mutation_scoring.selection import (  # noqa: E501
    parse_position_selection,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.thread.adapters.biohub_esmc import (
    BiohubEsmcClient,
    BiohubEsmcRequestError,
    biohub_query_hash,
    biohub_request_hash,
    load_biohub_credential,
)

from .constants import (
    DEFAULT_BIOHUB_API_BASE_URL,
    DEFAULT_BIOHUB_API_VERSION,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSITIONS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SEQUENCE_LIMIT,
    POSITION_PLL_FILE_NAME,
    REQUEST_MANIFEST_FILE_NAME,
    SEQUENCE_PLL_FILE_NAME,
    WT_SEQUENCE_ID,
    scoring_relative_root_for_model,
)
from .resume import CachedPseudolikelihoodRows, load_cached_rows


@dataclass(frozen=True)
class MaterializedBiohubEsmcSequencePseudolikelihoodArtifacts:
    """Paths emitted by one Eco1 sequence pseudo-likelihood materialization pass."""

    position_pll_path: Path
    sequence_pll_path: Path
    request_manifest_path: Path
    biohub_request_hash: str
    selected_sequence_count: int
    selected_position_count: int


def materialize_biohub_esmc_sequence_pseudolikelihood(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    sequence_limit: str = DEFAULT_SEQUENCE_LIMIT,
    candidate_ids: Sequence[str] = (),
    positions: str = DEFAULT_POSITIONS,
    biohub_api_base_url: str = DEFAULT_BIOHUB_API_BASE_URL,
    model: str = DEFAULT_MODEL,
    key_file: Path = DEFAULT_KEY_FILE,
    resume_existing: bool = False,
    max_new_requests: int | None = None,
    request_sleep_seconds: float = 0.0,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    biohub_client: Any | None = None,
    retrieved_at: str | None = None,
) -> MaterializedBiohubEsmcSequencePseudolikelihoodArtifacts:
    """Query Biohub sequence logits for Eco1 leave-one-out pseudo-likelihood scoring."""

    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be non-negative")
    if request_sleep_seconds < 0:
        raise ValueError("request_sleep_seconds must be non-negative")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    thread_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    scoring_root = thread_root / scoring_relative_root_for_model(model)
    selection = select_fold_accepted_biohub_esmc_sequences(output_root=thread_root, sequence_limit=sequence_limit)
    records = _select_records(selection.records, candidate_ids=tuple(candidate_ids))
    if not records or records[0].sequence_id != WT_SEQUENCE_ID:
        raise ValueError("Eco1 sequence pseudo-likelihood requires WT as the first selected sequence")
    selected_positions = parse_position_selection(positions, sequence_length=len(records[0].sequence))
    _require_uniform_sequence_lengths(records)
    request_hash = _biohub_request_hash(
        source_request_hash=selection.source_request_hash,
        records=records,
        positions=selected_positions,
        biohub_api_base_url=biohub_api_base_url,
        model=model,
    )
    timestamp = retrieved_at or _timestamp()
    client = biohub_client
    if client is None:
        credential = load_biohub_credential(key_file)
        client = BiohubEsmcClient(
            credential=credential,
            base_url=biohub_api_base_url,
            timeout_seconds=request_timeout_seconds,
        )
    key_label = str(getattr(getattr(client, "credential", None), "key_label", "injected-client"))
    cached = (
        load_cached_rows(scoring_root, request_hash=request_hash)
        if resume_existing
        else CachedPseudolikelihoodRows.empty()
    )
    token_map: dict[str, int] | None = None
    position_rows: list[dict[str, object]] = []
    logits_by_sequence_and_masked_sequence_hash: dict[tuple[str, str], dict[str, Any]] = {}
    new_request_count = 0
    for record in records:
        jobs = build_pseudolikelihood_jobs(
            sequence_id=record.sequence_id,
            sequence=record.sequence,
            positions=selected_positions,
        )
        for job in jobs:
            query_hash = _biohub_query_hash(
                source_request_hash=selection.source_request_hash,
                job_sequence_hash=job.sequence_hash,
                masked_sequence_hash=job.masked_sequence_hash,
                sequence_id=job.sequence_id,
                canonical_position=job.canonical_position,
                biohub_api_base_url=biohub_api_base_url,
                model=model,
            )
            cached_position = cached.position_row(job.sequence_id, job.canonical_position, query_hash)
            if cached_position is not None:
                position_rows.append(cached_position)
                continue
            response_cache_key = (job.sequence_hash, job.masked_sequence_hash)
            logits_response = logits_by_sequence_and_masked_sequence_hash.get(response_cache_key)
            if logits_response is None:
                if max_new_requests is not None and new_request_count >= max_new_requests:
                    position_rows.append(
                        build_error_pseudolikelihood_position_row(
                            job=job,
                            model=model,
                            biohub_request_hash=request_hash,
                            biohub_query_hash=query_hash,
                            retrieved_at=timestamp,
                            failure_reason="biohub_request_not_attempted_due_to_max_new_requests",
                        )
                    )
                    continue
                try:
                    if token_map is None:
                        token_map = dict(client.amino_acid_token_indices(model=model))
                    _encode_response, logits_response, _tokens = client.sequence_logits_for_sequence(
                        job.masked_sequence,
                        model=model,
                    )
                except (BiohubEsmcRequestError, OSError, ValueError) as error:
                    position_rows.append(
                        build_error_pseudolikelihood_position_row(
                            job=job,
                            model=model,
                            biohub_request_hash=request_hash,
                            biohub_query_hash=query_hash,
                            retrieved_at=timestamp,
                            failure_reason=str(error),
                        )
                    )
                    continue
                finally:
                    new_request_count += 1
                    if request_sleep_seconds:
                        sleep(request_sleep_seconds)
                logits_by_sequence_and_masked_sequence_hash[response_cache_key] = logits_response
            try:
                normalized = normalize_pseudolikelihood_response(
                    job=job,
                    logits_response=logits_response,
                    aa_token_indices=token_map,
                    model=model,
                    biohub_request_hash=request_hash,
                    biohub_query_hash=query_hash,
                    retrieved_at=timestamp,
                )
                position_rows.append(normalized.position_row)
            except ValueError as error:
                position_rows.append(
                    build_error_pseudolikelihood_position_row(
                        job=job,
                        model=model,
                        biohub_request_hash=request_hash,
                        biohub_query_hash=query_hash,
                        retrieved_at=timestamp,
                        failure_reason=str(error),
                    )
                )

    expected_lengths = {record.sequence_id: len(record.sequence) for record in records}
    sequence_rows = build_sequence_pseudolikelihood_rows(
        position_rows=position_rows,
        expected_lengths_by_sequence_id=expected_lengths,
        wt_sequence_id=WT_SEQUENCE_ID,
    )
    status_summary = _status_summary(sequence_rows)
    manifest = _request_manifest(
        request_hash=request_hash,
        source_request_hash=selection.source_request_hash,
        selected_sequence_count=len(records),
        selected_position_count=len(selected_positions),
        total_query_count=len(records) * len(selected_positions),
        biohub_api_base_url=biohub_api_base_url,
        model=model,
        key_label=key_label,
        request_timeout_seconds=request_timeout_seconds,
        token_map_hash=biohub_request_hash(token_map or {}),
        accepted_sequence_count=int(status_summary["accepted"]),
        partial_sequence_count=int(status_summary["partial"]),
        max_new_requests=max_new_requests,
        retrieved_at=timestamp,
    )
    artifacts = write_pseudolikelihood_artifacts(
        output_root=scoring_root,
        position_rows=position_rows,
        sequence_rows=sequence_rows,
        manifest=manifest,
        request_hash=request_hash,
        position_file_name=POSITION_PLL_FILE_NAME,
        sequence_file_name=SEQUENCE_PLL_FILE_NAME,
        manifest_file_name=REQUEST_MANIFEST_FILE_NAME,
    )
    issues = validate_pseudolikelihood_artifacts(
        artifacts=artifacts,
        expected_sequence_count=len(records),
        request_hash=request_hash,
    )
    if issues:
        raise ValueError("Biohub ESMC sequence pseudo-likelihood validation failed: " + "; ".join(issues))
    _require_complete_final_run(
        sequence_rows=sequence_rows,
        records=records,
        selected_positions=selected_positions,
        max_new_requests=max_new_requests,
    )
    return MaterializedBiohubEsmcSequencePseudolikelihoodArtifacts(
        position_pll_path=artifacts.position_pll_path,
        sequence_pll_path=artifacts.sequence_pll_path,
        request_manifest_path=artifacts.manifest_path,
        biohub_request_hash=request_hash,
        selected_sequence_count=len(records),
        selected_position_count=len(selected_positions),
    )


def _select_records(
    records: tuple[BiohubEsmcSequenceRecord, ...],
    *,
    candidate_ids: tuple[str, ...],
) -> tuple[BiohubEsmcSequenceRecord, ...]:
    if not candidate_ids:
        return records
    requested = {WT_SEQUENCE_ID, *[candidate_id.strip() for candidate_id in candidate_ids if candidate_id.strip()]}
    by_id = {record.sequence_id: record for record in records}
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"candidate_id(s) are not present in selected fold-check sequences: {missing}")
    return tuple(record for record in records if record.sequence_id in requested)


def _require_uniform_sequence_lengths(records: tuple[BiohubEsmcSequenceRecord, ...]) -> None:
    lengths = {len(record.sequence) for record in records}
    if len(lengths) != 1:
        raise ValueError(f"sequence pseudo-likelihood requires uniform sequence lengths, observed {sorted(lengths)}")


def _biohub_request_hash(
    *,
    source_request_hash: str,
    records: tuple[BiohubEsmcSequenceRecord, ...],
    positions: tuple[int, ...],
    biohub_api_base_url: str,
    model: str,
) -> str:
    return biohub_request_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_hashes": [record.sequence_hash for record in records],
            "sequence_ids": [record.sequence_id for record in records],
            "positions": positions,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "scoring_method_id": ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
        }
    )


def _biohub_query_hash(
    *,
    source_request_hash: str,
    job_sequence_hash: str,
    masked_sequence_hash: str,
    sequence_id: str,
    canonical_position: int,
    biohub_api_base_url: str,
    model: str,
) -> str:
    return biohub_query_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_id": sequence_id,
            "sequence_hash": job_sequence_hash,
            "masked_sequence_hash": masked_sequence_hash,
            "canonical_position": canonical_position,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "scoring_method_id": ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
        }
    )


def _request_manifest(
    *,
    request_hash: str,
    source_request_hash: str,
    selected_sequence_count: int,
    selected_position_count: int,
    total_query_count: int,
    biohub_api_base_url: str,
    model: str,
    key_label: str,
    request_timeout_seconds: float,
    token_map_hash: str,
    accepted_sequence_count: int,
    partial_sequence_count: int,
    max_new_requests: int | None,
    retrieved_at: str,
) -> dict[str, object]:
    return {
        "schema_id": "eco1_rt_repack.biohub_esmc_sequence_pseudolikelihood.request",
        "schema_version": 1,
        "biohub_request_hash": request_hash,
        "source_request_hash": source_request_hash,
        "selected_sequence_count": selected_sequence_count,
        "selected_position_count": selected_position_count,
        "total_query_count": total_query_count,
        "biohub_api_base_url": biohub_api_base_url,
        "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
        "endpoint_flow": ["POST /api/v1/encode", "POST /api/v1/logits"],
        "model": model,
        "scoring_method_id": ESMC_PSEUDOLIKELIHOOD_METHOD_ID,
        "method_summary": (
            "For each sequence and residue position, mask that residue, request Biohub ESMC sequence logits, "
            "and store log P(observed residue | remaining sequence). Per-sequence pseudo-log-likelihood is the "
            "sum over positions and is reported with a per-residue mean and pseudo-perplexity."
        ),
        "interpretation_limit": (
            "This is a masked-language-model pseudo-likelihood, not a calibrated probability that a protein "
            "occurs in nature, not a joint likelihood, not fold validation, and not activity evidence."
        ),
        "method_references": [
            {
                "title": "Biohub ESMC mutation-scoring notebook",
                "url": (
                    "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
                    "esmc_mutation_scoring.ipynb"
                ),
                "role": "Biohub encode-to-logits and masked-sequence scoring pattern",
            },
            {
                "title": "Biohub logits API reference",
                "url": "https://www.biohub.ai/api-reference/logits",
                "role": "public API endpoint for sequence logits",
            },
            {
                "title": "Salazar et al. 2020 masked language model scoring",
                "url": "https://aclanthology.org/2020.acl-main.240/",
                "role": "pseudo-log-likelihood scoring rationale for masked language models",
            },
        ],
        "key_label": key_label,
        "authorization": "<redacted>",
        "request_timeout_seconds": float(request_timeout_seconds),
        "token_map_hash": token_map_hash,
        "accepted_sequence_count": accepted_sequence_count,
        "partial_sequence_count": partial_sequence_count,
        "materialization_mode": "complete_required" if max_new_requests is None else "resumable_capped",
        "materialization_status": "complete" if partial_sequence_count == 0 else "partial",
        "max_new_requests": max_new_requests,
        "retrieved_at": retrieved_at,
    }


def _status_summary(sequence_rows: list[dict[str, object]]) -> dict[str, int]:
    accepted = sum(1 for row in sequence_rows if str(row.get("status") or "") == "accepted")
    partial = len(sequence_rows) - accepted
    return {"accepted": accepted, "partial": partial}


def _require_complete_final_run(
    *,
    sequence_rows: list[dict[str, object]],
    records: tuple[BiohubEsmcSequenceRecord, ...],
    selected_positions: tuple[int, ...],
    max_new_requests: int | None,
) -> None:
    if max_new_requests is not None:
        return
    sequence_length = len(records[0].sequence)
    if len(selected_positions) != sequence_length:
        raise ValueError(
            "Complete Biohub ESMC sequence pseudo-likelihood requires all positions; "
            "set --max-new-requests for a smoke run or pass --positions all for final scoring."
        )
    partial = [str(row["sequence_id"]) for row in sequence_rows if row.get("status") != "accepted"]
    if partial:
        preview = ", ".join(partial[:12])
        if len(partial) > 12:
            preview += f", ... (+{len(partial) - 12} more)"
        raise ValueError(
            "Complete Biohub ESMC sequence pseudo-likelihood requires every sequence to have accepted "
            f"position rows for all residues; partial sequences: {preview}"
        )


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
