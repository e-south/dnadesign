"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/pipeline.py

Materialize WT-only Biohub ESMC masked-marginal mutation scoring artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from dnadesign.permuter import (
    ProteinDmsRequest,
    build_error_position_row,
    build_masked_marginal_jobs,
    normalize_masked_marginal_response,
    render_masked_marginal_plots,
    validate_masked_marginal_artifacts,
    write_masked_marginal_artifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
    sha256,
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
    MASK_JOIN_FILE_NAME,
    MASK_SET_FILE_NAME,
    PLOTS_DIR_NAME,
    POSITION_ENTROPY_FILE_NAME,
    REQUEST_MANIFEST_FILE_NAME,
    SCORING_RELATIVE_ROOT,
    SUBSTITUTION_LLR_FILE_NAME,
)
from .mask_join import write_mask_join
from .plot_context import build_position_context_spans
from .resume import (
    CachedMutationScoringRows,
    load_cached_rows,
)
from .selection import select_wt_mutation_scoring_sequence


@dataclass(frozen=True)
class MaterializedBiohubEsmcWtMutationScoringArtifacts:
    """Paths emitted by one Eco1 WT mutation-scoring materialization pass."""

    position_entropy_path: Path
    substitution_llr_path: Path
    mask_join_path: Path
    request_manifest_path: Path
    plots_root: Path
    biohub_request_hash: str
    selected_position_count: int


def materialize_biohub_esmc_wt_mutation_scoring(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
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
) -> MaterializedBiohubEsmcWtMutationScoringArtifacts:
    """Query Biohub sequence logits for masked WT Eco1 contexts and write DMS tables."""

    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be non-negative")
    if request_sleep_seconds < 0:
        raise ValueError("request_sleep_seconds must be non-negative")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    thread_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    scoring_root = thread_root / SCORING_RELATIVE_ROOT
    selection = select_wt_mutation_scoring_sequence(output_root=thread_root, positions=positions)
    jobs = build_masked_marginal_jobs(
        ProteinDmsRequest(ref_name=selection.sequence_id, sequence=selection.sequence, positions=selection.positions),
        sequence_id=selection.sequence_id,
    )
    request_hash = _biohub_request_hash(
        source_request_hash=selection.source_request_hash,
        sequence_hash=selection.sequence_hash,
        positions=selection.positions,
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
        else CachedMutationScoringRows.empty()
    )
    position_rows: list[dict[str, object]] = []
    substitution_rows: list[dict[str, object]] = []
    token_map: dict[str, int] | None = None
    new_request_count = 0
    for job in jobs:
        query_hash = _biohub_query_hash(
            source_request_hash=selection.source_request_hash,
            masked_sequence_hash=job.masked_sequence_hash,
            canonical_position=job.canonical_position,
            biohub_api_base_url=biohub_api_base_url,
            model=model,
        )
        cached_position = cached.position_row(job.canonical_position, query_hash)
        if cached_position is not None:
            position_rows.append(cached_position)
            substitution_rows.extend(cached.substitution_rows(job.canonical_position, query_hash))
            continue
        if max_new_requests is not None and new_request_count >= max_new_requests:
            position_rows.append(
                build_error_position_row(
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
            normalized = normalize_masked_marginal_response(
                job=job,
                logits_response=logits_response,
                aa_token_indices=token_map,
                model=model,
                biohub_request_hash=request_hash,
                biohub_query_hash=query_hash,
                retrieved_at=timestamp,
            )
            position_rows.append(normalized.position_row)
            substitution_rows.extend(normalized.substitution_rows)
        except (BiohubEsmcRequestError, OSError, RuntimeError, ValueError) as error:
            position_rows.append(
                build_error_position_row(
                    job=job,
                    model=model,
                    biohub_request_hash=request_hash,
                    biohub_query_hash=query_hash,
                    retrieved_at=timestamp,
                    failure_reason=str(error),
                )
            )
        finally:
            new_request_count += 1
            if request_sleep_seconds:
                sleep(request_sleep_seconds)

    token_map_hash = biohub_request_hash(token_map or {})
    manifest = _request_manifest(
        request_hash=request_hash,
        source_request_hash=selection.source_request_hash,
        sequence_hash=selection.sequence_hash,
        sequence_id=selection.sequence_id,
        positions=selection.positions,
        biohub_api_base_url=biohub_api_base_url,
        model=model,
        key_label=key_label,
        request_timeout_seconds=request_timeout_seconds,
        token_map_hash=token_map_hash,
        mask_set_hash="sha256:" + sha256(thread_root / MASK_SET_FILE_NAME),
        retrieved_at=timestamp,
    )
    artifacts = write_masked_marginal_artifacts(
        output_root=scoring_root,
        position_rows=position_rows,
        substitution_rows=substitution_rows,
        manifest=manifest,
        request_hash=request_hash,
        position_file_name=POSITION_ENTROPY_FILE_NAME,
        substitution_file_name=SUBSTITUTION_LLR_FILE_NAME,
        manifest_file_name=REQUEST_MANIFEST_FILE_NAME,
    )
    issues = validate_masked_marginal_artifacts(
        artifacts=artifacts,
        expected_position_count=len(jobs),
        request_hash=request_hash,
    )
    if issues:
        raise ValueError("Biohub ESMC WT mutation scoring validation failed: " + "; ".join(issues))
    mask_join_path = write_mask_join(
        position_entropy_path=artifacts.position_entropy_path,
        mask_set_path=thread_root / MASK_SET_FILE_NAME,
        output_path=scoring_root / MASK_JOIN_FILE_NAME,
        request_hash=request_hash,
    )
    plot_artifacts = render_masked_marginal_plots(
        position_entropy_path=artifacts.position_entropy_path,
        substitution_llr_path=artifacts.substitution_llr_path,
        output_root=scoring_root / PLOTS_DIR_NAME,
        file_prefix="wt_",
        position_context_spans=build_position_context_spans(thread_root / MASK_SET_FILE_NAME),
    )
    del plot_artifacts
    return MaterializedBiohubEsmcWtMutationScoringArtifacts(
        position_entropy_path=artifacts.position_entropy_path,
        substitution_llr_path=artifacts.substitution_llr_path,
        mask_join_path=mask_join_path,
        request_manifest_path=artifacts.manifest_path,
        plots_root=scoring_root / PLOTS_DIR_NAME,
        biohub_request_hash=request_hash,
        selected_position_count=len(jobs),
    )


def _biohub_request_hash(
    *,
    source_request_hash: str,
    sequence_hash: str,
    positions: tuple[int, ...],
    biohub_api_base_url: str,
    model: str,
) -> str:
    return biohub_request_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_hash": sequence_hash,
            "positions": positions,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "scoring_method_id": "esmc_masked_marginal_v1",
        }
    )


def _biohub_query_hash(
    *,
    source_request_hash: str,
    masked_sequence_hash: str,
    canonical_position: int,
    biohub_api_base_url: str,
    model: str,
) -> str:
    return biohub_query_hash(
        {
            "source_request_hash": source_request_hash,
            "masked_sequence_hash": masked_sequence_hash,
            "canonical_position": canonical_position,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "scoring_method_id": "esmc_masked_marginal_v1",
        }
    )


def _request_manifest(
    *,
    request_hash: str,
    source_request_hash: str,
    sequence_hash: str,
    sequence_id: str,
    positions: tuple[int, ...],
    biohub_api_base_url: str,
    model: str,
    key_label: str,
    request_timeout_seconds: float,
    token_map_hash: str,
    mask_set_hash: str,
    retrieved_at: str,
) -> dict[str, object]:
    return {
        "schema_id": "eco1_rt_repack.biohub_esmc_wt_mutation_scoring.request",
        "schema_version": 1,
        "biohub_request_hash": request_hash,
        "source_request_hash": source_request_hash,
        "sequence_id": sequence_id,
        "sequence_hash": sequence_hash,
        "positions": list(positions),
        "position_count": len(positions),
        "biohub_api_base_url": biohub_api_base_url,
        "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
        "endpoint_flow": ["POST /api/v1/encode", "POST /api/v1/logits"],
        "model": model,
        "scoring_method_id": "esmc_masked_marginal_v1",
        "method_references": [
            {
                "title": "Biohub ESMC mutation-scoring notebook",
                "url": (
                    "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
                    "esmc_mutation_scoring.ipynb"
                ),
                "role": "masked-marginal entropy and zero-shot LLR method pattern",
            },
            {
                "title": "Biohub logits API reference",
                "url": "https://www.biohub.ai/api-reference/logits",
                "role": "public API endpoint for sequence logits",
            },
            {
                "title": "Candido et al. 2026 ESM world-model preprint",
                "url": "https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1",
                "role": "primary literature context for ESMC and model-derived protein representations",
            },
        ],
        "evidence_role": "model_constraint_audit_v1",
        "changes_current_mask": False,
        "key_label": key_label,
        "authorization": "<redacted>",
        "request_timeout_seconds": float(request_timeout_seconds),
        "token_map_hash": token_map_hash,
        "artifact_hashes": {"mask_set": mask_set_hash},
        "retrieved_at": retrieved_at,
    }


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
