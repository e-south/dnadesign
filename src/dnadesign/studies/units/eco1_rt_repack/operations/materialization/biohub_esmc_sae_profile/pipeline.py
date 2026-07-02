"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/pipeline.py

Materialize Eco1 Biohub ESMC SAE-profile artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.constants import (
    DEFAULT_BIOHUB_API_BASE_URL,
    DEFAULT_BIOHUB_API_VERSION,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    DEFAULT_NORMALIZE_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SAE_MODEL,
    DEFAULT_SEQUENCE_LIMIT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.resume import (
    cached_profile_row,
    load_existing_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.run_contract import (
    build_request_manifest,
    profile_status_summary,
    require_complete_final_run,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.selection import (
    select_fold_accepted_biohub_esmc_sequences,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.thread.adapters.biohub_esmc import (
    FEATURE_DESCRIPTION_SAE_MODEL,
    BiohubEsmcClient,
    BiohubEsmcRequestError,
    BiohubSaeFeatureDescriptionClient,
    BiohubSaeFeatureDescriptionError,
    biohub_query_hash,
    biohub_request_hash,
    build_error_profile_row,
    load_biohub_credential,
    normalize_logits_response,
    supports_feature_description_endpoint,
    validate_biohub_esmc_artifacts,
    write_biohub_esmc_artifacts,
    write_biohub_esmc_feature_catalog,
)

FEATURE_DESCRIPTION_MANIFEST_FILE_NAME = "biohub_esmc_feature_description_manifest.yaml"


@dataclass(frozen=True)
class MaterializedBiohubEsmcSaeProfileArtifacts:
    """Paths emitted by one Biohub ESMC SAE-profile materialization pass."""

    profile_path: Path
    protein_features_path: Path
    residue_features_path: Path
    feature_catalog_path: Path
    request_manifest_path: Path
    biohub_request_hash: str
    selected_sequence_count: int


@dataclass(frozen=True)
class MaterializedBiohubEsmcFeatureDescriptions:
    """Paths emitted by feature-catalog-only description enrichment."""

    feature_catalog_path: Path
    manifest_path: Path
    observed_feature_count: int
    enriched_feature_count: int


def materialize_biohub_esmc_sae_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    sequence_limit: str = DEFAULT_SEQUENCE_LIMIT,
    biohub_api_base_url: str = DEFAULT_BIOHUB_API_BASE_URL,
    model: str = DEFAULT_MODEL,
    sae_model: str = DEFAULT_SAE_MODEL,
    normalize_features: bool = DEFAULT_NORMALIZE_FEATURES,
    key_file: Path = DEFAULT_KEY_FILE,
    resume_existing: bool = False,
    max_new_requests: int | None = None,
    request_sleep_seconds: float = 0.0,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    biohub_client: Any | None = None,
    fetch_feature_descriptions: bool = False,
    feature_description_limit: int | None = None,
    feature_description_sleep_seconds: float = 0.0,
    feature_description_client: Any | None = None,
    retrieved_at: str | None = None,
) -> MaterializedBiohubEsmcSaeProfileArtifacts:
    """Query Biohub ESMC logits for fold-accepted Eco1 sequences and write sparse SAE tables."""

    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be non-negative")
    if request_sleep_seconds < 0:
        raise ValueError("request_sleep_seconds must be non-negative")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if feature_description_limit is not None and feature_description_limit < 0:
        raise ValueError("feature_description_limit must be non-negative")
    if feature_description_sleep_seconds < 0:
        raise ValueError("feature_description_sleep_seconds must be non-negative")
    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    selection = select_fold_accepted_biohub_esmc_sequences(output_root=out_root, sequence_limit=sequence_limit)
    request_hash = _biohub_request_hash(
        source_request_hash=selection.source_request_hash,
        sequence_ids=[record.sequence_id for record in selection.records],
        sequence_hashes=[record.sequence_hash for record in selection.records],
        biohub_api_base_url=biohub_api_base_url,
        model=model,
        sae_model=sae_model,
        normalize_features=normalize_features,
    )
    timestamp = retrieved_at or _timestamp()
    credential = None
    client = biohub_client
    if client is None:
        credential = load_biohub_credential(key_file)
        client = BiohubEsmcClient(
            credential=credential,
            base_url=biohub_api_base_url,
            timeout_seconds=request_timeout_seconds,
        )
    key_label = str(getattr(getattr(client, "credential", None), "key_label", "injected-client"))
    existing_rows = load_existing_rows(out_root) if resume_existing else None

    profile_rows: list[dict[str, object]] = []
    protein_feature_rows: list[dict[str, object]] = []
    residue_feature_rows: list[dict[str, object]] = []
    feature_catalog_rows: list[dict[str, object]] = []
    new_request_count = 0
    for record in selection.records:
        query_hash = _biohub_query_hash(
            source_request_hash=selection.source_request_hash,
            sequence_hash=record.sequence_hash,
            biohub_api_base_url=biohub_api_base_url,
            model=model,
            sae_model=sae_model,
            normalize_features=normalize_features,
        )
        cached = cached_profile_row(
            existing_rows=existing_rows,
            candidate_id=record.sequence_id,
            sequence_hash=record.sequence_hash,
            biohub_query_hash=query_hash,
            biohub_request_hash=request_hash,
            source_request_hash=selection.source_request_hash,
        )
        if cached is not None:
            cached_protein_rows = existing_rows.protein_feature_rows(record.sequence_id) if existing_rows else []
            cached_residue_rows = existing_rows.residue_feature_rows(record.sequence_id) if existing_rows else []
            if _cached_feature_rows_complete(
                profile_row=cached,
                protein_feature_rows=cached_protein_rows,
                residue_feature_rows=cached_residue_rows,
            ):
                profile_rows.append(cached)
                protein_feature_rows.extend(cached_protein_rows)
                residue_feature_rows.extend(cached_residue_rows)
                continue
        if max_new_requests is not None and new_request_count >= max_new_requests:
            profile_rows.append(
                build_error_profile_row(
                    candidate_id=record.sequence_id,
                    sequence=record.sequence,
                    sequence_hash=record.sequence_hash,
                    source_request_hash=selection.source_request_hash,
                    biohub_request_hash=request_hash,
                    biohub_query_hash=query_hash,
                    biohub_api_base_url=biohub_api_base_url,
                    biohub_api_version=DEFAULT_BIOHUB_API_VERSION,
                    model=model,
                    sae_model=sae_model,
                    normalize_features=normalize_features,
                    key_label=key_label,
                    retrieved_at=timestamp,
                    failure_reason="biohub_request_not_attempted_due_to_max_new_requests",
                )
            )
            continue
        try:
            encode_response, logits_response, _tokens = client.logits_for_sequence(
                record.sequence,
                model=model,
                sae_model=sae_model,
                normalize_features=normalize_features,
            )
            normalized = normalize_logits_response(
                candidate_id=record.sequence_id,
                sequence=record.sequence,
                sequence_hash=record.sequence_hash,
                encode_response=encode_response,
                logits_response=logits_response,
                source_request_hash=selection.source_request_hash,
                biohub_request_hash=request_hash,
                biohub_query_hash=query_hash,
                biohub_api_base_url=biohub_api_base_url,
                biohub_api_version=DEFAULT_BIOHUB_API_VERSION,
                model=model,
                sae_model=sae_model,
                normalize_features=normalize_features,
                key_label=key_label,
                retrieved_at=timestamp,
            )
            profile_rows.append(normalized.profile_row)
            protein_feature_rows.extend(normalized.protein_feature_rows)
            residue_feature_rows.extend(normalized.residue_feature_rows)
            feature_catalog_rows.extend(normalized.feature_catalog_rows)
        except (BiohubEsmcRequestError, OSError) as error:
            profile_rows.append(
                build_error_profile_row(
                    candidate_id=record.sequence_id,
                    sequence=record.sequence,
                    sequence_hash=record.sequence_hash,
                    source_request_hash=selection.source_request_hash,
                    biohub_request_hash=request_hash,
                    biohub_query_hash=query_hash,
                    biohub_api_base_url=biohub_api_base_url,
                    biohub_api_version=DEFAULT_BIOHUB_API_VERSION,
                    model=model,
                    sae_model=sae_model,
                    normalize_features=normalize_features,
                    key_label=key_label,
                    retrieved_at=timestamp,
                    failure_reason=str(error),
                )
            )
        finally:
            new_request_count += 1
            if request_sleep_seconds:
                sleep(request_sleep_seconds)

    if existing_rows is not None:
        feature_catalog_rows.extend(existing_rows.feature_catalog_rows)

    feature_catalog_rows, feature_description_summary = _maybe_enrich_feature_catalog_rows(
        feature_catalog_rows,
        sae_model=sae_model,
        retrieved_at=timestamp,
        fetch_feature_descriptions=fetch_feature_descriptions,
        feature_description_limit=feature_description_limit,
        feature_description_sleep_seconds=feature_description_sleep_seconds,
        feature_description_client=feature_description_client,
        biohub_api_base_url=biohub_api_base_url,
        request_timeout_seconds=request_timeout_seconds,
    )
    status_summary = profile_status_summary(profile_rows)
    request_manifest = build_request_manifest(
        request_hash=request_hash,
        source_request_hash=selection.source_request_hash,
        sequence_ids=[record.sequence_id for record in selection.records],
        biohub_api_base_url=biohub_api_base_url,
        model=model,
        sae_model=sae_model,
        normalize_features=normalize_features,
        key_label=key_label,
        selected_sequence_count=len(selection.records),
        accepted_sequence_count=int(status_summary["accepted"]),
        errored_sequence_count=int(status_summary["errored"]),
        max_new_requests=max_new_requests,
        request_timeout_seconds=request_timeout_seconds,
        feature_description_summary=feature_description_summary,
        retrieved_at=timestamp,
    )
    artifacts = write_biohub_esmc_artifacts(
        output_root=out_root,
        profile_rows=profile_rows,
        protein_feature_rows=protein_feature_rows,
        residue_feature_rows=residue_feature_rows,
        feature_catalog_rows=feature_catalog_rows,
        request_manifest=request_manifest,
        request_hash=request_hash,
    )
    issues = validate_biohub_esmc_artifacts(
        output_root=out_root,
        expected_candidate_ids={record.sequence_id for record in selection.records},
        request_hash=request_hash,
    )
    if issues:
        joined = "; ".join(f"{issue.check_id}: {issue.message}" for issue in issues)
        raise ValueError(f"Biohub ESMC SAE-profile validation failed: {joined}")
    require_complete_final_run(
        profile_rows=profile_rows,
        selected_sequence_ids=[record.sequence_id for record in selection.records],
        max_new_requests=max_new_requests,
    )
    return MaterializedBiohubEsmcSaeProfileArtifacts(
        profile_path=artifacts.profile_path,
        protein_features_path=artifacts.protein_features_path,
        residue_features_path=artifacts.residue_features_path,
        feature_catalog_path=artifacts.feature_catalog_path,
        request_manifest_path=artifacts.request_manifest_path,
        biohub_request_hash=request_hash,
        selected_sequence_count=len(selection.records),
    )


def enrich_existing_biohub_esmc_feature_catalog(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    biohub_api_base_url: str = DEFAULT_BIOHUB_API_BASE_URL,
    sae_model: str = DEFAULT_SAE_MODEL,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    feature_description_limit: int | None = None,
    feature_description_batch_size: int | None = 100,
    feature_description_sleep_seconds: float = 0.0,
    feature_description_client: Any | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    retrieved_at: str | None = None,
) -> MaterializedBiohubEsmcFeatureDescriptions:
    """Enrich an existing Biohub ESMC feature catalog without rebuilding large SAE tables."""

    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if feature_description_limit is not None and feature_description_limit < 0:
        raise ValueError("feature_description_limit must be non-negative")
    if feature_description_batch_size is not None and feature_description_batch_size <= 0:
        raise ValueError("feature_description_batch_size must be positive")
    if feature_description_sleep_seconds < 0:
        raise ValueError("feature_description_sleep_seconds must be non-negative")
    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    existing_rows = load_existing_rows(out_root)
    if existing_rows is None or existing_rows.feature_catalog_path is None:
        raise FileNotFoundError(out_root / "biohub_esmc_feature_catalog.parquet")
    feature_catalog_rows = existing_rows.feature_catalog_rows
    if not feature_catalog_rows:
        raise ValueError("Biohub ESMC feature catalog contains no rows to enrich")
    request_hash = _existing_request_hash(existing_rows.profile_rows_by_candidate)
    timestamp = retrieved_at or _timestamp()
    attempted_count = 0
    new_enriched_count = 0
    batch_count = 0
    remaining_limit = feature_description_limit
    feature_description_summary: dict[str, object] | None = None
    enriched_rows = feature_catalog_rows
    while True:
        effective_limit = feature_description_batch_size
        if remaining_limit is not None:
            effective_limit = remaining_limit if effective_limit is None else min(effective_limit, remaining_limit)
        if effective_limit == 0:
            break
        enriched_rows, batch_summary = _maybe_enrich_feature_catalog_rows(
            enriched_rows,
            sae_model=sae_model,
            retrieved_at=timestamp,
            fetch_feature_descriptions=True,
            feature_description_limit=effective_limit,
            feature_description_sleep_seconds=feature_description_sleep_seconds,
            feature_description_client=feature_description_client,
            biohub_api_base_url=biohub_api_base_url,
            request_timeout_seconds=request_timeout_seconds,
        )
        batch_count += 1
        attempted_count += int(batch_summary["attempted_feature_count"])
        new_enriched_count += int(batch_summary.get("new_enriched_feature_count", 0))
        feature_catalog_path = write_biohub_esmc_feature_catalog(
            existing_rows.feature_catalog_path,
            enriched_rows,
            request_hash=request_hash,
        )
        batch_summary.update(
            {
                "batch_count": batch_count,
                "cumulative_attempted_feature_count": attempted_count,
                "cumulative_new_enriched_feature_count": new_enriched_count,
                "feature_catalog_path": str(feature_catalog_path),
            }
        )
        if progress_callback is not None:
            progress_callback(batch_summary)
        feature_description_summary = batch_summary
        attempted_in_batch = int(batch_summary["attempted_feature_count"])
        if remaining_limit is not None:
            remaining_limit -= attempted_in_batch
        if (
            attempted_in_batch == 0
            or int(batch_summary.get("missing_feature_description_count", 0)) == 0
            or (remaining_limit is not None and remaining_limit <= 0)
        ):
            break
    if feature_description_summary is None:
        enriched_rows, feature_description_summary = _maybe_enrich_feature_catalog_rows(
            enriched_rows,
            sae_model=sae_model,
            retrieved_at=timestamp,
            fetch_feature_descriptions=True,
            feature_description_limit=0,
            feature_description_sleep_seconds=feature_description_sleep_seconds,
            feature_description_client=feature_description_client,
            biohub_api_base_url=biohub_api_base_url,
            request_timeout_seconds=request_timeout_seconds,
        )
    missing_count = int(feature_description_summary.get("missing_feature_description_count", 0))
    if missing_count:
        feature_description_summary["status"] = "limit_reached"
    else:
        feature_description_summary["status"] = "enriched"
    feature_description_summary.update(
        {
            "attempted_feature_count": attempted_count,
            "new_enriched_feature_count": new_enriched_count,
            "batch_size": feature_description_batch_size,
            "batch_count": batch_count,
        }
    )
    feature_catalog_path = write_biohub_esmc_feature_catalog(
        existing_rows.feature_catalog_path,
        enriched_rows,
        request_hash=request_hash,
    )
    manifest_path = out_root / FEATURE_DESCRIPTION_MANIFEST_FILE_NAME
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.biohub_esmc.feature_description_enrichment",
                "schema_version": 1,
                "status": feature_description_summary["status"],
                "feature_catalog_path": str(feature_catalog_path),
                "biohub_request_hash": request_hash,
                "feature_descriptions": feature_description_summary,
                "retrieved_at": timestamp,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return MaterializedBiohubEsmcFeatureDescriptions(
        feature_catalog_path=feature_catalog_path,
        manifest_path=manifest_path,
        observed_feature_count=int(feature_description_summary["observed_feature_count"]),
        enriched_feature_count=int(feature_description_summary["enriched_feature_count"]),
    )


def _biohub_request_hash(
    *,
    source_request_hash: str,
    sequence_ids: list[str],
    sequence_hashes: list[str],
    biohub_api_base_url: str,
    model: str,
    sae_model: str,
    normalize_features: bool,
) -> str:
    return biohub_request_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_ids": sequence_ids,
            "sequence_hashes": sequence_hashes,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "sae_model": sae_model,
            "normalize_features": normalize_features,
        }
    )


def _biohub_query_hash(
    *,
    source_request_hash: str,
    sequence_hash: str,
    biohub_api_base_url: str,
    model: str,
    sae_model: str,
    normalize_features: bool,
) -> str:
    return biohub_query_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_hash": sequence_hash,
            "biohub_api_base_url": biohub_api_base_url,
            "biohub_api_version": DEFAULT_BIOHUB_API_VERSION,
            "model": model,
            "sae_model": sae_model,
            "normalize_features": normalize_features,
        }
    )


def _existing_request_hash(rows_by_candidate: dict[str, dict[str, object]]) -> str:
    request_hashes = {str(row.get("biohub_request_hash") or "") for row in rows_by_candidate.values()}
    request_hashes.discard("")
    if len(request_hashes) != 1:
        raise ValueError("Existing Biohub ESMC profile rows must carry one biohub_request_hash")
    return next(iter(request_hashes))


def _maybe_enrich_feature_catalog_rows(
    rows: list[dict[str, object]],
    *,
    sae_model: str,
    retrieved_at: str,
    fetch_feature_descriptions: bool,
    feature_description_limit: int | None,
    feature_description_sleep_seconds: float,
    feature_description_client: Any | None,
    biohub_api_base_url: str,
    request_timeout_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    unique_indices = sorted({int(row["feature_index"]) for row in rows})
    summary = {
        "requested": bool(fetch_feature_descriptions),
        "source": "Biohub public feature-description endpoint",
        "endpoint_path": "/esm/protein/api/v1alpha1/features/{feature_index}",
        "supported_sae_model": FEATURE_DESCRIPTION_SAE_MODEL,
        "sae_model": sae_model,
        "status": "not_requested",
        "observed_feature_count": len(unique_indices),
        "attempted_feature_count": 0,
        "enriched_feature_count": 0,
    }
    if not fetch_feature_descriptions:
        return rows, summary
    if not supports_feature_description_endpoint(sae_model):
        raise ValueError(
            "Biohub feature-description enrichment was requested, but the public endpoint is "
            f"source-backed only for {FEATURE_DESCRIPTION_SAE_MODEL}; got {sae_model!r}"
        )
    client = feature_description_client or BiohubSaeFeatureDescriptionClient(
        base_url=biohub_api_base_url,
        timeout_seconds=request_timeout_seconds,
    )
    if feature_description_limit is None:
        indices_to_fetch = unique_indices
    else:
        indices_to_fetch = unique_indices[:feature_description_limit]
    existing_descriptions: dict[int, tuple[str, str, str]] = {}
    for row in rows:
        label = str(row.get("label") or "")
        description = str(row.get("description") or "")
        raw_feature_hash = str(row.get("raw_feature_hash") or "")
        if label or description or raw_feature_hash:
            existing_descriptions[int(row["feature_index"])] = (label, description, raw_feature_hash)
    preexisting_description_count = len(existing_descriptions)
    missing_indices = [feature_index for feature_index in unique_indices if feature_index not in existing_descriptions]
    if feature_description_limit is None:
        indices_to_fetch = missing_indices
    else:
        indices_to_fetch = missing_indices[:feature_description_limit]
    for feature_index in indices_to_fetch:
        try:
            description = client.fetch(sae_model=sae_model, feature_index=feature_index)
        except BiohubSaeFeatureDescriptionError as error:
            raise ValueError(str(error)) from error
        existing_descriptions[int(feature_index)] = (
            description.label,
            description.description,
            description.raw_feature_hash,
        )
        if feature_description_sleep_seconds:
            sleep(feature_description_sleep_seconds)
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        label, description, raw_feature_hash = existing_descriptions.get(int(row["feature_index"]), ("", "", ""))
        if label or description:
            enriched["label"] = label
            enriched["description"] = description
            enriched["raw_feature_hash"] = raw_feature_hash
            enriched["source_retrieved_at"] = retrieved_at
        enriched_rows.append(enriched)
    summary.update(
        {
            "status": "enriched",
            "attempted_feature_count": len(indices_to_fetch),
            "preexisting_feature_description_count": preexisting_description_count,
            "new_enriched_feature_count": len(existing_descriptions) - preexisting_description_count,
            "enriched_feature_count": len(existing_descriptions),
            "missing_feature_description_count": len(unique_indices) - len(existing_descriptions),
        }
    )
    return enriched_rows, summary


def _cached_feature_rows_complete(
    *,
    profile_row: dict[str, object],
    protein_feature_rows: list[dict[str, object]],
    residue_feature_rows: list[dict[str, object]],
) -> bool:
    """Return whether a cached accepted profile has its dependent sparse rows."""

    expected_protein_rows = int(profile_row.get("protein_feature_count") or 0)
    expected_residue_rows = int(profile_row.get("residue_feature_count") or 0)
    if expected_protein_rows <= 0 or expected_residue_rows <= 0:
        return False
    return len(protein_feature_rows) == expected_protein_rows and len(residue_feature_rows) == expected_residue_rows


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
