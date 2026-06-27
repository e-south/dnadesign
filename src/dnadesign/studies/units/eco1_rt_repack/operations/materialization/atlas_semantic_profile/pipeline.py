"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/pipeline.py

Materialize Eco1 ESM Atlas semantic-profile artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.constants import (
    ATLAS_API_BASE_URL,
    ATLAS_API_VERSION,
    DEFAULT_ALLOW_FOLD_ON_MISS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEQUENCE_LIMIT,
    DEFAULT_TOPK_FEATURES,
    STRUCTURE_PREDICTION_ROOT_RELATIVE_PATH,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.resume import (
    cached_profile_row,
    cached_structure_prediction_rows,
    load_existing_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile.selection import (
    select_fold_accepted_atlas_sequences,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.thread.adapters.esm_atlas import (
    AtlasClient,
    atlas_query_hash,
    atlas_request_hash,
    build_atlas_structure_prediction_row,
    build_error_profile_row,
    normalize_protein_lookup_response,
    sequence_md5,
    validate_atlas_semantic_artifacts,
    write_atlas_semantic_artifacts,
)
from dnadesign.thread.structure_predictions import (
    validate_structure_prediction_registry,
    write_structure_prediction_registry,
)


@dataclass(frozen=True)
class MaterializedAtlasSemanticProfileArtifacts:
    """Paths emitted by one Eco1 Atlas semantic-profile materialization pass."""

    profile_path: Path
    protein_activations_path: Path
    residue_activations_path: Path
    feature_catalog_path: Path
    structure_prediction_registry_path: Path
    atlas_request_hash: str
    selected_sequence_count: int


def materialize_atlas_semantic_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    sequence_limit: str = DEFAULT_SEQUENCE_LIMIT,
    atlas_api_base_url: str = ATLAS_API_BASE_URL,
    topk_features: int = DEFAULT_TOPK_FEATURES,
    allow_fold_on_miss: bool = DEFAULT_ALLOW_FOLD_ON_MISS,
    prediction_set_id: str | None = None,
    selection_manifest_path: Path | None = None,
    resume_existing: bool = False,
    max_new_requests: int | None = None,
    request_sleep_seconds: float = 0.0,
    atlas_client: Any | None = None,
    retrieved_at: str | None = None,
) -> MaterializedAtlasSemanticProfileArtifacts:
    """Query Atlas for fold-accepted Eco1 sequences and write compact sparse tables."""

    if allow_fold_on_miss and not (prediction_set_id or "").strip():
        raise ValueError("prediction_set_id is required when allow_fold_on_miss is true")
    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be non-negative")
    if request_sleep_seconds < 0:
        raise ValueError("request_sleep_seconds must be non-negative")
    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    structure_prediction_root = out_root / STRUCTURE_PREDICTION_ROOT_RELATIVE_PATH
    structure_payload_root = structure_prediction_root / "structures" / (prediction_set_id or "lookup_only")
    selection = select_fold_accepted_atlas_sequences(
        output_root=out_root,
        sequence_limit=sequence_limit,
        selection_manifest_path=selection_manifest_path,
    )
    request_hash = _atlas_request_hash(
        source_request_hash=selection.source_request_hash,
        sequence_ids=[record.sequence_id for record in selection.records],
        sequence_md5s=[sequence_md5(record.sequence) for record in selection.records],
        atlas_api_base_url=atlas_api_base_url,
        topk_features=topk_features,
        allow_fold_on_miss=allow_fold_on_miss,
    )
    timestamp = retrieved_at or _timestamp()
    client = atlas_client or AtlasClient(base_url=atlas_api_base_url)
    existing_rows = (
        load_existing_rows(out_root, structure_prediction_root=structure_prediction_root) if resume_existing else None
    )

    profile_rows: list[dict[str, object]] = []
    protein_activation_rows: list[dict[str, object]] = []
    residue_activation_rows: list[dict[str, object]] = []
    feature_catalog_rows: list[dict[str, object]] = []
    structure_prediction_rows: list[dict[str, object]] = []
    new_request_count = 0
    for record in selection.records:
        query_hash = _atlas_query_hash(
            source_request_hash=selection.source_request_hash,
            sequence_md5=sequence_md5(record.sequence),
            atlas_api_base_url=atlas_api_base_url,
            topk_features=topk_features,
            allow_fold_on_miss=allow_fold_on_miss,
        )
        cached = cached_profile_row(
            existing_rows=existing_rows,
            candidate_id=record.sequence_id,
            sequence_hash=record.sequence_hash,
            atlas_query_hash=query_hash,
            atlas_request_hash=request_hash,
            source_request_hash=selection.source_request_hash,
        )
        if cached is not None:
            cached_structure_rows = cached_structure_prediction_rows(
                existing_rows=existing_rows,
                candidate_id=record.sequence_id,
                prediction_set_id=str(prediction_set_id or ""),
                atlas_request_hash=request_hash,
            )
            if allow_fold_on_miss and bool(cached.get("folded_on_demand", False)) and not cached_structure_rows:
                cached = None
            else:
                structure_prediction_rows.extend(cached_structure_rows)
        if cached is not None:
            profile_rows.append(cached)
            protein_activation_rows.extend(
                existing_rows.protein_activation_rows_by_candidate.get(record.sequence_id, [])
            )
            residue_activation_rows.extend(
                existing_rows.residue_activation_rows_by_candidate.get(record.sequence_id, [])
            )
            feature_catalog_rows.extend(existing_rows.feature_catalog_rows)
            continue
        if max_new_requests is not None and new_request_count >= max_new_requests:
            profile_rows.append(
                build_error_profile_row(
                    candidate_id=record.sequence_id,
                    sequence=record.sequence,
                    sequence_hash=record.sequence_hash,
                    source_request_hash=selection.source_request_hash,
                    atlas_request_hash=request_hash,
                    atlas_query_hash=query_hash,
                    atlas_api_base_url=atlas_api_base_url,
                    atlas_api_version=ATLAS_API_VERSION,
                    retrieved_at=timestamp,
                    failure_reason="atlas_request_not_attempted_due_to_max_new_requests",
                )
            )
            continue
        try:
            response = client.protein_lookup_by_sequence(
                record.sequence,
                topk_features=topk_features,
                fold_on_miss=allow_fold_on_miss,
                normalize_features=True,
            )
            normalized = normalize_protein_lookup_response(
                candidate_id=record.sequence_id,
                sequence=record.sequence,
                sequence_hash=record.sequence_hash,
                response=response,
                source_request_hash=selection.source_request_hash,
                atlas_request_hash=request_hash,
                atlas_query_hash=query_hash,
                atlas_api_base_url=atlas_api_base_url,
                atlas_api_version=ATLAS_API_VERSION,
                retrieved_at=timestamp,
                allow_fold_on_miss=allow_fold_on_miss,
            )
            profile_rows.append(normalized.profile_row)
            protein_activation_rows.extend(normalized.protein_activation_rows)
            residue_activation_rows.extend(normalized.residue_activation_rows)
            feature_catalog_rows.extend(normalized.feature_catalog_rows)
        except Exception as error:  # noqa: BLE001 - API/schema failures become explicit artifact rows.
            profile_rows.append(
                build_error_profile_row(
                    candidate_id=record.sequence_id,
                    sequence=record.sequence,
                    sequence_hash=record.sequence_hash,
                    source_request_hash=selection.source_request_hash,
                    atlas_request_hash=request_hash,
                    atlas_query_hash=query_hash,
                    atlas_api_base_url=atlas_api_base_url,
                    atlas_api_version=ATLAS_API_VERSION,
                    retrieved_at=timestamp,
                    failure_reason=str(error),
                )
            )
            continue
        finally:
            new_request_count += 1
            if request_sleep_seconds:
                sleep(request_sleep_seconds)
        if allow_fold_on_miss:
            structure_row = build_atlas_structure_prediction_row(
                candidate_id=record.sequence_id,
                sequence_hash=record.sequence_hash,
                response=response,
                atlas_request_hash=request_hash,
                source_request_hash=selection.source_request_hash,
                prediction_set_id=str(prediction_set_id),
                output_root=structure_payload_root,
                atlas_api_base_url=atlas_api_base_url,
                atlas_api_version=ATLAS_API_VERSION,
            )
            if structure_row is not None:
                structure_prediction_rows.append(structure_row)

    artifacts = write_atlas_semantic_artifacts(
        output_root=out_root,
        profile_rows=profile_rows,
        protein_activation_rows=protein_activation_rows,
        residue_activation_rows=residue_activation_rows,
        feature_catalog_rows=feature_catalog_rows,
        request_hash=request_hash,
    )
    issues = validate_atlas_semantic_artifacts(
        output_root=out_root,
        expected_candidate_ids={record.sequence_id for record in selection.records},
        request_hash=request_hash,
        allow_fold_on_miss=allow_fold_on_miss,
    )
    if issues:
        joined = "; ".join(f"{issue.check_id}: {issue.message}" for issue in issues)
        raise ValueError(f"Atlas semantic-profile validation failed: {joined}")
    structure_artifacts = write_structure_prediction_registry(
        output_root=structure_prediction_root,
        rows=structure_prediction_rows,
        request_hash=request_hash,
    )
    structure_issues = validate_structure_prediction_registry(
        registry_path=structure_artifacts.registry_path,
        request_hash=request_hash,
    )
    if structure_issues:
        joined = "; ".join(f"{issue.check_id}: {issue.message}" for issue in structure_issues)
        raise ValueError(f"Atlas structure-prediction registry validation failed: {joined}")
    return MaterializedAtlasSemanticProfileArtifacts(
        profile_path=artifacts.profile_path,
        protein_activations_path=artifacts.protein_activations_path,
        residue_activations_path=artifacts.residue_activations_path,
        feature_catalog_path=artifacts.feature_catalog_path,
        structure_prediction_registry_path=structure_artifacts.registry_path,
        atlas_request_hash=request_hash,
        selected_sequence_count=len(selection.records),
    )


def _atlas_request_hash(
    *,
    source_request_hash: str,
    sequence_ids: list[str],
    sequence_md5s: list[str],
    atlas_api_base_url: str,
    topk_features: int,
    allow_fold_on_miss: bool,
) -> str:
    return atlas_request_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_ids": sequence_ids,
            "sequence_md5s": sequence_md5s,
            "atlas_api_base_url": atlas_api_base_url,
            "atlas_api_version": ATLAS_API_VERSION,
            "topk_features": topk_features,
            "fold_on_miss": allow_fold_on_miss,
            "normalize_features": True,
        }
    )


def _atlas_query_hash(
    *,
    source_request_hash: str,
    sequence_md5: str,
    atlas_api_base_url: str,
    topk_features: int,
    allow_fold_on_miss: bool,
) -> str:
    return atlas_query_hash(
        {
            "source_request_hash": source_request_hash,
            "sequence_md5": sequence_md5,
            "atlas_api_base_url": atlas_api_base_url,
            "atlas_api_version": ATLAS_API_VERSION,
            "topk_features": topk_features,
            "fold_on_miss": allow_fold_on_miss,
            "normalize_features": True,
        }
    )


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
