"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/sources.py

Load exact Reader evidence and resolve it through study-owned candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    READER_ALIAS_NAMESPACE,
    load_promoter_candidate_bindings,
    verify_promoter_candidate_bindings,
)

from .aggregation import VALUE_COLUMNS, ResponseWindowObservationPreview, aggregate_response_window_observations
from .policy import ResponseWindowObservationPolicy, load_response_window_observation_policy
from .reader_records import ReaderResponseRecords, load_reader_response_records

_BINDING_IDENTITY_COLUMNS = ("alias_namespace", "alias", "candidate_id", "sequence_sha256")
_BINDING_PROVENANCE_COLUMNS = (
    "display_label",
    "candidate_table_id",
    "candidate_selection_sha256",
    "sequence_authority_dataset_id",
    "sequence_authority_id",
    "sequence_authority_sha256",
    "source_class",
    "design_family",
    "baserender_adapter_kind",
)
_EXCLUSION_COLUMNS = ("design_id", "reason")
_EXCLUSION_REASON = "absent_from_study_candidate_bindings"


class ResponseWindowObservationSourceError(ValueError):
    """Raised when Reader evidence and study candidate authority do not agree exactly."""


@dataclass(frozen=True)
class ResolvedReaderCandidateEvidence:
    measurements: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    excluded_reader_designs: pd.DataFrame


@dataclass(frozen=True)
class ResponseWindowObservationEvidence:
    policy: ResponseWindowObservationPolicy
    resolved: ResolvedReaderCandidateEvidence
    preview: ResponseWindowObservationPreview
    reader_records: ReaderResponseRecords
    reader_catalog_path: Path
    reader_catalog_sha256: str
    reader_projection_path: Path
    reader_projection_sha256: str
    candidate_bindings_manifest_path: Path
    candidate_bindings_manifest_sha256: str
    candidate_bindings_path: Path


def preview_response_window_observation_evidence(
    *,
    reader_root: Path,
    reader_experiment_root: Path,
    reader_projection_path: Path,
    candidate_bindings_root: Path,
    policy_path: Path,
) -> ResponseWindowObservationEvidence:
    """Verify canonical Reader records and preview candidate observations."""

    policy = load_response_window_observation_policy(policy_path)
    reader = load_reader_response_records(
        reader_root=reader_root,
        experiment_root=reader_experiment_root,
        projection_path=reader_projection_path,
    )
    if reader.primary_reduction_id != policy.aggregation.primary_reduction_id:
        raise ResponseWindowObservationSourceError(
            "Reader primary reduction disagrees with the response-window observation policy."
        )
    binding_root = Path(candidate_bindings_root).expanduser().resolve()
    verification = verify_promoter_candidate_bindings(binding_root, allowed_root=binding_root)
    binding_digest = _sha256(verification.manifest_json)
    if binding_digest != policy.candidate_bindings_sha256:
        raise ResponseWindowObservationSourceError(
            "Candidate-binding manifest digest disagrees with the response-window observation policy."
        )
    bindings = load_promoter_candidate_bindings(binding_root, allowed_root=binding_root)
    resolved = resolve_reader_candidate_evidence(
        reader,
        binding_rows=bindings,
        unbound_reader_designs=policy.unbound_reader_designs,
    )
    preview = aggregate_response_window_observations(
        resolved.measurements,
        resolved.bootstrap_draws,
        policy=policy.aggregation,
        repeat_decisions=policy.repeat_decisions,
    )
    blockers = list(preview.blockers)
    if policy.approval_status != "approved":
        blockers.append("response-window observation policy requires study approval")
    preview = ResponseWindowObservationPreview(
        observations=preview.observations,
        contributions=preview.contributions,
        bootstrap_draws=preview.bootstrap_draws,
        uncertainty=preview.uncertainty,
        repeat_diagnostics=preview.repeat_diagnostics,
        reduction_sensitivity=preview.reduction_sensitivity,
        event_time_sensitivity=preview.event_time_sensitivity,
        blockers=tuple(sorted(blockers)),
    )
    return ResponseWindowObservationEvidence(
        policy=policy,
        resolved=resolved,
        preview=preview,
        reader_records=reader,
        reader_catalog_path=reader.catalog_path,
        reader_catalog_sha256=reader.catalog_sha256,
        reader_projection_path=reader.projection_path,
        reader_projection_sha256=reader.projection_sha256,
        candidate_bindings_manifest_path=verification.manifest_json,
        candidate_bindings_manifest_sha256=binding_digest,
        candidate_bindings_path=verification.bindings_parquet,
    )


def resolve_reader_candidate_evidence(
    records: ReaderResponseRecords,
    *,
    binding_rows: pd.DataFrame,
    unbound_reader_designs: pd.DataFrame,
) -> ResolvedReaderCandidateEvidence:
    """Resolve every non-reference Reader alias or reject the exact unbound-accounting set."""

    bindings = _reader_bindings(binding_rows)
    exclusions = _validated_exclusions(unbound_reader_designs)
    designs = records.designs.loc[~records.designs["is_reference"].astype(bool)].copy()
    observed_designs = set(designs["design_id"].astype(str))
    bound_designs = set(bindings["design_id"].astype(str))
    observed_unbound = observed_designs - bound_designs
    declared_unbound = set(exclusions["design_id"].astype(str))
    if observed_unbound != declared_unbound:
        raise ResponseWindowObservationSourceError(
            "Reader unbound design accounting disagrees; "
            f"observed={sorted(observed_unbound)}, declared={sorted(declared_unbound)}."
        )

    resolved_designs = _merge_evidence(
        designs.loc[~designs["design_id"].astype(str).isin(declared_unbound)],
        bindings=bindings,
        evidence_label="measurements",
    ).rename(columns={"experiment_id": "reader_experiment_id"})
    reader_draws = records.descriptive_resampling_draws
    draws = reader_draws.loc[~reader_draws["is_reference"].astype(bool)].copy()
    resolved_draws = _merge_evidence(
        draws.loc[~draws["design_id"].astype(str).isin(declared_unbound)],
        bindings=bindings,
        evidence_label="descriptive resampling draws",
    ).rename(columns={"experiment_id": "reader_experiment_id"})
    return ResolvedReaderCandidateEvidence(
        measurements=resolved_designs.reset_index(drop=True),
        bootstrap_draws=resolved_draws.reset_index(drop=True),
        excluded_reader_designs=exclusions.reset_index(drop=True),
    )


def _reader_bindings(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_BINDING_IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ResponseWindowObservationSourceError(f"candidate bindings lack identity columns: {missing}")
    projected = [*_BINDING_IDENTITY_COLUMNS, *(c for c in _BINDING_PROVENANCE_COLUMNS if c in frame.columns)]
    rows = frame.loc[frame["alias_namespace"].astype(str).eq(READER_ALIAS_NAMESPACE), projected].copy()
    for column in _BINDING_IDENTITY_COLUMNS:
        invalid = rows[column].isna() | rows[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ResponseWindowObservationSourceError(f"candidate binding field {column!r} contains empty values.")
        rows[column] = rows[column].astype(str)
    if rows["alias"].duplicated().any():
        raise ResponseWindowObservationSourceError("Reader candidate bindings contain duplicate exact aliases.")
    return rows.drop(columns="alias_namespace").rename(columns={"alias": "design_id"})


def _validated_exclusions(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(_EXCLUSION_COLUMNS):
        raise ResponseWindowObservationSourceError(
            f"unbound Reader design fields must be exactly {sorted(_EXCLUSION_COLUMNS)}."
        )
    result = frame.loc[:, _EXCLUSION_COLUMNS].copy()
    for column in _EXCLUSION_COLUMNS:
        invalid = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ResponseWindowObservationSourceError(f"unbound Reader design field {column!r} is empty.")
        result[column] = result[column].astype(str)
    if result["design_id"].duplicated().any() or set(result["reason"]) - {_EXCLUSION_REASON}:
        raise ResponseWindowObservationSourceError("unbound Reader design declarations are invalid.")
    return result


def _merge_evidence(frame: pd.DataFrame, *, bindings: pd.DataFrame, evidence_label: str) -> pd.DataFrame:
    result = frame.copy()
    result["design_id"] = result["design_id"].astype(str)
    result = result.merge(bindings, on="design_id", how="left", validate="many_to_one")
    if result["candidate_id"].isna().any():
        missing = sorted(result.loc[result["candidate_id"].isna(), "design_id"].astype(str).unique())
        raise ResponseWindowObservationSourceError(
            f"Reader {evidence_label} contain unresolved exact aliases: {missing}."
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "VALUE_COLUMNS",
    "ResolvedReaderCandidateEvidence",
    "ResponseWindowObservationEvidence",
    "ResponseWindowObservationSourceError",
    "preview_response_window_observation_evidence",
    "resolve_reader_candidate_evidence",
]
