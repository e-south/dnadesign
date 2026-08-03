"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator/regeneration.py

Canonical meta-study regeneration and exact live-state validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dnadesign.studies.core.reader_records import resolve_digest_verified_dataframe_record

from ....reader_evidence import build_reader_evidence_bindings, selected_experiments_for_route
from ....subject_bindings import load_registered_subject_bindings
from ...policy import ReporterResponseObservationPolicy
from ...profile.uncertainty import UncertaintyPolicy
from ..acquisition_projection import (
    AcquisitionProjection,
    acquisition_projection_payload,
    build_acquisition_projection,
)
from ..condition_ontology import DEFAULT_CONDITION_ONTOLOGY
from ..contracts._values import MetastudyContractError
from ..contracts.decision import MetastudyDecision
from ..contracts.decision_codec import decision_to_dict
from ..contracts.materialization import MaterializationAttemptReceipt, MaterializationBlocker
from ..contracts.objective import (
    DEFAULT_OBJECTIVE_READINESS,
    ObjectiveReadiness,
    objective_readiness_from_payload,
)
from ..contracts.profile import ProfileEvidence
from ..contracts.protocol import DEFAULT_PROTOCOL
from ..contracts.sensitivity import SensitivityEvaluation
from ..evaluation.readiness import readiness_from_live_bridge
from ..evaluation.selection import evaluate_metastudy
from ..materialize.service import materialize_record_evidence
from ..sensitivity import evaluate_sensitivity, sensitivity_evaluations_to_payload
from ..sensitivity_coverage import (
    SensitivityCoverageLedger,
    sensitivity_coverage_receipt_payload,
    validate_sensitivity_coverage_set,
)
from .checkout import require_active_dnadesign_checkout
from .state import (
    ROUTE_ID,
    ROUTE_REGISTRY_PATH,
    digest_file,
    require_sha256_digest,
    validate_source_controlled_state,
)

PROTOCOL_ID = "plate_reader/single_reporter_screen"
RECORD_ID = "sample_measurements/df"
RECORD_CONTRACT_ID = "plate_reader.annotated.v1"


@dataclass(frozen=True, slots=True)
class RegenerationResult:
    """Complete in-memory result; no output is written before construction succeeds."""

    route_registry_path: str
    route_registry_digest: str
    decision: MetastudyDecision
    primary_evidence: tuple[ProfileEvidence, ...]
    endpoint_sensitivity_evidence: tuple[ProfileEvidence, ...]
    centered_window_sensitivity_evidence: tuple[ProfileEvidence, ...]
    sensitivity_coverages: tuple[SensitivityCoverageLedger, ...]
    sensitivity_evaluations: tuple[SensitivityEvaluation, ...]
    attempts: tuple[MaterializationAttemptReceipt, ...]
    objective_readiness: ObjectiveReadiness
    acquisition_projection: AcquisitionProjection | None = field(init=False)

    def __post_init__(self) -> None:
        if self.route_registry_path != ROUTE_REGISTRY_PATH:
            raise MetastudyContractError("regeneration route registry path is not canonical")
        require_sha256_digest(
            self.route_registry_digest,
            label="regeneration route registry digest",
        )
        projection = (
            build_acquisition_projection(
                self.primary_evidence,
                selected_reduction=self.decision.selected_reduction,
            )
            if self.primary_evidence and self.decision.selected_reduction is not None
            else None
        )
        object.__setattr__(self, "acquisition_projection", projection)
        if self.sensitivity_evaluations != evaluate_sensitivity(self.sensitivity_evidence):
            raise MetastudyContractError(
                "regeneration sensitivity summaries differ from canonical sensitivity evidence"
            )
        validate_sensitivity_coverage_set(
            self.sensitivity_coverages,
            evidence=self.sensitivity_evidence,
            attempts=self.attempts,
        )
        if self.objective_readiness != DEFAULT_OBJECTIVE_READINESS:
            raise MetastudyContractError("regeneration objective readiness differs from the study-owned gate")

    @property
    def sensitivity_evidence(self) -> tuple[ProfileEvidence, ...]:
        return self.endpoint_sensitivity_evidence + self.centered_window_sensitivity_evidence


@dataclass(frozen=True, slots=True)
class LiveStateValidation:
    """One structurally valid state paired with its canonical live regeneration."""

    state: dict[str, object]
    regeneration: RegenerationResult


def regenerate_metastudy(
    *,
    phd_root: Path,
    dnadesign_root: Path | None = None,
    reader_executable: Path | None = None,
) -> RegenerationResult:
    """Reconstruct selected routes through public Reader records and canonical evaluation."""

    root = Path(phd_root).expanduser().resolve()
    repo_root = require_active_dnadesign_checkout(dnadesign_root)
    reader_root = root / "reader"
    route_registry = root / ROUTE_REGISTRY_PATH
    route_registry_digest = digest_file(route_registry)
    executable = Path(reader_executable).expanduser().resolve() if reader_executable is not None else None
    readiness = readiness_from_live_bridge(phd_root=root, reader_executable=executable)
    if not readiness.is_selection_authorized:
        raise MetastudyContractError("canonical regeneration requires owner-bound Reader readiness")
    ready_ids = set(readiness.ready_experiment_ids) & set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    if len(ready_ids) < DEFAULT_PROTOCOL.minimum_kinetic_experiments:
        raise MetastudyContractError("canonical regeneration requires at least 7 of 8 kinetic experiments ready")
    members = selected_experiments_for_route(route_registry, route_id=ROUTE_ID)
    if digest_file(route_registry) != route_registry_digest:
        raise MetastudyContractError("route registry changed during regeneration")
    member_ids = tuple(member.experiment_id for member in members)
    if member_ids != DEFAULT_PROTOCOL.planned_kinetic_experiment_ids:
        raise MetastudyContractError("selected route identities differ from the predeclared kinetic cohort")
    if (
        tuple(experiment_id for experiment_id in member_ids if experiment_id in ready_ids)
        != readiness.ready_experiment_ids
    ):
        raise MetastudyContractError("selected route order and live ready identities differ")
    if (
        tuple(experiment_id for experiment_id in member_ids if experiment_id not in ready_ids)
        != readiness.blocked_experiment_ids
    ):
        raise MetastudyContractError("selected route order and live blocked identities differ")
    registry = load_registered_subject_bindings(repo_root=repo_root)
    policy = ReporterResponseObservationPolicy(
        policy_id="rt_lnrna_reporter_response_observation_policy.v3",
        pairing_kind="pooled_controls_by_design",
        within_acquisition_reduction_statistic="median",
        biological_replicate_uncertainty_policy=UncertaintyPolicy(
            minimum_biological_replicates=2,
            biological_replicate_reduction_statistic="median",
        ),
    )
    attempts: list[MaterializationAttemptReceipt] = []
    primary_evidence: list[ProfileEvidence] = []
    endpoint_sensitivity_evidence: list[ProfileEvidence] = []
    centered_window_sensitivity_evidence: list[ProfileEvidence] = []
    sensitivity_coverages: list[SensitivityCoverageLedger] = []
    for member in members:
        if member.experiment_id not in ready_ids:
            attempts.append(
                MaterializationAttemptReceipt(
                    contract_id="rt_lnrna_reporter_response_materialization_attempt.v4",
                    experiment_id=member.experiment_id,
                    reader_record_identity=None,
                    evidence_binding_artifact_id=None,
                    evidence_binding_artifact_digest=None,
                    expected_subject_ids=(),
                    status="blocked",
                    candidate_profile_count=0,
                    candidate_profile_digests=(),
                    candidate_omissions=(),
                    blockers=(MaterializationBlocker("reader_records_not_ready"),),
                )
            )
            continue
        record = resolve_digest_verified_dataframe_record(
            (root / member.reader_config).resolve(),
            reader_root=reader_root,
            experiment_id=member.experiment_id,
            protocol_id=PROTOCOL_ID,
            record_id=RECORD_ID,
            contract_id=RECORD_CONTRACT_ID,
            reader_command=((str(executable),) if executable is not None else None),
        )
        bindings = build_reader_evidence_bindings(record=record, subject_registry=registry)
        materialized = materialize_record_evidence(
            record=record,
            bindings=bindings,
            ontology=DEFAULT_CONDITION_ONTOLOGY,
            observation_policy=policy,
        )
        attempts.append(materialized.attempt)
        primary_evidence.extend(materialized.candidate_evidence)
        endpoint_sensitivity_evidence.extend(materialized.endpoint_evidence)
        centered_window_sensitivity_evidence.extend(materialized.centered_window_evidence)
        if materialized.sensitivity_coverage is not None:
            sensitivity_coverages.append(materialized.sensitivity_coverage)
    decision = evaluate_metastudy(primary_evidence, readiness=readiness, attempts=attempts)
    sensitivity_evidence = (*endpoint_sensitivity_evidence, *centered_window_sensitivity_evidence)
    return RegenerationResult(
        route_registry_path=ROUTE_REGISTRY_PATH,
        route_registry_digest=route_registry_digest,
        decision=decision,
        primary_evidence=tuple(primary_evidence),
        endpoint_sensitivity_evidence=tuple(endpoint_sensitivity_evidence),
        centered_window_sensitivity_evidence=tuple(centered_window_sensitivity_evidence),
        sensitivity_coverages=tuple(sensitivity_coverages),
        sensitivity_evaluations=evaluate_sensitivity(sensitivity_evidence),
        attempts=tuple(attempts),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )


def validate_live_source_controlled_state(
    path: Path,
    *,
    phd_root: Path,
    dnadesign_root: Path | None = None,
    reader_executable: Path | None = None,
) -> LiveStateValidation:
    """Require exact parity between checked state and one canonical live regeneration."""

    repo_root = require_active_dnadesign_checkout(dnadesign_root)
    state = validate_source_controlled_state(path, phd_root=phd_root)
    regeneration = regenerate_metastudy(
        phd_root=phd_root,
        dnadesign_root=repo_root,
        reader_executable=reader_executable,
    )
    expected_decision = json.loads(json.dumps(decision_to_dict(regeneration.decision), allow_nan=False))
    if state["decision"] != expected_decision:
        raise MetastudyContractError("source-controlled meta-study state differs from canonical live regeneration")
    if objective_readiness_from_payload(state["objective_readiness"]) != regeneration.objective_readiness:
        raise MetastudyContractError("source-controlled objective readiness differs from canonical regeneration")
    expected_sensitivity = json.loads(
        json.dumps(
            sensitivity_evaluations_to_payload(regeneration.sensitivity_evaluations),
            allow_nan=False,
        )
    )
    if state["sensitivity_evaluations"] != expected_sensitivity:
        raise MetastudyContractError("source-controlled sensitivity state differs from canonical live regeneration")
    expected_coverage_receipts = json.loads(
        json.dumps(
            [sensitivity_coverage_receipt_payload(row) for row in regeneration.sensitivity_coverages],
            allow_nan=False,
        )
    )
    if state["sensitivity_coverage_receipts"] != expected_coverage_receipts:
        raise MetastudyContractError("source-controlled sensitivity coverage differs from canonical live regeneration")
    expected_acquisition = (
        json.loads(
            json.dumps(
                acquisition_projection_payload(regeneration.acquisition_projection),
                allow_nan=False,
            )
        )
        if regeneration.acquisition_projection is not None
        else None
    )
    if state["acquisition_projection"] != expected_acquisition:
        raise MetastudyContractError("source-controlled acquisition projection differs from canonical regeneration")
    return LiveStateValidation(state=state, regeneration=regeneration)
