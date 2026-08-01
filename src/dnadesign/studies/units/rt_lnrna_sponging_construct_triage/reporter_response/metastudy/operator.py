"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator.py

Canonical fail-closed regeneration and publication verification operator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

from dnadesign.studies.core.reader_records import resolve_digest_verified_dataframe_record

from ...reader_evidence import build_reader_evidence_bindings, selected_experiments_for_route
from ...subject_bindings import load_registered_subject_bindings
from .. import ReporterResponseObservationPolicy, UncertaintyPolicy
from .acquisition_projection import (
    AcquisitionProjection,
    acquisition_projection_payload,
    build_acquisition_projection,
    validate_acquisition_projection_payload,
)
from .condition_ontology import DEFAULT_CONDITION_ONTOLOGY
from .contracts import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    MaterializationAttemptReceipt,
    MaterializationBlocker,
    MetastudyContractError,
    MetastudyDecision,
    ObjectiveReadiness,
    ProfileEvidence,
    SensitivityEvaluation,
    decision_is_evidence_bearing,
    decision_to_dict,
    materialization_attempt_from_payload,
    objective_readiness_from_payload,
    validate_decision_payload,
)
from .evaluation import evaluate_metastudy, readiness_from_live_bridge
from .materialize import materialize_record_evidence
from .publication import publish_metastudy, verify_publication
from .sensitivity import evaluate_sensitivity, parse_sensitivity_evaluations, sensitivity_evaluations_to_payload
from .sensitivity_coverage import (
    SensitivityCoverageLedger,
    sensitivity_coverage_receipt_payload,
    validate_sensitivity_coverage_receipt_payloads,
    validate_sensitivity_coverage_set,
)

_ROUTE_ID = "rt_lnrna_reporter_response_metastudy"
_PROTOCOL_ID = "plate_reader/single_reporter_screen"
_RECORD_ID = "sample_measurements/df"
_RECORD_CONTRACT_ID = "plate_reader.annotated.v1"
_STATE_FILE = "metastudy-state.yaml"
_READINESS_SCHEMA_ID = "rt_lnrna_reporter_response_readiness_snapshot.v1"
_ROUTE_REGISTRY_PATH = ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
_RECEIPT_NORMALIZATION = "omit environment-specific reader_command before canonical JSON hashing"
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise MetastudyContractError(f"duplicate YAML key in combined meta-study state: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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
        if self.route_registry_path != _ROUTE_REGISTRY_PATH:
            raise MetastudyContractError("regeneration route registry path is not canonical")
        _require_sha256_digest(
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


def write_source_controlled_state(result: RegenerationResult, *, destination: Path) -> tuple[Path]:
    """Atomically replace one combined readiness-and-decision generation."""

    if not isinstance(result, RegenerationResult):
        raise MetastudyContractError("state publication requires one complete regeneration result")
    target = Path(destination).resolve()
    if not target.is_dir():
        raise MetastudyContractError("state destination must be an existing directory")
    decision_payload = json.loads(json.dumps(decision_to_dict(result.decision), allow_nan=False))
    validate_decision_payload(decision_payload)
    route_registry = (
        target.parents[5] / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    )
    if not route_registry.is_file():
        raise MetastudyContractError("state destination does not resolve to the canonical PhD route registry")
    if _digest_file(route_registry) != result.route_registry_digest:
        raise MetastudyContractError("route registry changed since regeneration")
    readiness = result.decision.readiness
    readiness_payload = {
        "schema_id": _READINESS_SCHEMA_ID,
        "source_identity": {
            "route_id": _ROUTE_ID,
            "route_registry_path": result.route_registry_path,
            "route_registry_digest": result.route_registry_digest,
            "normalized_full_receipt_digest": readiness.receipt_digest,
            "normalization": _RECEIPT_NORMALIZATION,
        },
        "last_verified": date.today().isoformat(),
        "selected_experiment_count": readiness.selected_experiment_count,
        "related_experiment_count": len(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
        "related_experiment_ids": list(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
        "ready_experiment_count": readiness.ready_experiment_count,
        "ready_experiment_ids": list(readiness.ready_experiment_ids),
        "blocked_experiment_ids": list(readiness.blocked_experiment_ids),
    }
    body = {
        "readiness": readiness_payload,
        "decision": decision_payload,
        "objective_readiness": asdict(result.objective_readiness),
        "sensitivity_evaluations": sensitivity_evaluations_to_payload(result.sensitivity_evaluations),
        "sensitivity_coverage_receipts": [
            sensitivity_coverage_receipt_payload(row) for row in result.sensitivity_coverages
        ],
        "acquisition_projection": (
            acquisition_projection_payload(result.acquisition_projection)
            if result.acquisition_projection is not None
            else None
        ),
    }
    state_payload = {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v6",
        "generation_digest": _canonical_digest(body),
        **body,
    }
    _validate_state_payload(state_payload, route_registry=route_registry)
    state_path = target / _STATE_FILE
    _atomic_replace_yaml(state_path, state_payload)
    return (state_path,)


def validate_source_controlled_state(path: Path, *, phd_root: Path) -> dict[str, object]:
    """Load and validate one atomic source-controlled state generation."""

    payload = _load_state_yaml(Path(path))
    route_registry = _route_registry_for_phd_root(phd_root)
    _validate_state_payload(payload, route_registry=route_registry)
    return payload


def _validate_state_payload(payload: object, *, route_registry: Path | None = None) -> None:
    if not isinstance(payload, dict):
        raise MetastudyContractError("combined meta-study state fields do not match the exact contract")
    expected_fields = {
        "schema_id",
        "generation_digest",
        "readiness",
        "decision",
        "objective_readiness",
        "sensitivity_evaluations",
        "sensitivity_coverage_receipts",
        "acquisition_projection",
    }
    if set(payload) != expected_fields:
        raise MetastudyContractError("combined meta-study state fields do not match the exact contract")
    if payload["schema_id"] != "rt_lnrna_reporter_response_metastudy_state.v6":
        raise MetastudyContractError("combined meta-study state schema_id changed")
    _require_sha256_digest(payload["generation_digest"], label="state generation digest")
    body = {
        key: payload[key]
        for key in (
            "readiness",
            "decision",
            "objective_readiness",
            "sensitivity_evaluations",
            "sensitivity_coverage_receipts",
            "acquisition_projection",
        )
        if key in payload
    }
    if payload["generation_digest"] != _canonical_digest(body):
        raise MetastudyContractError("combined meta-study state generation digest changed")
    decision = payload["decision"]
    objective_readiness = payload["objective_readiness"]
    readiness = payload["readiness"]
    if not isinstance(decision, dict) or not isinstance(objective_readiness, dict) or not isinstance(readiness, dict):
        raise MetastudyContractError("combined meta-study state projections must be objects")
    if set(readiness) != {
        "schema_id",
        "source_identity",
        "last_verified",
        "selected_experiment_count",
        "related_experiment_count",
        "related_experiment_ids",
        "ready_experiment_count",
        "ready_experiment_ids",
        "blocked_experiment_ids",
    }:
        raise MetastudyContractError("combined meta-study readiness fields changed")
    source_identity = readiness["source_identity"]
    if not isinstance(source_identity, dict) or set(source_identity) != {
        "route_id",
        "route_registry_path",
        "route_registry_digest",
        "normalized_full_receipt_digest",
        "normalization",
    }:
        raise MetastudyContractError("combined meta-study readiness source identity changed")
    _validate_readiness_snapshot(
        readiness,
        source_identity=source_identity,
        route_registry=route_registry,
    )
    validate_decision_payload(decision)
    if objective_readiness_from_payload(objective_readiness) != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("combined meta-study objective readiness changed")
    parse_sensitivity_evaluations(payload["sensitivity_evaluations"])
    attempts_payload = decision["materialization_attempts"]
    if not isinstance(attempts_payload, list):
        raise MetastudyContractError("combined meta-study attempts must be an array")
    attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts_payload)
    )
    validate_sensitivity_coverage_receipt_payloads(
        payload["sensitivity_coverage_receipts"],
        attempts=attempts,
    )
    acquisition_payload = payload["acquisition_projection"]
    selected_reduction = decision["selected_reduction"]
    if selected_reduction is not None:
        projection = validate_acquisition_projection_payload(acquisition_payload)
        if projection.selected_reduction != tuple(selected_reduction):
            raise MetastudyContractError("acquisition projection differs from the selected reduction")
    elif acquisition_payload is not None:
        raise MetastudyContractError("state without a selected reduction cannot contain a acquisition projection")
    expected_readiness = {
        "selected_experiment_count": readiness.get("selected_experiment_count"),
        "ready_experiment_count": readiness.get("ready_experiment_count"),
        "ready_experiment_ids": readiness.get("ready_experiment_ids"),
        "blocked_experiment_ids": readiness.get("blocked_experiment_ids"),
        "receipt_digest": source_identity["normalized_full_receipt_digest"],
    }
    if decision["readiness"] != expected_readiness:
        raise MetastudyContractError("combined meta-study readiness and decision generations differ")


def _validate_readiness_snapshot(
    readiness: dict[str, object],
    *,
    source_identity: dict[str, object],
    route_registry: Path | None,
) -> None:
    if readiness["schema_id"] != _READINESS_SCHEMA_ID:
        raise MetastudyContractError("combined meta-study readiness schema_id changed")
    if source_identity["route_id"] != _ROUTE_ID:
        raise MetastudyContractError("combined meta-study readiness route_id changed")
    if source_identity["route_registry_path"] != _ROUTE_REGISTRY_PATH:
        raise MetastudyContractError("combined meta-study readiness route registry path changed")
    if source_identity["normalization"] != _RECEIPT_NORMALIZATION:
        raise MetastudyContractError("combined meta-study readiness normalization changed")
    _require_sha256_digest(
        source_identity["route_registry_digest"],
        label="readiness route registry digest",
    )
    if route_registry is not None:
        expected_registry_digest = "sha256:" + hashlib.sha256(route_registry.read_bytes()).hexdigest()
        if source_identity["route_registry_digest"] != expected_registry_digest:
            raise MetastudyContractError("combined meta-study readiness route registry digest changed")
    _require_sha256_digest(
        source_identity["normalized_full_receipt_digest"],
        label="readiness normalized receipt digest",
    )
    verified_on = readiness["last_verified"]
    if not isinstance(verified_on, str):
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date")
    try:
        parsed_verified_on = date.fromisoformat(verified_on)
    except ValueError as exc:
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date") from exc
    if parsed_verified_on.isoformat() != verified_on:
        raise MetastudyContractError("combined meta-study readiness last_verified must be an ISO date")

    selected_count = readiness["selected_experiment_count"]
    if type(selected_count) is not int or selected_count != DEFAULT_PROTOCOL.planned_kinetic_experiments:
        raise MetastudyContractError("combined meta-study readiness selected experiment count changed")
    related_ids = readiness["related_experiment_ids"]
    related_count = readiness["related_experiment_count"]
    expected_related_ids = list(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids)
    if (
        type(related_count) is not int
        or related_count != len(expected_related_ids)
        or not isinstance(related_ids, list)
        or related_ids != expected_related_ids
    ):
        raise MetastudyContractError("combined meta-study readiness related experiment cohort changed")

    ready_ids = readiness["ready_experiment_ids"]
    blocked_ids = readiness["blocked_experiment_ids"]
    ready_count = readiness["ready_experiment_count"]
    if (
        type(ready_count) is not int
        or not isinstance(ready_ids, list)
        or not isinstance(blocked_ids, list)
        or not all(isinstance(value, str) for value in (*ready_ids, *blocked_ids))
    ):
        raise MetastudyContractError("combined meta-study readiness ready experiment count or cohort changed")
    ready_set = set(ready_ids)
    blocked_set = set(blocked_ids)
    planned_ids = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
    if (
        ready_count != len(ready_ids)
        or ready_count + len(blocked_ids) != selected_count
        or bool(ready_set & blocked_set)
        or ready_set | blocked_set != set(planned_ids)
        or ready_ids != [value for value in planned_ids if value in ready_set]
        or blocked_ids != [value for value in planned_ids if value in blocked_set]
    ):
        raise MetastudyContractError("combined meta-study readiness selected experiment cohort changed")


def _load_state_yaml(path: Path) -> object:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except MetastudyContractError:
        raise
    except yaml.YAMLError as exc:
        raise MetastudyContractError(f"cannot parse combined meta-study state YAML: {exc}") from exc


def _route_registry_for_phd_root(phd_root: Path) -> Path:
    root = Path(phd_root).expanduser().resolve()
    route_registry = root / _ROUTE_REGISTRY_PATH
    if not route_registry.is_file():
        raise MetastudyContractError("PhD root does not contain the canonical route registry")
    return route_registry


def _require_sha256_digest(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_DIGEST.fullmatch(value) is None:
        raise MetastudyContractError(f"combined meta-study {label} is not a canonical SHA-256 digest")


def _atomic_replace_yaml(path: Path, payload: dict[str, object]) -> None:
    canonical_payload = json.loads(json.dumps(payload, allow_nan=False))
    with tempfile.TemporaryDirectory(prefix=".metastudy-state-", dir=path.parent) as staging_name:
        staged = Path(staging_name) / path.name
        staged.write_text(yaml.safe_dump(canonical_payload, sort_keys=False), encoding="utf-8")
        if yaml.load(staged.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader) != canonical_payload:
            raise MetastudyContractError("staged combined meta-study state did not round-trip")
        os.replace(staged, path)


def _canonical_digest(payload: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _digest_file(path: Path) -> str:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise MetastudyContractError(f"cannot read canonical route registry: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def regenerate_metastudy(*, phd_root: Path) -> RegenerationResult:
    """Reconstruct selected routes through public Reader records and canonical evaluation."""

    root = Path(phd_root).expanduser().resolve()
    repo_root = root / "dnadesign"
    reader_root = root / "reader"
    route_registry = root / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    route_registry_digest = _digest_file(route_registry)
    readiness = readiness_from_live_bridge(phd_root=root)
    if not readiness.is_selection_authorized:
        raise MetastudyContractError("canonical regeneration requires owner-bound Reader readiness")
    ready_ids = set(readiness.ready_experiment_ids) & set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    if len(ready_ids) < DEFAULT_PROTOCOL.minimum_kinetic_experiments:
        raise MetastudyContractError("canonical regeneration requires at least 7 of 8 kinetic experiments ready")
    members = selected_experiments_for_route(route_registry, route_id=_ROUTE_ID)
    if _digest_file(route_registry) != route_registry_digest:
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
            protocol_id=_PROTOCOL_ID,
            record_id=_RECORD_ID,
            contract_id=_RECORD_CONTRACT_ID,
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
        route_registry_path=_ROUTE_REGISTRY_PATH,
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
) -> LiveStateValidation:
    """Require exact parity between checked state and one canonical live regeneration."""

    state = validate_source_controlled_state(path, phd_root=phd_root)
    regeneration = regenerate_metastudy(phd_root=phd_root)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    regenerate = subparsers.add_parser(
        "regenerate",
        help="Reconstruct and optionally publish one canonical meta-study generation",
    )
    regenerate.add_argument("--phd-root", type=Path, required=True)
    regenerate.add_argument("--publication", type=Path)
    regenerate.add_argument("--state-dir", type=Path)
    status = subparsers.add_parser("status", help="Validate and summarize one source-controlled state generation")
    status.add_argument("--phd-root", type=Path, required=True)
    status.add_argument("--state-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="Verify one create-only meta-study publication")
    verify.add_argument("--publication", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        verify_publication(args.publication)
        print(json.dumps({"ok": True, "publication": str(args.publication.resolve())}, sort_keys=True))
        return 0
    if args.command == "status":
        validation = validate_live_source_controlled_state(
            args.state_dir / _STATE_FILE,
            phd_root=args.phd_root,
        )
        state = validation.state
        decision = state["decision"]
        assert isinstance(decision, dict)
        print(
            json.dumps(
                {
                    "generation_digest": state["generation_digest"],
                    "status": decision["status"],
                    "selected_reduction": decision["selected_reduction"],
                    "blockers": decision["blockers"],
                    "limitations": decision["limitations"],
                    "objective_readiness": state["objective_readiness"],
                    "sensitivity_evaluations": state["sensitivity_evaluations"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = regenerate_metastudy(phd_root=args.phd_root)
    state_paths = None
    if args.state_dir is not None:
        state_paths = write_source_controlled_state(result, destination=args.state_dir)
    publication = None
    if args.publication is not None:
        publication = publish_metastudy(
            result.decision,
            args.publication,
            primary_evidence=(result.primary_evidence if decision_is_evidence_bearing(result.decision) else ()),
            sensitivity_evidence=result.sensitivity_evidence,
            sensitivity_evaluations=result.sensitivity_evaluations,
            sensitivity_coverages=result.sensitivity_coverages,
            objective_readiness=result.objective_readiness,
        )
    payload = decision_to_dict(result.decision)
    payload["decision_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    payload["publication"] = str(publication) if publication is not None else None
    payload["objective_readiness"] = asdict(result.objective_readiness)
    payload["sensitivity_evaluations"] = sensitivity_evaluations_to_payload(result.sensitivity_evaluations)
    payload["state_paths"] = [str(path) for path in state_paths] if state_paths is not None else None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveStateValidation",
    "RegenerationResult",
    "build_parser",
    "main",
    "regenerate_metastudy",
    "validate_live_source_controlled_state",
    "validate_source_controlled_state",
    "write_source_controlled_state",
]
