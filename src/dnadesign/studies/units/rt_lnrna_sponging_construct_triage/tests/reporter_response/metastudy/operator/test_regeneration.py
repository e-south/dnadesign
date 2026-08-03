"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_regeneration.py

Owner-aligned operator contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    MetastudyContractError,
    operator,
    validate_acquisition_projection_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    checkout as operator_checkout,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    regeneration as operator_regeneration,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.sensitivity import (
    parse_sensitivity_evaluations,
)

from ._support import (
    _checked_state_path,
    _digest,
    _state_for_external_registry,
)


def test_operator_preserves_complete_partial_and_blocked_materialization_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_ids = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[:-1]
    readiness = EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=8,
        ready_experiment_count=7,
        ready_experiment_ids=ready_ids,
        blocked_experiment_ids=(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[-1],),
        receipt_digest="sha256:" + "a" * 64,
    )
    monkeypatch.setattr(operator_regeneration, "readiness_from_live_bridge", lambda **_kwargs: readiness)
    monkeypatch.setattr(operator_regeneration, "digest_file", lambda _path: _digest("a"))
    members = tuple(
        SimpleNamespace(experiment_id=experiment_id, reader_config=f"reader/{experiment_id}/config.yaml")
        for experiment_id in DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
    )
    monkeypatch.setattr(
        operator_regeneration,
        "selected_experiments_for_route",
        lambda *_args, **_kwargs: members,
    )
    loaded_repo_roots: list[Path] = []

    def load_subjects(*, repo_root: Path) -> object:
        loaded_repo_roots.append(repo_root)
        return object()

    monkeypatch.setattr(operator_regeneration, "load_registered_subject_bindings", load_subjects)
    resolved_ids: list[str] = []
    reader_commands: list[tuple[str, ...] | None] = []

    def resolve_record(_config, **kwargs):
        resolved_ids.append(kwargs["experiment_id"])
        reader_commands.append(kwargs["reader_command"])
        return SimpleNamespace(experiment_id=kwargs["experiment_id"])

    monkeypatch.setattr(operator_regeneration, "resolve_digest_verified_dataframe_record", resolve_record)
    monkeypatch.setattr(operator_regeneration, "build_reader_evidence_bindings", lambda **_kwargs: object())
    monkeypatch.setattr(
        operator_regeneration,
        "materialize_record_evidence",
        lambda **kwargs: SimpleNamespace(
            attempt=SimpleNamespace(
                experiment_id=kwargs["record"].experiment_id,
                status=("partial" if kwargs["record"].experiment_id == ready_ids[0] else "complete"),
            ),
            candidate_evidence=(),
            endpoint_evidence=(f"endpoint:{kwargs['record'].experiment_id}",),
            centered_window_evidence=(f"centered:{kwargs['record'].experiment_id}",),
            sensitivity_coverage=f"coverage:{kwargs['record'].experiment_id}",
        ),
    )
    captured: dict[str, object] = {}

    def evaluate(evidence, *, readiness, attempts):
        captured.update(evidence=tuple(evidence), readiness=readiness, attempts=tuple(attempts))
        return object()

    monkeypatch.setattr(operator_regeneration, "evaluate_metastudy", evaluate)
    monkeypatch.setattr(
        operator_regeneration,
        "evaluate_sensitivity",
        lambda evidence: (SimpleNamespace(kind="endpoint", evidence=tuple(evidence)),),
    )
    monkeypatch.setattr(operator_regeneration, "validate_sensitivity_coverage_set", lambda *_args, **_kwargs: None)

    reader_executable = Path("/tmp/reader-cli")
    result = operator.regenerate_metastudy(
        phd_root=Path("unused"),
        reader_executable=reader_executable,
    )

    blocked_id = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[-1]
    assert result.route_registry_path == operator_state.ROUTE_REGISTRY_PATH
    assert result.route_registry_digest == _digest("a")
    assert loaded_repo_roots == [operator_checkout.active_dnadesign_checkout()]
    assert loaded_repo_roots != [(Path("unused").resolve() / "dnadesign")]
    assert resolved_ids == list(ready_ids)
    assert reader_commands == [(str(reader_executable.resolve()),)] * len(ready_ids)
    assert result.attempts == captured["attempts"]
    assert tuple(row.status for row in result.attempts) == (
        "partial",
        *("complete" for _ in ready_ids[1:]),
        "blocked",
    )
    assert result.primary_evidence == ()
    assert len(result.endpoint_sensitivity_evidence) == len(ready_ids)
    assert len(result.centered_window_sensitivity_evidence) == len(ready_ids)
    assert len(result.sensitivity_coverages) == len(ready_ids)
    assert result.objective_readiness == DEFAULT_OBJECTIVE_READINESS
    assert result.sensitivity_evaluations[0].evidence == (
        *result.endpoint_sensitivity_evidence,
        *result.centered_window_sensitivity_evidence,
    )
    blocked = next(row for row in result.attempts if row.experiment_id == blocked_id)
    assert blocked.status == "blocked"
    assert blocked.reader_record_identity is None
    assert tuple(row.code for row in blocked.blockers) == ("reader_records_not_ready",)


def test_operator_fails_before_source_resolution_below_minimum_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_ids = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[:-2]
    readiness = EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=8,
        ready_experiment_count=6,
        ready_experiment_ids=ready_ids,
        blocked_experiment_ids=DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[-2:],
        receipt_digest="sha256:" + "a" * 64,
    )
    monkeypatch.setattr(operator_regeneration, "readiness_from_live_bridge", lambda **_kwargs: readiness)
    monkeypatch.setattr(operator_regeneration, "digest_file", lambda _path: _digest("a"))
    monkeypatch.setattr(
        operator_regeneration,
        "selected_experiments_for_route",
        lambda *_args, **_kwargs: pytest.fail("source resolution must not start"),
    )

    with pytest.raises(MetastudyContractError, match="at least 7 of 8"):
        operator.regenerate_metastudy(phd_root=Path("unused"))


def test_regeneration_result_rejects_incoherent_sensitivity_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operator_regeneration, "evaluate_sensitivity", lambda _evidence: ("canonical",))

    with pytest.raises(MetastudyContractError, match="summaries differ"):
        operator.RegenerationResult(
            route_registry_path=operator_state.ROUTE_REGISTRY_PATH,
            route_registry_digest=_digest("a"),
            decision=SimpleNamespace(status="blocked"),
            primary_evidence=(),
            endpoint_sensitivity_evidence=("endpoint",),
            centered_window_sensitivity_evidence=(),
            sensitivity_coverages=(),
            sensitivity_evaluations=(),
            attempts=(),
            objective_readiness=DEFAULT_OBJECTIVE_READINESS,
        )


def test_live_state_validation_accepts_exact_canonical_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "exact-live-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    coverage_receipts = payload["sensitivity_coverage_receipts"]
    coverages = tuple(object() for _ in coverage_receipts)
    result = SimpleNamespace(
        decision=SimpleNamespace(status=payload["decision"]["status"]),
        primary_evidence=(),
        endpoint_sensitivity_evidence=(),
        centered_window_sensitivity_evidence=(),
        sensitivity_coverages=coverages,
        sensitivity_evaluations=parse_sensitivity_evaluations(payload["sensitivity_evaluations"]),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
        acquisition_projection=validate_acquisition_projection_payload(payload["acquisition_projection"]),
    )
    monkeypatch.setattr(operator_regeneration, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(
        operator_regeneration,
        "decision_to_dict",
        lambda _decision: json.loads(json.dumps(payload["decision"])),
    )
    monkeypatch.setattr(
        operator_regeneration,
        "sensitivity_coverage_receipt_payload",
        lambda coverage: coverage_receipts[coverages.index(coverage)],
    )
    validated = operator_regeneration.validate_live_source_controlled_state(state_path, phd_root=phd_root)

    assert validated.state == payload
    assert validated.regeneration is result


def test_live_state_validation_rejects_decision_drift_after_structural_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = yaml.safe_load(_checked_state_path().read_text(encoding="utf-8"))
    stale = json.loads(json.dumps(canonical))
    attempt = stale["decision"]["materialization_attempts"][0]
    attempt["candidate_profile_digests"][0] = _digest("0")
    attempt["candidate_profile_digests"].sort()
    attempt["attempt_digest"] = operator_state.canonical_digest(
        {key: value for key, value in attempt.items() if key != "attempt_digest"}
    )
    result = SimpleNamespace(
        decision=SimpleNamespace(status=canonical["decision"]["status"]),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator_regeneration, "validate_source_controlled_state", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(operator_regeneration, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(
        operator_regeneration,
        "decision_to_dict",
        lambda _decision: json.loads(json.dumps(canonical["decision"])),
    )

    with pytest.raises(MetastudyContractError, match="differs from canonical live regeneration"):
        operator_regeneration.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))


def test_live_state_validation_accepts_compact_selected_state_without_full_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = {"status": "selected"}
    state = {
        "decision": decision,
        "objective_readiness": {
            "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
            "status": DEFAULT_OBJECTIVE_READINESS.status,
            "objective_id": DEFAULT_OBJECTIVE_READINESS.objective_id,
            "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
        },
        "sensitivity_evaluations": [],
        "sensitivity_coverage_receipts": [],
        "acquisition_projection": None,
    }
    result = operator.RegenerationResult(
        route_registry_path=operator_state.ROUTE_REGISTRY_PATH,
        route_registry_digest=_digest("a"),
        decision=SimpleNamespace(status="selected"),
        primary_evidence=(),
        endpoint_sensitivity_evidence=(),
        centered_window_sensitivity_evidence=(),
        sensitivity_coverages=(),
        sensitivity_evaluations=(),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator_regeneration, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator_regeneration, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator_regeneration, "decision_to_dict", lambda _decision: decision)
    validated = operator_regeneration.validate_live_source_controlled_state(
        Path("unused"),
        phd_root=Path("unused"),
    )

    assert validated.state == state


def test_live_state_validation_rejects_sensitivity_summary_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = {"status": "blocked"}
    state = {
        "decision": decision,
        "objective_readiness": {
            "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
            "status": DEFAULT_OBJECTIVE_READINESS.status,
            "objective_id": DEFAULT_OBJECTIVE_READINESS.objective_id,
            "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
        },
        "sensitivity_evaluations": [{"kind": "endpoint"}],
        "sensitivity_coverage_receipts": [],
        "acquisition_projection": None,
    }
    result = operator.RegenerationResult(
        route_registry_path=operator_state.ROUTE_REGISTRY_PATH,
        route_registry_digest=_digest("a"),
        decision=SimpleNamespace(status="blocked"),
        primary_evidence=(),
        endpoint_sensitivity_evidence=(),
        centered_window_sensitivity_evidence=(),
        sensitivity_coverages=(),
        sensitivity_evaluations=(),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator_regeneration, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator_regeneration, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator_regeneration, "decision_to_dict", lambda _decision: decision)

    with pytest.raises(MetastudyContractError, match="sensitivity state differs"):
        operator_regeneration.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))


def test_live_state_validation_rejects_objective_readiness_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = {"status": "blocked"}
    state = {
        "decision": decision,
        "objective_readiness": {
            "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
            "status": "ready",
            "objective_id": "undeclared-objective",
            "blockers": [],
        },
        "sensitivity_evaluations": [],
        "sensitivity_coverage_receipts": [],
        "acquisition_projection": None,
    }
    result = operator.RegenerationResult(
        route_registry_path=operator_state.ROUTE_REGISTRY_PATH,
        route_registry_digest=_digest("a"),
        decision=SimpleNamespace(status="blocked"),
        primary_evidence=(),
        endpoint_sensitivity_evidence=(),
        centered_window_sensitivity_evidence=(),
        sensitivity_coverages=(),
        sensitivity_evaluations=(),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator_regeneration, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator_regeneration, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator_regeneration, "decision_to_dict", lambda _decision: decision)

    with pytest.raises(MetastudyContractError, match="objective readiness differs"):
        operator_regeneration.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))
