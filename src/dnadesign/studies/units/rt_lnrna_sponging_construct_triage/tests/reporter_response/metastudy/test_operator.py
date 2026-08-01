"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_operator.py

Canonical regeneration operator tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
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
    protocol_digest,
    validate_acquisition_projection_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _checked_state_path() -> Path:
    return next(
        parent
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy/metastudy-state.yaml"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )


def _acquisition_projection_payload() -> dict[str, object]:
    metric = {
        "estimate": 1.0,
        "method": "median_across_acquisitions",
        "acquisition_count": 1,
        "leave_one_acquisition_out_estimates": [],
    }
    body = {
        "contract_id": "rt_lnrna_reporter_response_acquisition_projection.v1",
        "selected_reduction": [6.0, 10.0],
        "coordinates": [
            {
                "subject_id": "synthetic-subject",
                "condition_role": "dose",
                "dose_uM": 500.0,
                "reduction_id": "window-6-10h",
                "reduction_digest": _digest("1"),
                "observation_policy_digest": _digest("2"),
                "acquisition_ids": [DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0]],
                "contributions": [
                    {
                        "acquisition_id": DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0],
                        "profile_id": "synthetic-profile",
                        "profile_digest": _digest("3"),
                        "declared_biological_replicate_ids": [],
                        "normalized_reporter_response": 1.0,
                        "relative_od": 1.0,
                    }
                ],
                "normalized_reporter_response": dict(metric),
                "relative_od": dict(metric),
            }
        ],
    }
    return {**body, "projection_digest": canonical_digest(body)}


def _state_for_external_registry(phd_root: Path) -> dict[str, object]:
    registry = phd_root / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text('{"routes": []}\n', encoding="utf-8")
    payload = yaml.safe_load(_checked_state_path().read_text(encoding="utf-8"))
    payload["schema_id"] = "rt_lnrna_reporter_response_metastudy_state.v6"
    payload["decision"]["policy_digest"] = protocol_digest()
    attempt_digests: dict[str, str] = {}
    for attempt in payload["decision"]["materialization_attempts"]:
        attempt["attempt_digest"] = canonical_digest(
            {key: value for key, value in attempt.items() if key != "attempt_digest"}
        )
        attempt_digests[attempt["experiment_id"]] = attempt["attempt_digest"]
    payload["objective_readiness"] = {
        "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
        "status": DEFAULT_OBJECTIVE_READINESS.status,
        "objective_id": DEFAULT_OBJECTIVE_READINESS.objective_id,
        "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
    }
    payload["sensitivity_evaluations"] = []
    payload.setdefault("sensitivity_coverage_receipts", [])
    for receipt in payload["sensitivity_coverage_receipts"]:
        receipt["materialization_attempt_digest"] = attempt_digests[receipt["experiment_id"]]
    payload["acquisition_projection"] = _acquisition_projection_payload()
    payload.pop("evidence", None)
    payload["readiness"]["source_identity"]["route_registry_digest"] = (
        "sha256:" + hashlib.sha256(registry.read_bytes()).hexdigest()
    )
    payload["generation_digest"] = operator._canonical_digest(
        {
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
    )
    return payload


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
    monkeypatch.setattr(operator, "readiness_from_live_bridge", lambda **_kwargs: readiness)
    monkeypatch.setattr(operator, "_digest_file", lambda _path: _digest("a"))
    members = tuple(
        SimpleNamespace(experiment_id=experiment_id, reader_config=f"reader/{experiment_id}/config.yaml")
        for experiment_id in DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
    )
    monkeypatch.setattr(
        operator,
        "selected_experiments_for_route",
        lambda *_args, **_kwargs: members,
    )
    monkeypatch.setattr(operator, "load_registered_subject_bindings", lambda **_kwargs: object())
    resolved_ids: list[str] = []

    def resolve_record(_config, **kwargs):
        resolved_ids.append(kwargs["experiment_id"])
        return SimpleNamespace(experiment_id=kwargs["experiment_id"])

    monkeypatch.setattr(operator, "resolve_digest_verified_dataframe_record", resolve_record)
    monkeypatch.setattr(operator, "build_reader_evidence_bindings", lambda **_kwargs: object())
    monkeypatch.setattr(
        operator,
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

    monkeypatch.setattr(operator, "evaluate_metastudy", evaluate)
    monkeypatch.setattr(
        operator,
        "evaluate_sensitivity",
        lambda evidence: (SimpleNamespace(kind="endpoint", evidence=tuple(evidence)),),
    )
    monkeypatch.setattr(operator, "validate_sensitivity_coverage_set", lambda *_args, **_kwargs: None)

    result = operator.regenerate_metastudy(phd_root=Path("unused"))

    blocked_id = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[-1]
    assert result.route_registry_path == operator._ROUTE_REGISTRY_PATH
    assert result.route_registry_digest == _digest("a")
    assert resolved_ids == list(ready_ids)
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
    monkeypatch.setattr(operator, "readiness_from_live_bridge", lambda **_kwargs: readiness)
    monkeypatch.setattr(operator, "_digest_file", lambda _path: _digest("a"))
    monkeypatch.setattr(
        operator,
        "selected_experiments_for_route",
        lambda *_args, **_kwargs: pytest.fail("source resolution must not start"),
    )

    with pytest.raises(MetastudyContractError, match="at least 7 of 8"):
        operator.regenerate_metastudy(phd_root=Path("unused"))


def test_regeneration_result_rejects_incoherent_sensitivity_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operator, "evaluate_sensitivity", lambda _evidence: ("canonical",))

    with pytest.raises(MetastudyContractError, match="summaries differ"):
        operator.RegenerationResult(
            route_registry_path=operator._ROUTE_REGISTRY_PATH,
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


def test_operator_parser_exposes_regenerate_status_and_verify() -> None:
    parser = operator.build_parser()

    regenerate = parser.parse_args(["regenerate", "--phd-root", "/tmp/phd"])
    status = parser.parse_args(["status", "--phd-root", "/tmp/phd", "--state-dir", "/tmp/state"])
    verify = parser.parse_args(["verify", "--publication", "/tmp/publication"])

    assert regenerate.command == "regenerate"
    assert status.command == "status"
    assert verify.command == "verify"


def test_regenerate_cli_emits_sibling_readiness_and_sensitivity_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = operator.RegenerationResult(
        route_registry_path=operator._ROUTE_REGISTRY_PATH,
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
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator, "decision_to_dict", lambda _decision: {"status": "blocked"})

    assert operator.main(["regenerate", "--phd-root", "unused"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["objective_readiness"] == {
        "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
        "status": "blocked",
        "objective_id": None,
        "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
    }
    assert payload["sensitivity_evaluations"] == []
    assert payload["publication"] is None
    assert payload["state_paths"] is None


def test_atomic_state_replace_preserves_prior_generation_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metastudy-state.yaml"
    original = {"generation": "prior"}
    target.write_text(yaml.safe_dump(original), encoding="utf-8")
    monkeypatch.setattr(operator.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("injected")))

    with pytest.raises(OSError, match="injected"):
        operator._atomic_replace_yaml(target, {"generation": "next"})

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == original


def test_state_publication_rejects_registry_drift_since_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phd_root = tmp_path / "phd"
    destination = (
        phd_root / "dnadesign/docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy"
    )
    destination.mkdir(parents=True)
    registry = phd_root / operator._ROUTE_REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text('{"generation": "destination"}\n', encoding="utf-8")
    source_digest = operator._canonical_digest({"generation": "source"})
    result = operator.RegenerationResult(
        route_registry_path=operator._ROUTE_REGISTRY_PATH,
        route_registry_digest=source_digest,
        decision=SimpleNamespace(
            status="blocked",
            readiness=SimpleNamespace(
                receipt_digest=_digest("a"),
                selected_experiment_count=8,
                ready_experiment_count=0,
                ready_experiment_ids=(),
                blocked_experiment_ids=DEFAULT_PROTOCOL.planned_kinetic_experiment_ids,
            ),
        ),
        primary_evidence=(),
        endpoint_sensitivity_evidence=(),
        centered_window_sensitivity_evidence=(),
        sensitivity_coverages=(),
        sensitivity_evaluations=(),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator, "decision_to_dict", lambda _decision: {"status": "blocked"})
    monkeypatch.setattr(operator, "validate_decision_payload", lambda _payload: None)

    with pytest.raises(MetastudyContractError, match="route registry changed since regeneration"):
        operator.write_source_controlled_state(result, destination=destination)

    assert not (destination / "metastudy-state.yaml").exists()


def test_state_validation_rejects_canonical_shaped_digest_for_wrong_route_registry(
    tmp_path: Path,
) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["readiness"]["source_identity"]["route_registry_digest"] = _digest("a")
    payload["generation_digest"] = operator._canonical_digest(
        {
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
    )
    state_path = tmp_path / "forged-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="route registry digest changed"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_accepts_exact_external_route_registry(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "exact-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert operator.validate_source_controlled_state(state_path, phd_root=phd_root) == payload


def test_state_validation_rejects_incomplete_sensitivity_coverage_receipt(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["profile_count"] -= 1
    payload["generation_digest"] = operator._canonical_digest(
        {
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
    )
    state_path = tmp_path / "incomplete-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="coordinate counts changed"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_sensitivity_receipt_attempt_drift(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["materialization_attempt_digest"] = _digest("f")
    payload["generation_digest"] = operator._canonical_digest(
        {
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
    )
    state_path = tmp_path / "drifted-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="exact materialization attempt"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


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
        sensitivity_evaluations=(),
        attempts=(),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
        acquisition_projection=validate_acquisition_projection_payload(payload["acquisition_projection"]),
    )
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(
        operator,
        "decision_to_dict",
        lambda _decision: json.loads(json.dumps(payload["decision"])),
    )
    monkeypatch.setattr(
        operator,
        "sensitivity_coverage_receipt_payload",
        lambda coverage: coverage_receipts[coverages.index(coverage)],
    )
    validated = operator.validate_live_source_controlled_state(state_path, phd_root=phd_root)

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
    attempt["attempt_digest"] = operator._canonical_digest(
        {key: value for key, value in attempt.items() if key != "attempt_digest"}
    )
    result = SimpleNamespace(
        decision=SimpleNamespace(status=canonical["decision"]["status"]),
        objective_readiness=DEFAULT_OBJECTIVE_READINESS,
    )
    monkeypatch.setattr(operator, "validate_source_controlled_state", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(
        operator,
        "decision_to_dict",
        lambda _decision: json.loads(json.dumps(canonical["decision"])),
    )

    with pytest.raises(MetastudyContractError, match="differs from canonical live regeneration"):
        operator.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))


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
        route_registry_path=operator._ROUTE_REGISTRY_PATH,
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
    monkeypatch.setattr(operator, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator, "decision_to_dict", lambda _decision: decision)
    validated = operator.validate_live_source_controlled_state(
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
        route_registry_path=operator._ROUTE_REGISTRY_PATH,
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
    monkeypatch.setattr(operator, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator, "decision_to_dict", lambda _decision: decision)

    with pytest.raises(MetastudyContractError, match="sensitivity state differs"):
        operator.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))


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
        route_registry_path=operator._ROUTE_REGISTRY_PATH,
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
    monkeypatch.setattr(operator, "validate_source_controlled_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(operator, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator, "decision_to_dict", lambda _decision: decision)

    with pytest.raises(MetastudyContractError, match="objective readiness differs"):
        operator.validate_live_source_controlled_state(Path("unused"), phd_root=Path("unused"))


def test_state_validation_fails_closed_without_external_route_registry(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    registry = phd_root / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    registry.unlink()

    with pytest.raises(MetastudyContractError, match="does not contain the canonical route registry"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_float_experiment_count(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    selected_count = payload["readiness"]["selected_experiment_count"]
    payload["readiness"]["selected_experiment_count"] = float(selected_count)
    payload["decision"]["readiness"]["selected_experiment_count"] = float(selected_count)
    payload["generation_digest"] = operator._canonical_digest(
        {
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
    )
    state_path = tmp_path / "float-count-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="selected experiment count"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "duplicate-key-state.yaml"
    state_path.write_text(
        "schema_id: shadowed-value\n" + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(MetastudyContractError, match="duplicate YAML key"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)
