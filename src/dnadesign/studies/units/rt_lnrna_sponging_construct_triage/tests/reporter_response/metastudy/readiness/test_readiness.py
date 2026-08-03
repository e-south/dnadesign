"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/readiness/test_readiness.py

Tests live bridge readiness authority and receipt validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    MetastudyContractError,
    decision_from_readiness,
    readiness_from_live_bridge,
    readiness_from_receipt,
)

from .._builders import (
    KINETIC_IDS,
    _digest,
    _evidence,
    evaluate_metastudy,
)


def test_zero_of_eight_reader_readiness_produces_typed_blocked_decision() -> None:
    readiness = EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=8,
        ready_experiment_count=0,
        ready_experiment_ids=(),
        blocked_experiment_ids=KINETIC_IDS,
        receipt_digest=_digest("9"),
    )
    decision = decision_from_readiness(readiness)

    assert decision.status == "blocked"
    assert decision.selected_reduction is None
    assert "reader_evidence_ready_0_of_8" in decision.blockers
    assert decision.policy_digest.startswith("sha256:")
    assert decision.evidence_digest.startswith("sha256:")


def test_arbitrary_ready_experiments_cannot_clear_the_7_of_8_kinetic_gate() -> None:
    arbitrary = tuple(f"arbitrary-experiment-{index}" for index in range(1, 8))
    readiness = EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=7,
        ready_experiment_count=7,
        ready_experiment_ids=arbitrary,
        blocked_experiment_ids=(),
        receipt_digest=_digest("7"),
    )

    decision = evaluate_metastudy((), readiness=readiness)

    assert decision.status == "blocked"
    assert "minimum_7_of_8_kinetic_experiments_not_met" in decision.blockers


def test_read_only_readiness_receipt_adapter_preserves_zero_of_eight() -> None:
    receipt = _readiness_receipt()

    readiness = readiness_from_receipt(receipt)

    assert readiness.selected_experiment_count == 8
    assert readiness.ready_experiment_count == 0
    assert readiness.ready_experiment_ids == ()
    assert len(readiness.blocked_experiment_ids) == 8


def test_readiness_receipt_digest_omits_only_environment_specific_reader_command() -> None:
    first = _readiness_receipt()
    second = json.loads(json.dumps(first))
    second["reader_command"] = ["/different/workstation/.venv/bin/reader"]

    assert readiness_from_receipt(first).receipt_digest == readiness_from_receipt(second).receipt_digest


def test_structurally_valid_synthetic_receipt_cannot_authorize_selection() -> None:
    readiness = readiness_from_receipt(_readiness_receipt(ready_ids=KINETIC_IDS))

    with pytest.raises(MetastudyContractError, match="owner-bound live bridge runner"):
        evaluate_metastudy(_evidence(), readiness=readiness)


def test_live_bridge_runner_is_the_selection_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_live_bridge_fixture(tmp_path)
    receipt = _readiness_receipt(ready_ids=KINETIC_IDS)

    observed_command: list[str] = []

    def run(command: list[str], **_kwargs: object) -> object:
        observed_command.extend(command)
        return type(
            "Completed",
            (),
            {"stdout": json.dumps(receipt), "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr("subprocess.run", run)

    reader_executable = tmp_path / "reader-cli"
    readiness = readiness_from_live_bridge(
        phd_root=tmp_path,
        reader_executable=reader_executable,
    )

    assert readiness.is_selection_authorized
    assert observed_command[-2:] == ["--reader-executable", str(reader_executable.resolve())]
    assert evaluate_metastudy(_evidence(), readiness=readiness).status == "selected"


def test_live_bridge_runner_rejects_ready_receipt_from_failed_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_live_bridge_fixture(tmp_path)
    receipt = _readiness_receipt(ready_ids=KINETIC_IDS)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: type(
            "Completed",
            (),
            {
                "stdout": json.dumps(receipt),
                "stderr": "bridge transport failed after writing receipt",
                "returncode": 1,
            },
        )(),
    )

    with pytest.raises(MetastudyContractError) as exc_info:
        readiness_from_live_bridge(phd_root=tmp_path)

    message = str(exc_info.value)
    assert "exited with status 1" in message
    assert 'stdout={"available_protocols"' in message
    assert "stderr=bridge transport failed after writing receipt" in message


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("bridge executable unavailable"), id="transport-error"),
        pytest.param(KeyboardInterrupt(), id="base-exception"),
    ],
)
def test_live_bridge_runner_preserves_execution_exception_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    _install_live_bridge_fixture(tmp_path)

    def fail_execution(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("subprocess.run", fail_execution)

    with pytest.raises(type(error)) as exc_info:
        readiness_from_live_bridge(phd_root=tmp_path)

    assert exc_info.value is error


def _install_live_bridge_fixture(root: Path) -> None:
    skill = root / ".agents/skills/retron-assay-study-bridge"
    registry = skill / "references/reader-experiment-routes.json"
    checker = skill / "scripts/check_reader_experiment_readiness.py"
    registry.parent.mkdir(parents=True)
    checker.parent.mkdir(parents=True)
    registry.write_text("{}\n", encoding="utf-8")
    checker.write_text("# fixture\n", encoding="utf-8")


def _readiness_receipt(*, ready_ids: tuple[str, ...] = ()) -> dict[str, object]:
    selected_ids = KINETIC_IDS
    related_ids = DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids
    blocked_ids = tuple(value for value in selected_ids if value not in ready_ids)
    return {
        "available_protocols": ["plate_reader/single_reporter_screen"],
        "contract_errors": [],
        "experiments": [
            {
                "experiment_id": experiment_id,
                "memberships": [
                    {
                        "membership": "selected" if experiment_id in selected_ids else "related",
                        "ready": experiment_id in ready_ids,
                        "required_reader_state": "records_ready",
                        "route_id": "rt_lnrna_reporter_response_metastudy",
                    }
                ],
            }
            for experiment_id in (*selected_ids, *related_ids)
        ],
        "ok": len(ready_ids) == len(selected_ids),
        "reader_command": ["reader"],
        "route_id": "rt_lnrna_reporter_response_metastudy",
        "summary": {
            "contract_error_count": 0,
            "experiment_count": 9,
            "membership_count": 9,
            "related_membership_count": 1,
            "selected_membership_count": 8,
            "selected_ready_count": len(ready_ids),
            "selected_blocker_count": len(blocked_ids),
        },
        "selected_blockers": [
            {"experiment_id": experiment_id, "route_id": "rt_lnrna_reporter_response_metastudy"}
            for experiment_id in blocked_ids
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update({"unexpected": True}), "top-level fields"),
        (lambda payload: payload["summary"].update({"unexpected": 1}), "summary fields"),
        (lambda payload: payload.update({"route_id": "wrong-route"}), "route_id"),
    ],
)
def test_readiness_receipt_rejects_contract_drift(mutation, match: str) -> None:
    receipt = _readiness_receipt()
    mutation(receipt)

    with pytest.raises(MetastudyContractError, match=match):
        readiness_from_receipt(receipt)


def test_readiness_receipt_rejects_errors_ok_drift_and_wrong_selected_identity() -> None:
    receipt = _readiness_receipt()
    receipt["contract_errors"] = [{"code": "invalid"}]
    receipt["summary"]["contract_error_count"] = 1
    with pytest.raises(MetastudyContractError, match="contains contract_errors"):
        readiness_from_receipt(receipt)

    receipt = _readiness_receipt()
    receipt["ok"] = True
    with pytest.raises(MetastudyContractError, match="ok does not match"):
        readiness_from_receipt(receipt)

    receipt = _readiness_receipt()
    receipt["experiments"][0]["experiment_id"] = "arbitrary-substitute"
    receipt["selected_blockers"][0]["experiment_id"] = "arbitrary-substitute"
    with pytest.raises(MetastudyContractError, match="predeclared route cohort"):
        readiness_from_receipt(receipt)
