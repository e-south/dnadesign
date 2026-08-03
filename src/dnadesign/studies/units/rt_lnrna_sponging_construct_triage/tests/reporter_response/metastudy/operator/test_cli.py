"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_cli.py

Owner-aligned operator contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    MetastudyContractError,
    operator,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    cli as operator_cli,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)

from ._support import (
    _digest,
)


def test_operator_parser_exposes_regenerate_status_and_verify() -> None:
    parser = operator_cli.build_parser()

    regenerate = parser.parse_args(
        [
            "regenerate",
            "--phd-root",
            "/tmp/phd",
            "--dnadesign-root",
            "/tmp/checkout",
            "--reader-executable",
            "/tmp/reader",
        ]
    )
    status = parser.parse_args(
        [
            "status",
            "--phd-root",
            "/tmp/phd",
            "--dnadesign-root",
            "/tmp/checkout",
            "--state-dir",
            "/tmp/state",
            "--reader-executable",
            "/tmp/reader",
        ]
    )
    verify = parser.parse_args(["verify", "--publication", "/tmp/publication"])

    assert regenerate.command == "regenerate"
    assert regenerate.dnadesign_root == Path("/tmp/checkout")
    assert regenerate.reader_executable == Path("/tmp/reader")
    assert status.command == "status"
    assert status.dnadesign_root == Path("/tmp/checkout")
    assert status.reader_executable == Path("/tmp/reader")
    assert verify.command == "verify"


@pytest.mark.parametrize(
    ("checkout_name", "create_checkout", "message"),
    (
        ("foreign", True, "must be the active source checkout"),
        ("missing", False, "does not exist"),
    ),
)
def test_cli_rejects_an_invalid_checkout_before_regeneration_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkout_name: str,
    create_checkout: bool,
    message: str,
) -> None:
    checkout = tmp_path / checkout_name
    if create_checkout:
        checkout.mkdir()
    monkeypatch.setattr(
        operator_cli,
        "regenerate_metastudy",
        lambda **_kwargs: pytest.fail("regeneration must not start"),
    )
    monkeypatch.setattr(
        operator_cli,
        "write_source_controlled_state",
        lambda *_args, **_kwargs: pytest.fail("state publication must not start"),
    )

    with pytest.raises(MetastudyContractError, match=message):
        operator_cli.main(
            [
                "regenerate",
                "--phd-root",
                str(tmp_path),
                "--dnadesign-root",
                str(checkout),
                "--state-dir",
                str(tmp_path),
            ]
        )


def test_regenerate_cli_emits_sibling_readiness_and_sensitivity_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    monkeypatch.setattr(operator_cli, "regenerate_metastudy", lambda **_kwargs: result)
    monkeypatch.setattr(operator_cli, "decision_to_dict", lambda _decision: {"status": "blocked"})

    assert operator_cli.main(["regenerate", "--phd-root", "unused"]) == 0

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
