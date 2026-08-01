"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_persistence.py

Owner-aligned operator contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    MetastudyContractError,
    operator,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    persistence as operator_persistence,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)

from ._support import (
    _digest,
)


def test_atomic_state_replace_preserves_prior_generation_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metastudy-state.yaml"
    original = {"generation": "prior"}
    target.write_text(yaml.safe_dump(original), encoding="utf-8")
    monkeypatch.setattr(
        operator_persistence.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("injected")),
    )

    with pytest.raises(OSError, match="injected"):
        operator_persistence.atomic_replace_yaml(target, {"generation": "next"})

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == original


def test_state_publication_rejects_registry_drift_since_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phd_root = tmp_path / "phd"
    destination = (
        phd_root / "dnadesign/.worktrees/feature/docs/studies/rt_lnrna_sponging_construct_triage/contexts/"
        "reporter-response-metastudy"
    )
    destination.mkdir(parents=True)
    registry = phd_root / operator_state.ROUTE_REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text('{"generation": "destination"}\n', encoding="utf-8")
    source_digest = operator_state.canonical_digest({"generation": "source"})
    result = operator.RegenerationResult(
        route_registry_path=operator_state.ROUTE_REGISTRY_PATH,
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
    monkeypatch.setattr(operator_persistence, "decision_to_dict", lambda _decision: {"status": "blocked"})
    monkeypatch.setattr(operator_persistence, "validate_decision_payload", lambda _payload: None)

    with pytest.raises(MetastudyContractError, match="route registry changed since regeneration"):
        operator.write_source_controlled_state(
            result,
            destination=destination,
            phd_root=phd_root,
        )

    assert not (destination / "metastudy-state.yaml").exists()
