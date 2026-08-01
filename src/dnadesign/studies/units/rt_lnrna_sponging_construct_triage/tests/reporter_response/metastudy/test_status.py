"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_status.py

Semantic status tests for the reporter-response meta-study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    MetastudyContractError,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    status as status_module,
)


def _state_path() -> Path:
    return next(
        parent
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy/metastudy-state.yaml"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )


def _live_readiness(*, ready_ids: tuple[str, ...], receipt_digest: str | None = None) -> EvidenceReadiness:
    selected_ids = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
    if receipt_digest is None:
        receipt_digest = "sha256:" + "a" * 64
    return EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=len(selected_ids),
        ready_experiment_count=len(ready_ids),
        ready_experiment_ids=ready_ids,
        blocked_experiment_ids=tuple(value for value in selected_ids if value not in ready_ids),
        receipt_digest=receipt_digest,
    )


def _state(*, decision_status: str = "blocked") -> dict[str, object]:
    selected = decision_status == "selected"
    return {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v6",
        "generation_digest": "sha256:" + "b" * 64,
        "decision": {
            "status": decision_status,
            "selected_reduction": [6.0, 10.0] if selected else None,
            "evidence_grade": "provisional_descriptive" if selected else "none",
            "limitations": ["retrospective_calibration_cohort"] if selected else [],
        },
        "objective_readiness": {
            "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
            "status": DEFAULT_OBJECTIVE_READINESS.status,
            "objective_id": DEFAULT_OBJECTIVE_READINESS.objective_id,
            "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
        },
    }


def _live_validation(
    *,
    state: dict[str, object] | None = None,
    readiness: EvidenceReadiness | None = None,
) -> SimpleNamespace:
    if state is None:
        state = _state()
    if readiness is None:
        readiness = _live_readiness(ready_ids=DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    return SimpleNamespace(
        state=state,
        regeneration=SimpleNamespace(decision=SimpleNamespace(readiness=readiness)),
    )


def test_status_reports_blocked_reduction_independently_from_available_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        status_module,
        "validate_live_source_controlled_state",
        lambda *_args, **_kwargs: _live_validation(),
    )

    payload = status_module.status_payload(phd_root=Path("unused"), state_path=_state_path())

    assert payload["status"] == "blocked"
    assert payload["semantic_blockers"] == ("reduction_recommendation_is_blocked",)
    assert payload["measurement_readiness"] == "ready"
    assert payload["descriptive_visualization_readiness"] == "ready"
    assert payload["reduction_recommendation_status"] == "blocked"
    assert payload["objective_readiness_status"] == "blocked"
    assert payload["objective_readiness_blockers"] == DEFAULT_OBJECTIVE_READINESS.blockers
    assert payload["selected_reduction"] is None
    assert payload["evidence_grade"] == "none"


def test_status_propagates_live_regeneration_parity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: object, **_kwargs: object) -> object:
        raise MetastudyContractError("source-controlled meta-study state differs from canonical live regeneration")

    monkeypatch.setattr(status_module, "validate_live_source_controlled_state", reject)

    with pytest.raises(MetastudyContractError, match="differs from canonical live regeneration"):
        status_module.status_payload(phd_root=Path("unused"), state_path=_state_path())


def test_status_uses_readiness_from_the_single_live_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_ids = (DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0],)
    readiness = _live_readiness(ready_ids=ready_ids)
    calls: list[tuple[Path, Path]] = []

    def validate(path: Path, *, phd_root: Path) -> SimpleNamespace:
        calls.append((path, phd_root))
        return _live_validation(readiness=readiness)

    monkeypatch.setattr(
        status_module,
        "validate_live_source_controlled_state",
        validate,
    )

    payload = status_module.status_payload(phd_root=Path("unused"), state_path=_state_path())

    assert calls == [(_state_path().resolve(), Path("unused"))]
    assert payload["measurement_readiness"] == "partial"
    assert payload["descriptive_visualization_readiness"] == "ready"
    assert payload["ready_experiment_ids"] == ready_ids
    assert payload["readiness_receipt_digest"] == readiness.receipt_digest


def test_status_reports_selected_exact_live_state_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(decision_status="selected")
    monkeypatch.setattr(
        status_module,
        "validate_live_source_controlled_state",
        lambda *_args, **_kwargs: _live_validation(state=state),
    )

    payload = status_module.status_payload(phd_root=Path("unused"), state_path=_state_path())

    assert payload["status"] == "ready"
    assert payload["semantic_blockers"] == ()
    assert payload["measurement_readiness"] == "ready"
    assert payload["descriptive_visualization_readiness"] == "ready"
    assert payload["reduction_recommendation_status"] == "ready"
    assert payload["objective_readiness_status"] == "blocked"
    assert payload["selected_reduction"] == (6.0, 10.0)
    assert payload["evidence_grade"] == "provisional_descriptive"
    assert payload["limitations"] == ("retrospective_calibration_cohort",)
