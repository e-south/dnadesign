"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/test_service.py

Tests the study-owned publication service over generic artifact mechanics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from dnadesign.artifacts import PublicationExistsError
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    EvidenceReadiness,
    MetastudyContractError,
    decision_from_readiness,
    publish_metastudy,
    verify_publication,
)

from .._builders import (
    KINETIC_IDS,
    _digest,
    _evidence,
    _quality_blocked_evidence,
    _ready,
    evaluate_metastudy,
)
from ._builders import (
    _publish_evaluated,
)


def test_publication_is_create_only_deterministic_and_verified(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "decision-v1"

    publish_metastudy(decision, destination)
    first = {path.name: path.read_bytes() for path in destination.iterdir()}
    verify_publication(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())

    with pytest.raises(PublicationExistsError):
        publish_metastudy(decision, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == first


def test_readiness_only_publication_rejects_primary_evidence(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )

    with pytest.raises(MetastudyContractError, match="readiness-only publication"):
        publish_metastudy(decision, tmp_path / "invalid", primary_evidence=_evidence())


def test_evidence_bearing_blocked_publication_round_trips_offline(tmp_path: Path) -> None:
    primary = _quality_blocked_evidence()
    decision = evaluate_metastudy(primary, readiness=_ready())
    destination = _publish_evaluated(decision, tmp_path / "evaluated-blocked", evidence=primary)

    assert decision.status == "blocked"
    assert decision.evaluations
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "report.md",
        "evidence.json",
        "sensitivity.json",
    }
    verify_publication(destination)


def test_evidence_bearing_blocked_publication_requires_primary_evidence(tmp_path: Path) -> None:
    primary = _quality_blocked_evidence()
    decision = evaluate_metastudy(primary, readiness=_ready())

    with pytest.raises(MetastudyContractError, match="evidence-bearing publication"):
        publish_metastudy(decision, tmp_path / "missing-evidence")


def test_final_verification_failure_rolls_back_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.publication import (
        service,
    )

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "failed-final-verification"
    original_verify = service.verify_publication
    call_count = 0

    def fail_final_verification(path: Path) -> None:
        nonlocal call_count
        call_count += 1
        original_verify(path)
        raise MetastudyContractError("simulated final verification failure")

    monkeypatch.setattr(service, "verify_publication", fail_final_verification)

    with pytest.raises(MetastudyContractError, match="simulated final verification failure"):
        publish_metastudy(decision, destination)

    assert not destination.exists()
    assert call_count == 1
