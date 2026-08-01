"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/test_atomicity.py

Tests create-only publication staging and atomic installation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

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

    with pytest.raises(FileExistsError):
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


def test_publication_installs_one_complete_staged_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "atomic-publication"
    original_install = publication._rename_directory_create_only
    observed_install: list[tuple[set[str], bool]] = []

    def inspect_install(stage: Path, target: Path) -> None:
        verify_publication(stage)
        observed_install.append(({entry.name for entry in stage.iterdir()}, target.exists()))
        original_install(stage, target)

    monkeypatch.setattr(publication, "_rename_directory_create_only", inspect_install)

    publish_metastudy(decision, destination)

    assert observed_install == [({"manifest.json", "report.md", "sensitivity.json"}, False)]
    verify_publication(destination)


def test_publication_target_race_preserves_competitor_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "raced-publication"
    original_install = publication._rename_directory_create_only
    competitor_inode: list[int] = []

    def race_install(stage: Path, target: Path) -> None:
        target.mkdir()
        competitor_inode.append(target.stat().st_ino)
        original_install(stage, target)

    monkeypatch.setattr(publication, "_rename_directory_create_only", race_install)

    with pytest.raises(FileExistsError, match="create-only"):
        publish_metastudy(decision, destination)

    assert destination.is_dir()
    assert destination.stat().st_ino == competitor_inode[0]
    assert list(destination.iterdir()) == []
    assert list(tmp_path.glob(".raced-publication.*")) == []


def test_publication_rejects_broken_destination_symlink_without_following_it(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    outside = tmp_path / "outside" / "redirected-publication"
    destination = tmp_path / "symlink-publication"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError, match="create-only"):
        publish_metastudy(decision, destination)

    assert destination.is_symlink()
    assert not outside.exists()
    assert list(tmp_path.glob(".symlink-publication.*")) == []


def test_publication_install_failure_cleans_private_staging_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "failed-publication"

    def fail_install(_stage: Path, _target: Path) -> None:
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr(publication, "_rename_directory_create_only", fail_install)

    with pytest.raises(OSError, match="simulated atomic rename failure"):
        publish_metastudy(decision, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".failed-publication.*")) == []


def test_termination_before_atomic_install_exposes_no_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "interrupted-publication"

    def interrupt_install(stage: Path, target: Path) -> None:
        verify_publication(stage)
        assert not target.exists()
        raise SystemExit("simulated process termination")

    monkeypatch.setattr(publication, "_rename_directory_create_only", interrupt_install)

    with pytest.raises(SystemExit, match="simulated process termination"):
        publish_metastudy(decision, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".interrupted-publication.*")) == []
