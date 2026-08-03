"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_experiment_routes.py

Contract tests for consuming bridge-owned Reader experiment routes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderExperimentRouteError,
    SelectedReaderExperiment,
    require_route_readiness,
    selected_experiments_for_route,
)

_RESPONSE_ROUTE_ID = "rt_lnrna_reporter_response_evidence"


def _write_registry(
    tmp_path: Path,
    *,
    experiments: list[dict[str, object]],
    memberships: list[dict[str, str]],
) -> Path:
    path = tmp_path / "reader-experiment-routes.json"
    path.write_text(
        json.dumps(
            {
                "schema": "phd.retron_reader_experiment_routes.v2",
                "owner": "phd-workspace",
                "routes": {
                    _RESPONSE_ROUTE_ID: {
                        "first_owner": "reader",
                        "continue_with": (
                            "dnadesign/docs/studies/rt_lnrna_sponging_construct_triage/routes/"
                            "reporter-response-evidence.md"
                        ),
                        "required_reader_state": "records_ready",
                    },
                    "rt_competence_subject_binding": {
                        "first_owner": "reader",
                        "continue_with": "dnadesign/.agents/skills/rt-lnrna-reporter-response/SKILL.md",
                        "required_reader_state": "records_ready",
                    },
                },
                "experiments": experiments,
                "memberships": memberships,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _entry(experiment_id: str) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "reader_config": f"reader/experiments/2026/{experiment_id}/config.yaml",
    }


def _membership(
    experiment_id: str,
    route_id: str,
    membership: str = "selected",
) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "route_id": route_id,
        "membership": membership,
    }


def test_selected_experiments_for_route_returns_exact_identity_and_config(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        experiments=[
            _entry("response_a"),
            _entry("competence_a"),
            _entry("response_b"),
            _entry("response_context"),
        ],
        memberships=[
            _membership("response_a", _RESPONSE_ROUTE_ID),
            _membership("competence_a", "rt_competence_subject_binding"),
            _membership("response_b", _RESPONSE_ROUTE_ID),
            _membership("response_context", _RESPONSE_ROUTE_ID, "related"),
        ],
    )

    assert selected_experiments_for_route(path, route_id=_RESPONSE_ROUTE_ID) == (
        SelectedReaderExperiment(
            experiment_id="response_a",
            reader_config="reader/experiments/2026/response_a/config.yaml",
        ),
        SelectedReaderExperiment(
            experiment_id="response_b",
            reader_config="reader/experiments/2026/response_b/config.yaml",
        ),
    )


def test_selected_experiments_for_route_rejects_empty_or_ambiguous_selection(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        experiments=[_entry("competence_a")],
        memberships=[_membership("competence_a", "rt_competence_subject_binding")],
    )
    with pytest.raises(ReaderExperimentRouteError, match="selects no Reader experiments"):
        selected_experiments_for_route(path, route_id=_RESPONSE_ROUTE_ID)

    duplicate = _write_registry(
        tmp_path,
        experiments=[
            _entry("response_a"),
            _entry("response_a"),
        ],
        memberships=[_membership("response_a", _RESPONSE_ROUTE_ID)],
    )
    with pytest.raises(ReaderExperimentRouteError, match="duplicate experiment_id"):
        selected_experiments_for_route(duplicate, route_id=_RESPONSE_ROUTE_ID)


def test_selected_experiments_for_route_rejects_duplicate_pairs_and_unknown_fields(tmp_path: Path) -> None:
    duplicate = _write_registry(
        tmp_path,
        experiments=[_entry("response_a")],
        memberships=[
            _membership("response_a", _RESPONSE_ROUTE_ID),
            _membership("response_a", _RESPONSE_ROUTE_ID, "related"),
        ],
    )
    with pytest.raises(ReaderExperimentRouteError, match="duplicate experiment-route membership"):
        selected_experiments_for_route(duplicate, route_id=_RESPONSE_ROUTE_ID)

    payload = json.loads(duplicate.read_text(encoding="utf-8"))
    payload["memberships"] = [_membership("response_a", _RESPONSE_ROUTE_ID)]
    payload["memberships"][0]["treatment"] = "forbidden science field"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReaderExperimentRouteError, match="unknown=treatment"):
        selected_experiments_for_route(duplicate, route_id=_RESPONSE_ROUTE_ID)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("experiment_id", "missing", "references unknown experiment"),
        ("route_id", "missing", "references unknown route"),
    ],
)
def test_selected_experiments_for_route_rejects_unknown_membership_references(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    membership = _membership("response_a", _RESPONSE_ROUTE_ID)
    membership[field] = value
    path = _write_registry(
        tmp_path,
        experiments=[_entry("response_a")],
        memberships=[membership],
    )

    with pytest.raises(ReaderExperimentRouteError, match=match):
        selected_experiments_for_route(path, route_id=_RESPONSE_ROUTE_ID)


def test_require_route_readiness_calls_bridge_for_only_the_requested_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_root = tmp_path / ".agents/skills/retron-assay-study-bridge"
    registry = skill_root / "references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}", encoding="utf-8")
    checker = skill_root / "scripts/check_reader_experiment_readiness.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("# fixture", encoding="utf-8")
    reader_root = tmp_path / "reader"
    reader_root.mkdir()
    observed_command: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        observed_command.extend(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "route_id": _RESPONSE_ROUTE_ID,
                    "selected_blockers": [],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    receipt = require_route_readiness(
        registry,
        route_id=_RESPONSE_ROUTE_ID,
        reader_root=reader_root,
    )

    assert receipt["ok"] is True
    assert observed_command[-2:] == ["--route-id", _RESPONSE_ROUTE_ID]


def test_require_route_readiness_rejects_noncanonical_registry_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_root = tmp_path / "untrusted/retron-assay-study-bridge"
    registry = skill_root / "references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}", encoding="utf-8")
    checker = skill_root / "scripts/check_reader_experiment_readiness.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("# must not execute", encoding="utf-8")
    reader_root = tmp_path / "reader"
    reader_root.mkdir()

    def unexpected_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("noncanonical bridge code must not execute")

    monkeypatch.setattr("subprocess.run", unexpected_run)

    with pytest.raises(ReaderExperimentRouteError, match="canonical bridge registry"):
        require_route_readiness(
            registry,
            route_id=_RESPONSE_ROUTE_ID,
            reader_root=reader_root,
        )


def test_require_route_readiness_rejects_checker_symlink_escape_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_root = tmp_path / ".agents/skills/retron-assay-study-bridge"
    registry = skill_root / "references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}", encoding="utf-8")
    outside_checker = tmp_path / "outside/check_reader_experiment_readiness.py"
    outside_checker.parent.mkdir()
    outside_checker.write_text("# must not execute", encoding="utf-8")
    checker = skill_root / "scripts/check_reader_experiment_readiness.py"
    checker.parent.mkdir(parents=True)
    checker.symlink_to(outside_checker)
    reader_root = tmp_path / "reader"
    reader_root.mkdir()

    def unexpected_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("escaped bridge checker must not execute")

    monkeypatch.setattr("subprocess.run", unexpected_run)

    with pytest.raises(ReaderExperimentRouteError, match="checker escapes"):
        require_route_readiness(
            registry,
            route_id=_RESPONSE_ROUTE_ID,
            reader_root=reader_root,
        )


def test_require_route_readiness_rejects_blocked_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_root = tmp_path / ".agents/skills/retron-assay-study-bridge"
    registry = skill_root / "references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}", encoding="utf-8")
    checker = skill_root / "scripts/check_reader_experiment_readiness.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("# fixture", encoding="utf-8")
    reader_root = tmp_path / "reader"
    reader_root.mkdir()
    blocker = {"experiment_id": "stale", "route_id": _RESPONSE_ROUTE_ID}
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout=json.dumps(
                {
                    "ok": False,
                    "route_id": _RESPONSE_ROUTE_ID,
                    "selected_blockers": [blocker],
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(ReaderExperimentRouteError, match="is not ready"):
        require_route_readiness(
            registry,
            route_id=_RESPONSE_ROUTE_ID,
            reader_root=reader_root,
        )
