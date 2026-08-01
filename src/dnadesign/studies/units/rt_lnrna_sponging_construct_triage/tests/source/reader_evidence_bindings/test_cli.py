"""CLI routing and fail-fast publication behavior."""

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import ReaderExperimentRouteError
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize import (
    ReaderEvidenceMaterializationError,
    main,
)

from ._fixtures import _resolve_record, _write_bridge_registry, _write_cli_reader_record


def test_cli_materializes_only_a_ready_selected_competence_experiment(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_root, experiment_id = _write_cli_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-1",
                "RFP/OD600": 7654.0,
            }
        ],
    )
    registry = _write_bridge_registry(tmp_path, selected_experiment_ids=[experiment_id])
    output = tmp_path / "bindings.json"
    observed_route: dict[str, object] = {}

    def fake_require_route_readiness(
        registry_path: Path,
        *,
        route_id: str,
        reader_root: Path,
    ) -> dict[str, object]:
        observed_route.update(
            {
                "registry_path": registry_path,
                "route_id": route_id,
                "reader_root": reader_root,
            }
        )
        return {"ok": True, "route_id": route_id, "selected_blockers": []}

    monkeypatch.setattr(
        "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize.require_route_readiness",
        fake_require_route_readiness,
    )
    experiment_dir = reader_root / "experiments" / "2026" / experiment_id
    monkeypatch.setattr(
        "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize.resolve_digest_verified_dataframe_record",
        lambda *_args, **_kwargs: _resolve_record(
            experiment_dir,
            replicate_kind="biological",
            replicate_identity_field="biological_replicate_id",
        ),
    )

    exit_code = main(
        [
            "--reader-root",
            str(reader_root),
            "--experiment-route-registry",
            str(registry),
            "--experiment-id",
            experiment_id,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["binding_count"] == 1
    assert payload["bindings"][0]["subject_id"] == ("rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO")
    assert "bindings=1 unbound=0" in capsys.readouterr().out
    assert observed_route == {
        "registry_path": registry,
        "route_id": "rt_competence_subject_binding",
        "reader_root": reader_root.resolve(),
    }


def test_cli_rejects_an_experiment_not_selected_by_the_competence_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_root, experiment_id = _write_cli_reader_record(tmp_path, [])
    registry = _write_bridge_registry(
        tmp_path,
        selected_experiment_ids=["20260728_other_retron_benchmark"],
    )
    output = tmp_path / "bindings.json"
    monkeypatch.setattr(
        "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize.require_route_readiness",
        lambda *_args, **_kwargs: pytest.fail("unselected evidence must fail before the live readiness check"),
    )

    with pytest.raises(ReaderEvidenceMaterializationError, match="is not selected exactly once by Reader route"):
        main(
            [
                "--reader-root",
                str(reader_root),
                "--experiment-route-registry",
                str(registry),
                "--experiment-id",
                experiment_id,
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_cli_blocked_competence_route_fails_before_reading_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_root, experiment_id = _write_cli_reader_record(tmp_path, [])
    registry = _write_bridge_registry(tmp_path, selected_experiment_ids=[experiment_id])
    output = tmp_path / "bindings.json"

    def blocked_route(*_args: object, **_kwargs: object) -> None:
        raise ReaderExperimentRouteError("Reader route 'rt_competence_subject_binding' is not ready")

    monkeypatch.setattr(
        "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize.require_route_readiness",
        blocked_route,
    )
    monkeypatch.setattr(
        "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize.resolve_digest_verified_dataframe_record",
        lambda *_args, **_kwargs: pytest.fail("blocked readiness must fail before Reader record loading"),
    )

    with pytest.raises(ReaderExperimentRouteError, match="is not ready"):
        main(
            [
                "--reader-root",
                str(reader_root),
                "--experiment-route-registry",
                str(registry),
                "--experiment-id",
                experiment_id,
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_cli_help_names_all_required_arguments(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])

    help_text = capsys.readouterr().out
    for option in (
        "--reader-root",
        "--experiment-route-registry",
        "--experiment-id",
        "--output",
    ):
        assert option in help_text
    assert "--replicate-identity-field" not in help_text
