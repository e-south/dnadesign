"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_evidence_bindings.py

Contracts for study-owned Reader evidence bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderDataframeRecordRef,
    ReaderEvidenceBindingError,
    ReaderExperimentRouteError,
    build_reader_evidence_bindings,
    materialize_reader_evidence_bindings_json,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import bindings as bindings_module
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize import (
    ReaderEvidenceMaterializationError,
    main,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import (
    SubjectBindingRegistry,
    load_registered_subject_bindings,
)

_REVISION_DIGEST = "sha256:" + ("a" * 64)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _write_reader_record(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    experiment_id = "20260720_retron_Eco1_26_180_201_202_203_204_benchmark"
    experiment = tmp_path / experiment_id
    artifact = experiment / "outputs" / "artifacts" / "ratio" / "df.parquet"
    artifact.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(artifact, index=False)
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    record = {
        "schema_version": 6,
        "record_id": "sample_measurements/df",
        "kind": "dataframe_artifact",
        "contract_id": "plate_reader.annotated.v1",
        "content_digest": digest,
        "path": "artifacts/sample_measurements/df.parquet",
        "revision": 1,
        "revision_digest": _REVISION_DIGEST,
    }
    manifest = experiment / "outputs" / "manifests" / "records.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "provenance_epoch_id": "epoch-fixture",
                "active_invocation_ledger": "manifests/invocations/epoch-fixture.jsonl",
                "latest": {"sample_measurements/df": record},
                "history": {"sample_measurements/df": [record]},
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _resolve_record(
    experiment: Path,
    *,
    replicate_kind: str = "biological",
    replicate_identity_field: str | None = None,
):
    artifact = experiment / "outputs" / "artifacts" / "ratio" / "df.parquet"
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    return ReaderDataframeRecordRef._from_source_closed_reader(
        reader_root=experiment.parent,
        experiment_id=experiment.name,
        protocol_id="plate_reader/single_reporter_screen",
        replicate_kind=replicate_kind,
        replicate_identity_field=replicate_identity_field,
        record_id="sample_measurements/df",
        record_kind="dataframe_artifact",
        record_schema_version=6,
        revision=1,
        revision_digest=_REVISION_DIGEST,
        contract_id="plate_reader.annotated.v1",
        reader_path="artifacts/sample_measurements/df.parquet",
        path=artifact,
        manifest_path=experiment / "outputs" / "manifests" / "records.json",
        content_digest=digest,
    )


def _write_cli_reader_record(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, str]:
    experiment_id = "20260727_retron_Eco1_26_D01_D02_P01_P03_DP01_DP03_benchmark"
    reader_root = tmp_path / "reader"
    source = _write_reader_record(tmp_path / "source", rows)
    destination = reader_root / "experiments" / "2026" / experiment_id
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    (destination / "config.yaml").write_text(
        f"schema: reader/v8\nexperiment:\n  id: {experiment_id}\n  title: fixture\n"
        "evidence:\n  replicate_kind: biological\n  replicate_identity_field: biological_replicate_id\n",
        encoding="utf-8",
    )
    return reader_root, experiment_id


def _write_bridge_registry(tmp_path: Path, *, selected_experiment_ids: list[str]) -> Path:
    registry = tmp_path / "reader-experiment-routes.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "phd.retron_reader_experiment_routes.v2",
                "owner": "phd-workspace",
                "routes": {
                    "rt_competence_subject_binding": {
                        "first_owner": "reader",
                        "continue_with": "dnadesign/.agents/skills/rt-lnrna-reporter-response/SKILL.md",
                        "required_reader_state": "records_ready",
                    }
                },
                "experiments": [
                    {
                        "experiment_id": experiment_id,
                        "reader_config": f"reader/experiments/2026/{experiment_id}/config.yaml",
                    }
                    for experiment_id in selected_experiment_ids
                ],
                "memberships": [
                    {
                        "experiment_id": experiment_id,
                        "route_id": "rt_competence_subject_binding",
                        "membership": "selected",
                    }
                    for experiment_id in selected_experiment_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_d01_exact_aliases_bind_and_retain_reader_provenance(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
                "time": 12.0,
                "RFP/OD600": 7654.0,
            },
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-2",
                "time": 12.0,
                "RFP/OD600": 7012.0,
            },
        ],
    )

    record = _resolve_record(experiment)
    registry = load_registered_subject_bindings(repo_root=_repo_root())
    binding_set = build_reader_evidence_bindings(record=record, subject_registry=registry)

    assert binding_set.unbound_count == 0
    assert len(binding_set.rows) == 1
    row = binding_set.rows[0]
    assert row.subject_id == "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"
    assert row.raw_design_id == "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp"
    assert row.raw_assay_subject_id == "retron-205-Eco1RT-G3-D01"
    assert row.reader_replicate_kind == "biological"
    assert row.reader_replicate_identity_field is None
    assert row.observation_identity_field == "position"
    assert row.observation_identity_values == ("colony-1", "colony-2")
    assert row.biological_replicate_identity_scopes == ()
    assert row.binding_state == "bound"
    assert row.binding_reason == "exact_subject_alias_match"
    assert row.reader_record_schema_version == 6
    assert row.reader_record_revision == 1
    assert row.reader_record_revision_digest == _REVISION_DIGEST
    assert row.reader_record_contract_id == "plate_reader.annotated.v1"
    assert row.reader_record_content_digest == _resolve_record(experiment).content_digest


def test_unknown_reader_evidence_remains_unknown_without_guessing(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment, replicate_kind="unknown"),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.reader_replicate_kind == "unknown"
    assert row.reader_replicate_identity_field is None
    assert row.observation_identity_field == "position"
    assert row.biological_replicate_identity_scopes == ()


def test_unknown_reader_evidence_rejects_a_declared_identity_field(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-1",
            }
        ],
    )
    record = _resolve_record(
        experiment,
        replicate_kind="unknown",
        replicate_identity_field="biological_replicate_id",
    )

    with pytest.raises(ReaderEvidenceBindingError, match="unknown replicate identity"):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_biological_reader_evidence_uses_the_declared_replicate_identity(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-1",
            },
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A2",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-2",
            },
        ],
    )
    record = _resolve_record(
        experiment,
        replicate_kind="biological",
        replicate_identity_field="biological_replicate_id",
    )

    binding_set = build_reader_evidence_bindings(
        record=record,
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.reader_replicate_kind == "biological"
    assert row.reader_replicate_identity_field == "biological_replicate_id"
    assert row.observation_identity_field == "position"
    assert row.observation_identity_values == ("A1", "A2")
    assert tuple(
        (scope.condition_value, scope.biological_replicate_id) for scope in row.biological_replicate_identity_scopes
    ) == (
        ("0.0 µM aTc + 0.0 µM IPTG", "culture-1"),
        ("0.0 µM aTc + 0.0 µM IPTG", "culture-2"),
    )


def test_unknown_alias_is_reported_unbound_without_guessing(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-999-unknown; pBbS2c-rfp",
                "assay_subject_id": "retron-999-unknown",
                "position": "colony-1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    assert binding_set.unbound_count == 1
    row = binding_set.rows[0]
    assert row.subject_id is None
    assert row.binding_state == "unbound"
    assert row.binding_reason == "no_exact_subject_alias_match"
    assert row.raw_design_id == "pES-retron-999-unknown; pBbS2c-rfp"


def test_binding_builder_rejects_directly_constructed_registry(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    loaded = load_registered_subject_bindings(repo_root=_repo_root())
    forged = SubjectBindingRegistry(
        schema_id=loaded.schema_id,
        study_id=loaded.study_id,
        binding_set_id=loaded.binding_set_id,
        subjects=loaded.subjects,
    )

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed registry"):
        build_reader_evidence_bindings(record=_resolve_record(experiment), subject_registry=forged)


def test_binding_builder_rejects_reader_record_without_source_closure(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    forged = replace(_resolve_record(experiment))

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed Reader record"):
        build_reader_evidence_bindings(
            record=forged,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_partial_alias_match_remains_unbound(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-999-typo",
                "position": "colony-1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.subject_id is None
    assert row.binding_state == "unbound"
    assert row.binding_reason == "partial_exact_subject_alias_match"


def test_conflicting_exact_aliases_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-206-Eco1RT-G3-D02",
                "position": "colony-1",
            }
        ],
    )

    with pytest.raises(ReaderEvidenceBindingError, match="conflicting exact aliases"):
        build_reader_evidence_bindings(
            record=_resolve_record(experiment),
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_binding_builder_rechecks_artifact_digest_before_reading(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    record = _resolve_record(experiment)
    record.path.write_bytes(b"drift after record resolution")

    with pytest.raises(ReaderEvidenceBindingError, match="content digest changed"):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"record_schema_version": 5}, "record schema v6"),
        ({"revision": 0}, "revision must be a positive integer"),
        ({"revision_digest": "sha256:" + ("A" * 64)}, "revision_digest must be a lowercase sha256 digest"),
        ({"content_digest": "sha256:" + ("A" * 64)}, "content_digest must be a lowercase sha256 digest"),
    ],
)
def test_binding_builder_rejects_invalid_exact_record_identity(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    record = replace(_resolve_record(experiment), **changes)

    with pytest.raises(ReaderEvidenceBindingError, match=message):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=SubjectBindingRegistry(
                schema_id="fixture",
                study_id="rt_lnrna_sponging_construct_triage",
                binding_set_id="fixture",
                subjects=(),
            ),
        )


def test_binding_builder_parses_the_same_bytes_it_digest_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    record = _resolve_record(experiment)
    replacement = tmp_path / "replacement.parquet"
    pd.DataFrame(
        [
            {
                "design_id": "pES-retron-206-Eco1RT-G3-D02; pBbS2c-rfp",
                "assay_subject_id": "retron-206-Eco1RT-G3-D02",
                "position": "colony-2",
            }
        ]
    ).to_parquet(replacement, index=False)
    replacement_bytes = replacement.read_bytes()
    read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        data = read_bytes(path)
        if path == record.path:
            record.path.write_bytes(replacement_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    binding_set = build_reader_evidence_bindings(
        record=record,
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    assert binding_set.rows[0].subject_id == "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"


def test_materialized_binding_rows_exclude_measurements_and_interpretations(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
                "OD600": 0.97,
                "RFP/OD600": 7654.0,
                "assay_score": 0.81,
            }
        ],
    )
    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )
    destination = tmp_path / "evidence-bindings.json"

    materialize_reader_evidence_bindings_json(binding_set, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    row = payload["bindings"][0]
    assert payload["artifact_id"] == binding_set.artifact_id
    assert payload["artifact_digest"] == binding_set.artifact_digest
    assert set(row) == {
        "reader_experiment_id",
        "reader_protocol_id",
        "reader_replicate_kind",
        "reader_replicate_identity_field",
        "reader_record_id",
        "reader_record_kind",
        "reader_record_schema_version",
        "reader_record_revision",
        "reader_record_revision_digest",
        "reader_record_contract_id",
        "reader_record_content_digest",
        "reader_record_path",
        "raw_design_id",
        "raw_assay_subject_id",
        "subject_id",
        "observation_identity_field",
        "observation_identity_values",
        "biological_replicate_identity_scopes",
        "binding_state",
        "binding_reason",
    }
    assert not ({"OD600", "RFP", "RFP/OD600", "assay_score", "measurement"} & set(row))
    assert payload["unbound_count"] == 0


def test_materialized_binding_loader_restores_source_closure_and_rejects_digest_drift(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    record = _resolve_record(experiment)
    registry = load_registered_subject_bindings(repo_root=_repo_root())
    binding_set = build_reader_evidence_bindings(record=record, subject_registry=registry)
    destination = tmp_path / "evidence-bindings.json"
    materialize_reader_evidence_bindings_json(binding_set, destination)

    loaded = bindings_module.load_reader_evidence_bindings_json(
        destination,
        record=record,
        subject_registry=registry,
    )

    assert loaded.is_source_closed
    assert loaded.artifact_id == binding_set.artifact_id
    assert loaded.artifact_digest == binding_set.artifact_digest

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["bindings"][0]["subject_id"] = "forged-subject"
    unsigned = dict(payload)
    unsigned.pop("artifact_digest")
    payload["artifact_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    )
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReaderEvidenceBindingError, match="no longer matches"):
        bindings_module.load_reader_evidence_bindings_json(
            destination,
            record=record,
            subject_registry=registry,
        )


def test_binding_publication_rejects_forged_sets_before_mutation_and_never_overwrites(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )
    forged_destination = tmp_path / "forged" / "bindings.json"

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed set"):
        materialize_reader_evidence_bindings_json(replace(binding_set), forged_destination)
    assert not forged_destination.parent.exists()

    destination = tmp_path / "bindings.json"
    materialize_reader_evidence_bindings_json(binding_set, destination)
    original = destination.read_bytes()
    with pytest.raises(ReaderEvidenceBindingError, match="already exists"):
        materialize_reader_evidence_bindings_json(binding_set, destination)
    assert destination.read_bytes() == original
    assert list(tmp_path.glob(".bindings.json.*.tmp")) == []


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
