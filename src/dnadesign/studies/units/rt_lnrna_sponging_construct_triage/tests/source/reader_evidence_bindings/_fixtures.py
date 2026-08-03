"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/reader_evidence_bindings/_fixtures.py

Shared fixtures for Reader evidence-binding contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dnadesign.studies.core.reader_records import ReaderRecordProducer
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderDataframeRecordRef,
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
        config_digest="sha256:" + ("b" * 64),
        producer_config_digest="sha256:" + ("c" * 64),
        producer=ReaderRecordProducer(
            kind="pipeline",
            id="sample_measurements",
            plugin="transform/sample_measurements",
        ),
        inputs=(),
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


__all__: list[str] = []
