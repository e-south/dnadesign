"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/reader_records/_fixtures.py

Shared fixtures for Reader-record boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dnadesign.contracts.reader_records import (
    resolve_digest_verified_dataframe_record,
)

_REVISION_DIGEST = "sha256:" + ("a" * 64)
_CONFIG_DIGEST = "sha256:" + ("b" * 64)
_PRODUCER_CONFIG_DIGEST = "sha256:" + ("c" * 64)
_INPUT_REVISION_DIGEST = "sha256:" + ("d" * 64)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    reader_root = tmp_path / "reader"
    experiment = reader_root / "experiments" / "2026" / "20260101_demo"
    config = experiment / "config.yaml"
    artifact = experiment / "outputs" / "artifacts" / "ratio" / "df.parquet"
    artifact.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")
    artifact.write_bytes(b"parquet fixture bytes")
    catalog = experiment / "outputs" / "manifests" / "records.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    return reader_root, config, artifact, digest


def _page(
    *,
    config: Path,
    artifact: Path,
    digest: str,
    records: list[dict[str, object]],
    total: int,
    truncated: bool = False,
    continuation: str | None = None,
    schema: str = "reader.cli/v1",
    evidence: object = ...,
) -> dict[str, object]:
    experiment = config.parent
    return {
        "schema": schema,
        "ok": True,
        "command": "records",
        "data": {
            "experiment": {
                "id": "20260101_demo",
                "title": "fixture",
                "lifecycle": "active",
                "protocol": "plate_reader/single_reporter_screen",
                "config": str(config),
                "root": str(experiment),
                "evidence": {
                    "data_class": "plate_reader_screen",
                    "data_class_reason": "fixture",
                    "replicate_kind": "biological",
                    "replicate_identity_field": "biological_replicate_id",
                }
                if evidence is ...
                else evidence,
            },
            "catalog": {
                "path": str(experiment / "outputs" / "manifests" / "records.json"),
                "outputs_root": str(experiment / "outputs"),
                "schema_version": 4,
                "provenance_epoch_id": "epoch-fixture",
                "active_invocation_ledger": str(
                    experiment / "outputs" / "manifests" / "invocations" / "epoch-fixture.jsonl"
                ),
            },
            "selection": {"include_history": False},
            "summary": {"records": total, "history": {"included": False, "revisions": None}},
            "records": records,
        },
        "error": None,
        "meta": {"projection": "full", "truncated": truncated, "continuation": continuation},
    }


def _record(
    *, digest: str, path: str = "artifacts/ratio/df.parquet", record_id: str = "ratio_reporter_normalizer/df"
) -> dict[str, object]:
    return {
        "schema_version": 6,
        "record_id": record_id,
        "kind": "dataframe_artifact",
        "contract_id": "plate_reader.annotated.v1",
        "content_digest": digest,
        "size_bytes": len(b"parquet fixture bytes"),
        "path": path,
        "revision": 1,
        "revision_digest": _REVISION_DIGEST,
        "config_digest": _CONFIG_DIGEST,
        "producer_config_digest": _PRODUCER_CONFIG_DIGEST,
        "producer": {
            "kind": "pipeline",
            "id": "ratio_reporter_normalizer",
            "plugin": "transform/ratio_reporter_normalizer",
            "source_recipe": {
                "recipe": "plate_reader/single_reporter_screen_base",
                "with": {"normalizer_channel": "OD600", "reporter_channel": "RFP"},
            },
        },
        "inputs": [
            {
                "label": "df",
                "kind": "record",
                "record": "labels/df",
                "discovery_policy": "record",
                "record_revision_digest": _INPUT_REVISION_DIGEST,
            }
        ],
    }


def _verify_page(
    *,
    status: str = "ok",
    record_id: str = "ratio_reporter_normalizer/df",
) -> dict[str, object]:
    record_status = "ok" if status == "ok" else status
    return {
        "schema": "reader.cli/v1",
        "ok": True,
        "command": "verify",
        "data": {
            "schema": "reader.verify/v1",
            "status": status,
            "summary": {
                "checked": 1,
                "failed": 0 if status == "ok" else 1,
                "unverifiable": 0,
                "invocations_checked": 1,
                "invocation_failures": 0,
            },
            "issues": [],
            "records": [
                {
                    "record_id": record_id,
                    "kind": "dataframe_artifact",
                    "schema_version": 6,
                    "status": record_status,
                    "issues": [],
                }
            ],
        },
        "error": None,
        "meta": {"projection": "full", "truncated": False, "continuation": None},
    }


def _reader_runner(
    records_payload: dict[str, object],
    *,
    verify_payload: dict[str, object] | None = None,
):
    verification = _verify_page() if verify_payload is None else verify_payload

    def run(command, **_kwargs):
        return verification if "verify" in command else records_payload

    return run


def _resolve(reader_root: Path, config: Path):
    return resolve_digest_verified_dataframe_record(
        config,
        reader_root=reader_root,
        experiment_id="20260101_demo",
        protocol_id="plate_reader/single_reporter_screen",
        record_id="ratio_reporter_normalizer/df",
        contract_id="plate_reader.annotated.v1",
        reader_command=("reader-fixture",),
    )


__all__: list[str] = []
