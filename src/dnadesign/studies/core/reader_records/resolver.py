"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/resolver.py

Canonical orchestration for digest-verified Reader record resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifacts import resolve_record
from .contracts import (
    READER_CATALOG_SCHEMA_VERSION,
    READER_RECORD_SCHEMA_VERSION,
    ReaderDataframeRecordRef,
    ReaderRecordExpectation,
    ReaderRecordSet,
)
from .transport import collect_record_pages, verify_record_store
from .transport import reader_command as resolve_reader_command
from .validation import (
    ReaderDataframeRecordError,
    ReaderRecordError,
    mapping,
    optional_text,
    require_contained,
    text,
)


def resolve_digest_verified_records(
    config_path: Path,
    *,
    reader_root: Path,
    experiment_id: str,
    protocol_id: str,
    expected_records: Mapping[str, ReaderRecordExpectation],
    reader_command: Sequence[str] | None = None,
) -> ReaderRecordSet:
    """Resolve exact records in one catalog snapshot and verify their bytes.

    Reader remains the lifecycle owner. This adapter consumes the public
    ``reader records --format json`` and ``reader verify --format json``
    contracts, canonical catalog v4, and schema-v6 record evidence. Public
    resolutions bracketing verification reject catalog or record identity
    changes during the read.
    """

    root = Path(reader_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    if not root.is_dir():
        raise ReaderRecordError(f"Reader root is missing or not a directory: {root}")
    require_contained(config, root, label="Reader config")
    if not config.is_file():
        raise ReaderRecordError(f"Reader config is missing: {config}")
    if not expected_records:
        raise ReaderRecordError("at least one Reader record expectation is required")
    if len({item.record_id for item in expected_records.values()}) != len(expected_records):
        raise ReaderRecordError("Reader record expectations must use unique record IDs")

    command = tuple(reader_command or resolve_reader_command(root))
    data, rows = collect_record_pages(command, config_path=config, cwd=root)
    context = resolve_context(
        data,
        reader_root=root,
        config_path=config,
        experiment_id=experiment_id,
        protocol_id=protocol_id,
    )
    catalog_path = context["catalog_path"]
    catalog_sha256 = sha256_file(catalog_path)
    initial_identity = record_set_identity(
        data,
        rows,
        expected_records=expected_records,
        catalog_sha256=catalog_sha256,
    )
    verify_record_store(command, config_path=config, cwd=root, expected_records=expected_records)
    confirmed_data, confirmed_rows = collect_record_pages(command, config_path=config, cwd=root)
    confirmed_context = resolve_context(
        confirmed_data,
        reader_root=root,
        config_path=config,
        experiment_id=experiment_id,
        protocol_id=protocol_id,
    )
    confirmed_catalog_path = confirmed_context["catalog_path"]
    confirmed_catalog_sha256 = sha256_file(confirmed_catalog_path)
    confirmed_identity = record_set_identity(
        confirmed_data,
        confirmed_rows,
        expected_records=expected_records,
        catalog_sha256=confirmed_catalog_sha256,
    )
    if confirmed_identity != initial_identity:
        raise ReaderRecordError(
            f"{experiment_id}: Reader catalog or exact requested record identity changed during resolution"
        )
    resolved = {
        name: resolve_record(
            one_record(confirmed_rows, record_id=expectation.record_id, experiment_id=experiment_id),
            expectation=expectation,
            outputs_root=confirmed_context["outputs_root"],
        )
        for name, expectation in expected_records.items()
    }
    return ReaderRecordSet(
        reader_root=root,
        experiment_root=confirmed_context["experiment_root"],
        config_path=config,
        outputs_root=confirmed_context["outputs_root"],
        catalog_path=confirmed_catalog_path,
        catalog_sha256=confirmed_catalog_sha256,
        catalog_schema_version=READER_CATALOG_SCHEMA_VERSION,
        provenance_epoch_id=confirmed_context["provenance_epoch_id"],
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        experiment_evidence=confirmed_context["experiment_evidence"],
        records=resolved,
    )


def resolve_digest_verified_dataframe_record(
    config_path: Path,
    *,
    reader_root: Path,
    experiment_id: str,
    protocol_id: str,
    record_id: str,
    contract_id: str,
    reader_command: Sequence[str] | None = None,
) -> ReaderDataframeRecordRef:
    """Resolve one assay dataframe through the shared multi-record adapter."""

    resolved = resolve_digest_verified_records(
        config_path,
        reader_root=reader_root,
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        expected_records={
            "dataframe": ReaderRecordExpectation(record_id=record_id, contract_id=contract_id),
        },
        reader_command=reader_command,
    )
    evidence = resolved.experiment_evidence
    if not evidence:
        raise ReaderDataframeRecordError("records.data.experiment.evidence must be an object")
    replicate_kind = text(evidence.get("replicate_kind"), label="records.data.experiment.evidence.replicate_kind")
    if replicate_kind not in {"biological", "technical", "mixed", "unknown"}:
        raise ReaderDataframeRecordError(
            f"{experiment_id}: replicate_kind must be biological, technical, mixed, or unknown for assay evidence"
        )
    replicate_field = optional_text(
        evidence.get("replicate_identity_field"),
        label="records.data.experiment.evidence.replicate_identity_field",
    )

    record = resolved.records["dataframe"]
    assert record.path is not None
    assert record.reader_path is not None
    assert record.content_digest is not None
    return ReaderDataframeRecordRef._from_source_closed_reader(
        reader_root=resolved.reader_root,
        experiment_id=experiment_id,
        protocol_id=resolved.protocol_id,
        replicate_kind=replicate_kind,
        replicate_identity_field=replicate_field,
        record_id=record_id,
        record_kind=record.kind,
        record_schema_version=READER_RECORD_SCHEMA_VERSION,
        revision=record.revision,
        revision_digest=record.revision_digest,
        config_digest=record.config_digest,
        producer_config_digest=record.producer_config_digest,
        producer=record.producer,
        inputs=record.inputs,
        contract_id=contract_id,
        reader_path=record.reader_path,
        path=record.path,
        manifest_path=resolved.catalog_path,
        content_digest=record.content_digest,
    )


def resolve_context(
    data: Mapping[str, object],
    *,
    reader_root: Path,
    config_path: Path,
    experiment_id: str,
    protocol_id: str,
) -> dict[str, object]:
    experiment = mapping(data.get("experiment"), label="records.data.experiment")
    observed_experiment_id = text(experiment.get("id"), label="records.data.experiment.id")
    if observed_experiment_id != experiment_id:
        raise ReaderRecordError(
            f"Reader experiment id mismatch: expected {experiment_id!r}, observed {observed_experiment_id!r}"
        )
    observed_protocol = text(experiment.get("protocol"), label="records.data.experiment.protocol")
    if observed_protocol != protocol_id:
        raise ReaderRecordError(f"{experiment_id}: protocol must equal {protocol_id!r}; observed {observed_protocol!r}")
    observed_config = Path(text(experiment.get("config"), label="records.data.experiment.config")).resolve()
    require_contained(observed_config, reader_root, label="Reader CLI config")
    if observed_config != config_path:
        raise ReaderRecordError(
            f"{experiment_id}: Reader CLI config identity changed; expected {config_path}, observed {observed_config}"
        )
    experiment_root = Path(text(experiment.get("root"), label="records.data.experiment.root")).resolve()
    require_contained(experiment_root, reader_root, label="Reader experiment root")
    if experiment_root != config_path.parent:
        raise ReaderRecordError(
            f"{experiment_id}: Reader experiment root must equal the config parent; "
            f"expected {config_path.parent}, observed {experiment_root}"
        )
    evidence_value = experiment.get("evidence")
    evidence = {} if evidence_value is None else dict(mapping(evidence_value, label="records.data.experiment.evidence"))

    catalog = mapping(data.get("catalog"), label="records.data.catalog")
    if catalog.get("schema_version") != READER_CATALOG_SCHEMA_VERSION:
        raise ReaderRecordError(
            f"{experiment_id}: requires Reader catalog schema v{READER_CATALOG_SCHEMA_VERSION}; "
            "regenerate and verify the Reader experiment"
        )
    outputs_root = Path(text(catalog.get("outputs_root"), label="records.data.catalog.outputs_root")).resolve()
    require_contained(outputs_root, reader_root, label="Reader outputs root")
    expected_outputs_root = experiment_root / "outputs"
    if outputs_root != expected_outputs_root:
        raise ReaderRecordError(
            f"{experiment_id}: Reader outputs root must equal the experiment outputs directory; "
            f"expected {expected_outputs_root}, observed {outputs_root}"
        )
    catalog_path = Path(text(catalog.get("path"), label="records.data.catalog.path")).resolve()
    require_contained(catalog_path, reader_root, label="Reader record manifest")
    expected_catalog_path = outputs_root / "manifests" / "records.json"
    if catalog_path != expected_catalog_path:
        raise ReaderRecordError(
            f"{experiment_id}: Reader record manifest must equal the canonical records manifest; "
            f"expected {expected_catalog_path}, observed {catalog_path}"
        )
    if not catalog_path.is_file():
        raise ReaderRecordError(f"{experiment_id}: Reader record manifest is missing: {catalog_path}")
    provenance_epoch_id = text(catalog.get("provenance_epoch_id"), label="records.data.catalog.provenance_epoch_id")
    return {
        "experiment_root": experiment_root,
        "outputs_root": outputs_root,
        "catalog_path": catalog_path,
        "provenance_epoch_id": provenance_epoch_id,
        "experiment_evidence": evidence,
    }


def record_set_identity(
    data: Mapping[str, object],
    rows: tuple[Mapping[str, object], ...],
    *,
    expected_records: Mapping[str, ReaderRecordExpectation],
    catalog_sha256: str,
) -> str:
    identity = {
        "experiment": data.get("experiment"),
        "catalog": data.get("catalog"),
        "catalog_sha256": catalog_sha256,
        "records": {
            name: one_record(rows, record_id=expectation.record_id, experiment_id="Reader").copy()
            for name, expectation in sorted(expected_records.items())
        },
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def one_record(
    rows: tuple[Mapping[str, object], ...],
    *,
    record_id: str,
    experiment_id: str,
) -> Mapping[str, object]:
    matches = [row for row in rows if row.get("record_id") == record_id]
    if len(matches) != 1:
        raise ReaderRecordError(
            f"{experiment_id}: public Reader records payload must contain exactly one {record_id!r}; "
            f"observed {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ReaderRecordError(f"could not read Reader catalog {path}: {exc}") from exc
    return hashlib.sha256(content).hexdigest()
