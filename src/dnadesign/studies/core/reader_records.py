"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records.py

Resolve digest-verified Reader records through Reader's public JSON CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

READER_CLI_SCHEMA = "reader.cli/v1"
READER_CATALOG_SCHEMA_VERSION = 4
READER_RECORD_SCHEMA_VERSION = 6
_PAGE_LIMIT = 100
_MAX_RECORD_PAGES = 100
_READER_CLI_TIMEOUT_SECONDS = 60
_SOURCE_CLOSURE_TOKEN = object()


class ReaderDataframeRecordError(ValueError):
    """Raised when Reader's public record handoff fails its contract."""


@dataclass(frozen=True, slots=True)
class ReaderDataframeRecordRef:
    """Verified public metadata and local bytes for one Reader dataframe."""

    experiment_id: str
    protocol_id: str
    replicate_kind: str
    replicate_identity_field: str | None
    record_id: str
    record_kind: str
    record_schema_version: int
    revision: int
    revision_digest: str
    contract_id: str
    reader_path: str
    path: Path
    manifest_path: Path
    content_digest: str
    _source_closure: object | None = field(default=None, init=False, repr=False, compare=False)
    _source_reader_root: Path | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def _from_source_closed_reader(
        cls,
        *,
        reader_root: Path,
        experiment_id: str,
        protocol_id: str,
        replicate_kind: str,
        replicate_identity_field: str | None,
        record_id: str,
        record_kind: str,
        record_schema_version: int,
        revision: int,
        revision_digest: str,
        contract_id: str,
        reader_path: str,
        path: Path,
        manifest_path: Path,
        content_digest: str,
    ) -> ReaderDataframeRecordRef:
        """Construct a reference only after the public Reader resolver closes its paths."""

        root = Path(reader_root).expanduser().resolve()
        artifact = Path(path).expanduser().resolve()
        manifest = Path(manifest_path).expanduser().resolve()
        _require_contained(artifact, root, label="Reader artifact")
        _require_contained(manifest, root, label="Reader record manifest")
        reference = cls(
            experiment_id=experiment_id,
            protocol_id=protocol_id,
            replicate_kind=replicate_kind,
            replicate_identity_field=replicate_identity_field,
            record_id=record_id,
            record_kind=record_kind,
            record_schema_version=record_schema_version,
            revision=revision,
            revision_digest=revision_digest,
            contract_id=contract_id,
            reader_path=reader_path,
            path=artifact,
            manifest_path=manifest,
            content_digest=content_digest,
        )
        object.__setattr__(reference, "_source_closure", _SOURCE_CLOSURE_TOKEN)
        object.__setattr__(reference, "_source_reader_root", root)
        return reference

    @property
    def is_source_closed(self) -> bool:
        """Whether this reference came from the canonical public Reader resolver."""

        return self._source_closure is _SOURCE_CLOSURE_TOKEN and self._source_reader_root is not None

    @property
    def ref(self) -> str:
        return f"{self.experiment_id}:{self.record_id}"


ReaderRecordError = ReaderDataframeRecordError


@dataclass(frozen=True, slots=True)
class ReaderRecordExpectation:
    """One exact record contract requested from a Reader experiment."""

    record_id: str
    kind: str = "dataframe_artifact"
    contract_id: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ReaderRecordError("Reader record expectation requires a non-empty record_id")
        if self.kind not in {"dataframe_artifact", "file_bundle"}:
            raise ReaderRecordError(f"unsupported Reader record kind {self.kind!r}")
        if self.kind == "dataframe_artifact" and not (self.contract_id or "").strip():
            raise ReaderRecordError("Reader dataframe expectations require a contract_id")
        if self.kind == "file_bundle" and self.contract_id is not None:
            raise ReaderRecordError("Reader file-bundle expectations do not accept a dataframe contract_id")


@dataclass(frozen=True, slots=True)
class ReaderArtifactFile:
    """Verified bytes for one file in a Reader record revision."""

    reader_path: str
    path: Path
    size_bytes: int
    content_digest: str
    content: bytes = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.reader_path,
            "size_bytes": self.size_bytes,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class ReaderResolvedRecord:
    """One source-closed schema-v6 Reader record revision."""

    record_id: str
    kind: str
    schema_version: int
    revision: int
    revision_digest: str
    contract_id: str | None
    producer: Mapping[str, object]
    producer_config_digest: str | None
    inputs: tuple[Mapping[str, object], ...]
    path: Path | None
    reader_path: str | None
    size_bytes: int | None
    content_digest: str | None
    content: bytes | None = field(repr=False, compare=False)
    files: tuple[ReaderArtifactFile, ...] = ()
    _source_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def _verified(cls, **values: object) -> ReaderResolvedRecord:
        result = cls(**values)  # type: ignore[arg-type]
        object.__setattr__(result, "_source_closure", _SOURCE_CLOSURE_TOKEN)
        return result

    @property
    def is_source_closed(self) -> bool:
        return self._source_closure is _SOURCE_CLOSURE_TOKEN

    def to_dict(self) -> dict[str, object]:
        if not self.is_source_closed:
            raise ReaderRecordError("Reader record was not resolved through the public CLI")
        result: dict[str, object] = {
            "record_id": self.record_id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "revision": self.revision,
            "revision_digest": self.revision_digest,
        }
        if self.contract_id is not None:
            result["contract_id"] = self.contract_id
        if self.reader_path is not None:
            result.update(
                {
                    "path": self.reader_path,
                    "size_bytes": self.size_bytes,
                    "content_digest": self.content_digest,
                }
            )
        if self.files:
            result.update(
                {
                    "producer_config_digest": self.producer_config_digest,
                    "file_evidence": [item.to_dict() for item in self.files],
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class ReaderRecordSet:
    """One verified Reader experiment identity and exact requested records."""

    reader_root: Path
    experiment_root: Path
    config_path: Path
    outputs_root: Path
    catalog_path: Path
    catalog_sha256: str
    catalog_schema_version: int
    provenance_epoch_id: str
    experiment_id: str
    protocol_id: str
    experiment_evidence: Mapping[str, object]
    records: Mapping[str, ReaderResolvedRecord]

    def source_receipt(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol_id,
            "catalog": {
                "schema_version": self.catalog_schema_version,
                "provenance_epoch_id": self.provenance_epoch_id,
                "sha256": self.catalog_sha256,
            },
            "records": {name: self.records[name].to_dict() for name in sorted(self.records)},
        }


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
    _require_contained(config, root, label="Reader config")
    if not config.is_file():
        raise ReaderRecordError(f"Reader config is missing: {config}")
    if not expected_records:
        raise ReaderRecordError("at least one Reader record expectation is required")
    if len({item.record_id for item in expected_records.values()}) != len(expected_records):
        raise ReaderRecordError("Reader record expectations must use unique record IDs")

    command = tuple(reader_command or _reader_command(root))
    data, rows = _collect_record_pages(command, config_path=config, cwd=root)
    context = _resolve_context(
        data,
        reader_root=root,
        config_path=config,
        experiment_id=experiment_id,
        protocol_id=protocol_id,
    )
    catalog_path = context["catalog_path"]
    catalog_sha256 = _sha256_file(catalog_path)
    initial_identity = _record_set_identity(
        data,
        rows,
        expected_records=expected_records,
        catalog_sha256=catalog_sha256,
    )
    _verify_record_store(
        command,
        config_path=config,
        cwd=root,
        expected_records=expected_records,
    )
    confirmed_data, confirmed_rows = _collect_record_pages(command, config_path=config, cwd=root)
    confirmed_context = _resolve_context(
        confirmed_data,
        reader_root=root,
        config_path=config,
        experiment_id=experiment_id,
        protocol_id=protocol_id,
    )
    confirmed_catalog_path = confirmed_context["catalog_path"]
    confirmed_catalog_sha256 = _sha256_file(confirmed_catalog_path)
    confirmed_identity = _record_set_identity(
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
        name: _resolve_record(
            _one_record(confirmed_rows, record_id=expectation.record_id, experiment_id=experiment_id),
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
    replicate_kind = _text(evidence.get("replicate_kind"), label="records.data.experiment.evidence.replicate_kind")
    if replicate_kind not in {"biological", "technical", "mixed", "unknown"}:
        raise ReaderDataframeRecordError(
            f"{experiment_id}: replicate_kind must be biological, technical, mixed, or unknown for assay evidence"
        )
    replicate_field = _optional_text(
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
        contract_id=contract_id,
        reader_path=record.reader_path,
        path=record.path,
        manifest_path=resolved.catalog_path,
        content_digest=record.content_digest,
    )


def _resolve_context(
    data: Mapping[str, object],
    *,
    reader_root: Path,
    config_path: Path,
    experiment_id: str,
    protocol_id: str,
) -> dict[str, object]:
    experiment = _mapping(data.get("experiment"), label="records.data.experiment")
    observed_experiment_id = _text(experiment.get("id"), label="records.data.experiment.id")
    if observed_experiment_id != experiment_id:
        raise ReaderRecordError(
            f"Reader experiment id mismatch: expected {experiment_id!r}, observed {observed_experiment_id!r}"
        )
    observed_protocol = _text(experiment.get("protocol"), label="records.data.experiment.protocol")
    if observed_protocol != protocol_id:
        raise ReaderRecordError(f"{experiment_id}: protocol must equal {protocol_id!r}; observed {observed_protocol!r}")
    observed_config = Path(_text(experiment.get("config"), label="records.data.experiment.config")).resolve()
    _require_contained(observed_config, reader_root, label="Reader CLI config")
    if observed_config != config_path:
        raise ReaderRecordError(
            f"{experiment_id}: Reader CLI config identity changed; expected {config_path}, observed {observed_config}"
        )
    experiment_root = Path(_text(experiment.get("root"), label="records.data.experiment.root")).resolve()
    _require_contained(experiment_root, reader_root, label="Reader experiment root")
    if experiment_root != config_path.parent:
        raise ReaderRecordError(
            f"{experiment_id}: Reader experiment root must equal the config parent; "
            f"expected {config_path.parent}, observed {experiment_root}"
        )
    evidence_value = experiment.get("evidence")
    evidence = (
        {} if evidence_value is None else dict(_mapping(evidence_value, label="records.data.experiment.evidence"))
    )

    catalog = _mapping(data.get("catalog"), label="records.data.catalog")
    if catalog.get("schema_version") != READER_CATALOG_SCHEMA_VERSION:
        raise ReaderRecordError(
            f"{experiment_id}: requires Reader catalog schema v{READER_CATALOG_SCHEMA_VERSION}; "
            "regenerate and verify the Reader experiment"
        )
    outputs_root = Path(_text(catalog.get("outputs_root"), label="records.data.catalog.outputs_root")).resolve()
    _require_contained(outputs_root, reader_root, label="Reader outputs root")
    expected_outputs_root = experiment_root / "outputs"
    if outputs_root != expected_outputs_root:
        raise ReaderRecordError(
            f"{experiment_id}: Reader outputs root must equal the experiment outputs directory; "
            f"expected {expected_outputs_root}, observed {outputs_root}"
        )
    catalog_path = Path(_text(catalog.get("path"), label="records.data.catalog.path")).resolve()
    _require_contained(catalog_path, reader_root, label="Reader record manifest")
    expected_catalog_path = outputs_root / "manifests" / "records.json"
    if catalog_path != expected_catalog_path:
        raise ReaderRecordError(
            f"{experiment_id}: Reader record manifest must equal the canonical records manifest; "
            f"expected {expected_catalog_path}, observed {catalog_path}"
        )
    if not catalog_path.is_file():
        raise ReaderRecordError(f"{experiment_id}: Reader record manifest is missing: {catalog_path}")
    provenance_epoch_id = _text(
        catalog.get("provenance_epoch_id"),
        label="records.data.catalog.provenance_epoch_id",
    )
    return {
        "experiment_root": experiment_root,
        "outputs_root": outputs_root,
        "catalog_path": catalog_path,
        "provenance_epoch_id": provenance_epoch_id,
        "experiment_evidence": evidence,
    }


def _resolve_record(
    record: Mapping[str, object],
    *,
    expectation: ReaderRecordExpectation,
    outputs_root: Path,
) -> ReaderResolvedRecord:
    record_id = expectation.record_id
    if record.get("schema_version") != READER_RECORD_SCHEMA_VERSION:
        raise ReaderRecordError(
            f"{record_id}: requires Reader record schema v{READER_RECORD_SCHEMA_VERSION}; "
            "regenerate and verify the Reader experiment"
        )
    kind = _text(record.get("kind"), label=f"{record_id}.kind")
    if kind != expectation.kind:
        raise ReaderRecordError(f"{record_id}: kind must equal {expectation.kind!r}")
    revision = _positive_revision(record.get("revision"), label=f"{record_id}.revision")
    revision_digest = _sha256_digest(record.get("revision_digest"), label=f"{record_id}.revision_digest")
    producer_value = record.get("producer")
    producer = {} if producer_value is None else dict(_mapping(producer_value, label=f"{record_id}.producer"))
    inputs_value = record.get("inputs", [])
    inputs = tuple(
        dict(_mapping(value, label=f"{record_id}.inputs[{index}]"))
        for index, value in enumerate(_list(inputs_value, label=f"{record_id}.inputs"))
    )
    if kind == "dataframe_artifact":
        if record.get("contract_id") != expectation.contract_id:
            raise ReaderRecordError(f"{record_id}: contract must equal {expectation.contract_id!r}")
        reader_path = _outputs_relative_path(record.get("path"), label=f"{record_id}.path")
        path = (outputs_root / reader_path).resolve()
        _require_contained(path, outputs_root, label=f"Reader record {record_id!r}")
        size_bytes = _nonnegative_integer(record.get("size_bytes"), label=f"{record_id}.size_bytes")
        content_digest = _sha256_digest(record.get("content_digest"), label=f"{record_id}.content_digest")
        content = _verified_bytes(
            path,
            expected_size=size_bytes,
            expected_digest=content_digest,
            label=f"Reader record {record_id!r}",
        )
        return ReaderResolvedRecord._verified(
            record_id=record_id,
            kind=kind,
            schema_version=READER_RECORD_SCHEMA_VERSION,
            revision=revision,
            revision_digest=revision_digest,
            contract_id=expectation.contract_id,
            producer=producer,
            producer_config_digest=None,
            inputs=inputs,
            path=path,
            reader_path=reader_path.as_posix(),
            size_bytes=size_bytes,
            content_digest=content_digest,
            content=content,
            files=(),
        )

    producer_config_digest = _sha256_digest(
        record.get("producer_config_digest"),
        label=f"{record_id}.producer_config_digest",
    )
    files = _resolve_file_evidence(record, outputs_root=outputs_root, record_id=record_id)
    return ReaderResolvedRecord._verified(
        record_id=record_id,
        kind=kind,
        schema_version=READER_RECORD_SCHEMA_VERSION,
        revision=revision,
        revision_digest=revision_digest,
        contract_id=None,
        producer=producer,
        producer_config_digest=producer_config_digest,
        inputs=inputs,
        path=None,
        reader_path=None,
        size_bytes=None,
        content_digest=None,
        content=None,
        files=files,
    )


def _resolve_file_evidence(
    record: Mapping[str, object],
    *,
    outputs_root: Path,
    record_id: str,
) -> tuple[ReaderArtifactFile, ...]:
    raw_files = _list(record.get("files"), label=f"{record_id}.files")
    raw_evidence = _list(record.get("file_evidence"), label=f"{record_id}.file_evidence")
    if not raw_files:
        raise ReaderRecordError(f"{record_id}: file bundle cannot be empty")
    file_paths = [
        _outputs_relative_path(value, label=f"{record_id}.files[{index}]") for index, value in enumerate(raw_files)
    ]
    if len(set(file_paths)) != len(file_paths):
        raise ReaderRecordError(f"{record_id}: file bundle paths must be unique")
    evidence_by_path: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(raw_evidence):
        item = _mapping(value, label=f"{record_id}.file_evidence[{index}]")
        if set(item) != {"path", "size_bytes", "content_digest"}:
            raise ReaderRecordError(f"{record_id}: file evidence fields are malformed")
        reader_path = _outputs_relative_path(
            item.get("path"),
            label=f"{record_id}.file_evidence[{index}].path",
        ).as_posix()
        if reader_path in evidence_by_path:
            raise ReaderRecordError(f"{record_id}: repeated file evidence path {reader_path!r}")
        evidence_by_path[reader_path] = item
    if {path.as_posix() for path in file_paths} != set(evidence_by_path):
        raise ReaderRecordError(f"{record_id}: files and file evidence disagree")
    result: list[ReaderArtifactFile] = []
    for relative in file_paths:
        reader_path = relative.as_posix()
        evidence = evidence_by_path[reader_path]
        size_bytes = _nonnegative_integer(evidence.get("size_bytes"), label=f"{reader_path}.size_bytes")
        content_digest = _sha256_digest(
            evidence.get("content_digest"),
            label=f"{reader_path}.content_digest",
        )
        path = (outputs_root / relative).resolve()
        _require_contained(path, outputs_root, label=f"Reader file {reader_path!r}")
        content = _verified_bytes(
            path,
            expected_size=size_bytes,
            expected_digest=content_digest,
            label=f"Reader file {reader_path!r}",
        )
        result.append(
            ReaderArtifactFile(
                reader_path=reader_path,
                path=path,
                size_bytes=size_bytes,
                content_digest=content_digest,
                content=content,
            )
        )
    return tuple(result)


def _record_set_identity(
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
            name: _one_record(rows, record_id=expectation.record_id, experiment_id="Reader").copy()
            for name, expectation in sorted(expected_records.items())
        },
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _one_record(
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


def _outputs_relative_path(value: object, *, label: str) -> Path:
    token = Path(_text(value, label=label))
    if token.is_absolute() or ".." in token.parts or token == Path("."):
        raise ReaderRecordError(f"{label} must be outputs-relative")
    return token


def _verified_bytes(
    path: Path,
    *,
    expected_size: int,
    expected_digest: str,
    label: str,
) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ReaderRecordError(f"could not read {label}: {exc}") from exc
    observed_digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if len(content) != expected_size or observed_digest != expected_digest:
        raise ReaderRecordError(f"{label} byte size or content digest mismatch")
    return content


def _sha256_file(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ReaderRecordError(f"could not read Reader catalog {path}: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReaderRecordError(f"{label} must be a nonnegative integer")
    return value


def _collect_record_pages(
    command: Sequence[str], *, config_path: Path, cwd: Path
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    continuation: str | None = None
    first_data: Mapping[str, object] | None = None
    records: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    seen_continuations: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        argv = [*command, "records", str(config_path), "--limit", str(_PAGE_LIMIT), "--format", "json"]
        if continuation is not None:
            argv.extend(("--continuation", continuation))
        payload = _run_reader_json(argv, cwd=cwd)
        if (
            payload.get("schema") != READER_CLI_SCHEMA
            or payload.get("command") != "records"
            or payload.get("ok") is not True
        ):
            raise ReaderDataframeRecordError(
                "Reader records command did not return a successful reader.cli/v1 records payload"
            )
        data = _mapping(payload.get("data"), label="records.data")
        page_records = _list(data.get("records"), label="records.data.records")
        if first_data is None:
            first_data = data
        else:
            for field in ("experiment", "catalog", "selection", "summary"):
                if data.get(field) != first_data.get(field):
                    raise ReaderDataframeRecordError(f"Reader records pagination changed data.{field}")
        for index, value in enumerate(page_records):
            record = _mapping(value, label=f"records.data.records[{index}]")
            current_id = _text(record.get("record_id"), label=f"records.data.records[{index}].record_id")
            if current_id in seen_ids:
                raise ReaderDataframeRecordError(f"Reader records pagination repeated record_id {current_id!r}")
            seen_ids.add(current_id)
            records.append(record)
        meta = _mapping(payload.get("meta"), label="records.meta")
        truncated = meta.get("truncated")
        if not isinstance(truncated, bool):
            raise ReaderDataframeRecordError("records.meta.truncated must be a boolean")
        next_token = meta.get("continuation")
        if not truncated:
            if next_token is not None:
                raise ReaderDataframeRecordError("records.meta.continuation must be null on the final page")
            break
        if not page_records:
            raise ReaderDataframeRecordError("Reader records truncated page must contain at least one record")
        next_continuation = _text(next_token, label="records.meta.continuation")
        if next_continuation in seen_continuations:
            raise ReaderDataframeRecordError(
                f"Reader records pagination repeated continuation token {next_continuation!r}"
            )
        if page_count >= _MAX_RECORD_PAGES:
            raise ReaderDataframeRecordError(
                f"Reader records pagination exceeded the {_MAX_RECORD_PAGES}-page safety bound"
            )
        seen_continuations.add(next_continuation)
        continuation = next_continuation
    assert first_data is not None
    summary = _mapping(first_data.get("summary"), label="records.data.summary")
    if summary.get("records") != len(records):
        raise ReaderDataframeRecordError(
            f"Reader records summary count {summary.get('records')!r} does not match collected count {len(records)}"
        )
    return first_data, tuple(records)


def _verify_record_store(
    command: Sequence[str],
    *,
    config_path: Path,
    cwd: Path,
    expected_records: Mapping[str, ReaderRecordExpectation],
) -> None:
    """Require Reader's full provenance verifier inside the stable catalog read."""

    payload = _run_reader_json(
        [*command, "verify", str(config_path), "--format", "json"],
        cwd=cwd,
    )
    if (
        payload.get("schema") != READER_CLI_SCHEMA
        or payload.get("command") != "verify"
        or payload.get("ok") is not True
    ):
        raise ReaderRecordError("Reader verify did not return a successful reader.cli/v1 verify payload")
    report = _mapping(payload.get("data"), label="verify.data")
    if set(report) != {"schema", "status", "summary", "issues", "records"}:
        raise ReaderRecordError("Reader verify report fields are malformed")
    if report.get("schema") != "reader.verify/v1" or report.get("status") != "ok":
        raise ReaderRecordError("Reader verify status must be ok before records can be consumed")
    report_issues = _list(report.get("issues"), label="verify.data.issues")
    if report_issues:
        raise ReaderRecordError("Reader verify reported issues despite status ok")
    summary = _mapping(report.get("summary"), label="verify.data.summary")
    rows = _list(report.get("records"), label="verify.data.records")
    summary_fields = {
        "checked",
        "failed",
        "unverifiable",
        "invocations_checked",
        "invocation_failures",
    }
    if set(summary) != summary_fields or any(type(summary.get(field)) is not int for field in summary_fields):
        raise ReaderRecordError("Reader verify summary is malformed")
    if (
        summary.get("checked") != len(rows)
        or summary.get("failed") != 0
        or summary.get("unverifiable") != 0
        or summary.get("invocation_failures") != 0
        or summary["invocations_checked"] < 1
    ):
        raise ReaderRecordError("Reader verify summary is not a complete successful provenance check")
    verified_by_id: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(rows):
        row = _mapping(value, label=f"verify.data.records[{index}]")
        if set(row) != {"record_id", "kind", "schema_version", "status", "issues"}:
            raise ReaderRecordError(f"Reader verify record row {index} fields are malformed")
        record_id = _text(row.get("record_id"), label=f"verify.data.records[{index}].record_id")
        row_issues = _list(row.get("issues"), label=f"verify.data.records[{index}].issues")
        if row.get("status") != "ok" or row_issues:
            raise ReaderRecordError(f"Reader verify record {record_id!r} is not cleanly verified")
        if record_id in verified_by_id:
            raise ReaderRecordError(f"Reader verify repeated record_id {record_id!r}")
        verified_by_id[record_id] = row
    for expectation in expected_records.values():
        row = verified_by_id.get(expectation.record_id)
        if (
            row is None
            or row.get("kind") != expectation.kind
            or row.get("schema_version") != READER_RECORD_SCHEMA_VERSION
        ):
            raise ReaderRecordError(
                f"Reader verify did not confirm expected record {expectation.record_id!r} as schema-v6 status ok"
            )


def _reader_command(reader_root: Path) -> tuple[str, ...]:
    repository_executable = reader_root / ".venv" / "bin" / "reader"
    if repository_executable.is_file():
        return (str(repository_executable),)
    installed = shutil.which("reader")
    if installed:
        return (installed,)
    uv = shutil.which("uv")
    if uv:
        return (uv, "run", "--project", str(reader_root), "reader")
    raise ReaderDataframeRecordError("Reader public CLI is unavailable")


def _run_reader_json(command: Sequence[str], *, cwd: Path) -> Mapping[str, object]:
    environment = os.environ.copy()
    environment.pop("__PYVENV_LAUNCHER__", None)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=_READER_CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderDataframeRecordError(
            f"Reader CLI command timed out after {_READER_CLI_TIMEOUT_SECONDS} seconds"
        ) from exc
    raw = completed.stdout.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReaderDataframeRecordError(
            f"Reader records command returned invalid JSON: {raw or completed.stderr.strip() or '<empty>'}"
        ) from exc
    result = _mapping(payload, label="Reader CLI envelope")
    if completed.returncode != 0:
        error = result.get("error")
        raise ReaderDataframeRecordError(f"Reader records command failed: {json.dumps(error, sort_keys=True)}")
    return result


def _require_contained(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReaderDataframeRecordError(f"{label} escapes {root}: {path}") from exc


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderDataframeRecordError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ReaderDataframeRecordError(f"{label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderDataframeRecordError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _sha256_digest(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderDataframeRecordError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderDataframeRecordError(f"{label} must be a lowercase sha256 digest")
    return token


def _positive_revision(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReaderDataframeRecordError(f"{label} must be a positive integer")
    return value
