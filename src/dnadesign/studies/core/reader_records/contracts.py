"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/contracts.py

Public value contracts for verified Reader records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .validation import ReaderRecordError, require_contained

READER_CLI_SCHEMA = "reader.cli/v1"
READER_CATALOG_SCHEMA_VERSION = 4
READER_RECORD_SCHEMA_VERSION = 6
_SOURCE_CLOSURE_TOKEN = object()


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
        require_contained(artifact, root, label="Reader artifact")
        require_contained(manifest, root, label="Reader record manifest")
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
