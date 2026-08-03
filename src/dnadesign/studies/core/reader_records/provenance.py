"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/provenance.py

Typed validation for public Reader record provenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .validation import (
    ReaderRecordError,
    list_value,
    mapping,
    nonnegative_integer,
    sha256_digest,
    text,
)

_PRODUCER_FIELDS = frozenset({"kind", "id", "plugin", "source_recipe"})
_PRODUCER_REQUIRED_FIELDS = frozenset({"kind", "id", "plugin"})
_RECIPE_FIELDS = frozenset({"recipe", "with"})
_LOCAL_INPUT_FIELDS = frozenset({"label", "kind", "record", "discovery_policy", "record_revision_digest"})
_SOURCE_INPUT_FIELDS = frozenset(
    {"label", "kind", "resource", "experiment", "record", "discovery_policy", "record_revision_digest"}
)
_FILE_INPUT_FIELDS = frozenset({"label", "kind", "discovery_policy", "artifact"})
_RESOURCE_INPUT_FIELDS = frozenset({"label", "kind", "resource", "discovery_policy", "artifact"})
_ARTIFACT_FIELDS = frozenset({"path", "size_bytes", "content_digest"})
_FILE_DISCOVERY_POLICIES = frozenset({"declared_file", "declared_resource", "plugin_discovery"})


@dataclass(frozen=True, slots=True)
class ReaderRecordRecipeSource:
    """Public recipe identity attached to a Reader record producer."""

    recipe: str
    with_: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"recipe": self.recipe, "with": deepcopy(dict(self.with_))}


@dataclass(frozen=True, slots=True)
class ReaderRecordProducer(Mapping[str, object]):
    """Validated public identity of the plugin that produced a record."""

    kind: str
    id: str
    plugin: str
    source_recipe: ReaderRecordRecipeSource | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind, "id": self.id, "plugin": self.plugin}
        if self.source_recipe is not None:
            result["source_recipe"] = self.source_recipe.to_dict()
        return result

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReaderInputArtifactEvidence:
    """Digest and size evidence for one Reader file or resource input."""

    reader_path: str
    size_bytes: int
    content_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.reader_path,
            "size_bytes": self.size_bytes,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class ReaderRecordInputEvidence(Mapping[str, object]):
    """One validated input edge from a public Reader record revision."""

    label: str
    kind: str
    discovery_policy: str
    record_id: str | None = None
    resource_id: str | None = None
    experiment_id: str | None = None
    record_revision_digest: str | None = None
    artifact: ReaderInputArtifactEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "label": self.label,
            "kind": self.kind,
            "discovery_policy": self.discovery_policy,
        }
        if self.resource_id is not None:
            result["resource"] = self.resource_id
        if self.experiment_id is not None:
            result["experiment"] = self.experiment_id
        if self.record_id is not None:
            result["record"] = self.record_id
        if self.record_revision_digest is not None:
            result["record_revision_digest"] = self.record_revision_digest
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        return result

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def parse_record_producer(value: object, *, record_id: str) -> ReaderRecordProducer:
    """Validate the exact schema-v6 producer shape published by Reader."""

    producer = mapping(value, label=f"{record_id}.producer")
    _require_fields(
        producer,
        allowed=_PRODUCER_FIELDS,
        required=_PRODUCER_REQUIRED_FIELDS,
        label=f"{record_id}.producer",
    )
    kind = text(producer.get("kind"), label=f"{record_id}.producer.kind")
    if kind not in {"pipeline", "plot", "export"}:
        raise ReaderRecordError(f"{record_id}.producer.kind is unsupported: {kind!r}")
    source_recipe_value = producer.get("source_recipe")
    source_recipe = None
    if source_recipe_value is not None:
        recipe = mapping(source_recipe_value, label=f"{record_id}.producer.source_recipe")
        _require_fields(
            recipe,
            allowed=_RECIPE_FIELDS,
            required=_RECIPE_FIELDS,
            label=f"{record_id}.producer.source_recipe",
        )
        with_block = mapping(recipe.get("with"), label=f"{record_id}.producer.source_recipe.with")
        source_recipe = ReaderRecordRecipeSource(
            recipe=text(recipe.get("recipe"), label=f"{record_id}.producer.source_recipe.recipe"),
            with_=MappingProxyType(deepcopy(dict(with_block))),
        )
    return ReaderRecordProducer(
        kind=kind,
        id=text(producer.get("id"), label=f"{record_id}.producer.id"),
        plugin=text(producer.get("plugin"), label=f"{record_id}.producer.plugin"),
        source_recipe=source_recipe,
    )


def parse_record_inputs(value: object, *, record_id: str) -> tuple[ReaderRecordInputEvidence, ...]:
    """Validate every public schema-v6 input edge without interpreting study meaning."""

    return tuple(
        _parse_record_input(item, record_id=record_id, index=index)
        for index, item in enumerate(list_value(value, label=f"{record_id}.inputs"))
    )


def _parse_record_input(value: object, *, record_id: str, index: int) -> ReaderRecordInputEvidence:
    label = f"{record_id}.inputs[{index}]"
    item = mapping(value, label=label)
    kind = text(item.get("kind"), label=f"{label}.kind")
    input_label = text(item.get("label"), label=f"{label}.label")
    policy = text(item.get("discovery_policy"), label=f"{label}.discovery_policy")
    if kind == "record":
        _require_fields(item, allowed=_LOCAL_INPUT_FIELDS, required=_LOCAL_INPUT_FIELDS, label=label)
        if policy != "record":
            raise ReaderRecordError(f"{label}.discovery_policy must equal 'record'")
        return ReaderRecordInputEvidence(
            label=input_label,
            kind=kind,
            discovery_policy=policy,
            record_id=text(item.get("record"), label=f"{label}.record"),
            record_revision_digest=sha256_digest(
                item.get("record_revision_digest"),
                label=f"{label}.record_revision_digest",
            ),
        )
    if kind == "source_record":
        _require_fields(item, allowed=_SOURCE_INPUT_FIELDS, required=_SOURCE_INPUT_FIELDS, label=label)
        if policy != "source_record":
            raise ReaderRecordError(f"{label}.discovery_policy must equal 'source_record'")
        return ReaderRecordInputEvidence(
            label=input_label,
            kind=kind,
            discovery_policy=policy,
            resource_id=text(item.get("resource"), label=f"{label}.resource"),
            experiment_id=text(item.get("experiment"), label=f"{label}.experiment"),
            record_id=text(item.get("record"), label=f"{label}.record"),
            record_revision_digest=sha256_digest(
                item.get("record_revision_digest"),
                label=f"{label}.record_revision_digest",
            ),
        )
    if kind not in {"file", "resource"}:
        raise ReaderRecordError(f"{label}.kind is unsupported: {kind!r}")
    fields = _RESOURCE_INPUT_FIELDS if kind == "resource" else _FILE_INPUT_FIELDS
    _require_fields(item, allowed=fields, required=fields, label=label)
    if policy not in _FILE_DISCOVERY_POLICIES:
        raise ReaderRecordError(f"{label}.discovery_policy is invalid for a {kind} input")
    return ReaderRecordInputEvidence(
        label=input_label,
        kind=kind,
        discovery_policy=policy,
        resource_id=text(item.get("resource"), label=f"{label}.resource") if kind == "resource" else None,
        artifact=_parse_input_artifact(item.get("artifact"), label=f"{label}.artifact"),
    )


def _parse_input_artifact(value: object, *, label: str) -> ReaderInputArtifactEvidence:
    artifact = mapping(value, label=label)
    _require_fields(artifact, allowed=_ARTIFACT_FIELDS, required=_ARTIFACT_FIELDS, label=label)
    reader_path = text(artifact.get("path"), label=f"{label}.path")
    path = Path(reader_path)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ReaderRecordError(f"{label}.path must be relative and confined")
    return ReaderInputArtifactEvidence(
        reader_path=path.as_posix(),
        size_bytes=nonnegative_integer(artifact.get("size_bytes"), label=f"{label}.size_bytes"),
        content_digest=sha256_digest(artifact.get("content_digest"), label=f"{label}.content_digest"),
    )


def _require_fields(
    value: Mapping[str, object],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        if missing:
            details.append("missing=" + ", ".join(missing))
        raise ReaderRecordError(f"{label} fields are malformed: {'; '.join(details)}")
