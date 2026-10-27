"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/api/infer_handoff.py

Non-executing Infer feature request manifests for Permuter handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_KIND = "permuter_infer_feature_request_v1"
_SOURCE_OWNER = "permuter"
_ALLOWED_SOURCE_OWNERS = frozenset({"construct", "permuter", "study"})
_EXECUTION_OWNER = "infer"
_WRITEBACK_OWNER = "infer"


@dataclass(frozen=True)
class InferFeatureSourceDataset:
    usr_root: str
    dataset_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "usr_root", _required_text(self.usr_root, label="source_dataset.usr_root"))
        dataset_id = _required_text(self.dataset_id, label="source_dataset.dataset_id")
        if "/" in dataset_id or "\\" in dataset_id:
            raise ValueError("source_dataset.dataset_id must be a flat USR dataset id, not a path")
        object.__setattr__(self, "dataset_id", dataset_id)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "InferFeatureSourceDataset":
        _reject_extra_keys(payload, allowed={"usr_root", "dataset_id"}, label="source_dataset")
        return cls(
            usr_root=_required_text(payload.get("usr_root"), label="source_dataset.usr_root"),
            dataset_id=_required_text(payload.get("dataset_id"), label="source_dataset.dataset_id"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "usr_root": self.usr_root,
            "dataset_id": self.dataset_id,
        }


@dataclass(frozen=True)
class InferSequenceViewSelector:
    view_name: str | None = None
    alias: str | None = None

    def __post_init__(self) -> None:
        view_name = _optional_text(self.view_name, label="sequence_view_selectors[].view_name")
        alias = _optional_text(self.alias, label="sequence_view_selectors[].alias")
        if (view_name is None) == (alias is None):
            raise ValueError("sequence_view_selectors[] must set exactly one of view_name or alias")
        object.__setattr__(self, "view_name", view_name)
        object.__setattr__(self, "alias", alias)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "InferSequenceViewSelector":
        forbidden = sorted(set(payload) & {"product_kind", "orientation"})
        if forbidden:
            raise ValueError(
                "sequence_view_selectors[] must select by explicit view_name or alias; "
                f"broad selector field(s) are not supported: {forbidden}"
            )
        _reject_extra_keys(payload, allowed={"view_name", "alias"}, label="sequence_view_selectors[]")
        return cls(
            view_name=_optional_text(payload.get("view_name"), label="sequence_view_selectors[].view_name"),
            alias=_optional_text(payload.get("alias"), label="sequence_view_selectors[].alias"),
        )

    def to_mapping(self) -> dict[str, str]:
        if self.view_name is not None:
            return {"view_name": self.view_name}
        if self.alias is not None:
            return {"alias": self.alias}
        raise ValueError("InferSequenceViewSelector is invalid: missing view_name/alias")


@dataclass(frozen=True)
class InferFeatureRequest:
    source_dataset: InferFeatureSourceDataset
    feature_bundle_ref: str
    sequence_view_selectors: tuple[InferSequenceViewSelector, ...]
    requested_outputs: tuple[str, ...]
    kind: Literal["permuter_infer_feature_request_v1"] = _KIND
    source_owner: str = _SOURCE_OWNER
    execution_owner: Literal["infer"] = _EXECUTION_OWNER
    writeback_owner: Literal["infer"] = _WRITEBACK_OWNER

    def __post_init__(self) -> None:
        if self.kind != _KIND:
            raise ValueError(f"kind must be {_KIND!r}")
        source_owner = _required_text(self.source_owner, label="source_owner")
        if source_owner not in _ALLOWED_SOURCE_OWNERS:
            allowed = ", ".join(sorted(_ALLOWED_SOURCE_OWNERS))
            raise ValueError(f"source_owner must be one of: {allowed}")
        object.__setattr__(self, "source_owner", source_owner)
        if self.execution_owner != _EXECUTION_OWNER:
            raise ValueError(f"execution_owner must be {_EXECUTION_OWNER!r}")
        if self.writeback_owner != _WRITEBACK_OWNER:
            raise ValueError(f"writeback_owner must be {_WRITEBACK_OWNER!r}")
        if not isinstance(self.source_dataset, InferFeatureSourceDataset):
            raise TypeError("source_dataset must be InferFeatureSourceDataset")
        object.__setattr__(
            self,
            "feature_bundle_ref",
            _required_text(self.feature_bundle_ref, label="feature_bundle_ref"),
        )
        selectors = tuple(self.sequence_view_selectors)
        if not selectors:
            raise ValueError("sequence_view_selectors must contain at least one explicit selector")
        if not all(isinstance(selector, InferSequenceViewSelector) for selector in selectors):
            raise TypeError("sequence_view_selectors must contain InferSequenceViewSelector objects")
        object.__setattr__(self, "sequence_view_selectors", selectors)
        requested_outputs = tuple(
            _required_text(value, label="requested_outputs[]") for value in self.requested_outputs
        )
        if not requested_outputs:
            raise ValueError("requested_outputs must contain at least one output id")
        if len(set(requested_outputs)) != len(requested_outputs):
            raise ValueError(f"requested_outputs must not contain duplicates: {requested_outputs}")
        object.__setattr__(self, "requested_outputs", requested_outputs)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "InferFeatureRequest":
        _reject_extra_keys(
            payload,
            allowed={
                "kind",
                "source_owner",
                "execution_owner",
                "writeback_owner",
                "source_dataset",
                "feature_bundle_ref",
                "sequence_view_selectors",
                "requested_outputs",
            },
            label="infer feature request",
        )
        kind = _required_text(payload.get("kind"), label="kind")
        if kind != _KIND:
            raise ValueError(f"kind must be {_KIND!r}")
        source_owner = _source_owner(payload)
        execution_owner = _owner(payload, "execution_owner", expected=_EXECUTION_OWNER)
        writeback_owner = _owner(payload, "writeback_owner", expected=_WRITEBACK_OWNER)
        source_dataset_payload = payload.get("source_dataset")
        if not isinstance(source_dataset_payload, Mapping):
            raise ValueError("source_dataset must be a mapping")
        selectors_payload = _mapping_sequence(payload.get("sequence_view_selectors"), label="sequence_view_selectors")
        outputs_payload = _string_sequence(payload.get("requested_outputs"), label="requested_outputs")
        return cls(
            kind=kind,
            source_owner=source_owner,
            execution_owner=execution_owner,
            writeback_owner=writeback_owner,
            source_dataset=InferFeatureSourceDataset.from_mapping(source_dataset_payload),
            feature_bundle_ref=_required_text(payload.get("feature_bundle_ref"), label="feature_bundle_ref"),
            sequence_view_selectors=tuple(
                InferSequenceViewSelector.from_mapping(selector) for selector in selectors_payload
            ),
            requested_outputs=tuple(outputs_payload),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_owner": self.source_owner,
            "execution_owner": self.execution_owner,
            "writeback_owner": self.writeback_owner,
            "source_dataset": self.source_dataset.to_mapping(),
            "feature_bundle_ref": self.feature_bundle_ref,
            "sequence_view_selectors": [selector.to_mapping() for selector in self.sequence_view_selectors],
            "requested_outputs": list(self.requested_outputs),
        }


def infer_feature_request_from_mapping(payload: Mapping[str, object]) -> InferFeatureRequest:
    """Parse a non-executing Permuter-to-Infer handoff manifest."""

    return InferFeatureRequest.from_mapping(payload)


def read_infer_feature_request_manifest(path: str | Path) -> InferFeatureRequest:
    manifest_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Infer feature request manifest must be a mapping: {manifest_path}")
    return InferFeatureRequest.from_mapping(payload)


def write_infer_feature_request_manifest(
    request: InferFeatureRequest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a non-executing handoff manifest without creating Infer sidecars."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.parent.exists():
        raise FileNotFoundError(f"Manifest parent directory not found: {manifest_path.parent}")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    manifest_path.write_text(yaml.safe_dump(request.to_mapping(), sort_keys=False), encoding="utf-8")
    return manifest_path


def _source_owner(payload: Mapping[str, object]) -> str:
    value = _required_text(payload.get("source_owner"), label="source_owner")
    if value not in _ALLOWED_SOURCE_OWNERS:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_OWNERS))
        raise ValueError(f"source_owner must be one of: {allowed}")
    return value


def _owner(payload: Mapping[str, object], key: str, *, expected: str) -> str:
    value = _required_text(payload.get(key), label=key)
    if value != expected:
        raise ValueError(f"{key} must be {expected!r}")
    return value


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty when provided")
    return text


def _reject_extra_keys(payload: Mapping[str, object], *, allowed: set[str], label: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"{label} has unsupported field(s): {extra}")


def _mapping_sequence(value: object, *, label: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of mappings")
    out: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        out.append(item)
    return tuple(out)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    return tuple(_required_text(item, label=f"{label}[]") for item in value)
