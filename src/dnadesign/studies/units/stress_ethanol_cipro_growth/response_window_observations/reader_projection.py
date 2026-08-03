"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_projection.py

Load one immutable study projection over canonical Reader records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from .contract_yaml import load_unique_yaml
from .display_contract import response_example_labels
from .reader_projection_contract import (
    READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
    READER_EVENT_WINDOW_PROTOCOL_ID,
    READER_EVENT_WINDOW_RECORD_CONTRACTS,
    STATE_ORDER,
    STUDY_PROJECTION_SCHEMA,
    ReaderResponseProjectionError,
    validate_display_artifact_spec,
    validate_projection_payload,
)


@dataclass(frozen=True, slots=True)
class ReaderResponseProjection:
    """One immutable, digest-addressed study interpretation of Reader records."""

    path: Path
    sha256: str
    payload: Mapping[str, object]

    @property
    def reader_experiment_id(self) -> str:
        return str(self.payload["reader_experiment_id"])

    @property
    def source_experiment_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["source_experiment_ids"])

    @property
    def primary_reduction_id(self) -> str:
        return str(self.payload["primary_reduction_id"])

    @property
    def reference_design_id(self) -> str:
        return str(self.payload["reference_design_id"])

    @property
    def source(self) -> Mapping[str, object]:
        return _mapping(self.payload["source"], label="projection.source")

    @property
    def event(self) -> Mapping[str, object]:
        return _mapping(self.payload["event"], label="projection.event")

    @property
    def aggregation(self) -> Mapping[str, object]:
        return _mapping(self.payload["aggregation"], label="projection.aggregation")

    @property
    def reductions(self) -> tuple[Mapping[str, object], ...]:
        value = self.payload["reductions"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ReaderResponseProjectionError("projection.reductions must be a sequence")
        return tuple(_mapping(item, label="projection.reductions[]") for item in value)

    @property
    def display(self) -> Mapping[str, object]:
        return _mapping(self.payload["display"], label="projection.display")

    @property
    def response_examples(self) -> dict[str, str]:
        return response_example_labels(self.display)

    def display_artifact_spec(self) -> dict[str, str]:
        value = self.payload.get("display_artifact")
        if value is None:
            raise ReaderResponseProjectionError(
                "study Reader projection has no display_artifact pin; run and verify the canonical Reader "
                "diagnostic, then pin its source experiment, design, producer-config digest, and path"
            )
        return validate_display_artifact_spec(value)


def load_reader_response_projection(path: Path) -> ReaderResponseProjection:
    """Load and validate one projection snapshot from the exact bytes hashed below."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReaderResponseProjectionError(f"study Reader projection is missing: {source}")
    try:
        source_bytes = source.read_bytes()
        payload = load_unique_yaml(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReaderResponseProjectionError(f"could not read study Reader projection {source}: {exc}") from exc
    validate_projection_payload(payload)
    frozen = cast(Mapping[str, object], _freeze(payload))
    return ReaderResponseProjection(
        path=source,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        payload=frozen,
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderResponseProjectionError(f"{label} must be an object")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = [
    "READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID",
    "READER_EVENT_WINDOW_PROTOCOL_ID",
    "READER_EVENT_WINDOW_RECORD_CONTRACTS",
    "STATE_ORDER",
    "STUDY_PROJECTION_SCHEMA",
    "ReaderResponseProjection",
    "ReaderResponseProjectionError",
    "load_reader_response_projection",
]
