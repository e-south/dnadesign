"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_config_attestation.py

Attest the Reader analysis settings that produced canonical response records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from dnadesign.studies.core.reader_records import ReaderRecordExpectation, ReaderRecordSet
from dnadesign.studies.core.reader_records.transport import (
    reader_command as resolve_reader_command,
)
from dnadesign.studies.core.reader_records.transport import (
    run_reader_json,
    verify_record_store,
)

from .reader_projection import ReaderResponseProjection
from .reader_projection_contract import READER_EVENT_WINDOW_RECORD_CONTRACTS

CONFIG_ATTESTATION_SCHEMA = "stress_ethanol_cipro_growth.reader_config_attestation.v1"


@dataclass(frozen=True, slots=True)
class ReaderResponseConfigAttestation:
    """Stable public-Reader evidence for one exact authoring contract."""

    config_sha256: str
    authoring_sha256: str
    analysis: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_ATTESTATION_SCHEMA,
            "config_sha256": self.config_sha256,
            "authoring_sha256": self.authoring_sha256,
            "analysis": _json_value(self.analysis),
        }


def attest_reader_response_config(
    source: ReaderRecordSet,
    projection: ReaderResponseProjection,
    *,
    reader_command: Sequence[str] | None = None,
) -> ReaderResponseConfigAttestation:
    """Bracket Reader verification with two identical public authoring reads."""

    command = tuple(reader_command or resolve_reader_command(source.reader_root))
    before = _read_attestation(source, projection=projection, command=command)
    verify_record_store(
        command,
        config_path=source.config_path,
        cwd=source.reader_root,
        expected_records=_record_expectations(),
    )
    after = _read_attestation(source, projection=projection, command=command)
    if after.to_dict() != before.to_dict():
        raise ValueError("Reader authoring or config bytes changed during semantic attestation.")
    return after


def expected_reader_analysis(projection: ReaderResponseProjection) -> dict[str, object]:
    """Project the study contract into Reader's public analysis shape."""

    aggregation = dict(projection.aggregation)
    quality = dict(cast(Mapping[str, object], aggregation.pop("quality")))
    reader_quality = {
        "positive_floor": quality["positive_floor"],
        "max_interior_gap_h": quality["allowed_max_interior_gap_h"],
        "min_observations_per_state": quality["required_min_observations_per_state"],
    }
    reductions = []
    for item in projection.reductions:
        reduction = dict(item)
        if reduction["pre_window_duration_h"] is None:
            del reduction["pre_window_duration_h"]
        reductions.append(reduction)
    return {
        "source": _json_value(projection.source),
        "event": _json_value(projection.event),
        "reductions": reductions,
        "aggregation": _json_value(aggregation),
        "quality": reader_quality,
    }


def _read_attestation(
    source: ReaderRecordSet,
    *,
    projection: ReaderResponseProjection,
    command: Sequence[str],
) -> ReaderResponseConfigAttestation:
    config_before = _sha256_file(source.config_path)
    payload = run_reader_json(
        [*command, "inspect", str(source.config_path), "--section", "authoring", "--format", "json"],
        cwd=source.reader_root,
    )
    config_after = _sha256_file(source.config_path)
    if config_before != config_after:
        raise ValueError("Reader config bytes changed during public authoring inspection.")
    if payload.get("schema") != "reader.cli/v1" or payload.get("command") != "inspect" or payload.get("ok") is not True:
        raise ValueError("Reader inspect did not return a successful reader.cli/v1 payload.")
    meta = _mapping(payload.get("meta"), label="Reader inspect meta")
    if meta != {"projection": "section:authoring", "truncated": False, "continuation": None}:
        raise ValueError("Reader inspect authoring projection metadata is malformed.")
    data = _mapping(payload.get("data"), label="Reader inspect data")
    if set(data) != {"experiment", "authoring"}:
        raise ValueError("Reader inspect authoring payload fields are malformed.")
    _validate_experiment(data["experiment"], source=source)
    authoring = _mapping(data["authoring"], label="Reader inspect authoring")
    if set(authoring) != {"inputs", "analysis", "outputs"}:
        raise ValueError("Reader inspect authoring fields are malformed.")
    analysis = _json_value(_mapping(authoring["analysis"], label="Reader inspect analysis"))
    expected = expected_reader_analysis(projection)
    if analysis != expected:
        raise ValueError("Reader analysis settings disagree with the study projection.")
    canonical_authoring = _canonical_json(_json_value(authoring))
    frozen = cast(Mapping[str, object], _freeze(analysis))
    return ReaderResponseConfigAttestation(
        config_sha256=config_after,
        authoring_sha256=hashlib.sha256(canonical_authoring).hexdigest(),
        analysis=frozen,
    )


def _validate_experiment(value: object, *, source: ReaderRecordSet) -> None:
    experiment = _mapping(value, label="Reader inspect experiment")
    observed_config = Path(str(experiment.get("config", ""))).expanduser().resolve()
    observed_root = Path(str(experiment.get("root", ""))).expanduser().resolve()
    if (
        experiment.get("id") != source.experiment_id
        or experiment.get("protocol") != source.protocol_id
        or observed_config != source.config_path
        or observed_root != source.experiment_root
    ):
        raise ValueError("Reader inspect experiment identity disagrees with the verified record source.")


def _record_expectations() -> dict[str, ReaderRecordExpectation]:
    return {
        name: ReaderRecordExpectation(record_id=record_id, contract_id=contract_id)
        for name, (record_id, contract_id) in READER_EVENT_WINDOW_RECORD_CONTRACTS.items()
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"could not read Reader config {path}: {exc}") from exc


__all__ = [
    "CONFIG_ATTESTATION_SCHEMA",
    "ReaderResponseConfigAttestation",
    "attest_reader_response_config",
    "expected_reader_analysis",
]
