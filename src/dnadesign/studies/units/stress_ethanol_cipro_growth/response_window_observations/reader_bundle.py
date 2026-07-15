"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_bundle.py

Consume the public Reader response-window bundle without Reader imports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .artifact_io import read_json_object
from .display_contract import READER_DISPLAY_SCHEMA, response_example_labels, validate_reader_display
from .reader_bundle_validation import validate_reader_bundle_frames

READER_BUNDLE_SCHEMA = "reader.response_window.bundle.v5"
EXPECTED_CONTRACTS = {
    "wells": "plate_reader.response_window.wells.v3",
    "designs": "plate_reader.response_window.designs.v3",
    "bootstrap_draws": "plate_reader.response_window.bootstrap_draws.v2",
    "traces": "plate_reader.response_window.traces.v3",
    "events": "plate_reader.response_window.events.v2",
}
EXPECTED_RECORD_ARTIFACTS = {record_id: f"tables/{record_id}.parquet" for record_id in EXPECTED_CONTRACTS}
STATE_ORDER = ("00", "10", "01", "11")
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER) + tuple(f"b{state}" for state in STATE_ORDER)


@dataclass(frozen=True)
class ReaderResponseBundle:
    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    designs: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    wells: pd.DataFrame
    traces: pd.DataFrame
    events: pd.DataFrame

    @property
    def primary_reduction_id(self) -> str:
        value = self.manifest.get("primary_reduction_id")
        if not isinstance(value, str) or not value:
            raise ValueError("Reader bundle lacks primary_reduction_id.")
        return value

    @property
    def response_examples(self) -> dict[str, str]:
        return response_example_labels(self.manifest.get("display"))

    @property
    def reference_design_id(self) -> str:
        display = self.manifest.get("display")
        channels = display.get("channels") if isinstance(display, dict) else None
        value = channels.get("reference_design_id") if isinstance(channels, dict) else None
        if not isinstance(value, str) or not value:
            raise ValueError("Reader bundle display contract lacks reference_design_id.")
        return value


def load_reader_response_bundle(path: Path, *, expected_request_path: Path) -> ReaderResponseBundle:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reader response-window manifest not found: {manifest_path}")
    payload = read_json_object(manifest_path, label="Reader response-window manifest")
    if not isinstance(payload, dict) or payload.get("schema_version") != READER_BUNDLE_SCHEMA:
        raise ValueError(f"Reader response-window bundle must use {READER_BUNDLE_SCHEMA!r}.")
    if payload.get("study_id") != "stress_ethanol_cipro_growth":
        raise ValueError("Reader response-window bundle study identity disagrees with this study.")
    contracts = payload.get("contracts")
    if contracts != EXPECTED_CONTRACTS:
        raise ValueError(f"Reader response-window contracts disagree: {contracts!r}.")
    if payload.get("state_order") != list(STATE_ORDER):
        raise ValueError(f"Reader response-window state order must be {list(STATE_ORDER)!r}.")
    validate_reader_display(payload.get("display"))
    request_path = Path(expected_request_path).expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"expected Reader response-window request not found: {request_path}")
    request = payload.get("request")
    if (
        not isinstance(request, dict)
        or request.get("artifact_id") != "request.yaml"
        or request.get("sha256") != _sha256(request_path)
    ):
        raise ValueError("Reader response-window request digest disagrees with the checked-in study request.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Reader response-window bundle has no artifact inventory.")
    artifact_paths: dict[str, Path] = {}
    for artifact_id, raw in artifacts.items():
        if not isinstance(artifact_id, str) or not isinstance(raw, dict):
            raise ValueError(f"Reader artifact {artifact_id!r} metadata must be a mapping.")
        relative = raw.get("path")
        expected = raw.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"Reader artifact {artifact_id!r} lacks path or digest.")
        if relative != artifact_id:
            raise ValueError(f"Reader artifact {artifact_id!r} path disagrees with its manifest identity.")
        artifact_path = (root / relative).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Reader artifact {artifact_id!r} escapes the bundle root.") from exc
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Reader artifact {artifact_id!r} is missing: {artifact_path}")
        if _sha256(artifact_path) != expected:
            raise ValueError(f"Reader artifact {artifact_id!r} digest mismatch.")
        artifact_paths[artifact_id] = artifact_path

    records = payload.get("records")
    expected_records = {
        record_id: {"contract_id": EXPECTED_CONTRACTS[record_id], "artifact_id": artifact_id}
        for record_id, artifact_id in EXPECTED_RECORD_ARTIFACTS.items()
    }
    if records != expected_records:
        raise ValueError(f"Reader response-window record map disagrees: {records!r}.")
    missing_record_artifacts = sorted(set(EXPECTED_RECORD_ARTIFACTS.values()) - set(artifacts))
    if missing_record_artifacts:
        raise ValueError(f"Reader response-window bundle lacks record artifacts: {missing_record_artifacts}.")
    designs = pd.read_parquet(_record_path(artifact_paths, records, record_id="designs"))
    draws = pd.read_parquet(_record_path(artifact_paths, records, record_id="bootstrap_draws"))
    wells = pd.read_parquet(_record_path(artifact_paths, records, record_id="wells"))
    traces = pd.read_parquet(_record_path(artifact_paths, records, record_id="traces"))
    events = pd.read_parquet(_record_path(artifact_paths, records, record_id="events"))
    validate_reader_bundle_frames(
        designs=designs, draws=draws, wells=wells, traces=traces, events=events, payload=payload
    )
    return ReaderResponseBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=payload,
        designs=designs,
        bootstrap_draws=draws,
        wells=wells,
        traces=traces,
        events=events,
    )


def build_all_primary_measurements(bundle: ReaderResponseBundle) -> pd.DataFrame:
    result = bundle.designs.loc[
        bundle.designs["reduction_id"].astype(str).eq(bundle.primary_reduction_id)
        & ~bundle.designs["is_reference"].astype(bool)
    ].copy()
    result = result.rename(columns={"experiment_id": "reader_experiment_id"})
    result["id"] = result["reader_experiment_id"].astype(str) + "::" + result["design_id"].astype(str)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _record_path(artifact_paths: dict[str, Path], records: object, *, record_id: str) -> Path:
    if not isinstance(records, dict):
        raise ValueError("Reader response-window bundle lacks a record map.")
    raw = records.get(record_id)
    if not isinstance(raw, dict) or not isinstance(raw.get("artifact_id"), str):
        raise ValueError(f"Reader response-window record {record_id!r} lacks an artifact id.")
    artifact_id = str(raw["artifact_id"])
    try:
        return artifact_paths[artifact_id]
    except KeyError as exc:
        raise ValueError(f"Reader response-window record {record_id!r} lacks a verified artifact.") from exc


__all__ = [
    "EXPECTED_CONTRACTS",
    "EXPECTED_RECORD_ARTIFACTS",
    "READER_BUNDLE_SCHEMA",
    "READER_DISPLAY_SCHEMA",
    "ReaderResponseBundle",
    "build_all_primary_measurements",
    "load_reader_response_bundle",
]
