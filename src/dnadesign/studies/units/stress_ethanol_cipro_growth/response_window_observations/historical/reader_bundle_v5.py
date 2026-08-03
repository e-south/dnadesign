"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/historical/reader_bundle_v5.py

Verify immutable bundle-v5 evidence for frozen campaign replay only.

This module is not an authoring source and must not be imported by the active
preview or materialization path.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

HISTORICAL_SCHEMA = "reader.response_window.bundle.v5"
_CONTRACTS = {
    "wells": "plate_reader.response_window.wells.v3",
    "designs": "plate_reader.response_window.designs.v3",
    "bootstrap_draws": "plate_reader.response_window.bootstrap_draws.v2",
    "traces": "plate_reader.response_window.traces.v3",
    "events": "plate_reader.response_window.events.v2",
}


@dataclass(frozen=True, slots=True)
class HistoricalReaderResponseBundleV5:
    """Verified legacy frames retained only to reproduce accepted evidence."""

    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    primary_reduction_id: str
    wells: pd.DataFrame
    designs: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    traces: pd.DataFrame
    events: pd.DataFrame


def load_historical_reader_response_bundle_v5(
    root: Path,
    *,
    expected_request_path: Path,
) -> HistoricalReaderResponseBundleV5:
    """Verify and load one frozen bundle without exposing a future-data API."""

    bundle_root = Path(root).expanduser().resolve()
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"historical Reader bundle manifest is missing: {manifest_path}")
    manifest = _read_unique_json(manifest_path)
    if manifest.get("schema_version") != HISTORICAL_SCHEMA:
        raise ValueError(f"historical Reader evidence must use {HISTORICAL_SCHEMA!r}")
    if manifest.get("contracts") != _CONTRACTS:
        raise ValueError("historical Reader bundle contracts disagree")
    request = manifest.get("request")
    if not isinstance(request, dict) or set(request) != {"artifact_id", "sha256"}:
        raise ValueError("historical Reader bundle request receipt is malformed")
    expected_request = Path(expected_request_path).expanduser().resolve().read_bytes()
    expected_request_digest = "sha256:" + hashlib.sha256(expected_request).hexdigest()
    if request.get("sha256") != expected_request_digest:
        raise ValueError("historical Reader request digest disagrees with its accepted evidence")

    artifacts = manifest.get("artifacts")
    records = manifest.get("records")
    if not isinstance(artifacts, dict) or not isinstance(records, dict) or set(records) != set(_CONTRACTS):
        raise ValueError("historical Reader bundle record inventory is malformed")
    frames: dict[str, pd.DataFrame] = {}
    for name, contract_id in _CONTRACTS.items():
        record = records[name]
        if not isinstance(record, dict) or record.get("contract_id") != contract_id:
            raise ValueError(f"historical Reader {name!r} record contract disagrees")
        artifact_id = record.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise ValueError(f"historical Reader {name!r} artifact identity is malformed")
        evidence = artifacts.get(artifact_id)
        if not isinstance(evidence, dict) or set(evidence) != {"path", "bytes", "sha256"}:
            raise ValueError(f"historical Reader {name!r} artifact evidence is malformed")
        if evidence.get("path") != artifact_id:
            raise ValueError(f"historical Reader {name!r} artifact path disagrees")
        path = (bundle_root / artifact_id).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"historical Reader {name!r} artifact escapes its bundle") from exc
        content = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if evidence.get("bytes") != len(content) or evidence.get("sha256") != digest:
            raise ValueError(f"historical Reader {name!r} artifact digest disagrees")
        try:
            frames[name] = pd.read_parquet(BytesIO(content))
        except Exception as exc:
            raise ValueError(f"historical Reader {name!r} artifact cannot be parsed: {exc}") from exc
    primary_reduction = manifest.get("primary_reduction_id")
    if not isinstance(primary_reduction, str) or not primary_reduction.strip():
        raise ValueError("historical Reader primary reduction is missing")
    return HistoricalReaderResponseBundleV5(
        root=bundle_root,
        manifest_path=manifest_path,
        manifest=manifest,
        primary_reduction_id=primary_reduction,
        wells=frames["wells"],
        designs=frames["designs"],
        bootstrap_draws=frames["bootstrap_draws"],
        traces=frames["traces"],
        events=frames["events"],
    )


def _read_unique_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"historical Reader manifest repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse historical Reader manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("historical Reader manifest must be an object")
    return value


__all__ = [
    "HISTORICAL_SCHEMA",
    "HistoricalReaderResponseBundleV5",
    "load_historical_reader_response_bundle_v5",
]
