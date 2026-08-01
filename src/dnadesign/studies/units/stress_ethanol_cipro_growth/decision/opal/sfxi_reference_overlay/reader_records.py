"""Verify and load selected neutral four-state Reader records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from dnadesign.usr import SchemaError

SELECTION_SCHEMA = "stress_sfxi_reader_record_selection.v1"
READER_CONTRACT = "logic.four_state_vector.v1"
READER_RECORD_ID = "four_state_vector/vector"
READER_RECORD_SCHEMA_VERSION = 6
READER_CATALOG_SCHEMA_VERSION = 4
READER_PRODUCER = {
    "id": "four_state_vector",
    "kind": "pipeline",
    "plugin": "transform/four_state_vector",
}


@dataclass(frozen=True, slots=True)
class VerifiedReaderSelection:
    frame: pd.DataFrame
    source_ref: str
    record_digests: tuple[str, ...]


def default_selection_path() -> Path:
    return Path(__file__).with_name("reader_records.json")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _json_digest(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise SchemaError(f"{label} must be a lowercase sha256 digest.")
    return str(value)


def _confined_file(root: Path, relative: object, *, label: str) -> Path:
    rel = Path(str(relative))
    if rel.is_absolute() or ".." in rel.parts:
        raise SchemaError(f"{label} must be a confined relative path.")
    candidate = (root / rel).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
        raise SchemaError(f"{label} is missing or outside the Reader root: {rel}")
    return candidate


def _experiment_identity(manifest: object) -> tuple[str, int, Path]:
    relative = Path(str(manifest))
    parts = relative.parts
    if (
        len(parts) != 6
        or parts[0] != "experiments"
        or parts[3:] != ("outputs", "manifests", "records.json")
        or len(parts[1]) != 4
        or not parts[1].isdigit()
    ):
        raise SchemaError("Reader record manifest must use the canonical experiment catalog location.")
    experiment_id = parts[2]
    date_prefix = experiment_id[:8]
    try:
        experiment_date = datetime.strptime(date_prefix, "%Y%m%d").date()
    except ValueError as exc:
        raise SchemaError("Reader experiment id must begin with a valid YYYYMMDD date.") from exc
    if str(experiment_date.year) != parts[1]:
        raise SchemaError("Reader experiment year directory must match its experiment id date.")
    return experiment_id, int(date_prefix), relative


def _verify_experiment_config(*, root: Path, manifest: Path, experiment_id: str) -> None:
    config_path = manifest.parents[2] / "config.yaml"
    if not config_path.is_file() or not config_path.resolve().is_relative_to(root):
        raise SchemaError("Reader experiment is missing its canonical config.yaml.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "reader/v8":
        raise SchemaError("Reader experiment config must use reader/v8.")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("id") != experiment_id:
        raise SchemaError("Reader experiment config id does not match its canonical directory.")


def load_verified_reader_selection(*, reader_root: Path, selection_path: Path) -> VerifiedReaderSelection:
    """Load only digest-pinned, latest canonical Reader records."""

    envelope_path = selection_path.expanduser().resolve()
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if envelope.get("schema_version") != SELECTION_SCHEMA:
        raise SchemaError(f"Reader selection must use {SELECTION_SCHEMA}.")
    selection_id = str(envelope.get("selection_id", "")).strip()
    records = envelope.get("records")
    if not selection_id or not isinstance(records, list) or not records:
        raise SchemaError("Reader selection requires an id and at least one record.")

    frames: list[pd.DataFrame] = []
    digests: list[str] = []
    root = reader_root.expanduser().resolve()
    for expected in records:
        if not isinstance(expected, dict):
            raise SchemaError("Reader selection records must be JSON objects.")
        experiment_id, experiment_date, manifest_relative = _experiment_identity(expected.get("manifest"))
        manifest_path = _confined_file(root, manifest_relative, label="Reader record manifest")
        _verify_experiment_config(root=root, manifest=manifest_path, experiment_id=experiment_id)
        catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
        if catalog.get("schema_version") != READER_CATALOG_SCHEMA_VERSION:
            raise SchemaError(f"Reader record catalog must use schema {READER_CATALOG_SCHEMA_VERSION}.")
        record_id = str(expected.get("record_id", ""))
        if record_id != READER_RECORD_ID:
            raise SchemaError(f"Reader selection record id must be {READER_RECORD_ID}.")
        if expected.get("contract_id") != READER_CONTRACT:
            raise SchemaError(f"Reader selection contract must be {READER_CONTRACT}.")
        if expected.get("record_schema_version") != READER_RECORD_SCHEMA_VERSION:
            raise SchemaError(f"Reader selection record schema must be {READER_RECORD_SCHEMA_VERSION}.")
        latest = catalog.get("latest", {}).get(record_id)
        history = catalog.get("history", {}).get(record_id)
        if not isinstance(latest, dict) or not isinstance(history, list) or not history:
            raise SchemaError(f"Reader latest record is missing: {record_id}")
        revision = expected.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SchemaError("Reader selection revision must be a positive integer.")
        if len(history) != revision:
            raise SchemaError(f"Reader record {record_id} revision count mismatch.")
        record = history[-1]
        if not isinstance(record, dict) or latest != record:
            raise SchemaError(f"Reader record {record_id} latest revision does not match history.")
        expected_revision_digest = _require_sha256(
            expected.get("revision_digest"), label="Reader selection revision_digest"
        )
        if _json_digest(record) != expected_revision_digest:
            raise SchemaError(f"Reader record {record_id} revision digest mismatch.")
        expected_digests = {
            field: _require_sha256(expected.get(field), label=f"Reader selection {field}")
            for field in ("content_digest", "config_digest", "code_digest")
        }
        for field in ("content_digest", "config_digest", "code_digest", "producer_config_digest"):
            _require_sha256(record.get(field), label=f"Reader record {field}")
        build_identity = record.get("build_identity")
        if isinstance(build_identity, dict):
            _require_sha256(build_identity.get("source_digest"), label="Reader record build_identity.source_digest")
        checks = {
            "record_id": record_id,
            "kind": "dataframe_artifact",
            "contract_id": READER_CONTRACT,
            "schema_version": READER_RECORD_SCHEMA_VERSION,
            **expected_digests,
            "producer": READER_PRODUCER,
        }
        for field, value in checks.items():
            if record.get(field) != value:
                raise SchemaError(f"Reader record {record_id} has unverified {field}.")
        artifact = _confined_file(manifest_path.parent.parent, record.get("path"), label="Reader record artifact")
        digest = _sha256(artifact)
        if digest != checks["content_digest"]:
            raise SchemaError(f"Reader record {record_id} content digest mismatch.")
        frame = pq.read_table(artifact).to_pandas()
        if "design_id" not in frame.columns:
            raise SchemaError(f"Reader record {record_id} is missing design_id.")
        design_ids = [str(value) for value in expected.get("design_ids", [])]
        selected = frame.loc[frame["design_id"].astype(str).isin(design_ids)].copy()
        observed = selected["design_id"].astype(str).tolist()
        if len(observed) != len(design_ids) or set(observed) != set(design_ids):
            raise SchemaError(f"Reader record {record_id} does not contain its exact selected design ids.")
        selected["experiment_id"] = experiment_id
        selected["experiment_date"] = experiment_date
        frames.append(selected)
        digests.append(digest)

    combined = pd.concat(frames, ignore_index=True)
    if combined["design_id"].astype(str).duplicated().any():
        raise SchemaError("Reader selection contains duplicate design ids across records.")
    envelope_digest = _sha256(envelope_path)
    source_ref = f"reader-record-selection:{selection_id}@{envelope_digest}"
    return VerifiedReaderSelection(frame=combined, source_ref=source_ref, record_digests=tuple(digests))
