"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/artifacts.py

Defines round artifact paths and write helpers for OPAL outputs. Provides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict

import pandas as pd

from ..core.utils import OpalError, ensure_dir, file_sha256
from .parquet_io import write_parquet_df

PROGRESS_EVENT_SCHEMA_VERSION = "opal.progress_event.v1"
PROGRESS_EVENT_PHASES = frozenset({"command", "preflight", "run", "abort", "finalize"})
PROGRESS_EVENT_SEVERITIES = frozenset({"info", "warning", "error"})
LABELS_USED_ARTIFACT_KEY = "labels/labels_used.parquet"
OBSERVED_EVENTS_ARTIFACT_KEY = "labels/observed_events.parquet"
RUN_ARTIFACTS_DIRECTORY = "run_artifacts"
_RUN_ARTIFACT_STEM_MAX_LENGTH = 48


@dataclass
class ArtifactPaths:
    model: Path
    model_meta_json: Path
    selections_parquet: Path
    selection_batch_parquet: Path
    selection_allocation_trace_parquet: Path
    round_log_jsonl: Path
    round_ctx_json: Path
    objective_meta_json: Path
    labels_used_parquet: Path
    observed_events_parquet: Path
    feature_importance_csv: Path


def run_artifact_slug(run_id: str) -> str:
    """Return a deterministic, path-safe run key with a strong hash suffix."""

    canonical_run_id = str(run_id)
    if not canonical_run_id or canonical_run_id != canonical_run_id.strip():
        raise OpalError("Run artifact paths require a canonical, non-blank run_id.")
    stem = re.sub(r"[^a-z0-9]+", "-", canonical_run_id.lower()).strip("-")
    stem = stem[:_RUN_ARTIFACT_STEM_MAX_LENGTH].rstrip("-") or "run"
    digest = hashlib.sha256(canonical_run_id.encode("utf-8")).hexdigest()
    return f"{stem}-{digest}"


def _run_artifact_directory(round_dir: Path, *, run_id: str) -> Path:
    round_root = Path(round_dir).resolve()
    directory = (round_root / RUN_ARTIFACTS_DIRECTORY / run_artifact_slug(run_id)).resolve()
    try:
        directory.relative_to(round_root)
    except ValueError as exc:
        raise OpalError(f"Run artifact directory is outside its round directory: {directory}") from exc
    return directory


def reserve_run_artifact_directory(round_dir: Path, *, run_id: str) -> Path:
    """Atomically reserve one create-only run snapshot directory."""

    directory = _run_artifact_directory(round_dir, run_id=run_id)
    ensure_dir(directory.parent)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise OpalError(f"Run artifact snapshot already exists for run_id={run_id!r}: {directory}") from exc
    except OSError as exc:
        raise OpalError(f"Could not reserve run artifact snapshot for run_id={run_id!r}: {exc}") from exc
    return directory


def run_scoped_artifact_path(
    round_dir: Path,
    *,
    run_id: str,
    artifact_key: str,
) -> Path:
    """Resolve one logical artifact key beneath an immutable run directory."""

    key = str(artifact_key)
    parts = key.split("/")
    logical_path = PurePosixPath(key)
    if (
        not key
        or key != key.strip()
        or logical_path.is_absolute()
        or "\\" in key
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise OpalError(f"Run artifact key must be a canonical relative path: {artifact_key!r}.")
    round_root = Path(round_dir).resolve()
    run_directory = _run_artifact_directory(round_root, run_id=run_id)
    path = (run_directory / Path(*logical_path.parts)).resolve()
    try:
        path.relative_to(round_root)
    except ValueError as exc:
        raise OpalError(f"Run artifact path is outside its round directory: {path}") from exc
    return path


def snapshot_run_artifacts(
    round_dir: Path,
    *,
    run_id: str,
    artifacts: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Copy every run-ledger artifact into its create-only run snapshot."""

    round_root = Path(round_dir).resolve()
    snapshots: dict[str, tuple[str, str]] = {}
    for artifact_key, (expected_sha256, raw_source) in artifacts.items():
        source = Path(raw_source).expanduser().resolve()
        try:
            source.relative_to(round_root)
        except ValueError as exc:
            raise OpalError(f"Run artifact source is outside its round directory: {source}") from exc
        if not source.is_file():
            raise OpalError(f"Run artifact source not found: {source}")
        source_sha256 = file_sha256(source)
        if source_sha256 != str(expected_sha256):
            raise OpalError(
                f"Run artifact source digest disagrees for {artifact_key!r}: "
                f"expected={expected_sha256}, actual={source_sha256}."
            )

        destination = run_scoped_artifact_path(
            round_root,
            run_id=run_id,
            artifact_key=artifact_key,
        )
        if destination != source:
            ensure_dir(destination.parent)
            try:
                with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        destination_sha256 = file_sha256(destination)
        if destination_sha256 != source_sha256:
            if destination != source:
                destination.unlink(missing_ok=True)
            raise OpalError(f"Run artifact snapshot digest disagrees for {artifact_key!r}.")
        snapshots[str(artifact_key)] = (destination_sha256, str(destination))
    return snapshots


def write_selection_parquet(path: Path, frame: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    write_parquet_df(path, frame, index=False)
    return file_sha256(path)


def write_feature_importance_csv(path: Path, df: pd.DataFrame) -> str:
    """
    Persist per-feature importances. Expected columns:
      - feature_index (int)
      - importance    (float; should sum to 1.0)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return file_sha256(path)


def append_round_log_event(path: Path, event: dict) -> None:
    ensure_dir(path.parent)
    event = dict(event)
    stage = str(event.get("stage", "")).strip()
    event.setdefault("schema_version", PROGRESS_EVENT_SCHEMA_VERSION)
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("phase", _infer_progress_phase(stage))
    event.setdefault("severity", "info")
    validate_progress_event(event)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def validate_progress_event(event: Any, *, position: int | None = None) -> None:
    prefix = f"Progress event {position}" if position is not None else "Progress event"
    if not isinstance(event, dict):
        raise OpalError(f"{prefix} must be a JSON object.")
    required = {"schema_version", "event_id", "phase", "severity", "stage", "ts"}
    missing = sorted(required - set(event))
    if missing:
        raise OpalError(f"{prefix} missing required fields: {missing}")
    if event["schema_version"] != PROGRESS_EVENT_SCHEMA_VERSION:
        raise OpalError(
            f"{prefix} schema_version must be {PROGRESS_EVENT_SCHEMA_VERSION!r}; observed {event['schema_version']!r}."
        )
    for field in ("event_id", "stage"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise OpalError(f"{prefix} field {field!r} must be a non-empty string.")
    if not isinstance(event["phase"], str) or event["phase"] not in PROGRESS_EVENT_PHASES:
        raise OpalError(f"{prefix} phase must be one of {sorted(PROGRESS_EVENT_PHASES)}; observed {event['phase']!r}.")
    if not isinstance(event["severity"], str) or event["severity"] not in PROGRESS_EVENT_SEVERITIES:
        raise OpalError(
            f"{prefix} severity must be one of {sorted(PROGRESS_EVENT_SEVERITIES)}; observed {event['severity']!r}."
        )
    try:
        timestamp = datetime.fromisoformat(str(event["ts"]))
    except ValueError as exc:
        raise OpalError(f"{prefix} ts must be an ISO-8601 timestamp; observed {event['ts']!r}.") from exc
    if timestamp.tzinfo is None:
        raise OpalError(f"{prefix} ts must include a UTC offset; observed {event['ts']!r}.")


def _infer_progress_phase(stage: str) -> str:
    if stage.startswith("command_"):
        return "command"
    if (
        stage.startswith("records_load")
        or stage.startswith("x_validate")
        or stage.startswith("lock_")
        or stage in {"preflight", "preflight_start", "preflight_done"}
    ):
        return "preflight"
    if stage in {"abort", "aborted"} or stage.endswith("_abort"):
        return "abort"
    if stage.startswith("finalize"):
        return "finalize"
    return "run"


def write_round_ctx(path: Path, ctx: dict) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(ctx, indent=2))
    return file_sha256(path)


def write_objective_meta(path: Path, meta: Dict[str, Any]) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(meta, indent=2))
    return file_sha256(path)


def write_model_meta(path: Path, meta: Dict[str, Any]) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(meta, indent=2))
    return file_sha256(path)


def write_labels_used_parquet(path: Path, df: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    write_parquet_df(path, df, index=False)
    return file_sha256(path)


def write_observed_events_parquet(path: Path, df: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    write_parquet_df(path, df, index=False)
    return file_sha256(path)
