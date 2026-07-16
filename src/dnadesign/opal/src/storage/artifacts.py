"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/artifacts.py

Defines round artifact paths and write helpers for OPAL outputs. Provides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..core.utils import OpalError, ensure_dir, file_sha256
from .parquet_io import write_parquet_df

PROGRESS_EVENT_SCHEMA_VERSION = "opal.progress_event.v1"
PROGRESS_EVENT_PHASES = frozenset({"command", "preflight", "run", "abort", "finalize"})
PROGRESS_EVENT_SEVERITIES = frozenset({"info", "warning", "error"})


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
    feature_importance_csv: Path


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
