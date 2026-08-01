"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/recording.py

USR event payload assembly and append-only recording.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..registry import registry_hash as _registry_hash
from ..storage.parquet import now_utc
from ..version import __version__
from .actor import _normalize_actor
from .append import append_event_line
from .defaults import USR_EVENT_VERSION, _event_defaults
from .fingerprint import fingerprint_parquet
from .redaction import _redact_args


def _normalized_event_metadata(
    action: str,
    *,
    args: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    maintenance: dict[str, Any] | None = None,
    actor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _event_defaults(str(action))
    return {
        "action": str(action),
        "args": _redact_args(args),
        "metrics": {**defaults["metrics"], **dict(metrics or {})},
        "artifacts": {**defaults["artifacts"], **dict(artifacts or {})},
        "maintenance": dict(maintenance or {}),
        "actor": _normalize_actor(actor),
    }


def validate_event_metadata(
    action: str,
    *,
    args: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    maintenance: dict[str, Any] | None = None,
    actor: dict[str, Any] | None = None,
) -> None:
    """Fail before mutation when caller-controlled event metadata is invalid."""

    metadata = _normalized_event_metadata(
        action,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        maintenance=maintenance,
        actor=actor,
    )
    json.dumps(metadata, separators=(",", ":"))


def record_event(
    event_path: Path,
    action: str,
    *,
    dataset: str,
    args: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    maintenance: dict[str, Any] | None = None,
    target_path: Path | None = None,
    dataset_root: Path | None = None,
    registry_hash: str | None = None,
    actor: dict[str, Any] | None = None,
) -> None:
    if target_path is None:
        raise ValueError("target_path is required for event fingerprinting.")
    metadata = _normalized_event_metadata(
        action,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        maintenance=maintenance,
        actor=actor,
    )
    if registry_hash is None and dataset_root is not None:
        registry_hash = _registry_hash(Path(dataset_root), required=False)
    payload = {
        "event_version": USR_EVENT_VERSION,
        "timestamp_utc": now_utc(),
        "action": metadata["action"],
        "dataset": {
            "name": str(dataset),
            "root": str(dataset_root) if dataset_root else None,
        },
        "args": metadata["args"],
        "metrics": metadata["metrics"],
        "artifacts": metadata["artifacts"],
        "maintenance": metadata["maintenance"],
        "fingerprint": fingerprint_parquet(target_path).to_dict(),
        "registry_hash": registry_hash,
        "actor": metadata["actor"],
        "version": __version__,
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    append_event_line(event_path, encoded)
