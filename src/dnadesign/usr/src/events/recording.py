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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..registry import registry_hash as _registry_hash
from ..storage.parquet import now_utc
from ..version import __version__
from .actor import _normalize_actor
from .append import append_event_payload, encode_event_line
from .defaults import USR_EVENT_VERSION, _event_defaults
from .fingerprint import fingerprint_parquet
from .redaction import _redact_args


@dataclass(frozen=True, slots=True)
class PreparedEvent:
    """Fully validated and encoded event ready for one append attempt."""

    payload: bytes


@dataclass(slots=True)
class EventAppendAttempt:
    """Expose whether a prepared event may have reached the append boundary."""

    event: PreparedEvent
    _started: bool = field(default=False, init=False, repr=False)
    _completed: bool = field(default=False, init=False, repr=False)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def completed(self) -> bool:
        return self._completed

    def _mark_started(self) -> None:
        self._started = True

    def append_to(self, event_path: Path) -> None:
        if self._started:
            raise RuntimeError("Prepared event append has already been attempted.")
        append_event_payload(event_path, self.event.payload, on_start=self._mark_started)
        self._completed = True


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


def prepare_event(
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
) -> PreparedEvent:
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
    return PreparedEvent(payload=encode_event_line(encoded))


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
    prepared = prepare_event(
        action,
        dataset=dataset,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        maintenance=maintenance,
        target_path=target_path,
        dataset_root=dataset_root,
        registry_hash=registry_hash,
        actor=actor,
    )
    EventAppendAttempt(prepared).append_to(event_path)
