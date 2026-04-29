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
from .defaults import USR_EVENT_VERSION, _event_defaults
from .fingerprint import fingerprint_parquet
from .redaction import _redact_args


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
    defaults = _event_defaults(str(action))
    metrics = {**defaults["metrics"], **dict(metrics or {})}
    artifacts = {**defaults["artifacts"], **dict(artifacts or {})}
    maintenance = dict(maintenance or {})
    if registry_hash is None and dataset_root is not None:
        registry_hash = _registry_hash(Path(dataset_root), required=False)
    actor_value = _normalize_actor(actor)
    payload = {
        "event_version": USR_EVENT_VERSION,
        "timestamp_utc": now_utc(),
        "action": str(action),
        "dataset": {
            "name": str(dataset),
            "root": str(dataset_root) if dataset_root else None,
        },
        "args": _redact_args(args),
        "metrics": metrics,
        "artifacts": artifacts,
        "maintenance": maintenance,
        "fingerprint": fingerprint_parquet(target_path).to_dict(),
        "registry_hash": registry_hash,
        "actor": actor_value,
        "version": __version__,
    }
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
