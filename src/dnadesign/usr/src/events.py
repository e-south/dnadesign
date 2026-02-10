"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/events.py

Structured JSONL event logging for USR dataset mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping, Optional

import pyarrow.parquet as pq

from .event_schema import USR_EVENT_VERSION
from .io import now_utc
from .registry import registry_hash as _registry_hash
from .types import Fingerprint
from .version import __version__

_REDACTED_VALUE = "***REDACTED***"
_SENSITIVE_ARG_KEY_TOKENS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "webhook",
    "auth",
    "credential",
    "bearer",
    "cookie",
    "session",
)


def _arg_key_is_sensitive(key: str) -> bool:
    key_norm = str(key or "").strip().lower().replace("-", "_")
    if not key_norm:
        return False
    return any(token in key_norm for token in _SENSITIVE_ARG_KEY_TOKENS)


def _redact_arg_value(value: Any, *, force_redact: bool = False) -> Any:
    if force_redact:
        return _REDACTED_VALUE
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = _redact_arg_value(item, force_redact=_arg_key_is_sensitive(key_text))
        return redacted
    if isinstance(value, (list, tuple)):
        redacted_items: list[Any] = []
        redact_next_value = False
        for item in value:
            if redact_next_value:
                redacted_items.append(_REDACTED_VALUE)
                redact_next_value = False
                continue
            if isinstance(item, str):
                token = str(item).strip()
                if token.startswith("-") and "=" in token:
                    flag, _raw_value = token.split("=", 1)
                    flag_key = str(flag).lstrip("-").replace("-", "_")
                    if _arg_key_is_sensitive(flag_key):
                        redacted_items.append(f"{flag}={_REDACTED_VALUE}")
                        continue
                if token.startswith("-"):
                    flag_key = token.lstrip("-").replace("-", "_")
                    if _arg_key_is_sensitive(flag_key):
                        redacted_items.append(token)
                        redact_next_value = True
                        continue
            redacted_items.append(_redact_arg_value(item, force_redact=False))
        return redacted_items
    return value


def _redact_args(args: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if args is None:
        return {}
    if not isinstance(args, Mapping):
        raise TypeError("event args must be a mapping when provided")
    return dict(_redact_arg_value(args, force_redact=False))


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_parquet(path: Path) -> Fingerprint:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fingerprint path does not exist: {p}")
    pf = pq.ParquetFile(str(p))
    meta = pf.metadata
    sha256 = _sha256_file(p) if os.getenv("USR_EVENT_SHA256") == "1" else None
    return Fingerprint(
        rows=meta.num_rows,
        cols=meta.num_columns,
        size_bytes=int(p.stat().st_size),
        sha256=sha256,
    )


def record_event(
    event_path: Path,
    action: str,
    *,
    dataset: str,
    args: Optional[dict] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[dict] = None,
    maintenance: Optional[dict] = None,
    target_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    registry_hash: Optional[str] = None,
    actor: Optional[dict] = None,
) -> None:
    if target_path is None:
        raise ValueError("target_path is required for event fingerprinting.")
    defaults = _event_defaults(str(action))
    metrics = {**defaults["metrics"], **dict(metrics or {})}
    artifacts = {**defaults["artifacts"], **dict(artifacts or {})}
    maintenance = dict(maintenance or {})
    if registry_hash is None and dataset_root is not None:
        registry_hash = _registry_hash(Path(dataset_root), required=False)
    if actor is None:
        tool = os.getenv("USR_ACTOR_TOOL") or "usr"
        run_id = os.getenv("USR_ACTOR_RUN_ID")
        actor = {
            "tool": tool,
            "run_id": run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
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
        "actor": actor,
        "version": __version__,
    }
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _event_defaults(action: str) -> dict:
    action = str(action)
    defaults = {
        "init": {
            "metrics": {"rows": 0},
            "artifacts": {"overlays": []},
        },
        "import_rows": {
            "metrics": {"rows_written": 0, "rows_skipped": 0},
            "artifacts": {"base": {}},
        },
        "attach": {
            "metrics": {"rows_incoming": 0, "rows_matched": 0, "rows_missing": 0},
            "artifacts": {"overlay": {}},
        },
        "write_overlay_part": {
            "metrics": {"rows_incoming": 0, "rows_written": 0, "rows_missing": 0},
            "artifacts": {"overlay": {}},
        },
        "tombstone": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr"}},
        },
        "restore": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr"}},
        },
        "state_set": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr_state"}},
        },
        "state_clear": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr_state"}},
        },
        "snapshot": {
            "metrics": {"rows": 0},
            "artifacts": {"snapshot": {}},
        },
        "materialize": {
            "metrics": {"overlays": 0, "rows": 0},
            "artifacts": {"overlays": []},
        },
        "dedupe": {
            "metrics": {"rows_total": 0, "rows_dropped": 0, "groups": 0},
            "artifacts": {"base": {}},
        },
        "merge": {
            "metrics": {
                "rows_added": 0,
                "duplicates_total": 0,
                "duplicates_skipped": 0,
                "duplicates_replaced": 0,
            },
            "artifacts": {"src": None, "dest": None},
        },
        "registry_freeze": {
            "metrics": {"updated": 0},
            "artifacts": {"registry_snapshot": ""},
        },
        "overlay_compact": {
            "metrics": {"parts_in": 0, "parts_out": 0, "rows": 0},
            "artifacts": {"overlay": {}},
        },
        "remove_overlay": {
            "metrics": {"removed": 0},
            "artifacts": {"overlay": {}},
        },
    }
    return defaults.get(action, {"metrics": {}, "artifacts": {}})
