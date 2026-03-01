"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/spool.py

Spool directory and payload persistence helpers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import stat
import time
from pathlib import Path

from ..errors import NotifyConfigError


def ensure_private_directory(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to create {label}: {path}") from exc
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on {label}: {path}") from exc
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"{label} must not be group/world-accessible (expected mode 700): {path} (mode={oct(mode)})"
        )


def spool_payload(
    spool_dir: Path,
    *,
    provider: str,
    payload: dict[str, object],
) -> Path:
    ensure_private_directory(spool_dir, label="spool_dir")
    body = {"provider": provider, "payload": payload}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    file_name = f"{int(time.time() * 1000)}-{digest}.json"
    out_path = spool_dir / file_name
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(raw, encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on spool temp file: {tmp_path}") from exc
    tmp_path.replace(out_path)
    try:
        out_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on spool file: {out_path}") from exc
    return out_path
