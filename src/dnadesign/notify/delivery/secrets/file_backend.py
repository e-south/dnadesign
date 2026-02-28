"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/file_backend.py

Filesystem secret read/write operations for notify secret references.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ...errors import NotifyConfigError


def ensure_private_secret_parent(path: Path) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to create secret directory: {parent}") from exc
    if os.name == "nt":
        return
    try:
        parent.chmod(0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on secret directory: {parent}") from exc
    mode = stat.S_IMODE(parent.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"secret directory must not be group/world-accessible (expected mode 700): {parent} (mode={oct(mode)})"
        )


def assert_private_secret_file(path: Path) -> None:
    if not path.exists():
        raise NotifyConfigError(f"secret value not found for reference: {path.as_uri()}")
    if not path.is_file():
        raise NotifyConfigError(f"secret path is not a file: {path}")
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"secret file must not be group/world-accessible (expected mode 600): {path} (mode={oct(mode)})"
        )


def read_file_secret(path: Path) -> str:
    assert_private_secret_file(path)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise NotifyConfigError(f"failed to read secret file: {path}") from exc
    if not value:
        raise NotifyConfigError(f"secret value not found for reference: {path.as_uri()}")
    return value


def write_file_secret(path: Path, value: str) -> None:
    ensure_private_secret_parent(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(value, encoding="utf-8")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to write secret file: {path}") from exc
    assert_private_secret_file(path)
