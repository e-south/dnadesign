from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .utils import OpalError


def _ensure_writable_dir(path: Path, *, ctx: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OpalError(f"{ctx} could not create directory: {path}") from exc
    if not path.is_dir():
        raise OpalError(f"{ctx} is not a directory: {path}")
    if not os.access(path, os.W_OK):
        raise OpalError(f"{ctx} is not writable: {path}")
    return path


def resolve_opal_tmpdir(*, workdir: Path | None = None) -> Path:
    """Return the authoritative OPAL tmp/cache directory (created if needed).

    Override with OPAL_TMPDIR for debugging, but do not require it for normal use.
    """

    override = os.getenv("OPAL_TMPDIR")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            base = workdir or Path.cwd()
            override_path = base / override_path
        return _ensure_writable_dir(override_path, ctx="OPAL_TMPDIR")

    if workdir is not None:
        candidate = workdir / ".opal" / "tmp"
        return _ensure_writable_dir(candidate, ctx="OPAL tmpdir")

    candidate = Path(tempfile.gettempdir()) / "opal" / "tmp"
    return _ensure_writable_dir(candidate, ctx="OPAL tmpdir")
