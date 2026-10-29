"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/sources/paths.py

Filesystem and SeqLedger root resolution for Construct source loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.seqledger import default_seqledger_root as usr_default_root
from dnadesign.seqledger import normalize_seqledger_root

from ..contracts.errors import ValidationError


def default_seqledger_root() -> Path:
    return usr_default_root()


def resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_usr_root(base_dir: Path, value: str | None, *, label: str) -> Path:
    resolved = resolve_optional_path(base_dir, value)
    if resolved is None:
        raise ValidationError(f"{label} is required for SeqLedger-backed construct jobs.")
    return normalize_seqledger_root(resolved)
