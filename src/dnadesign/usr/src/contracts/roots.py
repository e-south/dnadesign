"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/contracts/roots.py

Validates explicit operator-managed USR storage coordinates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def require_explicit_usr_root(root: str | Path) -> Path:
    """Validate an operator-supplied USR coordinate without path inference."""

    target = Path(root).expanduser()
    if not target.is_absolute():
        raise ValueError("explicit USR root must be absolute")
    if target.is_symlink():
        raise ValueError("explicit USR root must not be a symbolic link")
    if not target.is_dir():
        raise ValueError("explicit USR root must name an existing directory")
    return target.resolve(strict=True)


__all__ = ["require_explicit_usr_root"]
