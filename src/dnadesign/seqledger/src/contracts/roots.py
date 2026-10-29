"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/src/contracts/roots.py

Validates explicit operator-managed SeqLedger storage coordinates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def require_explicit_seqledger_root(root: str | Path) -> Path:
    """Validate an operator-supplied SeqLedger coordinate without path inference."""

    target = Path(root).expanduser()
    if not target.is_absolute():
        raise ValueError("explicit SeqLedger root must be absolute")
    if target.is_symlink():
        raise ValueError("explicit SeqLedger root must not be a symbolic link")
    if not target.is_dir():
        raise ValueError("explicit SeqLedger root must name an existing directory")
    return target.resolve(strict=True)


__all__ = ["require_explicit_seqledger_root"]
