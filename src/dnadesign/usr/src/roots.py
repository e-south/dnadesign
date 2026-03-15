"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/roots.py

USR root resolution helpers for canonical dataset storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def pkg_usr_root() -> Path:
    """Return the installed dnadesign/usr package directory."""
    return Path(__file__).resolve().parents[1]


def default_usr_root(*, pkg_root: Path | None = None) -> Path:
    """Return the canonical USR datasets root."""
    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root().resolve()
    return (base / "datasets").resolve()


def normalize_usr_root(root: str | Path | None, *, pkg_root: Path | None = None) -> Path:
    """
    Accept either the package root (.../dnadesign/usr) or the datasets root
    (.../dnadesign/usr/datasets) and normalize to the canonical datasets root.
    """
    datasets_root = default_usr_root(pkg_root=pkg_root)
    if root is None:
        return datasets_root

    target = Path(root).expanduser().resolve()
    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root().resolve()
    if target == base:
        return datasets_root
    if (target / "__init__.py").exists():
        return (target / "datasets").resolve()
    return target
