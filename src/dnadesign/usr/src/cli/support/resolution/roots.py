"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/support/resolution/roots.py

Internal USR root-resolution helpers for canonical dataset storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def pkg_usr_root() -> Path:
    """Return the installed dnadesign/usr package directory."""
    return Path(__file__).resolve().parents[4]


def default_usr_root(*, pkg_root: Path | None = None) -> Path:
    """Return the canonical USR datasets root."""
    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root()
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
    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_usr_root()
    if target == base:
        return datasets_root
    if (target / "__init__.py").exists():
        return (target / "datasets").resolve()
    return target


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


def resolve_usr_root_from_env(*, env_var: str = "DNADESIGN_USR_ROOT", pkg_root: Path | None = None) -> Path | None:
    """Resolve the configured USR root from an environment variable, if present."""
    value = str(os.environ.get(env_var, "")).strip()
    if not value:
        return None
    return normalize_usr_root(value, pkg_root=pkg_root)


def resolve_usr_root_from_config(
    root: object,
    *,
    config_path: Path,
    label: str,
    pkg_root: Path | None = None,
) -> Path | None:
    """
    Resolve an optional USR root declared in a config file relative to that
    file's directory, then normalize package-root inputs to datasets roots.
    """
    if root is None:
        return None
    text = str(root).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return normalize_usr_root(candidate.resolve(), pkg_root=pkg_root)
    resolved_config_path = config_path.expanduser().resolve()
    return normalize_usr_root((resolved_config_path.parent / candidate).resolve(), pkg_root=pkg_root)


__all__ = [
    "default_usr_root",
    "normalize_usr_root",
    "pkg_usr_root",
    "require_explicit_usr_root",
    "resolve_usr_root_from_config",
    "resolve_usr_root_from_env",
]
