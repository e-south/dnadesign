"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/src/cli/support/resolution/roots.py

Internal SeqLedger root-resolution helpers for canonical dataset storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def pkg_seqledger_root() -> Path:
    """Return the installed dnadesign/seqledger package directory."""
    return Path(__file__).resolve().parents[4]


def default_seqledger_root(*, pkg_root: Path | None = None) -> Path:
    """Return the project-local data root without creating it.

    ``pkg_root`` is retained for callers sharing root-resolution options; package
    location no longer determines the default writable data location.
    """
    return (Path.cwd() / "seqledger_datasets").resolve()


def normalize_seqledger_root(root: str | Path | None, *, pkg_root: Path | None = None) -> Path:
    """
    Accept either the package root (.../dnadesign/seqledger) or the datasets root
    (.../dnadesign/seqledger/datasets) and normalize to the canonical datasets root.
    """
    if root is None:
        return default_seqledger_root(pkg_root=pkg_root)

    target = Path(root).expanduser().resolve()
    base = Path(pkg_root).resolve() if pkg_root is not None else pkg_seqledger_root()
    if target == base:
        return (base / "datasets").resolve()
    if (target / "__init__.py").exists():
        return (target / "datasets").resolve()
    return target


def resolve_seqledger_root_from_env(
    *, env_var: str = "DNADESIGN_SEQLEDGER_ROOT", pkg_root: Path | None = None
) -> Path | None:
    """Resolve the configured SeqLedger root from an environment variable, if present."""
    value = str(os.environ.get(env_var, "")).strip()
    if not value:
        return None
    return normalize_seqledger_root(value, pkg_root=pkg_root)


def resolve_seqledger_root_from_config(
    root: object,
    *,
    config_path: Path,
    label: str,
    pkg_root: Path | None = None,
) -> Path | None:
    """
    Resolve an optional SeqLedger root declared in a config file relative to that
    file's directory, then normalize package-root inputs to datasets roots.
    """
    if root is None:
        return None
    text = str(root).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return normalize_seqledger_root(candidate.resolve(), pkg_root=pkg_root)
    resolved_config_path = config_path.expanduser().resolve()
    return normalize_seqledger_root((resolved_config_path.parent / candidate).resolve(), pkg_root=pkg_root)


__all__ = [
    "default_seqledger_root",
    "normalize_seqledger_root",
    "pkg_seqledger_root",
    "resolve_seqledger_root_from_config",
    "resolve_seqledger_root_from_env",
]
