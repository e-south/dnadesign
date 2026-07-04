"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/paths.py

Path, YAML, and hash helpers for contact-geometry materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.constants import (
    _DEFAULT_OUTPUT_ROOT,
)

try:
    from yaml import CSafeDumper as _SafeDumper
    from yaml import CSafeLoader as _SafeLoader
except ImportError:  # pragma: no cover - depends on optional LibYAML bindings
    from yaml import SafeDumper as _SafeDumper
    from yaml import SafeLoader as _SafeLoader


def resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    """Resolve the runtime output root against the repository root."""

    resolved = output_root or repo_root / _DEFAULT_OUTPUT_ROOT
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def resolve_source_ref(repo_root: Path, source_ref: str) -> Path:
    """Resolve repo-local and sibling source references."""

    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (repo_root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def require_hash(path: Path, expected_sha256: str) -> None:
    """Fail when a materialization input hash differs from the selected source contract."""

    if not path.exists():
        raise FileNotFoundError(path)
    observed = sha256(path)
    expected = expected_sha256.removeprefix("sha256:")
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: {observed} != {expected}")


def sha256(path: Path) -> str:
    """Return a SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_SafeLoader)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def dump_yaml(payload: object) -> str:
    """Dump YAML with safe representers while preserving insertion order."""

    return yaml.dump(payload, Dumper=_SafeDumper, sort_keys=False)


def write_yaml(path: Path, payload: object) -> None:
    """Write a safe YAML payload to disk."""

    path.write_text(dump_yaml(payload), encoding="utf-8")


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a value to be a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def require_text(payload: Mapping[str, Any], field: str) -> str:
    """Require a non-empty string field from a mapping."""

    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def find_repo_root(start: Path) -> Path:
    """Find the nearest repository root from a starting path."""

    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
