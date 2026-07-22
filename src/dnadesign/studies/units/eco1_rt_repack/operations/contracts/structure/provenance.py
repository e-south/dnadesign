"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/provenance.py

Shared structure-provenance helpers for Eco1 RT repack contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


def validate_upstream_artifact_hashes(
    recorded_hashes: Mapping[str, Any] | None,
    expected_paths: Mapping[str, Path],
    *,
    path: Path,
    check_id: str,
    artifact_label: str,
) -> list[ContractIssue]:
    """Validate recorded sha256 URIs against current upstream files."""

    issues: list[ContractIssue] = []
    if not isinstance(recorded_hashes, Mapping):
        return [
            ContractIssue(
                check_id=check_id,
                message=f"{artifact_label} must declare upstream_artifact_hashes as a mapping",
                path=str(path),
            )
        ]
    for key, upstream_path in expected_paths.items():
        recorded = recorded_hashes.get(key)
        if not isinstance(recorded, str) or not recorded.startswith("sha256:"):
            issues.append(
                ContractIssue(
                    check_id=check_id,
                    message=f"{artifact_label} must record sha256 hash for upstream artifact {key!r}",
                    path=f"{path}:upstream_artifact_hashes.{key}",
                )
            )
            continue
        if not upstream_path.exists():
            issues.append(
                ContractIssue(
                    check_id=check_id,
                    message=f"{artifact_label} upstream artifact {key!r} is missing: {upstream_path}",
                    path=f"{path}:upstream_artifact_hashes.{key}",
                )
            )
            continue
        observed = sha256_uri(upstream_path)
        if recorded != observed:
            issues.append(
                ContractIssue(
                    check_id=check_id,
                    message=f"{artifact_label} upstream artifact {key!r} hash is stale",
                    path=f"{path}:upstream_artifact_hashes.{key}",
                )
            )
    return issues


def json_metadata_mapping(value: bytes | None) -> dict[str, Any] | None:
    """Parse a Parquet schema JSON metadata mapping."""

    if not value:
        return None
    try:
        loaded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail if the file is not object-shaped."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def resolve_contract_ref(repo_root: Path, source_ref: str) -> Path:
    """Resolve repo, sibling, absolute, or cwd-relative contract references."""

    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (repo_root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def sha256_uri(path: Path) -> str:
    """Return a sha256 URI for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
