"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/load.py

Load cassette specs and resolve workspace-relative paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.cassette.errors import CassetteSpecError
from dnadesign.cruncher.cassette.models import HairpinCassetteSpec, HairpinCassetteSpecDocument


def resolve_workspace_root_for_spec(spec_path: Path) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Cassette spec not found: {resolved}")
    if resolved.name.count(".") < 2 or not resolved.name.endswith(".cassette.yaml"):
        raise CassetteSpecError("--spec must point to a <workspace>/configs/cassettes/<name>.cassette.yaml file.")
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise CassetteSpecError("--spec must live under a workspace configs/ tree.")


def resolve_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise CassetteSpecError(f"{label} must not traverse outside the workspace: {raw_path}")
    return (workspace_root / path).resolve()


def load_cassette_spec(path: str | Path) -> tuple[HairpinCassetteSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_spec(spec_path)
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CassetteSpecError(f"Invalid YAML in cassette spec {spec_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CassetteSpecError(f"Cassette spec {spec_path} must be a YAML mapping with top-level key 'cassette'.")
    try:
        document = HairpinCassetteSpecDocument.model_validate(payload)
    except Exception as exc:
        raise CassetteSpecError(f"Cassette schema validation failed for {spec_path}: {exc}") from exc
    spec = document.cassette
    return spec, spec_path, workspace_root
