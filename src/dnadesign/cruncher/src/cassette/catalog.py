"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/catalog.py

Nickase catalog loading for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.cassette.errors import NickaseCatalogError
from dnadesign.cruncher.cassette.load import resolve_workspace_relative_path
from dnadesign.cruncher.cassette.models import HairpinCassetteSpec, NickaseCatalog, NickaseCatalogDocument


def resolve_catalog_path(spec: HairpinCassetteSpec, *, workspace_root: Path) -> Path:
    return resolve_workspace_relative_path(spec.catalog.path, workspace_root=workspace_root, label="catalog.path")


def load_nickase_catalog(path: Path) -> NickaseCatalog:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Nickase catalog not found: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NickaseCatalogError(f"Invalid YAML in nickase catalog {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NickaseCatalogError(f"Nickase catalog {resolved} must be a YAML mapping with top-level key 'nickases'.")
    try:
        document = NickaseCatalogDocument.model_validate(payload)
    except Exception as exc:
        raise NickaseCatalogError(f"Nickase catalog validation failed for {resolved}: {exc}") from exc
    return document.nickases
