"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/catalog.py

Validated loading for optional YIU protocol catalogs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.load import resolve_workspace_relative_path
from dnadesign.cruncher.yiu.models import (
    YiuAdapterCatalogDocument,
    YiuAdapterCatalogEntry,
    YiuEnzymeCatalogEntry,
    YiuNickaseCatalogDocument,
    YiuProcessSpec,
    YiuRestrictionCatalogDocument,
)


@dataclass(frozen=True)
class LoadedYiuCatalogs:
    restriction_enzymes: dict[str, YiuEnzymeCatalogEntry]
    nickases: dict[str, YiuEnzymeCatalogEntry]
    adapters: dict[str, YiuAdapterCatalogEntry]
    paths: tuple[Path, ...] = ()


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {label} catalog {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} catalog {path} must be a YAML mapping.")
    return payload


def _resolve_catalog_path(raw_path: Path | None, *, workspace_root: Path, label: str) -> Path | None:
    if raw_path is None:
        return None
    resolved = resolve_workspace_relative_path(raw_path, workspace_root=workspace_root, label=label)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_yiu_catalogs(spec: YiuProcessSpec, *, workspace_root: Path) -> LoadedYiuCatalogs:
    restriction_path = _resolve_catalog_path(
        spec.catalogs.restriction_enzymes,
        workspace_root=workspace_root,
        label="catalogs.restriction_enzymes",
    )
    nickase_path = _resolve_catalog_path(
        spec.catalogs.nickases,
        workspace_root=workspace_root,
        label="catalogs.nickases",
    )
    adapter_path = _resolve_catalog_path(
        spec.catalogs.adapters,
        workspace_root=workspace_root,
        label="catalogs.adapters",
    )

    restriction_entries: dict[str, YiuEnzymeCatalogEntry] = {}
    nickase_entries: dict[str, YiuEnzymeCatalogEntry] = {}
    adapter_entries: dict[str, YiuAdapterCatalogEntry] = {}

    if restriction_path is not None:
        payload = _load_yaml_mapping(restriction_path, label="restriction")
        try:
            document = YiuRestrictionCatalogDocument.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"YIU restriction catalog validation failed for {restriction_path}: {exc}") from exc
        restriction_entries = {entry.id: entry for entry in document.restriction_enzymes.entries}

    if nickase_path is not None:
        payload = _load_yaml_mapping(nickase_path, label="nickase")
        try:
            document = YiuNickaseCatalogDocument.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"YIU nickase catalog validation failed for {nickase_path}: {exc}") from exc
        nickase_entries = {entry.id: entry for entry in document.nickases.entries}

    if adapter_path is not None:
        payload = _load_yaml_mapping(adapter_path, label="adapter")
        try:
            document = YiuAdapterCatalogDocument.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"YIU adapter catalog validation failed for {adapter_path}: {exc}") from exc
        adapter_entries = {entry.id: entry for entry in document.adapters.entries}

    paths = tuple(path for path in (restriction_path, nickase_path, adapter_path) if path is not None)
    return LoadedYiuCatalogs(
        restriction_enzymes=restriction_entries,
        nickases=nickase_entries,
        adapters=adapter_entries,
        paths=paths,
    )
