"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/catalog.py

Validated loading for optional YIU protocol catalogs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.load import resolve_workspace_relative_path
from dnadesign.cruncher.yiu.models import (
    YiuAdapterCatalogDocument,
    YiuAdapterCatalogEntry,
    YiuBackboneCatalogDocument,
    YiuBackboneCatalogEntry,
    YiuEnzymeCatalogEntry,
    YiuGenericEnzymeCatalogDocument,
    YiuNickaseCatalogDocument,
    YiuOligoPartCatalogDocument,
    YiuOligoPartCatalogEntry,
    YiuProcessSpec,
    YiuProcessSpecV2,
    YiuRestrictionCatalogDocument,
)


@dataclass(frozen=True)
class LoadedYiuCatalogs:
    restriction_enzymes: dict[str, YiuEnzymeCatalogEntry] = field(default_factory=dict)
    nickases: dict[str, YiuEnzymeCatalogEntry] = field(default_factory=dict)
    adapters: dict[str, YiuAdapterCatalogEntry] = field(default_factory=dict)
    enzymes: dict[str, YiuEnzymeCatalogEntry] = field(default_factory=dict)
    oligo_parts: dict[str, YiuOligoPartCatalogEntry] = field(default_factory=dict)
    backbones: dict[str, YiuBackboneCatalogEntry] = field(default_factory=dict)
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


def load_yiu_catalogs(spec: YiuProcessSpec | YiuProcessSpecV2, *, workspace_root: Path) -> LoadedYiuCatalogs:
    if isinstance(spec, YiuProcessSpecV2):
        enzyme_path = _resolve_catalog_path(
            spec.catalogs.enzymes,
            workspace_root=workspace_root,
            label="catalogs.enzymes",
        )
        oligo_parts_path = _resolve_catalog_path(
            spec.catalogs.oligo_parts,
            workspace_root=workspace_root,
            label="catalogs.oligo_parts",
        )
        backbone_path = _resolve_catalog_path(
            spec.catalogs.backbones,
            workspace_root=workspace_root,
            label="catalogs.backbones",
        )

        enzyme_entries: dict[str, YiuEnzymeCatalogEntry] = {}
        oligo_part_entries: dict[str, YiuOligoPartCatalogEntry] = {}
        backbone_entries: dict[str, YiuBackboneCatalogEntry] = {}

        if enzyme_path is not None:
            payload = _load_yaml_mapping(enzyme_path, label="enzyme")
            try:
                document = YiuGenericEnzymeCatalogDocument.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"YIU enzyme catalog validation failed for {enzyme_path}: {exc}") from exc
            enzyme_entries = {entry.id: entry for entry in document.enzymes.entries}

        if oligo_parts_path is not None:
            payload = _load_yaml_mapping(oligo_parts_path, label="oligo_parts")
            try:
                document = YiuOligoPartCatalogDocument.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"YIU oligo-parts catalog validation failed for {oligo_parts_path}: {exc}") from exc
            oligo_part_entries = {entry.id: entry for entry in document.oligo_parts.entries}

        if backbone_path is not None:
            payload = _load_yaml_mapping(backbone_path, label="backbone")
            try:
                document = YiuBackboneCatalogDocument.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"YIU backbone catalog validation failed for {backbone_path}: {exc}") from exc
            backbone_entries = {entry.id: entry for entry in document.backbones.entries}

        paths = tuple(path for path in (enzyme_path, oligo_parts_path, backbone_path) if path is not None)
        return LoadedYiuCatalogs(
            restriction_enzymes={},
            nickases={},
            adapters={},
            enzymes=enzyme_entries,
            oligo_parts=oligo_part_entries,
            backbones=backbone_entries,
            paths=paths,
        )

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
        enzymes={},
        oligo_parts={},
        backbones={},
        paths=paths,
    )
