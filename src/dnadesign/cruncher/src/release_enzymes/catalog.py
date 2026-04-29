"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/release_enzymes/catalog.py

Release-enzyme catalog loading and merging for released-product snapback
workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.nickases.catalog import resolve_workspace_relative_path
from dnadesign.cruncher.release_enzymes.errors import ReleaseEnzymeCatalogError
from dnadesign.cruncher.release_enzymes.models import (
    ReleaseEnzymeCatalog,
    ReleaseEnzymeCatalogDocument,
)

_PRESET_RESOURCE_ROOT = ("resources", "cassette", "catalogs")


def _normalize_preset_ids(value: Any, *, label: str) -> list[str]:
    raw_ids = value or []
    if not isinstance(raw_ids, list):
        raise ReleaseEnzymeCatalogError(f"{label} must be a list when present.")
    preset_ids = [str(item or "").strip() for item in raw_ids]
    if any(not item for item in preset_ids):
        raise ReleaseEnzymeCatalogError(f"{label} must not contain blank values.")
    if len(set(preset_ids)) != len(preset_ids):
        raise ReleaseEnzymeCatalogError(f"{label} must not repeat preset ids.")
    return preset_ids


def _load_catalog_from_mapping(payload: dict[str, Any], *, source_label: str) -> ReleaseEnzymeCatalog:
    try:
        document = ReleaseEnzymeCatalogDocument.model_validate(payload)
    except Exception as exc:
        raise ReleaseEnzymeCatalogError(f"Release-enzyme catalog validation failed for {source_label}: {exc}") from exc
    return document.release_enzymes


def _normalize_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "release_enzymes" in payload:
        release_root = payload["release_enzymes"]
        if not isinstance(release_root, dict):
            raise ReleaseEnzymeCatalogError("release_enzymes must be a mapping.")
        return {"release_enzymes": release_root}
    variants = payload.get("variants")
    if variants is None:
        raise ReleaseEnzymeCatalogError(
            "Release-enzyme catalog must define top-level key 'release_enzymes' or 'variants'."
        )
    if not isinstance(variants, list):
        raise ReleaseEnzymeCatalogError("Preset release-enzyme variants must be a list.")
    for item in variants:
        if not isinstance(item, dict):
            raise ReleaseEnzymeCatalogError("Preset release-enzyme variants must contain mappings.")
    normalized = {
        "release_enzymes": {
            "schema_version": int(payload.get("schema_version", 1)),
            "preset_id": payload.get("preset_id"),
            "preset_ids": _normalize_preset_ids(payload.get("preset_ids"), label="preset_ids"),
            "catalog_version": payload.get("catalog_version"),
            "generated_from": payload.get("generated_from"),
            "generated_on": payload.get("generated_on"),
            "normalization_policy": payload.get("normalization_policy"),
            "entries": variants,
        }
    }
    return normalized


def _load_catalog_from_text(text: str, *, source_label: str) -> ReleaseEnzymeCatalog:
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ReleaseEnzymeCatalogError(f"Invalid YAML in release-enzyme catalog {source_label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEnzymeCatalogError(f"Release-enzyme catalog {source_label} must be a YAML mapping.")
    normalized = _normalize_catalog_payload(payload)
    return _load_catalog_from_mapping(normalized, source_label=source_label)


def resolve_builtin_catalog_resource(preset_id: str) -> resources.abc.Traversable:
    resource = resources.files("dnadesign.cruncher")
    for part in _PRESET_RESOURCE_ROOT:
        resource = resource.joinpath(part)
    resource = resource.joinpath(f"{preset_id}.yaml")
    if not resource.is_file():
        raise ReleaseEnzymeCatalogError(f"Unknown built-in release-enzyme preset: {preset_id}")
    return resource


def load_builtin_release_enzyme_catalog_preset(preset_id: str) -> ReleaseEnzymeCatalog:
    resource = resolve_builtin_catalog_resource(preset_id)
    return _load_catalog_from_text(resource.read_text(encoding="utf-8"), source_label=f"preset:{preset_id}")


def read_builtin_release_enzyme_catalog_preset_text(preset_id: str) -> str:
    resource = resolve_builtin_catalog_resource(preset_id)
    return resource.read_text(encoding="utf-8")


def load_release_enzyme_catalog(path: Path) -> ReleaseEnzymeCatalog:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Release-enzyme catalog not found: {resolved}")
    return _load_catalog_from_text(resolved.read_text(encoding="utf-8"), source_label=str(resolved))


def merge_release_enzyme_catalogs(*catalogs: ReleaseEnzymeCatalog) -> ReleaseEnzymeCatalog:
    entries = []
    seen_variant_ids: set[str] = set()
    preset_id: str | None = None
    preset_ids: list[str] = []
    catalog_version: int | None = None
    generated_from: str | None = None
    generated_on: str | None = None
    normalization_policy: str | None = None
    for catalog in catalogs:
        if catalog.preset_id is not None and preset_id is None:
            preset_id = catalog.preset_id
        for catalog_preset_id in catalog.preset_ids:
            if catalog_preset_id not in preset_ids:
                preset_ids.append(catalog_preset_id)
        if catalog.catalog_version is not None and catalog_version is None:
            catalog_version = catalog.catalog_version
        if catalog.generated_from is not None and generated_from is None:
            generated_from = catalog.generated_from
        if catalog.generated_on is not None and generated_on is None:
            generated_on = catalog.generated_on
        if catalog.normalization_policy is not None and normalization_policy is None:
            normalization_policy = catalog.normalization_policy
        for entry in catalog.entries:
            if entry.variant_id in seen_variant_ids:
                raise ReleaseEnzymeCatalogError(
                    f"Duplicate release-enzyme variant id across merged catalogs: {entry.variant_id}"
                )
            entries.append(entry)
            seen_variant_ids.add(entry.variant_id)
    return ReleaseEnzymeCatalog(
        schema_version=1,
        entries=entries,
        preset_id=preset_id,
        preset_ids=preset_ids,
        catalog_version=catalog_version,
        generated_from=generated_from,
        generated_on=generated_on,
        normalization_policy=normalization_policy,
    )


def load_merged_release_enzyme_catalog(
    *,
    preset_id: str | None,
    additional_preset_ids: list[str] | None = None,
    additional_paths: list[Path],
    workspace_root: Path,
) -> tuple[ReleaseEnzymeCatalog, list[Path]]:
    catalogs: list[ReleaseEnzymeCatalog] = []
    resolved_paths: list[Path] = []
    for builtin_preset_id in [preset_id, *(additional_preset_ids or [])]:
        if builtin_preset_id:
            catalogs.append(load_builtin_release_enzyme_catalog_preset(builtin_preset_id))
    for raw_path in additional_paths:
        resolved = resolve_workspace_relative_path(
            raw_path,
            workspace_root=workspace_root,
            label="release_sources.additional_paths",
        )
        resolved_paths.append(resolved)
        catalogs.append(load_release_enzyme_catalog(resolved))
    if not catalogs:
        raise ReleaseEnzymeCatalogError(
            "Release-enzyme sources must define at least one preset or additional catalog path."
        )
    return merge_release_enzyme_catalogs(*catalogs), resolved_paths


def dump_release_enzyme_catalog_payload(catalog: ReleaseEnzymeCatalog) -> dict[str, Any]:
    return {
        "release_enzymes": {
            "schema_version": catalog.schema_version,
            "preset_id": catalog.preset_id,
            "preset_ids": list(catalog.preset_ids),
            "catalog_version": catalog.catalog_version,
            "generated_from": catalog.generated_from,
            "generated_on": catalog.generated_on,
            "normalization_policy": catalog.normalization_policy,
            "entries": [entry.model_dump(mode="json") for entry in catalog.entries],
        }
    }


def dump_release_enzyme_catalog_yaml(catalog: ReleaseEnzymeCatalog) -> str:
    return yaml.safe_dump(dump_release_enzyme_catalog_payload(catalog), sort_keys=False)


__all__ = [
    "dump_release_enzyme_catalog_payload",
    "dump_release_enzyme_catalog_yaml",
    "load_builtin_release_enzyme_catalog_preset",
    "load_merged_release_enzyme_catalog",
    "load_release_enzyme_catalog",
    "merge_release_enzyme_catalogs",
    "read_builtin_release_enzyme_catalog_preset_text",
    "resolve_builtin_catalog_resource",
]
