"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/provider_sources.py

Owner-confined procedure metadata from installed Ops provider packages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from importlib.util import find_spec
from pathlib import Path

from .constants import REGISTRY_METADATA_SUFFIX

_CATALOG_REGISTRY_ENTRY_POINT_GROUP = "dnadesign.ops.catalog_registries"


@dataclass(frozen=True)
class CatalogRegistrySource:
    """One provider-owned procedure sidecar and its package root."""

    provider_id: str
    path: Path
    package_root: Path


def discover_external_catalog_registry_sources() -> tuple[CatalogRegistrySource, ...]:
    """Load procedure sidecars registered by installed provider packages."""

    sources: list[CatalogRegistrySource] = []
    seen_paths: set[Path] = set()
    discovered = entry_points(group=_CATALOG_REGISTRY_ENTRY_POINT_GROUP)
    for entry_point in sorted(discovered, key=lambda item: (item.name, item.value)):
        loader = entry_point.load()
        if not callable(loader):
            raise ValueError(f"catalog registry entry point must resolve to a callable: {entry_point.name}")
        provider_id = _entry_point_provider_id(entry_point=entry_point)
        package_root = _entry_point_package_root(entry_point=entry_point)
        raw_paths = loader()
        if isinstance(raw_paths, (str, Path)):
            raw_paths = (raw_paths,)
        if not isinstance(raw_paths, (list, tuple)):
            raise ValueError(f"catalog registry entry point must return paths: {entry_point.name}")
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.name.endswith(REGISTRY_METADATA_SUFFIX) or not path.is_file():
                raise ValueError(
                    f"catalog registry entry point returned an invalid metadata path: {entry_point.name}: {path}"
                )
            if not path.is_relative_to(package_root):
                raise ValueError(
                    "catalog registry path must stay under its entry-point package: "
                    f"{entry_point.name}: {path} package_root={package_root}"
                )
            if path in seen_paths:
                raise ValueError(f"catalog registry path registered more than once: {path}")
            seen_paths.add(path)
            sources.append(CatalogRegistrySource(provider_id=provider_id, path=path, package_root=package_root))
    return tuple(sources)


def _entry_point_provider_id(*, entry_point: EntryPoint) -> str:
    module_name = entry_point.value.partition(":")[0].strip()
    distribution = getattr(entry_point, "dist", None)
    owner = str(getattr(distribution, "name", "") or module_name.partition(".")[0]).strip()
    if not owner:
        raise ValueError(f"catalog registry entry point has no provider identity: {entry_point.name}")
    return f"{owner}:{entry_point.name}"


def _entry_point_package_root(*, entry_point: EntryPoint) -> Path:
    module_name = entry_point.value.partition(":")[0].strip()
    module_parts = tuple(part for part in module_name.split(".") if part)
    if not module_parts:
        raise ValueError(f"catalog registry entry point has no import package: {entry_point.name}")
    spec = find_spec(module_name)
    if spec is None:
        raise ValueError(f"catalog registry entry-point module cannot be resolved: {entry_point.name}: {module_name}")
    origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen"}:
        raise ValueError(
            f"catalog registry entry point must resolve to a package file: {entry_point.name}: {module_name}"
        )
    package_root = Path(origin).expanduser().resolve().parent
    is_package = bool(getattr(spec, "submodule_search_locations", None))
    parent_steps = len(module_parts) - (1 if is_package else 2)
    if parent_steps < 0:
        raise ValueError(
            f"catalog registry entry point must belong to an import package: {entry_point.name}: {module_name}"
        )
    for _ in range(parent_steps):
        package_root = package_root.parent
    if package_root.name != module_parts[0]:
        raise ValueError(
            f"catalog registry package root is ambiguous: {entry_point.name}: {module_name}: {package_root}"
        )
    return package_root


__all__ = ["CatalogRegistrySource", "discover_external_catalog_registry_sources"]
