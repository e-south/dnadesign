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
        package_roots = _entry_point_package_roots(entry_point=entry_point)
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
            package_root = next((root for root in package_roots if path.is_relative_to(root)), None)
            if package_root is None:
                raise ValueError(
                    "catalog registry path must stay under its entry-point package: "
                    f"{entry_point.name}: {path} package_roots={package_roots}"
                )
            if path in seen_paths:
                raise ValueError(f"catalog registry path registered more than once: {path}")
            seen_paths.add(path)
            sources.append(CatalogRegistrySource(path=path, package_root=package_root))
    return tuple(sources)


def _entry_point_package_roots(*, entry_point: EntryPoint) -> tuple[Path, ...]:
    module_name = entry_point.value.partition(":")[0].strip()
    top_level_package = module_name.partition(".")[0]
    if not top_level_package:
        raise ValueError(f"catalog registry entry point has no import package: {entry_point.name}")
    spec = find_spec(top_level_package)
    if spec is None:
        raise ValueError(
            f"catalog registry entry-point package cannot be resolved: {entry_point.name}: {top_level_package}"
        )
    search_locations = tuple(spec.submodule_search_locations or ())
    if search_locations:
        return tuple(sorted({Path(location).expanduser().resolve() for location in search_locations}))
    raise ValueError(
        f"catalog registry entry point must belong to an import package: {entry_point.name}: {top_level_package}"
    )


__all__ = ["CatalogRegistrySource", "discover_external_catalog_registry_sources"]
