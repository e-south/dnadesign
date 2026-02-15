"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/transforms.py

Record transform/plugin loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Protocol, Sequence

from ..config import PluginSpec
from ..core import PluginError, Record
from .attach_motifs_from_config import AttachMotifsFromConfigTransform
from .attach_motifs_from_cruncher_lockfile import AttachMotifsFromCruncherLockfileTransform
from .attach_motifs_from_library import AttachMotifsFromLibraryTransform
from .sigma70 import Sigma70Transform


class Transform(Protocol):
    def apply(self, record: Record) -> Record: ...


def _load_class(spec: str):
    mod_name, _, obj_name = spec.partition(":")
    if not obj_name:
        raise PluginError(f"Transform spec must be 'module:Class', got: {spec}")
    module = import_module(mod_name)
    cls = getattr(module, obj_name, None)
    if cls is None:
        raise PluginError(f"Could not find transform class '{obj_name}' in module '{mod_name}'")
    return cls


def load_transforms(requested: Sequence[PluginSpec]) -> tuple[Transform, ...]:
    transforms: list[Transform] = []
    for spec in requested:
        name = spec.name
        params = spec.params
        if name == "sigma70":
            transforms.append(Sigma70Transform(**params))
            continue
        if name == "attach_motifs_from_config":
            transforms.append(AttachMotifsFromConfigTransform(**params))
            continue
        if name == "attach_motifs_from_cruncher_lockfile":
            transforms.append(AttachMotifsFromCruncherLockfileTransform(**params))
            continue
        if name == "attach_motifs_from_library":
            transforms.append(AttachMotifsFromLibraryTransform(**params))
            continue

        if ":" in name:
            cls = _load_class(name)
            transforms.append(cls(**params))
            continue

        eps = entry_points(group="baserender.plugins")
        found = [ep for ep in eps if ep.name == name]
        if not found:
            raise PluginError(
                f"Unknown transform '{name}'. "
                "Use 'sigma70', 'module:Class', or an installed "
                "baserender.plugins entry point."
            )
        if len(found) > 1:
            raise PluginError(f"Multiple entry points found for transform '{name}'")
        cls = found[0].load()
        transforms.append(cls(**params))
    return tuple(transforms)


def apply_transforms(record: Record, transforms: Sequence[Transform]) -> Record:
    out = record
    for transform in transforms:
        out = transform.apply(out)
    return out
