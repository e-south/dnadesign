"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/plugins/registry.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Mapping, Protocol, Sequence, Union

from ..contracts import PluginError
from ..model import SeqRecord


class DerivedAnnotationPlugin(Protocol):
    name: str

    def apply(self, record: SeqRecord) -> SeqRecord: ...


class PalettePlugin(Protocol):
    name: str

    def color_for(self, tag: str): ...


@dataclass(frozen=True)
class PluginSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


PluginLike = Union[str, Mapping[str, Any], PluginSpec]


def _as_specs(requested: Sequence[PluginLike]) -> list[PluginSpec]:
    specs: list[PluginSpec] = []
    for item in requested:
        if isinstance(item, PluginSpec):
            specs.append(item)
        elif isinstance(item, str):
            specs.append(PluginSpec(name=item, params={}))
        elif isinstance(item, Mapping):
            if not item:
                raise PluginError("Plugin mapping cannot be empty.")
            if len(item) != 1:
                raise PluginError(f"Plugin mapping must have a single key: {item}")
            name, params = next(iter(item.items()))
            if not isinstance(params, Mapping):
                raise PluginError(f"Plugin params must be a mapping for '{name}'")
            specs.append(PluginSpec(name=str(name), params=dict(params)))
        else:
            raise PluginError(f"Unsupported plugin spec type: {type(item).__name__}")
    return specs


def _load_by_module(path: str):
    mod_name, _, obj = path.partition(":")
    if not obj:
        raise PluginError(f"Plugin spec must be 'module:object', got {path}")
    mod = import_module(mod_name)
    plugin = getattr(mod, obj, None)
    if plugin is None:
        raise PluginError(f"Could not find object '{obj}' in module '{mod_name}'")
    return plugin


def load_plugins(requested: Sequence[PluginLike]) -> Sequence[DerivedAnnotationPlugin]:
    specs = _as_specs(requested)
    plugins: list[DerivedAnnotationPlugin] = []
    for spec in specs:
        name, params = spec.name, spec.params
        if name == "sigma70":
            from .builtin.sigma70 import Sigma70Plugin

            plugins.append(Sigma70Plugin(**params))
        elif ":" in name:
            cls = _load_by_module(name)
            plugins.append(cls(**params))
        else:
            eps = entry_points(group="baserender.plugins")
            found = False
            for ep in eps:
                if ep.name == name:
                    cls = ep.load()
                    plugins.append(cls(**params))
                    found = True
                    break
            if not found:
                raise PluginError(
                    f"Unknown plugin '{name}'. Use 'module:Class', an installed entry point name, "
                    "or a builtin ('sigma70')."
                )
    return plugins
