"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/models.py

Model registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..exceptions import RegistryError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[dict, dict], object]  # (params, target_scaler_cfg) -> model


_REG: Dict[str, ModelSpec] = {}


def register_model(name: str):
    def _wrap(factory: Callable[[dict, dict], object]):
        if name in _REG:
            raise RegistryError(f"Duplicate model: {name}")
        _REG[name] = ModelSpec(name=name, factory=factory)
        return factory

    return _wrap


def get_model(name: str, params: dict, target_scaler_cfg: dict):
    try:
        return _REG[name].factory(params, target_scaler_cfg)
    except KeyError:
        raise RegistryError(f"Unknown model: {name}. Choices: {list(_REG)}")


def list_models() -> list[str]:
    return sorted(_REG.keys())
