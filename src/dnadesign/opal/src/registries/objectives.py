"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/objectives.py

Objective registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..exceptions import RegistryError


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    fn: Callable[..., object]


_REG: Dict[str, ObjectiveSpec] = {}


def register_objective(name: str):
    def _wrap(fn: Callable[..., object]):
        if name in _REG:
            raise RegistryError(f"Duplicate objective: {name}")
        _REG[name] = ObjectiveSpec(name=name, fn=fn)
        return fn

    return _wrap


def get_objective(name: str) -> Callable[..., object]:
    try:
        return _REG[name].fn
    except KeyError:
        raise RegistryError(f"Unknown objective: {name}. Choices: {list(_REG)}")


def list_objectives() -> list[str]:
    return sorted(_REG.keys())
