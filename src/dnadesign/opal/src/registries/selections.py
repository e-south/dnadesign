"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/selections.py

Selection strategy registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..exceptions import RegistryError


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    fn: Callable[..., object]


_REG: Dict[str, SelectionSpec] = {}


def register_selection(name: str):
    def _wrap(fn: Callable[..., object]):
        if name in _REG:
            raise RegistryError(f"Duplicate selection strategy: {name}")
        _REG[name] = SelectionSpec(name=name, fn=fn)
        return fn

    return _wrap


def get_selection(name: str) -> Callable[..., object]:
    try:
        return _REG[name].fn
    except KeyError:
        raise RegistryError(
            f"Unknown selection strategy: {name}. Choices: {list(_REG)}"
        )


def list_selections() -> list[str]:
    return sorted(_REG.keys())
