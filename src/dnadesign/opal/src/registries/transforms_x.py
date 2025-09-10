"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_x.py

Transforms-X registry (raw -> model-ready X).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..exceptions import RegistryError


@dataclass(frozen=True)
class RepTransformSpec:
    name: str
    factory: Callable[
        [dict], Any
    ]  # returns object with .transform(series)->(np.ndarray, d)


_REG: Dict[str, RepTransformSpec] = {}


def register_rep_transform(name: str):
    def _wrap(factory: Callable[[dict], Any]):
        if name in _REG:
            raise RegistryError(f"Duplicate rep transform: {name}")
        _REG[name] = RepTransformSpec(name, factory)
        return factory

    return _wrap


def get_rep_transform(name: str, params: dict):
    try:
        return _REG[name].factory(params)
    except KeyError:
        raise RegistryError(f"Unknown rep transform: {name}. Choices: {list(_REG)}")
