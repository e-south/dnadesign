"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_y.py

Transforms-Y registry (raw -> model-ready y).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..exceptions import RegistryError


@dataclass(frozen=True)
class IngestTransformSpec:
    name: str
    fn: Callable[
        [Any, dict, list[float]], Any
    ]  # (df_tidy, params, setpoint) -> df[id,y]


_REG: Dict[str, IngestTransformSpec] = {}


def register_ingest_transform(name: str):
    def _wrap(fn: Callable[[Any, dict, list[float]], Any]):
        if name in _REG:
            raise RegistryError(f"Duplicate ingest transform: {name}")
        _REG[name] = IngestTransformSpec(name, fn)
        return fn

    return _wrap


def get_ingest_transform(name: str) -> Callable[..., Any]:
    try:
        return _REG[name].fn
    except KeyError:
        raise RegistryError(f"Unknown ingest transform: {name}. Choices: {list(_REG)}")
