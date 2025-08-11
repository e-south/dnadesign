"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/__init__.py

Protocol registry: register() decorator and get_protocol() factory.
Also provides lazy import on cache miss so implementations can be loaded
on demand by name (e.g., "scan_dna", "scan_codon", "scan_stem_loop").

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from typing import Dict, Type

from .base import Protocol

_REGISTRY: Dict[str, Type[Protocol]] = {}


def register(cls: Type[Protocol]) -> Type[Protocol]:
    if not getattr(cls, "id", None):
        raise ValueError(
            f"Protocol class {cls.__name__} must define class attribute 'id'"
        )
    _REGISTRY[cls.id] = cls
    return cls


def get_protocol(name: str) -> Protocol:
    if name not in _REGISTRY:
        # lazy-load module by conventional name
        importlib.import_module(f"dnadesign.permuter.protocols.{name}")
    if name not in _REGISTRY:
        raise KeyError(f"Unknown protocol '{name}' (not registered)")
    return _REGISTRY[name]()  # instantiate a fresh instance
