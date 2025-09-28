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

from typing import Any, Dict, List, Protocol


class _ModelFactory(Protocol):
    def __call__(
        self, params: Dict[str, Any], target_normalizer: Dict[str, Any] | None = None
    ): ...


# Registry: name -> factory(params, target_normalizer) -> model_instance
_REG_M: Dict[str, _ModelFactory] = {}


def register_model(name: str):
    """Decorator to register a model factory. The factory is called with (params, target_normalizer)."""

    def _wrap(factory: _ModelFactory):
        if name in _REG_M:
            raise ValueError(f"model '{name}' already registered")
        _REG_M[name] = factory
        return factory

    return _wrap


def get_model(
    name: str, params: Dict[str, Any], target_normalizer: Dict[str, Any] | None = None
):
    """Instantiate a model via its registered factory.

    We try common call patterns to be tolerant of factory shapes:
      • factory(params, target_normalizer)
      • factory(params)
      • class with .from_params(params, target_normalizer) or .from_params(params)
    """
    if name not in _REG_M:
        raise KeyError(f"model '{name}' not found. Available: {sorted(_REG_M)}")
    factory = _REG_M[name]
    # try direct factory(params, target_normalizer)
    try:
        return factory(params=params, target_normalizer=target_normalizer)
    except TypeError:
        pass
    # try factory(params)
    try:
        return factory(params=params)  # type: ignore[call-arg]
    except TypeError:
        pass
    # try classmethod from_params
    if hasattr(factory, "from_params"):
        fp = getattr(factory, "from_params")
        try:
            return fp(params, target_normalizer)  # type: ignore[misc]
        except TypeError:
            return fp(params)  # type: ignore[misc]
    # final fall-back: call with no kwargs
    try:
        return factory(params, target_normalizer)  # type: ignore[misc]
    except Exception as e:
        raise TypeError(f"cannot construct model '{name}': {e}") from e


def list_models() -> List[str]:
    return sorted(_REG_M)
