"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/contracts.py

Small contract helpers for strict validation and unknown-key rejection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

from .errors import ContractError


def ensure(cond: bool, msg: str, exc: type[Exception] = ContractError) -> None:
    if not cond:
        raise exc(msg)


def require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ContractError(f"{ctx} must be a mapping/dict")
    return obj


def reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str] | Set[str], ctx: str) -> None:
    extra = sorted(set(mapping.keys()) - set(allowed))
    if extra:
        raise ContractError(f"Unknown keys in {ctx}: {extra}")


def require_one_of(val: str, allowed_set: set[str] | Set[str], ctx: str) -> None:
    if val not in set(allowed_set):
        allowed = "|".join(sorted(allowed_set))
        raise ContractError(f"{ctx} must be one of: {allowed}")
