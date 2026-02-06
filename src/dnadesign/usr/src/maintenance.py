"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/maintenance.py

Maintenance context enforcement for destructive USR operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from .errors import SchemaError


@dataclass(frozen=True)
class MaintenanceContext:
    reason: Optional[str]
    actor: Optional[dict]


_MAINTENANCE_CTX: contextvars.ContextVar[MaintenanceContext | None] = contextvars.ContextVar(
    "usr_maintenance_ctx",
    default=None,
)


def current_maintenance() -> MaintenanceContext | None:
    return _MAINTENANCE_CTX.get()


@contextmanager
def maintenance(reason: Optional[str] = None, *, actor: Optional[dict] = None) -> Iterator[MaintenanceContext]:
    ctx = MaintenanceContext(reason=reason, actor=actor)
    token = _MAINTENANCE_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _MAINTENANCE_CTX.reset(token)


def require_maintenance(op_name: str) -> MaintenanceContext:
    ctx = current_maintenance()
    if ctx is None:
        raise SchemaError(f"{op_name} is a maintenance-only operation. Use Dataset.maintenance(...).")
    return ctx
