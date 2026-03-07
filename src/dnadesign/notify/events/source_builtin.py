"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/events/source_builtin.py

Built-in tool config resolvers for notify USR events source discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dnadesign._contracts import (
    resolve_densegen_usr_output_contract,
    resolve_infer_usr_output_contract,
)

from ..errors import NotifyConfigError

ToolEventsSourceRegister = Callable[..., None]


def _resolve_densegen_events_from_config(config_path: Path) -> Path:
    try:
        contract = resolve_densegen_usr_output_contract(config_path)
    except ValueError as exc:
        raise NotifyConfigError(str(exc)) from exc
    return (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def _resolve_infer_events_from_config(config_path: Path) -> Path:
    try:
        contract = resolve_infer_usr_output_contract(config_path)
    except ValueError as exc:
        raise NotifyConfigError(str(exc)) from exc
    return (contract.usr_root / contract.usr_dataset / ".events.log").resolve()


def register_builtin_tool_events_sources(register: ToolEventsSourceRegister) -> None:
    if not callable(register):
        raise TypeError("register must be callable")
    register(
        tool="densegen",
        resolver=_resolve_densegen_events_from_config,
        default_policy="densegen",
    )
    register(
        tool="infer",
        resolver=_resolve_infer_events_from_config,
        default_policy="infer",
    )
