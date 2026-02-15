"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_event_packs_builtin.py

Built-in tool-event pack installers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable

from .tool_events_densegen import register_densegen_handlers

ToolEventRegister = Callable[..., None]
ToolEventPackRegister = Callable[..., None]


def _install_densegen_pack(register: ToolEventRegister) -> None:
    register_densegen_handlers(
        register_status_override=lambda action, override: register(action=action, status_override=override),
        register_message_override=lambda action, renderer: register(action=action, message_override=renderer),
        register_evaluator=lambda action, evaluator: register(action=action, evaluator=evaluator),
    )


def register_builtin_tool_event_packs(register_pack: ToolEventPackRegister) -> None:
    if not callable(register_pack):
        raise TypeError("register_pack must be callable")
    register_pack(pack="densegen", installer=_install_densegen_pack)
