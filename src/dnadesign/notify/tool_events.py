"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events.py

Tool-agnostic registration and dispatch for notify tool-event behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import NotifyConfigError
from .tool_event_types import ToolEventDecision, ToolEventState
from .tool_events_densegen import register_densegen_handlers

StatusOverride = Callable[[dict[str, Any]], str | None]
MessageOverride = Callable[..., str]
Evaluator = Callable[[dict[str, Any], str, ToolEventState], ToolEventDecision]


class ToolEventRegistry:
    def __init__(self) -> None:
        self._status_overrides: dict[str, StatusOverride] = {}
        self._message_overrides: dict[str, MessageOverride] = {}
        self._evaluators: dict[str, Evaluator] = {}

    def _normalize_action(self, action: str) -> str:
        value = str(action or "").strip()
        if not value:
            raise NotifyConfigError("tool-event action must be a non-empty string")
        return value

    def register_status_override(self, action: str, override: StatusOverride) -> None:
        action_name = self._normalize_action(action)
        if action_name in self._status_overrides:
            raise NotifyConfigError(f"status override already registered for action '{action_name}'")
        self._status_overrides[action_name] = override

    def register_message_override(self, action: str, renderer: MessageOverride) -> None:
        action_name = self._normalize_action(action)
        if action_name in self._message_overrides:
            raise NotifyConfigError(f"message override already registered for action '{action_name}'")
        self._message_overrides[action_name] = renderer

    def register_evaluator(self, action: str, evaluator: Evaluator) -> None:
        action_name = self._normalize_action(action)
        if action_name in self._evaluators:
            raise NotifyConfigError(f"evaluator already registered for action '{action_name}'")
        self._evaluators[action_name] = evaluator

    def register(
        self,
        *,
        action: str,
        status_override: StatusOverride | None = None,
        message_override: MessageOverride | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        if status_override is None and message_override is None and evaluator is None:
            raise NotifyConfigError("register_tool_event_handlers requires at least one handler")
        if status_override is not None:
            self.register_status_override(action, status_override)
        if message_override is not None:
            self.register_message_override(action, message_override)
        if evaluator is not None:
            self.register_evaluator(action, evaluator)

    def status_override(self, action: str, event: dict[str, Any]) -> str | None:
        override = self._status_overrides.get(action)
        if override is None:
            return None
        return override(event)

    def message_override(
        self,
        action: str,
        event: dict[str, Any],
        *,
        run_id: str,
        duration_seconds: float | None,
    ) -> str | None:
        renderer = self._message_overrides.get(action)
        if renderer is None:
            return None
        return renderer(event, run_id=run_id, duration_seconds=duration_seconds)

    def evaluate(self, action: str, event: dict[str, Any], *, run_id: str, state: ToolEventState) -> ToolEventDecision:
        evaluator = self._evaluators.get(action)
        if evaluator is None:
            return ToolEventDecision(emit=True, duration_seconds=None)
        return evaluator(event, str(run_id), state)


_DEFAULT_REGISTRY = ToolEventRegistry()

register_densegen_handlers(
    register_status_override=_DEFAULT_REGISTRY.register_status_override,
    register_message_override=_DEFAULT_REGISTRY.register_message_override,
    register_evaluator=_DEFAULT_REGISTRY.register_evaluator,
)


def register_tool_event_handlers(
    *,
    action: str,
    status_override: StatusOverride | None = None,
    message_override: MessageOverride | None = None,
    evaluator: Evaluator | None = None,
) -> None:
    _DEFAULT_REGISTRY.register(
        action=action,
        status_override=status_override,
        message_override=message_override,
        evaluator=evaluator,
    )


def tool_event_status_override(action: str, event: dict[str, Any]) -> str | None:
    return _DEFAULT_REGISTRY.status_override(action, event)


def tool_event_message_override(
    action: str,
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str | None:
    return _DEFAULT_REGISTRY.message_override(
        action,
        event,
        run_id=run_id,
        duration_seconds=duration_seconds,
    )


def evaluate_tool_event(action: str, event: dict[str, Any], *, run_id: str, state: ToolEventState) -> ToolEventDecision:
    return _DEFAULT_REGISTRY.evaluate(action, event, run_id=run_id, state=state)
