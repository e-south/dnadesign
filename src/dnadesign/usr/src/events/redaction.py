"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/redaction.py

Argument redaction helpers for USR event payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

_REDACTED_VALUE = "***REDACTED***"
_SENSITIVE_ARG_KEY_TOKENS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "webhook",
    "auth",
    "credential",
    "bearer",
    "cookie",
    "session",
)


def _arg_key_is_sensitive(key: str) -> bool:
    key_norm = str(key or "").strip().lower().replace("-", "_")
    if not key_norm:
        return False
    return any(token in key_norm for token in _SENSITIVE_ARG_KEY_TOKENS)


def _redact_arg_value(value: Any, *, force_redact: bool = False) -> Any:
    if force_redact:
        return _REDACTED_VALUE
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = _redact_arg_value(item, force_redact=_arg_key_is_sensitive(key_text))
        return redacted
    if isinstance(value, (list, tuple)):
        redacted_items: list[Any] = []
        redact_next_value = False
        for item in value:
            if redact_next_value:
                redacted_items.append(_REDACTED_VALUE)
                redact_next_value = False
                continue
            if isinstance(item, str):
                token = str(item).strip()
                if token.startswith("-") and "=" in token:
                    flag, _raw_value = token.split("=", 1)
                    flag_key = str(flag).lstrip("-").replace("-", "_")
                    if _arg_key_is_sensitive(flag_key):
                        redacted_items.append(f"{flag}={_REDACTED_VALUE}")
                        continue
                if token.startswith("-"):
                    flag_key = token.lstrip("-").replace("-", "_")
                    if _arg_key_is_sensitive(flag_key):
                        redacted_items.append(token)
                        redact_next_value = True
                        continue
            redacted_items.append(_redact_arg_value(item, force_redact=False))
        return redacted_items
    return value


def _redact_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    if args is None:
        return {}
    if not isinstance(args, Mapping):
        raise TypeError("event args must be a mapping when provided")
    return dict(_redact_arg_value(args, force_redact=False))
