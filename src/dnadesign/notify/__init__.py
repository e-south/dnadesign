"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/__init__.py

Stable Notify integration contracts, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.contracts import (  # noqa: F401
        DEFAULT_NOTIFY_WEBHOOK_SOURCES,
        DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
        TLSCABundleResolutionError,
        parse_notify_profile_webhook,
        resolve_file_secret_ref_path,
        resolve_tls_ca_bundle_path,
    )
    from .core.errors import (  # noqa: F401
        NotifyConfigError,
        NotifyDeliveryError,
        NotifyError,
        NotifyValidationError,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_NOTIFY_WEBHOOK_SOURCES": (".core.contracts", "DEFAULT_NOTIFY_WEBHOOK_SOURCES"),
    "DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES": (
        ".core.contracts",
        "DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES",
    ),
    "TLSCABundleResolutionError": (".core.contracts", "TLSCABundleResolutionError"),
    "parse_notify_profile_webhook": (".core.contracts", "parse_notify_profile_webhook"),
    "resolve_file_secret_ref_path": (".core.contracts", "resolve_file_secret_ref_path"),
    "resolve_tls_ca_bundle_path": (".core.contracts", "resolve_tls_ca_bundle_path"),
    "NotifyError": (".core.errors", "NotifyError"),
    "NotifyConfigError": (".core.errors", "NotifyConfigError"),
    "NotifyValidationError": (".core.errors", "NotifyValidationError"),
    "NotifyDeliveryError": (".core.errors", "NotifyDeliveryError"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
