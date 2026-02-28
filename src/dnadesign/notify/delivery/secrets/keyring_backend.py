"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/keyring_backend.py

Keyring backend detection and compatibility checks for notify secrets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .contract import BACKEND_KEYCHAIN


def load_keyring_module() -> ModuleType | None:
    try:
        return importlib.import_module("keyring")
    except Exception:
        return None


def keyring_backend_descriptor(keyring_module: ModuleType) -> str:
    try:
        backend = keyring_module.get_keyring()
    except Exception:
        return ""
    return f"{type(backend).__module__}.{type(backend).__name__}".lower()


def keyring_backend_matches(*, backend: str, descriptor: str) -> bool:
    descriptor_value = str(descriptor or "").strip().lower()
    if not descriptor_value:
        return False
    if backend == BACKEND_KEYCHAIN:
        return "keychain" in descriptor_value or "macos" in descriptor_value
    return "secretservice" in descriptor_value


def keyring_client_for_backend(backend: str) -> ModuleType | None:
    keyring_module = load_keyring_module()
    if keyring_module is None:
        return None
    descriptor = keyring_backend_descriptor(keyring_module)
    if not keyring_backend_matches(backend=backend, descriptor=descriptor):
        return None
    if not hasattr(keyring_module, "get_password") or not hasattr(keyring_module, "set_password"):
        return None
    return keyring_module
