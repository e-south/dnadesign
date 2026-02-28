"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/__init__.py

Delivery-layer helpers for webhook transport, payloads, secrets, and validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .http import post_json
from .payload import ALLOWED_STATUSES, build_payload
from .secrets import (
    SecretReference,
    is_secret_backend_available,
    parse_secret_ref,
    resolve_secret_ref,
    store_secret_ref,
)
from .validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url

__all__ = [
    "ALLOWED_STATUSES",
    "SecretReference",
    "build_payload",
    "is_secret_backend_available",
    "parse_secret_ref",
    "post_json",
    "resolve_secret_ref",
    "resolve_tls_ca_bundle",
    "resolve_webhook_url",
    "store_secret_ref",
    "validate_provider_webhook_url",
]
