"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/schema/__init__.py

Public profile-schema API surface for notify profile parsing and resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contract import (
    PROFILE_ALLOWED_KEYS,
    PROFILE_REQUIRED_KEYS,
    PROFILE_VERSION,
    WEBHOOK_SOURCES,
    validate_events_source_config,
    validate_webhook_config,
)
from .reader import read_profile
from .resolver import resolve_profile_events_source, resolve_profile_webhook_source

__all__ = [
    "PROFILE_ALLOWED_KEYS",
    "PROFILE_REQUIRED_KEYS",
    "PROFILE_VERSION",
    "WEBHOOK_SOURCES",
    "read_profile",
    "resolve_profile_events_source",
    "resolve_profile_webhook_source",
    "validate_events_source_config",
    "validate_webhook_config",
]
