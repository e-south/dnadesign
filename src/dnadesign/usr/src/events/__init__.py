"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/__init__.py

Structured JSONL event logging for USR dataset mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .actor import _default_actor, _normalize_actor
from .defaults import USR_EVENT_VERSION, _event_defaults
from .fingerprint import _sha256_file, fingerprint_parquet
from .gardening import EventLogGardenResult, garden_event_log
from .recording import record_event, validate_event_metadata
from .redaction import _arg_key_is_sensitive, _redact_arg_value, _redact_args

__all__ = [
    "USR_EVENT_VERSION",
    "_arg_key_is_sensitive",
    "_default_actor",
    "_event_defaults",
    "_normalize_actor",
    "_redact_arg_value",
    "_redact_args",
    "_sha256_file",
    "EventLogGardenResult",
    "fingerprint_parquet",
    "garden_event_log",
    "record_event",
    "validate_event_metadata",
]
