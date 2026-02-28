"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/__init__.py

Secret reference parsing and secret backend operations for webhook URLs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contract import SecretReference, parse_secret_ref
from .ops import is_secret_backend_available, resolve_secret_ref, store_secret_ref

__all__ = [
    "SecretReference",
    "is_secret_backend_available",
    "parse_secret_ref",
    "resolve_secret_ref",
    "store_secret_ref",
]
