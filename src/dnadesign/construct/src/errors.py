"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/errors.py

Construct-specific exception types.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class ConstructError(Exception):
    """Base error for construct."""


class ConfigError(ConstructError):
    """Raised when construct configuration is invalid."""


class ValidationError(ConstructError):
    """Raised when construct inputs violate runtime contracts."""


class ExecutionError(ConstructError):
    """Raised when construct execution cannot complete."""
