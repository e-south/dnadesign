"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/errors.py

Error types for notification delivery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class NotifyError(Exception):
    """Base error for notifier failures."""


class NotifyConfigError(NotifyError):
    """Raised when configuration or inputs are invalid."""


class NotifyValidationError(NotifyError):
    """Raised when payload validation fails."""


class NotifyDeliveryError(NotifyError):
    """Raised when delivery to a webhook fails."""
