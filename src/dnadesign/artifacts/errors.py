"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/errors.py

Shared artifact publication errors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""


class PublicationError(RuntimeError):
    """Raised when an immutable artifact bundle cannot be published safely."""


__all__ = ["PublicationError"]
