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


class PublicationExistsError(PublicationError, FileExistsError):
    """Raised when create-only publication would replace an existing artifact."""


__all__ = ["PublicationError", "PublicationExistsError"]
