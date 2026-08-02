"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/__init__.py

Expose tool-neutral artifact publication primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .errors import PublicationError, PublicationExistsError
from .portable_paths import portable_path_identity
from .publication import (
    CreateOnlyDirectoryPublication,
    preflight_create_only_directory_publication,
)

__all__ = [
    "CreateOnlyDirectoryPublication",
    "PublicationError",
    "PublicationExistsError",
    "portable_path_identity",
    "preflight_create_only_directory_publication",
]
