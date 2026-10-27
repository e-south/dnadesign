"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/datasets/__init__.py

USR CLI dataset-discovery and resolution helper package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .catalog import list_datasets
from .resolution import _normalize_dataset_id, resolve_dataset_name_interactive, resolve_existing_dataset_id

__all__ = [
    "list_datasets",
    "resolve_existing_dataset_id",
    "resolve_dataset_name_interactive",
    "_normalize_dataset_id",
]
