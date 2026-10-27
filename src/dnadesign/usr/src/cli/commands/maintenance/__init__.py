"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/__init__.py

USR CLI maintenance command family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .cli import register_maintenance_commands
from .dedupe import cmd_dedupe_sequences
from .events import cmd_event_log_garden
from .merge import MergeDeps, cmd_merge_datasets
from .overlay import cmd_overlay_compact, cmd_overlay_project, cmd_overlay_refresh_metadata, cmd_overlay_remove
from .registry import MaintenanceDeps, cmd_registry_freeze

__all__ = [
    "MaintenanceDeps",
    "MergeDeps",
    "cmd_dedupe_sequences",
    "cmd_event_log_garden",
    "cmd_merge_datasets",
    "cmd_overlay_compact",
    "cmd_overlay_project",
    "cmd_overlay_refresh_metadata",
    "cmd_overlay_remove",
    "cmd_registry_freeze",
    "register_maintenance_commands",
]
