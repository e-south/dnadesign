"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/lifecycle/__init__.py

USR CLI dataset lifecycle command family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .cli import register_lifecycle_commands
from .materialize import MaterializeDeps, cmd_materialize
from .snapshot import SnapshotDeps, cmd_snapshot
from .state import cmd_delete, cmd_restore, cmd_state_clear, cmd_state_get, cmd_state_set
from .write import cmd_attach, cmd_import, cmd_init

__all__ = [
    "MaterializeDeps",
    "SnapshotDeps",
    "cmd_attach",
    "cmd_delete",
    "cmd_import",
    "cmd_init",
    "cmd_materialize",
    "cmd_restore",
    "cmd_snapshot",
    "cmd_state_clear",
    "cmd_state_get",
    "cmd_state_set",
    "register_lifecycle_commands",
]
