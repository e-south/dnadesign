"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/runtime/__init__.py

Runtime command registration helpers for Notify watch and spool groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .spool_cmd import register_spool_drain_command
from .watch_cmd import register_usr_events_watch_command

__all__ = ["register_spool_drain_command", "register_usr_events_watch_command"]
