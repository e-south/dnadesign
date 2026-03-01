"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/__init__.py

Runtime primitives for notify watch loops, cursor state, spooling, and runner orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .cursor import acquire_cursor_lock, iter_file_lines, load_cursor_offset, save_cursor_offset
from .runner import run_spool_drain, run_usr_events_watch
from .spool import ensure_private_directory, spool_payload
from .watch import watch_usr_events_loop

__all__ = [
    "acquire_cursor_lock",
    "ensure_private_directory",
    "iter_file_lines",
    "load_cursor_offset",
    "run_spool_drain",
    "run_usr_events_watch",
    "save_cursor_offset",
    "spool_payload",
    "watch_usr_events_loop",
]
