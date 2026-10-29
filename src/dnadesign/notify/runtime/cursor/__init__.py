"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/cursor/__init__.py

Cursor offset, lock, and follow-loop primitives for notify runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .iteration import iter_file_lines
from .locking import acquire_cursor_lock
from .offsets import load_cursor_offset, save_cursor_offset

__all__ = [
    "acquire_cursor_lock",
    "iter_file_lines",
    "load_cursor_offset",
    "save_cursor_offset",
]
