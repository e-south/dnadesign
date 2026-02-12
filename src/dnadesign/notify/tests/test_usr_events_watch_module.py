"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_usr_events_watch_module.py

Module contract test for the notify USR events watch loop entrypoint.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.notify.usr_events_watch import watch_usr_events_loop


def test_watch_usr_events_loop_exported() -> None:
    assert callable(watch_usr_events_loop)
