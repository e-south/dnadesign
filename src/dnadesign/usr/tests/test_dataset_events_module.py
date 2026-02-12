"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_events_module.py

Module contract test for dataset-scoped USR event recording helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr.src.dataset_events import record_dataset_event


def test_record_dataset_event_exported() -> None:
    assert callable(record_dataset_event)
