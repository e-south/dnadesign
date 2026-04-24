"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/defaults.py

Default event payload fragments for USR event records.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

USR_EVENT_VERSION = 1


def _event_defaults(action: str) -> dict:
    action = str(action)
    defaults = {
        "init": {
            "metrics": {"rows": 0},
            "artifacts": {"overlays": []},
        },
        "import_rows": {
            "metrics": {"rows_written": 0, "rows_skipped": 0},
            "artifacts": {"base": {}},
        },
        "attach": {
            "metrics": {"rows_incoming": 0, "rows_matched": 0, "rows_missing": 0},
            "artifacts": {"overlay": {}},
        },
        "write_overlay_part": {
            "metrics": {"rows_incoming": 0, "rows_written": 0, "rows_missing": 0},
            "artifacts": {"overlay": {}},
        },
        "tombstone": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr"}},
        },
        "restore": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr"}},
        },
        "state_set": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr_state"}},
        },
        "state_clear": {
            "metrics": {"rows": 0},
            "artifacts": {"overlay": {"namespace": "usr_state"}},
        },
        "snapshot": {
            "metrics": {"rows": 0},
            "artifacts": {"snapshot": {}},
        },
        "materialize": {
            "metrics": {"overlays": 0, "rows": 0},
            "artifacts": {"overlays": []},
        },
        "dedupe": {
            "metrics": {"rows_total": 0, "rows_dropped": 0, "groups": 0},
            "artifacts": {"base": {}},
        },
        "merge": {
            "metrics": {
                "rows_added": 0,
                "duplicates_total": 0,
                "duplicates_skipped": 0,
                "duplicates_replaced": 0,
            },
            "artifacts": {"src": None, "dest": None},
        },
        "registry_freeze": {
            "metrics": {"updated": 0},
            "artifacts": {"registry_snapshot": ""},
        },
        "overlay_compact": {
            "metrics": {"parts_in": 0, "parts_out": 0, "rows": 0},
            "artifacts": {"overlay": {}},
        },
        "remove_overlay": {
            "metrics": {"removed": 0},
            "artifacts": {"overlay": {}},
        },
    }
    return defaults.get(action, {"metrics": {}, "artifacts": {}})
