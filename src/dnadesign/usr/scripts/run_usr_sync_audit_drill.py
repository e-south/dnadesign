"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py

Backward-compatible wrapper for the package-owned USR sync audit drill entrypoint.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr.ops.sync_audit_drill import main

if __name__ == "__main__":
    raise SystemExit(main())
