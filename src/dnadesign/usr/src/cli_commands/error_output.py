"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/error_output.py

User-facing CLI error rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..errors import DuplicateIDError, SequencesError


def _abbrev_id(rid: str, n: int = 8) -> str:
    return (rid[:n] + "…") if isinstance(rid, str) and len(rid) > n else rid


def _print_dup_groups(title: str, groups) -> None:
    if not groups:
        return
    print(title)
    for g in groups:
        rid = _abbrev_id(g.id)
        rows = ",".join(str(i) for i in g.rows) if g.rows else "n/a"
        seq = g.sequence
        print(f"  • id={rid:<10}  count={g.count:<3}  input rows=[{rows}]")
        print(f"    sequence: {seq}")


def print_user_error(e: SequencesError) -> None:
    print(f"ERROR: {e}")

    if isinstance(e, DuplicateIDError):
        if getattr(e, "groups", None):
            _print_dup_groups("Top duplicates (showing up to 5):", e.groups[:5])
        if getattr(e, "casefold_groups", None):
            _print_dup_groups(
                "Case-insensitive duplicates (same letters, different capitalization):",
                e.casefold_groups[:5],
            )
        if getattr(e, "hint", None):
            print(f"Hint: {e.hint}")
        return

    hint = getattr(e, "hint", None)
    if hint:
        print(f"Hint: {hint}")
