"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/cache_readiness.py

Cache readiness hints for lock and target selection errors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def cache_refresh_hint(*, pwm_source: str | None) -> str:
    if pwm_source == "sites":
        return "Run `cruncher fetch sites` first."
    if pwm_source == "matrix":
        return "Run `cruncher fetch motifs` first, or run `cruncher fetch sites` then `cruncher discover motifs`."
    return "Run `cruncher fetch motifs`, or run `cruncher fetch sites` then `cruncher discover motifs`."


def lock_refresh_hint() -> str:
    return (
        "Hint: run `cruncher fetch motifs` or run `cruncher fetch sites` then "
        "`cruncher discover motifs` before locking."
    )
