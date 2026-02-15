"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/telemetry.py

Bridge optimizer telemetry updates to run status writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.artifacts.status import RunStatusWriter


class RunTelemetry:
    def __init__(self, writer: RunStatusWriter) -> None:
        self._writer = writer

    def update(self, **fields: Any) -> None:
        self._writer.update(**fields)
