"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/telemetry.py

Send optimizer telemetry updates to a caller-supplied sink.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Protocol


class OptimizerTelemetry(Protocol):
    def update(self, **fields: Any) -> None: ...


class NullTelemetry:
    def update(self, **_fields: Any) -> None:
        return None
