"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/store/motif_store.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.ingest.models import MotifDescriptor, MotifQuery


@dataclass(frozen=True, slots=True)
class MotifRef:
    source: str
    motif_id: str

    def __str__(self) -> str:
        return f"{self.source}:{self.motif_id}"


class MotifStore(Protocol):
    def get_pwm(self, ref: MotifRef) -> PWM: ...

    def list(self, query: MotifQuery) -> list[MotifDescriptor]: ...
