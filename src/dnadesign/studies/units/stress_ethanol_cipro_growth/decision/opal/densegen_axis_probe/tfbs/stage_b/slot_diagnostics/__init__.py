"""Slot-position diagnostics with explicit count-confound controls."""

from __future__ import annotations

from .contracts import (
    INSUFFICIENT_NONDETERMINISTIC_SELECTION,
    MAX_TFBS_SLOT_COUNT,
    NOT_SEPARATED_AFTER_COUNT_RESTRICTION,
    POSITION_SIGNAL_AFTER_COUNT_RESTRICTION,
    SLOT_DIAGNOSTIC_SCHEMA_VERSION,
    SLOT_LABEL_SPECS,
    SlotLabelSpec,
    TfbsStageBSlotDiagnosticResult,
)
from .materialization import build_tfbs_stage_b_slot_diagnostics

__all__ = [
    "INSUFFICIENT_NONDETERMINISTIC_SELECTION",
    "MAX_TFBS_SLOT_COUNT",
    "NOT_SEPARATED_AFTER_COUNT_RESTRICTION",
    "POSITION_SIGNAL_AFTER_COUNT_RESTRICTION",
    "SLOT_DIAGNOSTIC_SCHEMA_VERSION",
    "SLOT_LABEL_SPECS",
    "SlotLabelSpec",
    "TfbsStageBSlotDiagnosticResult",
    "build_tfbs_stage_b_slot_diagnostics",
]
