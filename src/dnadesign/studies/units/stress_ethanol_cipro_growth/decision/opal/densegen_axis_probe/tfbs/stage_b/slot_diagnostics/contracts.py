"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/slot_diagnostics/contracts.py

Contracts for Stage B slot-count confound diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...schema import TFBS_LEARNABILITY_SCHEMA_VERSION

SLOT_DIAGNOSTIC_SCHEMA_VERSION = f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.stage_b_slot_diagnostics"
MAX_TFBS_SLOT_COUNT = 3
POSITION_SIGNAL_AFTER_COUNT_RESTRICTION = "position_signal_after_count_restriction"
NOT_SEPARATED_AFTER_COUNT_RESTRICTION = "not_separated_after_count_restriction"
INSUFFICIENT_NONDETERMINISTIC_SELECTION = "insufficient_nondeterministic_selection"


@dataclass(frozen=True)
class SlotLabelSpec:
    """Count column and deterministic strata for a slot-family target label."""

    label_name: str
    target_family_count_column: str
    max_target_family_count: int = MAX_TFBS_SLOT_COUNT

    @property
    def deterministic_counts(self) -> tuple[int, int]:
        return (0, int(self.max_target_family_count))


@dataclass(frozen=True)
class TfbsStageBSlotDiagnosticResult:
    """Paths for a materialized Stage B slot-count diagnostic bundle."""

    status: str
    review_dir: Path
    trajectory_csv_path: Path
    count_distribution_csv_path: Path
    pair_summary_csv_path: Path
    plot_manifest_json_path: Path
    summary_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "count_distribution_csv_path": str(self.count_distribution_csv_path),
            "pair_summary_csv_path": str(self.pair_summary_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
            "summary_json_path": str(self.summary_json_path),
        }


SLOT_LABEL_SPECS: dict[str, SlotLabelSpec] = {
    "lexA_in_slot0": SlotLabelSpec("lexA_in_slot0", "lexA_count"),
    "lexA_in_slot1": SlotLabelSpec("lexA_in_slot1", "lexA_count"),
    "lexA_in_slot2": SlotLabelSpec("lexA_in_slot2", "lexA_count"),
    "cpxR_or_baeR_in_slot0": SlotLabelSpec("cpxR_or_baeR_in_slot0", "cpxR_or_baeR_count"),
    "cpxR_or_baeR_in_slot1": SlotLabelSpec("cpxR_or_baeR_in_slot1", "cpxR_or_baeR_count"),
    "cpxR_or_baeR_in_slot2": SlotLabelSpec("cpxR_or_baeR_in_slot2", "cpxR_or_baeR_count"),
}
