"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/local_structure_sensitivity.py

Threshold-sensitivity rows for Eco1 RT local-structure gates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
)


@dataclass(frozen=True)
class LocalStructureThresholdScenario:
    """One threshold multiplier used to audit local RMSD gate sensitivity."""

    scenario_id: str
    label: str
    multiplier: float


LOCAL_STRUCTURE_THRESHOLD_SCENARIOS = (
    LocalStructureThresholdScenario("tighter_80_percent", "Tighter 0.8x", 0.80),
    LocalStructureThresholdScenario("declared_threshold", "Declared 1.0x", 1.00),
    LocalStructureThresholdScenario("looser_120_percent", "Looser 1.2x", 1.20),
)


def build_local_structure_threshold_sensitivity_rows(
    *,
    local_structure_rows: Sequence[Mapping[str, object]],
    selected_candidate_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Summarize pass/fail counts under tighter, declared, and looser local RMSD thresholds."""

    selected = {str(candidate_id) for candidate_id in selected_candidate_ids}
    rows_by_region: dict[str, list[Mapping[str, object]]] = {region_id: [] for region_id in LOCAL_STRUCTURE_REGION_IDS}
    labels_by_region: dict[str, str] = {}
    for row in local_structure_rows:
        region_id = str(row.get("region_id") or "")
        if region_id not in rows_by_region:
            continue
        rows_by_region[region_id].append(row)
        labels_by_region.setdefault(region_id, str(row.get("region_label") or region_id.replace("_", " ")))

    rows: list[dict[str, object]] = []
    for region_id in LOCAL_STRUCTURE_REGION_IDS:
        region_rows = rows_by_region[region_id]
        available_rows = [
            row
            for row in region_rows
            if str(row.get("status") or "") == "available" and row.get("local_ca_rmsd_angstrom") is not None
        ]
        for scenario in LOCAL_STRUCTURE_THRESHOLD_SCENARIOS:
            threshold = round(LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id] * scenario.multiplier, 3)
            failed = [row for row in available_rows if float(row["local_ca_rmsd_angstrom"]) > threshold]
            selected_failed = [row for row in failed if str(row.get("candidate_id") or "") in selected]
            rows.append(
                {
                    "threshold_policy_id": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
                    "scenario_id": scenario.scenario_id,
                    "scenario_label": scenario.label,
                    "threshold_multiplier": scenario.multiplier,
                    "region_id": region_id,
                    "region_label": labels_by_region.get(region_id, region_id.replace("_", " ")),
                    "threshold_angstrom": threshold,
                    "candidate_count": len(region_rows),
                    "available_count": len(available_rows),
                    "unavailable_count": len(region_rows) - len(available_rows),
                    "pass_count": len(available_rows) - len(failed),
                    "failure_count": len(failed),
                    "selected_failure_count": len(selected_failed),
                }
            )
    return rows


__all__ = [
    "LOCAL_STRUCTURE_THRESHOLD_SCENARIOS",
    "LocalStructureThresholdScenario",
    "build_local_structure_threshold_sensitivity_rows",
]
