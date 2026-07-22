"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/selection.py

Select review and Atlas subsets from Eco1 fold-check ranking rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    WT_SEQUENCE_ID,
)


def select_structure_panel_rows(ranking_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select a compact, contrastive structure-review panel."""

    selected: list[dict[str, Any]] = []
    _add_first(
        selected,
        "best_rmsd",
        sorted(ranking_rows, key=lambda row: (_float(row["wt_runtime_ca_rmsd"]), -_float(row["plddt"]))),
    )
    _add_first(
        selected,
        "best_plddt",
        sorted(ranking_rows, key=lambda row: (-_float(row["plddt"]), _float(row["wt_runtime_ca_rmsd"]))),
    )
    _add_first(
        selected,
        "high_mutation_good_fold",
        sorted(
            [row for row in ranking_rows if _float(row["wt_runtime_ca_rmsd"]) <= 1.5],
            key=lambda row: (-int(row["mutation_count"]), _float(row["wt_runtime_ca_rmsd"])),
        ),
    )
    for row in sorted(
        [row for row in ranking_rows if str(row["review_class"]) == "structural_outlier"],
        key=lambda item: -_float(item["wt_runtime_ca_rmsd"]),
    )[:3]:
        _add_row(selected, "rmsd_outlier", row)
    for row in sorted(
        [row for row in ranking_rows if str(row["review_class"]) == "low_confidence"],
        key=lambda item: _float(item["plddt"]),
    )[:3]:
        _add_row(selected, "low_plddt", row)
    for row in sorted(
        [row for row in ranking_rows if str(row["review_class"]) == "review_band"],
        key=lambda item: -_float(item["wt_runtime_ca_rmsd"]),
    )[:2]:
        _add_row(selected, "intermediate_rmsd", row)
    for row in sorted(
        [row for row in ranking_rows if str(row["review_class"]) in {"good_fold_preserved", "strong_fold_preserved"}],
        key=lambda item: str(item["candidate_id"]),
    )[:2]:
        _add_row(selected, "deterministic_fold_preserved_control", row)
    return selected


def build_atlas_subset_rows(structure_panel_rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return WT plus selected structure-panel ids for Atlas staging."""

    rows = [{"sequence_id": WT_SEQUENCE_ID, "selection_stratum": "wild_type_baseline"}]
    seen = {WT_SEQUENCE_ID}
    for row in structure_panel_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        rows.append({"sequence_id": candidate_id, "selection_stratum": str(row["selection_stratum"])})
    return rows


def _add_first(selected: list[dict[str, Any]], stratum: str, candidates: list[Mapping[str, Any]]) -> None:
    for row in candidates:
        if _add_row(selected, stratum, row):
            return


def _add_row(selected: list[dict[str, Any]], stratum: str, row: Mapping[str, Any]) -> bool:
    candidate_id = str(row["candidate_id"])
    if candidate_id in {str(item["candidate_id"]) for item in selected}:
        return False
    copied = dict(row)
    copied["selection_stratum"] = stratum
    selected.append(copied)
    return True


def _float(value: Any) -> float:
    if value is None:
        return 9999.0
    return float(value)
