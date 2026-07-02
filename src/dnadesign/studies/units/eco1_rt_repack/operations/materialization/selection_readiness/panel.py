"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/panel.py

Six-variant panel selection for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    SELECTION_POLICY_ID,
)


def build_selection_panel_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Select one feasible fold-preserved representative per design class."""

    sequence_by_id = {str(row["candidate_id"]): str(row.get("sequence") or "") for row in candidate_rows}
    eligible_rows = [row for row in triage_rows if row["hard_gate_status"] == "eligible"]
    selected: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for spec in ALL_SPECS:
        class_rows = [row for row in eligible_rows if row["design_class_id"] == spec.design_class_id]
        if not class_rows:
            continue
        chosen, nearest_distance = _choose_representative(
            class_rows=class_rows,
            selected_rows=selected,
            sequence_by_id=sequence_by_id,
        )
        selected.append(chosen)
        panel_rows.append(_panel_row(chosen, nearest_distance=nearest_distance, input_hashes=input_hashes))
    return panel_rows


def _choose_representative(
    *,
    class_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    sequence_by_id: dict[str, str],
) -> tuple[dict[str, object], int | None]:
    selected_sequences = [sequence_by_id[str(row["candidate_id"])] for row in selected_rows]

    def sort_key(row: dict[str, object]) -> tuple[object, ...]:
        nearest_distance = _nearest_distance(sequence_by_id[str(row["candidate_id"])], selected_sequences)
        return (
            _fold_rank(str(row["fold_review_class"])),
            -(nearest_distance if nearest_distance is not None else 0),
            -float(row.get("mean_plddt") or 0.0),
            float(row.get("wt_runtime_ca_rmsd") or 9999.0),
            float(row.get("cryoem_mapped_ca_rmsd") or 9999.0),
            -float(row.get("esmc_6b_additive_llr_total") or -9999.0),
            int(row.get("mutation_count_total") or 0),
            str(row["sequence_hash"]),
        )

    chosen = sorted(class_rows, key=sort_key)[0]
    return chosen, _nearest_distance(sequence_by_id[str(chosen["candidate_id"])], selected_sequences)


def _panel_row(
    row: dict[str, object],
    *,
    nearest_distance: int | None,
    input_hashes: dict[str, str | None],
) -> dict[str, object]:
    reason = (
        f"Selected as the {row['design_class_id']} representative after feasibility and fold gates. "
        "SAE windows were WT-like and not used for selection."
    )
    trace = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "design_class_id": row["design_class_id"],
        "nearest_selected_distance_aa": nearest_distance,
        "fold_review_class": row["fold_review_class"],
        "mean_plddt": row["mean_plddt"],
        "wt_runtime_ca_rmsd": row["wt_runtime_ca_rmsd"],
        "cryoem_mapped_ca_rmsd": row["cryoem_mapped_ca_rmsd"],
        "esmc_6b_additive_llr_total": row["esmc_6b_additive_llr_total"],
        "sae_window_status": row["sae_window_status"],
    }
    return {
        "candidate_id": row["candidate_id"],
        "sequence_hash": row["sequence_hash"],
        "design_class_id": row["design_class_id"],
        "eligible_for_handoff": True,
        "selection_slot": row["design_class_id"],
        "slot_rank": 1,
        "selected_for_panel": True,
        "selection_reason": reason,
        "tie_break_trace_json": json.dumps(trace, sort_keys=True),
        "nearest_selected_distance_aa": nearest_distance,
        "fold_review_class": row["fold_review_class"],
        "feasibility_status": row["feasibility_status"],
        "esmc_penalty_rank": None,
        "sae_window_contrast_rank": None,
        "hard_gate_status": row["hard_gate_status"],
        "input_candidate_triage_table_hash": input_hashes["candidate_triage_table"],
        "input_foldcheck_review_hash": input_hashes["foldcheck_review"],
        "input_feasibility_report_hash": input_hashes["feasibility_report"],
        "input_sae_window_summary_hash": input_hashes.get("sae_window_summary"),
    }


def _nearest_distance(sequence: str, selected_sequences: list[str]) -> int | None:
    if not selected_sequences:
        return None
    return min(_hamming_distance(sequence, selected) for selected in selected_sequences)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


def _fold_rank(review_class: str) -> int:
    if review_class == "strong_fold_preserved":
        return 0
    if review_class == "good_fold_preserved":
        return 1
    return 2
