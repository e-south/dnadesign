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
from collections import Counter
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    SELECTION_POLICY_ID,
)

NA_FACING_BURDEN_TARGET_RATIO = 0.40
NA_FACING_LOW_BURDEN_RATIO = 0.05
NA_FACING_HIGH_BURDEN_RATIO = 0.75


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
    validate_exact_panel_coverage(
        panel_rows,
        expected_design_classes=[spec.design_class_id for spec in ALL_SPECS],
    )
    return panel_rows


def validate_exact_panel_coverage(
    panel_rows: Sequence[dict[str, object]],
    *,
    expected_design_classes: Sequence[str],
) -> None:
    """Fail unless the panel has exactly one row per declared design class."""

    expected = [str(design_class_id) for design_class_id in expected_design_classes]
    observed = [str(row.get("design_class_id") or "") for row in panel_rows]
    counts = Counter(observed)
    missing = [design_class_id for design_class_id in expected if counts[design_class_id] == 0]
    duplicate = [design_class_id for design_class_id in expected if counts[design_class_id] > 1]
    unexpected = sorted(design_class_id for design_class_id in counts if design_class_id not in set(expected))
    if len(panel_rows) == len(expected) and not missing and not duplicate and not unexpected:
        return
    raise ValueError(
        "Panel coverage failed: expected exactly one selected row for each of "
        f"{len(expected)} design classes. Missing classes: {_format_class_list(missing)}. "
        f"Duplicate classes: {_format_class_list(duplicate)}. "
        f"Unexpected classes: {_format_class_list(unexpected)}. Selected rows: {len(panel_rows)}."
    )


def panel_coverage_summary(
    panel_rows: Sequence[dict[str, object]],
    *,
    expected_design_classes: Sequence[str],
) -> dict[str, object]:
    """Return manifest-ready exact panel coverage fields."""

    expected = [str(design_class_id) for design_class_id in expected_design_classes]
    observed = [str(row.get("design_class_id") or "") for row in panel_rows]
    counts = Counter(observed)
    missing = [design_class_id for design_class_id in expected if counts[design_class_id] == 0]
    duplicate = [design_class_id for design_class_id in expected if counts[design_class_id] > 1]
    unexpected = sorted(design_class_id for design_class_id in counts if design_class_id not in set(expected))
    return {
        "expected_design_class_count": len(expected),
        "selected_row_count": len(panel_rows),
        "required_rows_per_class": 1,
        "missing_design_classes": missing,
        "duplicate_design_classes": duplicate,
        "unexpected_design_classes": unexpected,
        "valid": len(panel_rows) == len(expected) and not missing and not duplicate and not unexpected,
    }


def _format_class_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


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
            -_float_value(row.get("selection_support_alt_observed_fraction")),
            -_float_value(row.get("selection_support_alt_frequency_mean")),
            _int_value(row.get("selection_support_unobserved_mutation_count"), default=9999),
            _int_value(row.get("nucleic_acid_facing_chemistry_warning_count"), default=9999),
            _na_facing_burden_sort_values(row),
            _local_structure_sort_values(row),
            -(nearest_distance if nearest_distance is not None else 0),
            -float(row.get("mean_plddt") or 0.0),
            float(row.get("wt_runtime_ca_rmsd") or 9999.0),
            float(row.get("cryoem_mapped_ca_rmsd") or 9999.0),
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
        f"Selected as the {row['design_class_id']} representative after feasibility, fold, and local-structure gates. "
        "The tie-breaks use MSA support, near-DNA/RNA chemistry warnings, moderate regional mutation burden, "
        "local/global fold metrics, and sequence nonredundancy. ESMC and SAE rows were retained for review "
        "but not used for selection."
    )
    na_facing_count, na_facing_ratio = _na_facing_count_and_ratio(row)
    trace = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "design_class_id": row["design_class_id"],
        "selection_support_profile_id": row["selection_support_profile_id"],
        "selection_support_alt_observed_fraction": row["selection_support_alt_observed_fraction"],
        "selection_support_alt_frequency_mean": row["selection_support_alt_frequency_mean"],
        "selection_support_unobserved_mutation_count": row["selection_support_unobserved_mutation_count"],
        "mutation_count_total": row["mutation_count_total"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "nucleic_acid_facing_burden_ratio": na_facing_ratio,
        "nucleic_acid_facing_burden_band": _na_facing_burden_band(na_facing_count, na_facing_ratio),
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_charge_delta": row["nucleic_acid_facing_charge_delta"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_max_ca_rmsd_angstrom": row["local_structure_max_ca_rmsd_angstrom"],
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": row[
            "local_structure_catalytic_initiation_context_ca_rmsd_angstrom"
        ],
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": row[
            "local_structure_thumb_contact_track_context_ca_rmsd_angstrom"
        ],
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": row[
            "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom"
        ],
        "nearest_selected_distance_aa": nearest_distance,
        "fold_review_class": row["fold_review_class"],
        "mean_plddt": row["mean_plddt"],
        "wt_runtime_ca_rmsd": row["wt_runtime_ca_rmsd"],
        "cryoem_mapped_ca_rmsd": row["cryoem_mapped_ca_rmsd"],
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


def _na_facing_burden_sort_values(row: dict[str, object]) -> tuple[int, float, int]:
    """Prefer controlled substrate-proximal burden instead of maximizing it."""

    count, ratio = _na_facing_count_and_ratio(row)
    band_rank = {
        "moderate": 0,
        "low": 1,
        "none": 2,
        "broad": 3,
    }[_na_facing_burden_band(count, ratio)]
    return (band_rank, abs(ratio - NA_FACING_BURDEN_TARGET_RATIO), count)


def _na_facing_count_and_ratio(row: dict[str, object]) -> tuple[int, float]:
    count = _int_value(row.get("nucleic_acid_facing_mutation_count"))
    total = max(_int_value(row.get("mutation_count_total")), 0)
    return count, count / total if total else 0.0


def _na_facing_burden_band(count: int, ratio: float) -> str:
    if count == 0:
        return "none"
    if ratio < NA_FACING_LOW_BURDEN_RATIO:
        return "low"
    if ratio <= NA_FACING_HIGH_BURDEN_RATIO:
        return "moderate"
    return "broad"


def _local_structure_sort_values(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        _float_value(row.get("local_structure_catalytic_initiation_context_ca_rmsd_angstrom"), default=9999.0),
        _float_value(row.get("local_structure_thumb_contact_track_context_ca_rmsd_angstrom"), default=9999.0),
        _float_value(row.get("local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom"), default=9999.0),
        _float_value(row.get("local_structure_max_ca_rmsd_angstrom"), default=9999.0),
    )


def _float_value(value: object, *, default: float = -1.0) -> float:
    return default if value is None else float(value)


def _int_value(value: object, *, default: int = 0) -> int:
    return default if value is None else int(value)
