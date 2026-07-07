"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/panel.py

Primary-panel selection for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    PRIMARY_C_TERMINAL_LOCAL_RMSD_MAX_ANGSTROM,
    PRIMARY_PANEL_SIZE,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    nearest_jaccard_distance,
)

NA_FACING_LOW_BURDEN_RATIO = 0.05
NA_FACING_HIGH_BURDEN_RATIO = 0.75

_PRIMARY_RANK_FIELDS = (
    (
        "proximal_unobserved_support",
        "proximal_review_unobserved_mutation_count",
        "lower",
    ),
    (
        "proximal_rare_support",
        "proximal_review_rare_or_unobserved_mutation_count",
        "lower",
    ),
    (
        "acidic_gain_near_retained_dna_rna",
        "nucleic_acid_facing_acidic_gain_count",
        "lower",
    ),
    (
        "basic_loss_near_retained_dna_rna",
        "nucleic_acid_facing_basic_loss_count",
        "lower",
    ),
    (
        "proline_glycine_gain_near_retained_dna_rna",
        "nucleic_acid_facing_proline_glycine_gain_count",
        "lower",
    ),
    (
        "nearest_selected_mutation_position_jaccard_distance",
        "nearest_selected_mutation_position_jaccard_distance",
        "higher",
    ),
    (
        "nearest_selected_mutation_token_jaccard_distance",
        "nearest_selected_mutation_token_jaccard_distance",
        "higher",
    ),
    (
        "c_terminal_primer_rna_recognition_rmsd",
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom",
        "lower",
    ),
    (
        "substrate_relevant_local_rmsd",
        "local_structure_substrate_relevant_max_ca_rmsd_angstrom",
        "lower",
    ),
    (
        "chemistry_warning_count",
        "nucleic_acid_facing_chemistry_warning_count",
        "lower",
    ),
    (
        "near_retained_dna_rna_burden",
        "nucleic_acid_facing_mutation_count",
        "lower",
    ),
    (
        "selection_msa_observed_fraction",
        "selection_support_alt_observed_fraction",
        "higher",
    ),
    (
        "selection_msa_alt_frequency",
        "selection_support_alt_frequency_mean",
        "higher",
    ),
    ("mean_plddt", "mean_plddt", "higher"),
    ("cryoem_mapped_rmsd", "cryoem_mapped_ca_rmsd", "lower"),
    ("sequence_hash", "sequence_hash", "lexicographic"),
)


def build_selection_panel_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Select the primary conservative panel from globally eligible primary candidates."""

    sequence_by_id = {str(row["candidate_id"]): str(row.get("sequence") or "") for row in candidate_rows}
    mutation_tokens_by_id = {
        str(row["candidate_id"]): canonical_mutation_tokens(row.get("canonical_mutations")) for row in candidate_rows
    }
    mutation_positions_by_id = {
        str(row["candidate_id"]): canonical_mutation_positions(row.get("canonical_mutations")) for row in candidate_rows
    }
    primary_rows = [
        row for row in triage_rows if str(row.get("selection_candidate_tier") or "") == "primary_panel_candidate"
    ]
    if len(primary_rows) < PRIMARY_PANEL_SIZE:
        raise ValueError(
            "Primary panel selection failed: "
            f"requires {PRIMARY_PANEL_SIZE} primary-panel candidates but found {len(primary_rows)}."
        )
    selected: list[dict[str, object]] = []
    remaining = list(primary_rows)
    panel_rows: list[dict[str, object]] = []
    for slot_rank in range(1, PRIMARY_PANEL_SIZE + 1):
        chosen, nearest_distance = _choose_primary_candidate(
            candidate_rows=remaining,
            selected_rows=selected,
            sequence_by_id=sequence_by_id,
            mutation_tokens_by_id=mutation_tokens_by_id,
            mutation_positions_by_id=mutation_positions_by_id,
        )
        selected.append(chosen)
        remaining = [row for row in remaining if str(row["candidate_id"]) != str(chosen["candidate_id"])]
        panel_rows.append(
            _panel_row(
                chosen,
                nearest_distance=nearest_distance,
                input_hashes=input_hashes,
                slot_rank=slot_rank,
            )
        )
    validate_primary_panel(panel_rows, required_panel_size=PRIMARY_PANEL_SIZE)
    return panel_rows


def build_primary_panel_selection_trace_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    panel_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return stage counts for the primary-panel funnel and boundary-candidate split."""

    all_rows = list(triage_rows)
    trace_rows: list[dict[str, object]] = [
        _trace_row(
            stage_order=1,
            stage_id="candidate_pool",
            stage_label="Accepted candidate pool",
            selector_role="input_pool",
            filter_rule="Accepted ProteinMPNN candidate rows before protein-level selection checks.",
            input_count=len(all_rows),
            remaining_count=len(all_rows),
            is_hard_gate=False,
        )
    ]
    broad_rows = [row for row in all_rows if str(row.get("hard_gate_status") or "") == "eligible"]
    trace_rows.append(
        _trace_row(
            stage_order=2,
            stage_id="broad_contract_pool",
            stage_label="Broad protein contract",
            selector_role="hard_gate",
            filter_rule=(
                "Keep strong-fold candidates with feasible rows, no protected/core/direct-contact/thumb-track edits, "
                "passed declared local RMSD gates, substrate-relevant local RMSD <= 3.0 A, and minimum directional "
                "near retained DNA/RNA chemistry."
            ),
            input_count=len(all_rows),
            remaining_count=len(broad_rows),
            is_hard_gate=True,
        )
    )
    primary_rows = [
        row for row in broad_rows if str(row.get("selection_candidate_tier") or "") == "primary_panel_candidate"
    ]
    trace_rows.append(
        _trace_row(
            stage_order=3,
            stage_id="primary_panel_candidate_pool",
            stage_label="Primary candidate pool",
            selector_role="preservation_contract",
            filter_rule=(
                "Keep broad-contract rows with C-terminal primer-RNA recognition-region C-alpha RMSD <= "
                f"{PRIMARY_C_TERMINAL_LOCAL_RMSD_MAX_ANGSTROM:.1f} A after the global mapped C-alpha fit. "
                "Proximal MSA support and near-region chemistry warnings remain selector fields, not separate "
                "displayed gates."
            ),
            input_count=len(broad_rows),
            remaining_count=len(primary_rows),
            is_hard_gate=False,
        )
    )
    trace_rows.append(
        _trace_row(
            stage_order=4,
            stage_id="global_conservative_diverse_selection",
            stage_label="Conservative-diverse six-row selection",
            selector_role="global_rank",
            filter_rule=(
                "Select six rows globally from primary candidates by proximal MSA support, near retained DNA/RNA and "
                "thumb-track chemistry warnings, mutation-set dissimilarity to already selected rows, local RMSD, "
                "fold metrics, and a deterministic tie-break. Design class is context, not a quota."
            ),
            input_count=len(primary_rows),
            remaining_count=len(panel_rows),
            is_hard_gate=False,
        )
    )
    return trace_rows


def validate_primary_panel(
    panel_rows: Sequence[dict[str, object]],
    *,
    required_panel_size: int = PRIMARY_PANEL_SIZE,
) -> None:
    """Fail unless the selected primary panel has the required size and unique candidate ids."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    duplicates = sorted(candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1)
    wrong_tier = [
        str(row.get("candidate_id") or "")
        for row in panel_rows
        if str(row.get("selection_candidate_tier") or "") != "primary_panel_candidate"
    ]
    if len(panel_rows) == required_panel_size and not duplicates and not wrong_tier:
        return
    raise ValueError(
        "Primary panel validation failed: "
        f"expected {required_panel_size} selected rows. Selected rows: {len(panel_rows)}. "
        f"Duplicate candidate ids: {_format_list(duplicates)}. "
        f"Non-primary selected rows: {_format_list(wrong_tier)}."
    )


def panel_coverage_summary(panel_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return manifest-ready primary-panel coverage fields."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    design_class_counts = Counter(str(row.get("design_class_id") or "") for row in panel_rows)
    duplicate_candidate_ids = sorted(
        candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1
    )
    non_primary = [
        str(row.get("candidate_id") or "")
        for row in panel_rows
        if str(row.get("selection_candidate_tier") or "") != "primary_panel_candidate"
    ]
    return {
        "required_primary_panel_size": PRIMARY_PANEL_SIZE,
        "selected_row_count": len(panel_rows),
        "design_class_quota_enforced": False,
        "selected_design_class_counts": {key: design_class_counts[key] for key in sorted(design_class_counts)},
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "non_primary_selected_candidate_ids": non_primary,
        "valid": len(panel_rows) == PRIMARY_PANEL_SIZE and not duplicate_candidate_ids and not non_primary,
    }


def _choose_primary_candidate(
    *,
    candidate_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    sequence_by_id: dict[str, str],
    mutation_tokens_by_id: dict[str, frozenset[str]] | None = None,
    mutation_positions_by_id: dict[str, frozenset[int]] | None = None,
) -> tuple[dict[str, object], int | None]:
    selected_sequences = [sequence_by_id[str(row["candidate_id"])] for row in selected_rows]
    mutation_tokens_by_id = mutation_tokens_by_id or {}
    mutation_positions_by_id = mutation_positions_by_id or {}
    selected_token_sets = [mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows]
    selected_position_sets = [
        mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows
    ]
    nearest_distance_by_id = {
        str(row["candidate_id"]): _nearest_distance(
            sequence_by_id.get(str(row["candidate_id"]), ""),
            selected_sequences,
        )
        for row in candidate_rows
    }
    nearest_token_jaccard_by_id = {
        str(row["candidate_id"]): nearest_jaccard_distance(
            mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_token_sets,
        )
        for row in candidate_rows
    }
    nearest_position_jaccard_by_id = {
        str(row["candidate_id"]): nearest_jaccard_distance(
            mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_position_sets,
        )
        for row in candidate_rows
    }
    chosen = min(
        candidate_rows,
        key=lambda row: _primary_sort_key(
            row,
            nearest_distance=nearest_distance_by_id[str(row["candidate_id"])],
            nearest_mutation_token_jaccard=nearest_token_jaccard_by_id[str(row["candidate_id"])],
            nearest_mutation_position_jaccard=nearest_position_jaccard_by_id[str(row["candidate_id"])],
        ),
    )
    chosen["nearest_selected_mutation_token_jaccard_distance"] = nearest_token_jaccard_by_id[
        str(chosen["candidate_id"])
    ]
    chosen["nearest_selected_mutation_position_jaccard_distance"] = nearest_position_jaccard_by_id[
        str(chosen["candidate_id"])
    ]
    return chosen, nearest_distance_by_id[str(chosen["candidate_id"])]


def _panel_row(
    row: dict[str, object],
    *,
    nearest_distance: int | None,
    input_hashes: dict[str, str | None],
    slot_rank: int,
) -> dict[str, object]:
    reason = (
        "Selected for the primary conservative panel after protein-contract gates and a stricter C-terminal/thumb "
        "primer-RNA recognition RMSD check. The final panel is selected globally using proximal MSA support, "
        "near retained DNA/RNA chemistry risk, mutation-set dissimilarity, local structure, and fold metrics; design "
        "classes remain review context rather than quotas. ESMC and SAE rows were retained for review but not used "
        "for selection."
    )
    na_facing_count, na_facing_ratio = _na_facing_count_and_ratio(row)
    trace = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "selection_candidate_tier": row.get("selection_candidate_tier"),
        "primary_panel_candidate": row.get("primary_panel_candidate"),
        "primary_panel_failure_reasons_json": row.get("primary_panel_failure_reasons_json"),
        "design_class_id": row["design_class_id"],
        "proximal_review_unobserved_mutation_count": row.get("proximal_review_unobserved_mutation_count"),
        "proximal_review_rare_or_unobserved_mutation_count": row.get(
            "proximal_review_rare_or_unobserved_mutation_count"
        ),
        "selection_support_profile_id": row["selection_support_profile_id"],
        "selection_support_alt_observed_fraction": row["selection_support_alt_observed_fraction"],
        "selection_support_alt_frequency_mean": row["selection_support_alt_frequency_mean"],
        "selection_support_unobserved_mutation_count": row["selection_support_unobserved_mutation_count"],
        "mutation_count_total": row["mutation_count_total"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "nucleic_acid_facing_burden_ratio": na_facing_ratio,
        "nucleic_acid_facing_burden_band": _na_facing_burden_band(na_facing_count, na_facing_ratio),
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_chemistry_compatible": row.get("nucleic_acid_facing_chemistry_compatible"),
        "nucleic_acid_facing_chemistry_gate_status": row.get("nucleic_acid_facing_chemistry_gate_status"),
        "near_retained_dna_rna_acidic_gain_review_status": row.get("near_retained_dna_rna_acidic_gain_review_status"),
        "primary_c_terminal_local_rmsd_gate_status": row.get("primary_c_terminal_local_rmsd_gate_status"),
        "proximal_msa_support_review_status": row.get("proximal_msa_support_review_status"),
        "nucleic_acid_facing_charge_delta": row["nucleic_acid_facing_charge_delta"],
        "nucleic_acid_facing_acidic_gain_count": row["nucleic_acid_facing_acidic_gain_count"],
        "nucleic_acid_facing_basic_loss_count": row["nucleic_acid_facing_basic_loss_count"],
        "nucleic_acid_facing_proline_glycine_gain_count": row["nucleic_acid_facing_proline_glycine_gain_count"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "c_terminal_primer_rna_recognition_mutation_count": row["c_terminal_primer_rna_recognition_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_max_ca_rmsd_angstrom": row["local_structure_max_ca_rmsd_angstrom"],
        "local_structure_substrate_relevant_max_ca_rmsd_angstrom": row.get(
            "local_structure_substrate_relevant_max_ca_rmsd_angstrom"
        ),
        "local_structure_substrate_relevant_max_gate_status": row.get(
            "local_structure_substrate_relevant_max_gate_status"
        ),
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": row[
            "local_structure_catalytic_initiation_context_ca_rmsd_angstrom"
        ],
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": row[
            "local_structure_thumb_contact_track_context_ca_rmsd_angstrom"
        ],
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": row[
            "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom"
        ],
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": row[
            "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom"
        ],
        "nearest_selected_distance_aa": nearest_distance,
        "nearest_selected_mutation_token_jaccard_distance": row.get("nearest_selected_mutation_token_jaccard_distance"),
        "nearest_selected_mutation_position_jaccard_distance": row.get(
            "nearest_selected_mutation_position_jaccard_distance"
        ),
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
        "selection_slot": f"primary_panel_{slot_rank:02d}",
        "slot_rank": slot_rank,
        "selected_for_panel": True,
        "selection_reason": reason,
        "tie_break_trace_json": json.dumps(trace, sort_keys=True),
        "nearest_selected_distance_aa": nearest_distance,
        "fold_review_class": row["fold_review_class"],
        "feasibility_status": row["feasibility_status"],
        "hard_gate_status": row["hard_gate_status"],
        "primary_panel_candidate": bool(row.get("primary_panel_candidate")),
        "selection_candidate_tier": str(row.get("selection_candidate_tier") or ""),
        "primary_panel_failure_reasons_json": row.get("primary_panel_failure_reasons_json"),
        "near_retained_dna_rna_acidic_gain_review_status": row.get("near_retained_dna_rna_acidic_gain_review_status"),
        "primary_c_terminal_local_rmsd_gate_status": row.get("primary_c_terminal_local_rmsd_gate_status"),
        "proximal_msa_support_review_status": row.get("proximal_msa_support_review_status"),
        "nearest_selected_mutation_token_jaccard_distance": row.get("nearest_selected_mutation_token_jaccard_distance"),
        "nearest_selected_mutation_position_jaccard_distance": row.get(
            "nearest_selected_mutation_position_jaccard_distance"
        ),
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_threshold_policy_id": row["local_structure_threshold_policy_id"],
        "local_structure_threshold_failed_region_count": row["local_structure_threshold_failed_region_count"],
        "local_structure_max_ca_rmsd_angstrom": row["local_structure_max_ca_rmsd_angstrom"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "c_terminal_primer_rna_recognition_mutation_count": row["c_terminal_primer_rna_recognition_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_chemistry_compatible": row.get("nucleic_acid_facing_chemistry_compatible"),
        "nucleic_acid_facing_chemistry_gate_status": row.get("nucleic_acid_facing_chemistry_gate_status"),
        "nucleic_acid_facing_acidic_gain_count": row.get("nucleic_acid_facing_acidic_gain_count"),
        "proximal_review_unobserved_mutation_count": row.get("proximal_review_unobserved_mutation_count"),
        "proximal_review_rare_or_unobserved_mutation_count": row.get(
            "proximal_review_rare_or_unobserved_mutation_count"
        ),
        "local_structure_substrate_relevant_max_ca_rmsd_angstrom": row.get(
            "local_structure_substrate_relevant_max_ca_rmsd_angstrom"
        ),
        "local_structure_substrate_relevant_max_gate_status": row.get(
            "local_structure_substrate_relevant_max_gate_status"
        ),
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": row[
            "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom"
        ],
        "input_candidate_triage_table_hash": input_hashes["candidate_triage_table"],
        "input_foldcheck_review_hash": input_hashes["foldcheck_review"],
        "input_feasibility_report_hash": input_hashes["feasibility_report"],
        "input_sae_window_summary_hash": input_hashes.get("sae_window_summary"),
    }


def _primary_sort_key(
    row: dict[str, object],
    *,
    nearest_distance: int | None,
    nearest_mutation_token_jaccard: float | None = None,
    nearest_mutation_position_jaccard: float | None = None,
) -> tuple[object, ...]:
    values: list[object] = []
    for _stage_id, field_name, direction in _PRIMARY_RANK_FIELDS:
        if field_name == "nearest_selected_distance_aa":
            value: object = nearest_distance if nearest_distance is not None else 0
        elif field_name == "nearest_selected_mutation_token_jaccard_distance":
            value = nearest_mutation_token_jaccard if nearest_mutation_token_jaccard is not None else 0.0
        elif field_name == "nearest_selected_mutation_position_jaccard_distance":
            value = nearest_mutation_position_jaccard if nearest_mutation_position_jaccard is not None else 0.0
        else:
            value = row.get(field_name)
        if direction == "lexicographic":
            values.append(str(value or ""))
        elif direction == "lower":
            values.append(_float_value(value, default=9999.0))
        elif direction == "higher":
            values.append(-_float_value(value, default=-9999.0))
        else:
            raise ValueError(f"Unknown primary-panel rank direction: {direction}")
    return tuple(values)


def _trace_row(
    *,
    stage_order: int,
    stage_id: str,
    stage_label: str,
    selector_role: str,
    filter_rule: str,
    input_count: int,
    remaining_count: int,
    is_hard_gate: bool,
) -> dict[str, object]:
    return {
        "selection_policy_id": SELECTION_POLICY_ID,
        "stage_order": stage_order,
        "stage_id": stage_id,
        "stage_label": stage_label,
        "selector_role": selector_role,
        "filter_rule": filter_rule,
        "input_count": input_count,
        "removed_count": max(input_count - remaining_count, 0),
        "remaining_count": remaining_count,
        "is_hard_gate": is_hard_gate,
    }


def _nearest_distance(sequence: str, selected_sequences: list[str]) -> int | None:
    if not selected_sequences:
        return None
    return min(_hamming_distance(sequence, selected) for selected in selected_sequences)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


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


def _format_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _float_value(value: object, *, default: float = -1.0) -> float:
    return default if value is None else float(value)


def _int_value(value: object, *, default: int = 0) -> int:
    return default if value is None else int(value)
