"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/triage.py

Candidate triage row construction for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    ALLOWED_FOLD_CLASSES,
    REVIEW_ONLY_FOLD_CLASSES,
    SAE_WINDOW_SELECTION_THRESHOLD,
)


def build_triage_rows(
    *,
    candidate_rows: Sequence[dict[str, object]],
    fold_review_rows: Sequence[dict[str, object]],
    feasibility_rows: Sequence[dict[str, object]],
    llr_300m_rows: Sequence[dict[str, object]],
    llr_6b_rows: Sequence[dict[str, object]],
    sae_window_rows: Sequence[dict[str, object]],
    review_axis_by_candidate: dict[str, dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Build the flat reviewer-facing triage table."""

    fold_by_id = {str(row["candidate_id"]): row for row in fold_review_rows}
    feasibility_by_id = {str(row["candidate_id"]): row for row in feasibility_rows}
    llr300_by_id = _llr_by_candidate(llr_300m_rows)
    llr6b_by_id = _llr_by_candidate(llr_6b_rows)
    sae_by_id = _sae_by_candidate(sae_window_rows)
    rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if str(candidate.get("status")) != "accepted":
            continue
        candidate_id = str(candidate["candidate_id"])
        fold = fold_by_id.get(candidate_id)
        feasibility = feasibility_by_id.get(candidate_id)
        sae = sae_by_id.get(candidate_id)
        if candidate_id not in review_axis_by_candidate:
            raise ValueError(f"Missing Eco1 selection review axes for candidate: {candidate_id}")
        review_axes = review_axis_by_candidate[candidate_id]
        hard_gate_status, reasons = _hard_gate_status(candidate=candidate, fold=fold, feasibility=feasibility)
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": str(candidate["sequence_hash"]),
                "design_class_id": str(candidate["design_class_id"]),
                "mask_policy_id": str(candidate.get("mask_policy_id") or candidate["design_class_id"]),
                "mutation_count_total": int(candidate.get("mutation_count") or 0),
                "sequence_distance_to_wt": int(candidate.get("mutation_count") or 0),
                "nearest_selected_distance_aa": None,
                "fold_review_class": str((fold or {}).get("review_class") or ""),
                "mean_plddt": _float_or_none((fold or {}).get("plddt")),
                "wt_runtime_ca_rmsd": _float_or_none((fold or {}).get("wt_runtime_ca_rmsd")),
                "cryoem_mapped_ca_rmsd": _float_or_none((fold or {}).get("cryoem_mapped_ca_rmsd")),
                "esmc_300m_additive_llr_total": _float_or_none((llr300_by_id.get(candidate_id) or {}).get("llr_total")),
                "esmc_6b_additive_llr_total": _float_or_none((llr6b_by_id.get(candidate_id) or {}).get("llr_total")),
                "sae_window_status": _sae_status(sae),
                "sae_mechanistic_contrast_window_id": None,
                "sae_mechanistic_contrast_rank": None,
                **_review_axis_fields(review_axes),
                "feasibility_status": str((feasibility or {}).get("feasibility_status") or ""),
                "hard_gate_status": hard_gate_status,
                "hard_gate_failure_reasons_json": json.dumps(reasons, sort_keys=True),
                "slot_candidate_status": _slot_candidate_status(hard_gate_status),
                "seed": int(candidate.get("seed") or 0),
                "temperature": _float_or_none(candidate.get("temperature")),
                "input_candidate_pool_hash": input_hashes["candidate_pool"],
                "input_foldcheck_review_hash": input_hashes["foldcheck_review"],
                "input_feasibility_report_hash": input_hashes["feasibility_report"],
                "input_sae_window_summary_hash": input_hashes.get("sae_window_summary"),
                "input_conservation_profile_hash": input_hashes["conservation_profile"],
                "input_clade9_alignment_hash": input_hashes["clade9_alignment"],
                "input_subtype_alignment_hash": input_hashes["subtype_alignment"],
                "input_contact_geometry_profile_hash": input_hashes["contact_geometry_profile"],
            }
        )
    return rows


def _hard_gate_status(
    *,
    candidate: dict[str, object],
    fold: dict[str, object] | None,
    feasibility: dict[str, object] | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if str(candidate.get("status")) != "accepted":
        reasons.append("candidate_status_not_accepted")
    if int(candidate.get("protected_mutation_count") or 0):
        reasons.append("protected_mutation_violation")
    if fold is None:
        reasons.append("missing_fold_review_row")
    elif str(fold.get("foldcheck_status")) != "accepted":
        reasons.append("foldcheck_status_not_accepted")
    if feasibility is None:
        reasons.append("missing_feasibility_row")
    elif str(feasibility.get("feasibility_status")) != "feasible":
        reasons.append("feasibility_not_feasible")
    review_class = str((fold or {}).get("review_class") or "")
    if any(reason.startswith("missing_") for reason in reasons):
        return "missing_inputs", sorted(reasons)
    if reasons:
        if review_class in REVIEW_ONLY_FOLD_CLASSES:
            reasons.append("fold_review_class_requires_manual_review")
        return "ineligible", sorted(reasons)
    if review_class in REVIEW_ONLY_FOLD_CLASSES:
        return "needs_review", ["fold_review_class_requires_manual_review"]
    if review_class and review_class not in ALLOWED_FOLD_CLASSES:
        reasons.append("fold_review_class_not_allowed")
    return ("ineligible", sorted(reasons)) if reasons else ("eligible", [])


def _slot_candidate_status(hard_gate_status: str) -> str:
    if hard_gate_status == "eligible":
        return "eligible_for_class_representative"
    if hard_gate_status == "needs_review":
        return "manual_reserve_only"
    return "not_panel_eligible"


def _llr_by_candidate(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["candidate_id"]): row for row in rows}


def _sae_by_candidate(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    return grouped


def _sae_status(rows: list[dict[str, object]] | None) -> str:
    if not rows:
        return "missing"
    max_distance = max(float(row.get("cosine_distance_to_wt") or 0.0) for row in rows)
    if max_distance < SAE_WINDOW_SELECTION_THRESHOLD:
        return "wt_like_not_used_for_selection"
    return "review_only_window_shift_not_gating"


def _review_axis_fields(values: dict[str, object]) -> dict[str, object]:
    fields = {
        "clade9_alt_observed_fraction": None,
        "clade9_alt_frequency_mean": None,
        "clade9_unobserved_mutation_count": None,
        "clade9_rare_or_unobserved_mutation_count": None,
        "subtype_alt_observed_fraction": None,
        "subtype_alt_frequency_mean": None,
        "subtype_unobserved_mutation_count": None,
        "subtype_rare_or_unobserved_mutation_count": None,
        "selection_support_profile_id": "",
        "selection_support_alt_observed_fraction": None,
        "selection_support_alt_frequency_mean": None,
        "selection_support_unobserved_mutation_count": None,
        "catalytic_or_direct_contact_mutation_count": None,
        "nucleic_acid_facing_mutation_count": None,
        "thumb_contact_track_mutation_count": None,
        "distal_scaffold_mutation_count": None,
        "nucleic_acid_facing_charge_delta": None,
        "nucleic_acid_facing_basic_gain_count": None,
        "nucleic_acid_facing_basic_loss_count": None,
        "nucleic_acid_facing_acidic_gain_count": None,
        "nucleic_acid_facing_proline_glycine_gain_count": None,
        "nucleic_acid_facing_chemistry_warning_count": None,
    }
    fields.update({key: values.get(key) for key in fields if key in values})
    return fields


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)
