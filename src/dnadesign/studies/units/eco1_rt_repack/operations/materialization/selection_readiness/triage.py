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
    PRIMARY_C_TERMINAL_LOCAL_RMSD_MAX_ANGSTROM,
    SAE_WINDOW_SELECTION_THRESHOLD,
    SUBSTRATE_RELEVANT_LOCAL_RMSD_MAX_ANGSTROM,
)

_SUBSTRATE_RELEVANT_LOCAL_STRUCTURE_FIELDS = (
    "local_structure_catalytic_initiation_context_ca_rmsd_angstrom",
    "local_structure_retron_x_naxxh_context_ca_rmsd_angstrom",
    "local_structure_retron_y_vtg_context_ca_rmsd_angstrom",
    "local_structure_thumb_contact_track_context_ca_rmsd_angstrom",
    "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom",
    "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom",
)
_PROXIMAL_REGION_MSA_SUPPORT_IDS = frozenset(
    {
        "catalytic_or_direct_contact",
        "near_retained_dna_rna_region",
        "thumb_contact_track",
        "c_terminal_primer_rna_recognition_region",
    }
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
    local_structure_review_by_candidate: dict[str, dict[str, object]],
    region_msa_support_rows: Sequence[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Build the flat reviewer-facing triage table."""

    fold_by_id = {str(row["candidate_id"]): row for row in fold_review_rows}
    feasibility_by_id = {str(row["candidate_id"]): row for row in feasibility_rows}
    llr300_by_id = _llr_by_candidate(llr_300m_rows)
    llr6b_by_id = _llr_by_candidate(llr_6b_rows)
    sae_by_id = _sae_by_candidate(sae_window_rows)
    proximal_support_by_id = _proximal_region_support_by_candidate(region_msa_support_rows)
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
        local_structure_review = local_structure_review_by_candidate.get(candidate_id)
        hard_gate_status, reasons = _hard_gate_status(
            candidate=candidate,
            fold=fold,
            feasibility=feasibility,
            review_axes=review_axes,
            local_structure_review=local_structure_review,
        )
        row = {
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
            **_proximal_region_support_fields(proximal_support_by_id.get(candidate_id)),
            **_local_structure_fields(local_structure_review),
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
        row.update(_primary_panel_candidate_fields(row))
        rows.append(row)
    return rows


def _hard_gate_status(
    *,
    candidate: dict[str, object],
    fold: dict[str, object] | None,
    feasibility: dict[str, object] | None,
    review_axes: dict[str, object] | None = None,
    local_structure_review: dict[str, object] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if str(candidate.get("status")) != "accepted":
        reasons.append("candidate_status_not_accepted")
    if int(candidate.get("protected_mutation_count") or 0):
        reasons.append("protected_mutation_violation")
    if int((review_axes or {}).get("catalytic_or_direct_contact_mutation_count") or 0):
        reasons.append("catalytic_or_direct_contact_mutation")
    if int((review_axes or {}).get("thumb_contact_track_mutation_count") or 0):
        reasons.append("thumb_contact_track_mutation")
    if not _chemistry_compatible(review_axes):
        reasons.append("nucleic_acid_facing_chemistry_incompatible")
    if fold is None:
        reasons.append("missing_fold_review_row")
    elif str(fold.get("foldcheck_status")) != "accepted":
        reasons.append("foldcheck_status_not_accepted")
    if feasibility is None:
        reasons.append("missing_feasibility_row")
    elif str(feasibility.get("feasibility_status")) != "feasible":
        reasons.append("feasibility_not_feasible")
    if local_structure_review is None:
        reasons.append("missing_local_structure_review")
    else:
        local_structure_status = str(local_structure_review.get("local_structure_gate_status") or "")
        if local_structure_status == "unavailable":
            reasons.append("local_structure_gate_unavailable")
        elif local_structure_status == "threshold_exceeded":
            reasons.append("local_structure_threshold_exceeded")
        elif local_structure_status != "passed":
            reasons.append("local_structure_gate_not_passed")
        if local_structure_status == "passed":
            substrate_rmsd = _substrate_relevant_local_rmsd(local_structure_review)
            if substrate_rmsd is None:
                reasons.append("missing_substrate_relevant_local_structure_rmsd")
            elif substrate_rmsd > SUBSTRATE_RELEVANT_LOCAL_RMSD_MAX_ANGSTROM:
                reasons.append("local_structure_substrate_relevant_rmsd_exceeded")
    review_class = str((fold or {}).get("review_class") or "")
    if review_class and review_class not in ALLOWED_FOLD_CLASSES:
        reasons.append("fold_review_class_not_allowed")
    if any(reason.startswith("missing_") for reason in reasons) or "local_structure_gate_unavailable" in reasons:
        return "missing_inputs", sorted(reasons)
    if reasons:
        return "ineligible", sorted(reasons)
    return "eligible", []


def _slot_candidate_status(hard_gate_status: str) -> str:
    if hard_gate_status == "eligible":
        return "passes_broad_protein_contract"
    return "not_panel_eligible"


def _primary_panel_candidate_fields(row: dict[str, object]) -> dict[str, object]:
    reasons: list[str] = []
    if str(row.get("hard_gate_status") or "") != "eligible":
        reasons.append("broad_protein_contract_not_met")

    c_terminal_rmsd = _float_or_none(
        row.get("local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom")
    )
    if c_terminal_rmsd is None:
        reasons.append("missing_c_terminal_primer_rna_recognition_rmsd")
        c_terminal_status = "missing"
    elif c_terminal_rmsd > PRIMARY_C_TERMINAL_LOCAL_RMSD_MAX_ANGSTROM:
        reasons.append("c_terminal_primer_rna_recognition_rmsd_exceeded")
        c_terminal_status = "threshold_exceeded"
    else:
        c_terminal_status = "passed"

    acidic_gain_count = int(row.get("nucleic_acid_facing_acidic_gain_count") or 0)
    chemistry_status = "acidic_gain_present" if acidic_gain_count else "passed"

    proximal_unobserved_count = row.get("proximal_review_unobserved_mutation_count")
    if proximal_unobserved_count is None:
        support_status = "missing"
    elif int(proximal_unobserved_count) != 0:
        support_status = "unobserved_substitution_present"
    else:
        support_status = "passed"

    primary = not reasons
    if primary:
        tier = "primary_panel_candidate"
    elif str(row.get("hard_gate_status") or "") == "eligible":
        tier = "boundary_candidate"
    else:
        tier = "not_panel_candidate"
    return {
        "primary_panel_candidate": primary,
        "selection_candidate_tier": tier,
        "primary_panel_failure_reasons_json": json.dumps(sorted(reasons), sort_keys=True),
        "primary_c_terminal_local_rmsd_gate_status": c_terminal_status,
        "near_retained_dna_rna_acidic_gain_review_status": chemistry_status,
        "proximal_msa_support_review_status": support_status,
    }


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
        "c_terminal_primer_rna_recognition_mutation_count": None,
        "distal_scaffold_mutation_count": None,
        "nucleic_acid_facing_charge_delta": None,
        "nucleic_acid_facing_basic_gain_count": None,
        "nucleic_acid_facing_basic_loss_count": None,
        "nucleic_acid_facing_acidic_gain_count": None,
        "nucleic_acid_facing_proline_glycine_gain_count": None,
        "nucleic_acid_facing_chemistry_warning_count": None,
        "nucleic_acid_facing_chemistry_compatible": None,
        "nucleic_acid_facing_chemistry_gate_status": None,
    }
    fields.update({key: values.get(key) for key in fields if key in values})
    return fields


def _local_structure_fields(values: dict[str, object] | None) -> dict[str, object]:
    fields = {
        "local_structure_gate_status": "missing",
        "local_structure_gate_failure_reasons_json": "[]",
        "local_structure_region_count": None,
        "local_structure_available_region_count": None,
        "local_structure_unavailable_region_count": None,
        "local_structure_threshold_failed_region_count": None,
        "local_structure_threshold_policy_id": "",
        "local_structure_max_ca_rmsd_angstrom": None,
        "local_structure_mean_ca_rmsd_angstrom": None,
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": None,
        "local_structure_retron_x_naxxh_context_ca_rmsd_angstrom": None,
        "local_structure_retron_y_vtg_context_ca_rmsd_angstrom": None,
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": None,
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": None,
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": None,
        "local_structure_distal_scaffold_control_ca_rmsd_angstrom": None,
        "local_structure_substrate_relevant_max_ca_rmsd_angstrom": None,
        "local_structure_substrate_relevant_max_gate_status": "missing",
    }
    if values is not None:
        fields.update({key: values.get(key) for key in fields if key in values})
        substrate_max = _substrate_relevant_local_rmsd(values)
        fields["local_structure_substrate_relevant_max_ca_rmsd_angstrom"] = substrate_max
        if substrate_max is None:
            fields["local_structure_substrate_relevant_max_gate_status"] = "missing"
        else:
            fields["local_structure_substrate_relevant_max_gate_status"] = (
                "passed" if substrate_max <= SUBSTRATE_RELEVANT_LOCAL_RMSD_MAX_ANGSTROM else "threshold_exceeded"
            )
    return fields


def _proximal_region_support_by_candidate(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, int]]:
    by_candidate: dict[str, dict[str, int]] = {}
    for row in rows:
        if str(row.get("region_id") or "") not in _PROXIMAL_REGION_MSA_SUPPORT_IDS:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        summary = by_candidate.setdefault(
            candidate_id,
            {
                "proximal_review_unobserved_mutation_count": 0,
                "proximal_review_rare_or_unobserved_mutation_count": 0,
            },
        )
        summary["proximal_review_unobserved_mutation_count"] += int(row.get("unobserved_mutation_count") or 0)
        summary["proximal_review_rare_or_unobserved_mutation_count"] += int(
            row.get("rare_or_unobserved_mutation_count") or 0
        )
    return by_candidate


def _proximal_region_support_fields(values: dict[str, int] | None) -> dict[str, object]:
    return {
        "proximal_review_unobserved_mutation_count": None
        if values is None
        else values["proximal_review_unobserved_mutation_count"],
        "proximal_review_rare_or_unobserved_mutation_count": None
        if values is None
        else values["proximal_review_rare_or_unobserved_mutation_count"],
    }


def _chemistry_compatible(review_axes: dict[str, object] | None) -> bool:
    if review_axes is None:
        return True
    compatible = review_axes.get("nucleic_acid_facing_chemistry_compatible")
    if compatible is not None:
        return bool(compatible)
    charge_delta = int(review_axes.get("nucleic_acid_facing_charge_delta") or 0)
    acidic_gain = int(review_axes.get("nucleic_acid_facing_acidic_gain_count") or 0)
    basic_gain = int(review_axes.get("nucleic_acid_facing_basic_gain_count") or 0)
    return charge_delta >= 0 and acidic_gain <= basic_gain


def _substrate_relevant_local_rmsd(values: dict[str, object]) -> float | None:
    numeric_values = [
        float(values[field]) for field in _SUBSTRATE_RELEVANT_LOCAL_STRUCTURE_FIELDS if values.get(field) is not None
    ]
    return round(max(numeric_values), 3) if numeric_values else None


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)
