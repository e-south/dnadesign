"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/tables.py

CSV handoff table writers for scar_nick runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dnadesign.cruncher.scar_nick.artifacts import (
    candidate_pair_call_table_path,
    candidate_table_path,
    nickase_geometry_audit_table_path,
)
from dnadesign.cruncher.scar_nick.models import ScarNickEvaluationReport


def write_nickase_geometry_audit_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "variant_id",
        "specificity_id",
        "orientation",
        "strand",
        "terminal_candidate",
        "motif_top_5to3",
        "source_site_start",
        "source_site_end",
        "boundary",
        "terminal_boundary",
        "exact_terminal",
        "policy_rejection_reasons",
        "rejection_reasons",
        "release_overlap_conflicts",
        "downstream_symbols",
        "retained_scar_domains",
        "feasible_scar_count",
        "upstream_flank_sequence",
        "type_iis_offset_sequence",
        "compatible",
    ]
    with nickase_geometry_audit_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report.nickase_geometry_audit:
            writer.writerow(
                {
                    "variant_id": entry.variant_id,
                    "specificity_id": entry.specificity_id,
                    "orientation": entry.orientation or "",
                    "strand": entry.strand or "",
                    "terminal_candidate": entry.terminal_candidate,
                    "motif_top_5to3": entry.motif_top_5to3,
                    "source_site_start": "" if entry.source_site_start is None else entry.source_site_start,
                    "source_site_end": "" if entry.source_site_end is None else entry.source_site_end,
                    "boundary": "" if entry.boundary is None else entry.boundary,
                    "terminal_boundary": entry.terminal_boundary,
                    "exact_terminal": entry.exact_terminal,
                    "policy_rejection_reasons": json.dumps(entry.policy_rejection_reasons),
                    "rejection_reasons": json.dumps(entry.rejection_reasons),
                    "release_overlap_conflicts": json.dumps(
                        [conflict.model_dump(mode="json") for conflict in entry.release_overlap_conflicts]
                    ),
                    "downstream_symbols": json.dumps(
                        [symbol.model_dump(mode="json") for symbol in entry.downstream_symbols]
                    ),
                    "retained_scar_domains": json.dumps(
                        [domain.model_dump(mode="json") for domain in entry.retained_scar_domains]
                    ),
                    "feasible_scar_count": entry.feasible_scar_count,
                    "upstream_flank_sequence": entry.upstream_flank_sequence,
                    "type_iis_offset_sequence": entry.type_iis_offset_sequence,
                    "compatible": entry.compatible,
                }
            )


def write_candidate_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "left_base",
        "right_base",
        "profile_order",
        "profile_policy_status",
        "profile_policy_reason",
        "profile_s3s2s1s0",
        "profile_payload_outward",
        "s0_match_required",
        "s3_pair_identity",
        "s2_pair_identity",
        "s1_pair_identity",
        "s0_pair_identity",
        "s3_pair_type",
        "s2_pair_type",
        "s1_pair_type",
        "s0_pair_type",
        "pair_classes",
        "m_count",
        "w_count",
        "x_count",
        "non_watson_crick_count",
        "middle_hard_count",
        "middle_wobble_count",
        "worst_hard_mismatch_tier",
        "hard_mismatch_tier_sum",
        "middle_hard_mismatch_tier_sum",
        "edge_hard_mismatch_tier_sum",
        "ligation_support",
        "effective_disruption",
        "tnna_flag",
        "release_placement",
        "release_variant_id",
        "release_recognition_sequence",
        "release_recognition_site_start",
        "release_recognition_site_end",
        "release_top_cut_boundary",
        "release_bottom_cut_boundary",
        "release_recognition_site_excised",
        "retained_scar",
        "nickase_site",
        "nickase_variant_id",
        "nickase_motif_top_5to3",
        "nickase_strand",
        "nickase_source_site_start",
        "nickase_source_site_end",
        "nickase_exact_terminal",
        "nicked_strand",
        "surviving_strand",
        "retained_scar_source",
        "discarded_strand_enzyme_burden",
        "nickase_vendor",
        "nickase_source_url",
        "nickase_commercial_confidence",
        "nickase_warning_codes",
        "nick_boundary",
        "gc_fraction",
        "reference_control_distance",
        "rejection_reasons",
    ]
    with candidate_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in report.candidates:
            release = candidate.release_placement
            pair_type_by_site = {pair.site: pair.class_label for pair in candidate.pair_classes}
            release_placement = (
                ""
                if release is None
                else (
                    f"{release.variant_id}:{release.orientation}"
                    f"[{release.recognition_site_start},{release.recognition_site_end})"
                    f" top={release.top_cut_boundary} bottom={release.bottom_cut_boundary}"
                )
            )
            nickase = candidate.nickase_placement
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "candidate_id": candidate.candidate_id,
                    "left_base": candidate.left_base,
                    "right_base": candidate.right_base,
                    "profile_order": candidate.profile_order,
                    "profile_policy_status": candidate.profile_policy_status,
                    "profile_policy_reason": candidate.profile_policy_reason,
                    "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
                    "profile_payload_outward": candidate.profile_payload_outward,
                    "s0_match_required": candidate.s0_match_required,
                    "s3_pair_identity": candidate.s3_pair_identity,
                    "s2_pair_identity": candidate.s2_pair_identity,
                    "s1_pair_identity": candidate.s1_pair_identity,
                    "s0_pair_identity": candidate.s0_pair_identity,
                    "s3_pair_type": pair_type_by_site["S3"],
                    "s2_pair_type": pair_type_by_site["S2"],
                    "s1_pair_type": pair_type_by_site["S1"],
                    "s0_pair_type": pair_type_by_site["S0"],
                    "pair_classes": json.dumps([entry.model_dump(mode="json") for entry in candidate.pair_classes]),
                    "m_count": candidate.m_count,
                    "w_count": candidate.w_count,
                    "x_count": candidate.x_count,
                    "non_watson_crick_count": candidate.non_watson_crick_count,
                    "middle_hard_count": candidate.middle_hard_count,
                    "middle_wobble_count": candidate.middle_wobble_count,
                    "worst_hard_mismatch_tier": candidate.worst_hard_mismatch_tier,
                    "hard_mismatch_tier_sum": candidate.hard_mismatch_tier_sum,
                    "middle_hard_mismatch_tier_sum": candidate.middle_hard_mismatch_tier_sum,
                    "edge_hard_mismatch_tier_sum": candidate.edge_hard_mismatch_tier_sum,
                    "ligation_support": f"{candidate.ligation_support:.6f}",
                    "effective_disruption": f"{candidate.effective_disruption:.6f}",
                    "tnna_flag": candidate.tnna_flag,
                    "release_placement": release_placement,
                    "release_variant_id": "" if release is None else release.variant_id,
                    "release_recognition_sequence": "" if release is None else release.recognition_sequence,
                    "release_recognition_site_start": "" if release is None else release.recognition_site_start,
                    "release_recognition_site_end": "" if release is None else release.recognition_site_end,
                    "release_top_cut_boundary": "" if release is None else release.top_cut_boundary,
                    "release_bottom_cut_boundary": "" if release is None else release.bottom_cut_boundary,
                    "release_recognition_site_excised": "" if release is None else release.recognition_site_excised,
                    "retained_scar": candidate.retained_scar,
                    "nickase_site": candidate.nickase_site,
                    "nickase_variant_id": "" if nickase is None else nickase.variant_id,
                    "nickase_motif_top_5to3": "" if nickase is None else nickase.motif_top_5to3,
                    "nickase_strand": "" if nickase is None else nickase.strand,
                    "nickase_source_site_start": "" if nickase is None else nickase.source_site_start,
                    "nickase_source_site_end": "" if nickase is None else nickase.source_site_end,
                    "nickase_exact_terminal": "" if nickase is None else nickase.exact_terminal,
                    "nicked_strand": candidate.nicked_strand or "",
                    "surviving_strand": candidate.surviving_strand or "",
                    "retained_scar_source": candidate.retained_scar_source,
                    "discarded_strand_enzyme_burden": candidate.discarded_strand_enzyme_burden or "",
                    "nickase_vendor": "" if nickase is None else nickase.vendor,
                    "nickase_source_url": "" if nickase is None else nickase.source_url,
                    "nickase_commercial_confidence": "" if nickase is None else nickase.commercial_confidence,
                    "nickase_warning_codes": json.dumps([] if nickase is None else nickase.warning_codes),
                    "nick_boundary": candidate.nick_boundary,
                    "gc_fraction": f"{candidate.gc_fraction:.6f}",
                    "reference_control_distance": candidate.reference_control_distance,
                    "rejection_reasons": json.dumps(candidate.rejection_reasons),
                }
            )


def write_candidate_pair_call_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "left_base",
        "right_base",
        "profile_order",
        "profile_policy_status",
        "profile_policy_reason",
        "profile_s3s2s1s0",
        "site",
        "position",
        "position_class",
        "left_offset_5to3",
        "right_offset_5to3",
        "left_nt",
        "right_nt",
        "aligned_right_nt",
        "pair_identity",
        "class_label",
        "is_watson_crick",
        "is_wobble",
        "is_hard_mismatch",
        "canonical_mismatch_class",
        "class_tier_t4",
        "m_count",
        "w_count",
        "x_count",
        "non_watson_crick_count",
        "middle_hard_count",
        "middle_wobble_count",
        "worst_hard_mismatch_tier",
        "hard_mismatch_tier_sum",
        "middle_hard_mismatch_tier_sum",
        "edge_hard_mismatch_tier_sum",
        "ligation_support",
        "effective_disruption",
        "nicked_strand",
        "surviving_strand",
        "retained_scar",
        "nickase_site",
    ]
    with candidate_pair_call_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in report.candidates:
            for pair in candidate.pair_classes:
                writer.writerow(
                    {
                        "rank": candidate.rank,
                        "candidate_id": candidate.candidate_id,
                        "left_base": candidate.left_base,
                        "right_base": candidate.right_base,
                        "profile_order": candidate.profile_order,
                        "profile_policy_status": candidate.profile_policy_status,
                        "profile_policy_reason": candidate.profile_policy_reason,
                        "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
                        "site": pair.site,
                        "position": pair.position,
                        "position_class": pair.position_class,
                        "left_offset_5to3": pair.source_offset,
                        "right_offset_5to3": 3 - pair.source_offset,
                        "left_nt": pair.left_base,
                        "right_nt": pair.right_base,
                        "aligned_right_nt": pair.aligned_right_base,
                        "pair_identity": f"{pair.left_base}:{pair.aligned_right_base}",
                        "class_label": pair.class_label,
                        "is_watson_crick": pair.class_label == "M",
                        "is_wobble": pair.class_label == "W",
                        "is_hard_mismatch": pair.class_label == "X",
                        "canonical_mismatch_class": pair.canonical_mismatch_class or "",
                        "class_tier_t4": pair.class_tier_t4,
                        "m_count": candidate.m_count,
                        "w_count": candidate.w_count,
                        "x_count": candidate.x_count,
                        "non_watson_crick_count": candidate.non_watson_crick_count,
                        "middle_hard_count": candidate.middle_hard_count,
                        "middle_wobble_count": candidate.middle_wobble_count,
                        "worst_hard_mismatch_tier": candidate.worst_hard_mismatch_tier,
                        "hard_mismatch_tier_sum": candidate.hard_mismatch_tier_sum,
                        "middle_hard_mismatch_tier_sum": candidate.middle_hard_mismatch_tier_sum,
                        "edge_hard_mismatch_tier_sum": candidate.edge_hard_mismatch_tier_sum,
                        "ligation_support": f"{candidate.ligation_support:.6f}",
                        "effective_disruption": f"{candidate.effective_disruption:.6f}",
                        "nicked_strand": candidate.nicked_strand or "",
                        "surviving_strand": candidate.surviving_strand or "",
                        "retained_scar": candidate.retained_scar,
                        "nickase_site": candidate.nickase_site,
                    }
                )


__all__ = [
    "write_candidate_pair_call_table",
    "write_candidate_table",
    "write_nickase_geometry_audit_table",
]
