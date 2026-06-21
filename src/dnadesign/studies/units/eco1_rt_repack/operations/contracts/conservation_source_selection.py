"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation_source_selection.py

Selection-rule checks for Eco1 conservation source groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _as_string_list,
    _is_positive_number,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _FORBIDDEN_CONSERVATION_DENOMINATOR_RULES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


def validate_conservation_selection_rule(
    issues: list[ContractIssue],
    *,
    source_group: Mapping[str, Any],
    index: int,
) -> None:
    """Validate one conservation source-group selection rule."""

    selection_rule = source_group.get("selection_rule")
    profile_id = str(source_group.get("profile_id", "")).strip()
    if not isinstance(selection_rule, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_selection_rule",
                message=f"source group {profile_id!r} must declare selection_rule",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].selection_rule",
            )
        )
        return
    _validate_shared_filter_fields(issues, selection_rule=selection_rule, profile_id=profile_id, index=index)
    if profile_id == "ec86_clade9_conservation_v1":
        _validate_ec86_clade9_rule(issues, selection_rule=selection_rule, index=index)
    if profile_id == "ec86_iia3_cluster42_1_conservation_v1":
        _validate_ec86_iia3_cluster42_1_rule(issues, selection_rule=selection_rule, index=index)


def _validate_shared_filter_fields(
    issues: list[ContractIssue],
    *,
    selection_rule: Mapping[str, Any],
    profile_id: str,
    index: int,
) -> None:
    included_records = selection_rule.get("included_records")
    if included_records in _FORBIDDEN_CONSERVATION_DENOMINATOR_RULES:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.forbidden_full_roster_denominator",
                message=(
                    "Mestre S1 full-roster rows may be retained as classification context, "
                    "but not as a Phase 1 conservation scoring denominator"
                ),
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.included_records"
                ),
            )
        )
    if not _is_positive_number(selection_rule.get("query_coverage_minimum")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_query_coverage_minimum",
                message=f"source group {profile_id!r} must declare positive query_coverage_minimum",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.query_coverage_minimum"
                ),
            )
        )
    if len(selection_rule.get("identity_range") or []) != 2 or len(selection_rule.get("length_range_aa") or []) != 2:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_filter_ranges",
                message=f"source group {profile_id!r} must declare two-value identity and length ranges",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].selection_rule",
            )
        )
    if not _as_string_list(selection_rule.get("motif_qc_markers")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_motif_qc_markers",
                message=f"source group {profile_id!r} must declare motif QC markers",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.motif_qc_markers"
                ),
            )
        )
    hard_reject_filters = set(_as_string_list(selection_rule.get("hard_reject_filters")))
    for required_filter in {
        "missing_catalytic_rt_core",
        "below_query_coverage_minimum",
        "outside_identity_range",
        "outside_length_range",
        "obvious_fragment",
        "unresolved_long_fusion",
    }:
        if required_filter not in hard_reject_filters:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.missing_hard_reject_filter",
                    message=f"source group {profile_id!r} must declare hard reject filter {required_filter!r}",
                    path=(
                        "workbench/provenance/conservation-sources.yaml:"
                        f"source_groups[{index}].selection_rule.hard_reject_filters"
                    ),
                )
            )


def _validate_ec86_clade9_rule(
    issues: list[ContractIssue],
    *,
    selection_rule: Mapping[str, Any],
    index: int,
) -> None:
    if selection_rule.get("included_records") != "mestre_s1_ec86_rt_clade9_after_qc":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.ec86_clade9_roster_scope_mismatch",
                message=(
                    "ec86_clade9_conservation_v1 must use the Mestre Ec86 RT clade 9 homolog panel, "
                    "not the full Mestre S1 roster or an arbitrary cap-first subset"
                ),
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.included_records"
                ),
            )
        )
    if selection_rule.get("parent_rt_clade") != 9:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.ec86_clade9_roster_scope_mismatch",
                message="ec86_clade9_conservation_v1 must declare parent_rt_clade=9",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.parent_rt_clade"
                ),
            )
        )
    if selection_rule.get("full_roster_role") != "context_only_not_conservation_denominator":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_full_roster_context_boundary",
                message="ec86_clade9_conservation_v1 must state that the full Mestre roster is context-only",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.full_roster_role"
                ),
            )
        )
    if selection_rule.get("candidate_pool_records") != "mestre_s1_all_retron_rt_records_context":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_ec86_clade9_candidate_pool",
                message="ec86_clade9_conservation_v1 must declare the full Mestre roster as a candidate pool only",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.candidate_pool_records"
                ),
            )
        )


def _validate_ec86_iia3_cluster42_1_rule(
    issues: list[ContractIssue],
    *,
    selection_rule: Mapping[str, Any],
    index: int,
) -> None:
    expected_pairs = {
        "parent_rt_clade": 9,
        "retron_subtype": "II-A3",
        "cluster_domain": "42_1",
    }
    for field, expected in expected_pairs.items():
        if selection_rule.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.ec86_iia3_cluster42_1_roster_scope_mismatch",
                    message=f"ec86_iia3_cluster42_1_conservation_v1 must declare {field}={expected!r}",
                    path=(
                        f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].selection_rule.{field}"
                    ),
                )
            )
