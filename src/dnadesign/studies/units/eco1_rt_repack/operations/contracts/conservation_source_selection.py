"""Selection-rule checks for Eco1 conservation source groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _as_string_list,
    _is_positive_int,
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
    if profile_id == "broad_tao_homolog_rt":
        _validate_broad_tao_homolog_rule(issues, selection_rule=selection_rule, index=index)
    if profile_id == "eco1_like_retron_rt":
        _validate_eco1_like_rule(issues, selection_rule=selection_rule, index=index)


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
    if not _as_string_list(selection_rule.get("required_motifs")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_required_motifs",
                message=f"source group {profile_id!r} must declare motif requirements",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].selection_rule.required_motifs",
            )
        )


def _validate_broad_tao_homolog_rule(
    issues: list[ContractIssue],
    *,
    selection_rule: Mapping[str, Any],
    index: int,
) -> None:
    if selection_rule.get("included_records") != "mestre_s1_target_centered_bounded_homologs_after_filters":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.broad_tao_roster_scope_mismatch",
                message=(
                    "broad_tao_homolog_rt must use a target-centered bounded homolog selection, "
                    "not the full Mestre S1 roster"
                ),
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.included_records"
                ),
            )
        )
    if not _is_positive_int(selection_rule.get("nonredundant_cap")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_broad_tao_cap",
                message="broad_tao_homolog_rt must declare a positive nonredundant_cap",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.nonredundant_cap"
                ),
            )
        )
    elif int(selection_rule["nonredundant_cap"]) > 250:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.broad_tao_cap_too_large",
                message="broad_tao_homolog_rt nonredundant_cap should stay <=250 before alignment",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.nonredundant_cap"
                ),
            )
        )
    if selection_rule.get("full_roster_role") != "context_only_not_conservation_denominator":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_full_roster_context_boundary",
                message="broad_tao_homolog_rt must state that the full Mestre roster is context-only",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.full_roster_role"
                ),
            )
        )
    if selection_rule.get("candidate_pool_records") != "all_mestre_s1_rt_records_after_filters":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_broad_tao_candidate_pool",
                message="broad_tao_homolog_rt must declare the full Mestre roster as a candidate pool only",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    f"source_groups[{index}].selection_rule.candidate_pool_records"
                ),
            )
        )


def _validate_eco1_like_rule(
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
                    check_id="eco1_rt.conservation.eco1_like_roster_scope_mismatch",
                    message=f"Eco1-like source group must declare {field}={expected!r}",
                    path=(
                        f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].selection_rule.{field}"
                    ),
                )
            )
