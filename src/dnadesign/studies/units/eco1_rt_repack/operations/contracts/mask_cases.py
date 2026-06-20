"""Eco1 RT repack contract validation primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _require_known_phase
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _REQUIRED_MASK_CASES,
    _STUDY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_conservative_mask_cases_payload(
    payload: Mapping[str, Any],
    *,
    phase: str = "phase0_scaffold",
) -> ContractReport:
    """Validate conservative-mask pass/fail case coverage."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []

    if payload.get("study_id") != _STUDY_ID:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask_cases.study_id_mismatch",
                message=f"mask case study_id must be {_STUDY_ID!r}",
                path="study_id",
            )
        )

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return ContractReport(
            phase=phase,
            issues=(
                *issues,
                ContractIssue(
                    check_id="eco1_rt.mask_cases.missing_cases",
                    message="mask cases payload must contain a cases list",
                    path="cases",
                ),
            ),
        )

    seen_case_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask_cases.invalid_case",
                    message="each mask case must be a mapping",
                    path=f"cases[{index}]",
                )
            )
            continue
        case_id = str(case.get("id", "")).strip()
        expected = str(case.get("expected", "")).strip()
        reason = str(case.get("reason", "")).strip()
        if not case_id:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask_cases.missing_case_id",
                    message="each mask case must declare an id",
                    path=f"cases[{index}].id",
                )
            )
        else:
            seen_case_ids.add(case_id)
        if expected not in {"fail", "pass"}:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask_cases.invalid_expected_value",
                    message="mask case expected value must be 'fail' or 'pass'",
                    path=f"cases[{index}].expected",
                )
            )
        if not reason:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask_cases.missing_reason",
                    message="each mask case must declare a reason",
                    path=f"cases[{index}].reason",
                )
            )

    for case_id in sorted(_REQUIRED_MASK_CASES - seen_case_ids):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask_cases.missing_required_case",
                message=f"conservative mask cases must include {case_id!r}",
                path="cases",
            )
        )

    return ContractReport(phase=phase, issues=tuple(issues))
