"""Runtime application of candidate eligibility rules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config.types import CandidateEligibilityBlock
from ..core.utils import OpalError
from ..registries.eligibility import get_candidate_eligibility_rule
from .contracts import CandidateEligibilityResult, CandidateEligibilityRuleResult


def apply_candidate_eligibility(
    frame: pd.DataFrame,
    eligibility: CandidateEligibilityBlock | None,
) -> CandidateEligibilityResult:
    """Apply configured candidate eligibility rules in order."""

    if eligibility is None or not eligibility.rules:
        return CandidateEligibilityResult(frame=frame, reports=())
    current = frame
    reports: list[dict[str, Any]] = []
    for rule_ref in eligibility.rules:
        rule_name = str(rule_ref.name).strip()
        if not rule_name:
            raise OpalError("candidate_eligibility rule name must be non-empty")
        rule = get_candidate_eligibility_rule(rule_name)
        try:
            result = rule(frame=current, params=dict(rule_ref.params or {}))
        except OpalError:
            raise
        except Exception as exc:
            raise OpalError(f"candidate eligibility rule '{rule_name}' failed: {exc}") from exc
        if not isinstance(result, CandidateEligibilityRuleResult):
            raise OpalError(
                f"candidate eligibility rule '{rule_name}' must return CandidateEligibilityRuleResult, "
                f"got {type(result).__name__}"
            )
        current = result.frame
        reports.append(dict(result.report))
    return CandidateEligibilityResult(frame=current, reports=tuple(reports))
