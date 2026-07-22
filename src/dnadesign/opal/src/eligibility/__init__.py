"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/eligibility/__init__.py

Candidate eligibility primitives for OPAL.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .candidate_ids import candidate_id_exclusion
from .contracts import (
    CandidateEligibilityResult,
    CandidateEligibilityRuleResult,
    RestrictionSiteHit,
    RestrictionSiteScanReport,
    RestrictionSiteSpec,
)
from .restriction_sites import scan_restriction_sites
from .runtime import apply_candidate_eligibility

__all__ = [
    "CandidateEligibilityResult",
    "CandidateEligibilityRuleResult",
    "RestrictionSiteHit",
    "RestrictionSiteScanReport",
    "RestrictionSiteSpec",
    "apply_candidate_eligibility",
    "candidate_id_exclusion",
    "scan_restriction_sites",
]
