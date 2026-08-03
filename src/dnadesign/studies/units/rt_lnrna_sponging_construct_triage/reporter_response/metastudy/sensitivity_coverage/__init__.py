"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity_coverage/__init__.py

Canonical sensitivity-coverage contracts, codecs, construction, and validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .building import build_sensitivity_coverage
from .contracts import (
    SENSITIVITY_COVERAGE_CONTRACT_ID,
    SensitivityCoverageEntry,
    SensitivityCoverageLedger,
    SensitivitySubjectCoordinate,
    declared_sensitivity_reduction_ids,
)
from .serialization import (
    parse_sensitivity_coverage,
    sensitivity_coverage_payload,
    sensitivity_coverage_receipt_payload,
    validate_sensitivity_coverage_receipt_payloads,
)
from .validation import (
    sensitivity_profile_coordinate_key,
    validate_sensitivity_coverage,
    validate_sensitivity_coverage_set,
)

__all__ = [
    "SENSITIVITY_COVERAGE_CONTRACT_ID",
    "SensitivityCoverageEntry",
    "SensitivityCoverageLedger",
    "SensitivitySubjectCoordinate",
    "build_sensitivity_coverage",
    "declared_sensitivity_reduction_ids",
    "parse_sensitivity_coverage",
    "sensitivity_coverage_payload",
    "sensitivity_coverage_receipt_payload",
    "sensitivity_profile_coordinate_key",
    "validate_sensitivity_coverage",
    "validate_sensitivity_coverage_receipt_payloads",
    "validate_sensitivity_coverage_set",
]
