"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/errors.py

Strict YIU payload-lane error helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

YIU_CONTRACT_UNKNOWN = "YIU_CONTRACT_UNKNOWN"
YIU_SCHEMA_VERSION_UNSUPPORTED = "YIU_SCHEMA_VERSION_UNSUPPORTED"
YIU_INPUT_KIND_UNKNOWN = "YIU_INPUT_KIND_UNKNOWN"
YIU_INPUT_MUTUALLY_EXCLUSIVE = "YIU_INPUT_MUTUALLY_EXCLUSIVE"
YIU_SEQUENCE_INVALID = "YIU_SEQUENCE_INVALID"
YIU_PATH_INVALID = "YIU_PATH_INVALID"
YIU_SAMPLE_HIT_SEQUENCE_MISSING = "YIU_SAMPLE_HIT_SEQUENCE_MISSING"
YIU_SAMPLE_HIT_AMBIGUOUS = "YIU_SAMPLE_HIT_AMBIGUOUS"
YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE = "YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE"
YIU_SAMPLE_HIT_PWM_MISSING = "YIU_SAMPLE_HIT_PWM_MISSING"
YIU_PWM_CONTEXT_INVALID = "YIU_PWM_CONTEXT_INVALID"
YIU_PWM_CONTEXT_REQUIRED = "YIU_PWM_CONTEXT_REQUIRED"
YIU_SPLIT_POLICY_UNSAT = "YIU_SPLIT_POLICY_UNSAT"
YIU_SPLIT_INDEX_INVALID = "YIU_SPLIT_INDEX_INVALID"
YIU_OVERHANG_INCOMPATIBLE = "YIU_OVERHANG_INCOMPATIBLE"
YIU_BULGE_MASK_INVALID = "YIU_BULGE_MASK_INVALID"
YIU_JUNCTION_INVALID = "YIU_JUNCTION_INVALID"
YIU_MISMATCH_INVALID = "YIU_MISMATCH_INVALID"
YIU_NO_FEASIBLE_PLAN = "YIU_NO_FEASIBLE_PLAN"
YIU_RENDER_FAILED = "YIU_RENDER_FAILED"
YIU_BUNDLE_INVALID = "YIU_BUNDLE_INVALID"


class YiuContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


def raise_yiu_error(code: str, message: str) -> None:
    raise YiuContractError(code, message)


class NoFeasiblePlanError(YiuContractError):
    def __init__(self, message: str):
        super().__init__(YIU_NO_FEASIBLE_PLAN, message)
