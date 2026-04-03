"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/payload.py

Compatibility re-exports for YIU payload and spec models.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.yiu.domain_models import (
    JunctionSelection as JunctionWindow,
)
from dnadesign.cruncher.yiu.domain_models import (
    MismatchSelection as MismatchSite,
)
from dnadesign.cruncher.yiu.domain_models import (
    NormalizedMotifContext,
    NormalizedPayload,
    OptimizationDecision,
    OptimizationObjective,
    OptimizationWinner,
)
from dnadesign.cruncher.yiu.spec_models import (
    InputSpec,
    MismatchesSpec,
    OptimizationSpec,
    OutputSpec,
    PwmObjectiveSpec,
    PwmOptimizationSpec,
    PwmSourceSpec,
    SampleHitInput,
    UserSequenceInput,
    YiuPayloadRenderingSpec,
    YiuPwmContextV1,
    YiuPwmMotifInstanceV1,
    YiuSpecRoot,
)
from dnadesign.cruncher.yiu.spec_models import (
    JunctionOptimizationSpec as JunctionSelectionSpec,
)

__all__ = [
    "InputSpec",
    "JunctionSelectionSpec",
    "JunctionWindow",
    "MismatchSite",
    "MismatchesSpec",
    "NormalizedMotifContext",
    "NormalizedPayload",
    "OptimizationDecision",
    "OptimizationObjective",
    "OptimizationSpec",
    "OptimizationWinner",
    "OutputSpec",
    "PwmObjectiveSpec",
    "PwmOptimizationSpec",
    "PwmSourceSpec",
    "SampleHitInput",
    "UserSequenceInput",
    "YiuPayloadRenderingSpec",
    "YiuPwmContextV1",
    "YiuPwmMotifInstanceV1",
    "YiuSpecRoot",
]
