"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/payload.py

Compatibility re-exports for YIU payload and spec models.
Prefer `dnadesign.cruncher.yiu.domain_models` and
`dnadesign.cruncher.yiu.spec_models`.

Module Author(s): Eric J. South
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
from dnadesign.cruncher.yiu.spec_input_models import (
    InputSpec,
    SampleHitInput,
    UserSequenceInput,
    YiuSpecRoot,
)
from dnadesign.cruncher.yiu.spec_pwm_models import (
    PwmObjectiveSpec,
    PwmOptimizationSpec,
    PwmSourceSpec,
    YiuPwmContextV1,
    YiuPwmMotifInstanceV1,
)
from dnadesign.cruncher.yiu.spec_rendering_models import (
    JunctionOptimizationSpec as JunctionSelectionSpec,
)
from dnadesign.cruncher.yiu.spec_rendering_models import (
    MismatchesSpec,
    OptimizationSpec,
    OutputSpec,
    YiuPayloadRenderingSpec,
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
