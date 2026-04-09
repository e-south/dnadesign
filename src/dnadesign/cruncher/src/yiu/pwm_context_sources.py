"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context_sources.py

Concrete YIU PWM context source loaders behind the public resolution facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.errors import YIU_PWM_CONTEXT_INVALID, raise_yiu_error
from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload
from dnadesign.cruncher.yiu.pwm_context_io import load_yaml_mapping, resolve_workspace_file_path
from dnadesign.cruncher.yiu.pwm_context_sample_context import sample_context_to_model
from dnadesign.cruncher.yiu.spec_pwm_models import PwmOptimizationSpec, YiuPwmContextV1


def resolve_context_model(
    *,
    pwm_spec: PwmOptimizationSpec,
    resolved_input: ResolvedInputPayload,
    workspace_root: Path,
    spec_name: str,
) -> YiuPwmContextV1:
    if pwm_spec.source.kind == "inline":
        assert pwm_spec.source.inline_context is not None
        return pwm_spec.source.inline_context
    if pwm_spec.source.kind == "file":
        assert pwm_spec.source.path is not None
        path = resolve_workspace_file_path(pwm_spec.source.path, workspace_root=workspace_root)
        return YiuPwmContextV1.model_validate(load_yaml_mapping(path))
    if pwm_spec.source.kind == "sample_context":
        return sample_context_to_model(resolved_input=resolved_input, spec_name=spec_name)
    raise_yiu_error(
        YIU_PWM_CONTEXT_INVALID,
        f"unsupported PWM source kind for active PWM mode: {pwm_spec.source.kind!r}",
    )


__all__ = ["resolve_context_model"]
