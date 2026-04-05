"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context.py

Resolve YIU PWM context from inline specs, local files, or Sample metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.domain_models import NormalizedMotifContext
from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload
from dnadesign.cruncher.yiu.pwm_context_sources import resolve_context_model
from dnadesign.cruncher.yiu.spec_pwm_models import PwmOptimizationSpec


def resolve_motif_context(
    *,
    pwm_spec: PwmOptimizationSpec,
    resolved_input: ResolvedInputPayload,
    workspace_root: Path,
    spec_name: str,
) -> NormalizedMotifContext:
    """Resolve PWM context with an explicit degraded-mode contract.

    `use_if_available` records a visible fallback reason instead of hiding source
    failures. `require` continues to fail fast.
    """
    if pwm_spec.mode == "none":
        return NormalizedMotifContext(
            requested_mode="none",
            effective=False,
            source_kind="none",
            fallback_reason=None,
            motifs=[],
        )

    try:
        context_model = resolve_context_model(
            pwm_spec=pwm_spec,
            resolved_input=resolved_input,
            workspace_root=workspace_root,
            spec_name=spec_name,
        )
    except Exception as exc:
        if pwm_spec.mode == "require":
            raise
        return NormalizedMotifContext(
            requested_mode=pwm_spec.mode,
            effective=False,
            source_kind=pwm_spec.source.kind,
            fallback_reason=str(exc),
            motifs=[],
        )

    return NormalizedMotifContext(
        requested_mode=pwm_spec.mode,
        effective=True,
        source_kind=pwm_spec.source.kind,
        fallback_reason=None,
        motifs=context_model.motifs,
    )
