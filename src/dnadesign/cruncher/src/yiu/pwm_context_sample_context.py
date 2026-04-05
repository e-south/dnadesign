"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context_sample_context.py

Sample-backed PWM context resolution for payload-centric YIU workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.yiu.errors import (
    YIU_PWM_CONTEXT_INVALID,
    YIU_PWM_CONTEXT_REQUIRED,
    raise_yiu_error,
)
from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload
from dnadesign.cruncher.yiu.pwm_context_io import load_yaml_mapping
from dnadesign.cruncher.yiu.pwm_context_sample_motifs import (
    coerce_detail_entries,
    motif_from_occurrence_row,
    motif_from_sample_detail,
    parse_per_tf_json,
)
from dnadesign.cruncher.yiu.pwm_context_sample_occurrences import load_selected_occurrence_rows
from dnadesign.cruncher.yiu.spec_pwm_models import (
    YiuPwmContextV1,
    YiuPwmMotifInstanceV1,
)


def sample_context_to_model(
    *,
    resolved_input: ResolvedInputPayload,
    spec_name: str,
) -> YiuPwmContextV1:
    if resolved_input.hit_row is None or resolved_input.sample_workspace_root is None:
        raise_yiu_error(
            YIU_PWM_CONTEXT_REQUIRED,
            "sample_context PWM resolution requires a sample_hit input with a resolved hit row and source workspace",
        )
    config_used_path = resolved_input.sample_workspace_root / "outputs" / "meta" / "config_used.yaml"
    if not config_used_path.exists():
        raise_yiu_error(
            YIU_PWM_CONTEXT_REQUIRED,
            f"sample_context PWM resolution expected config_used.yaml at {config_used_path}",
        )
    payload = load_yaml_mapping(config_used_path)
    cruncher_cfg = payload.get("cruncher")
    if not isinstance(cruncher_cfg, dict):
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"missing cruncher block in {config_used_path}")
    pwms_info = cruncher_cfg.get("pwms_info")
    if not isinstance(pwms_info, dict) or not pwms_info:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"missing cruncher.pwms_info in {config_used_path}")
    motifs: list[YiuPwmMotifInstanceV1] = []
    hit_id = str(resolved_input.provenance.get("hit_id", "")).strip()
    if hit_id:
        occurrences_path = (
            resolved_input.sample_workspace_root / "outputs" / "optimize" / "tables" / "elites_occurrences.parquet"
        )
        for row in load_selected_occurrence_rows(
            sample_workspace_root=resolved_input.sample_workspace_root,
            elite_id=hit_id,
        ):
            pwm_info = pwms_info.get(str(row.get("tf", "")).strip())
            if not isinstance(pwm_info, dict):
                continue
            motifs.append(
                motif_from_occurrence_row(
                    row=row,
                    pwm_info=pwm_info,
                    source_ref=str(occurrences_path.resolve()),
                )
            )
    if not motifs:
        per_tf_payload = parse_per_tf_json(resolved_input.hit_row.get("per_tf_json"))
        if not per_tf_payload:
            raise_yiu_error(
                YIU_PWM_CONTEXT_REQUIRED,
                "sample_context PWM resolution requires per_tf_json details or selected elites_occurrences rows "
                "for the selected sample-hit row",
            )
        for tf_name, raw_detail in sorted(per_tf_payload.items()):
            pwm_info = pwms_info.get(tf_name)
            if not isinstance(pwm_info, dict):
                continue
            for index, detail in enumerate(coerce_detail_entries(raw_detail)):
                motifs.append(
                    motif_from_sample_detail(
                        tf_name=tf_name,
                        pwm_info=pwm_info,
                        detail=detail,
                        source_ref=str(config_used_path.resolve()),
                        index=index,
                    )
                )
    if not motifs:
        raise_yiu_error(
            YIU_PWM_CONTEXT_REQUIRED,
            "sample_context PWM resolution did not yield any usable motif instances from the selected hit row",
        )
    return YiuPwmContextV1(
        contract="yiu_pwm_context_v1", schema_version=1, name=f"{spec_name}_sample_context", motifs=motifs
    )


__all__ = ["sample_context_to_model"]
