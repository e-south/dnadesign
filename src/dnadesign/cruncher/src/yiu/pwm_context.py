"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context.py

Resolve YIU PWM context from inline specs, local files, or Sample metadata.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from dnadesign.cruncher.yiu.domain_models import NormalizedMotifContext
from dnadesign.cruncher.yiu.errors import (
    YIU_PATH_INVALID,
    YIU_PWM_CONTEXT_INVALID,
    YIU_PWM_CONTEXT_REQUIRED,
    raise_yiu_error,
)
from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload
from dnadesign.cruncher.yiu.spec_pwm_models import (
    PwmOptimizationSpec,
    YiuPwmContextV1,
    YiuPwmMotifInstanceV1,
    YiuPwmProbabilities,
    YiuPwmProvenance,
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"invalid PWM context YAML at {path} ({exc})")
    if not isinstance(payload, dict):
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"PWM context file must be a YAML mapping: {path}")
    return payload


def _resolve_file_path(raw_path: str, *, workspace_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise_yiu_error(YIU_PATH_INVALID, "optimization.pwm.source.path must not traverse outside the workspace")
    return (workspace_root / path).resolve()


def _coerce_detail_entries(tf_name: str, raw: object) -> Iterable[dict[str, Any]]:
    if isinstance(raw, dict):
        yield dict(raw)
        return
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield dict(item)
        return
    return []


def _parse_per_tf_json(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"sample-hit per_tf_json is invalid JSON ({exc})")
    if not isinstance(payload, dict):
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, "sample-hit per_tf_json must decode to a mapping")
    return {str(key): value for key, value in payload.items()}


def _rows_from_pwm_info(tf_name: str, pwm_info: dict[str, Any]) -> list[list[float]]:
    matrix = pwm_info.get("pwm_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"sample context is missing pwm_matrix for TF {tf_name!r}")
    rows: list[list[float]] = []
    for idx, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) < 4:
            raise_yiu_error(
                YIU_PWM_CONTEXT_INVALID,
                f"sample context pwm_matrix[{idx}] for TF {tf_name!r} must contain at least 4 values",
            )
        parsed = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        if any(not math.isfinite(item) or item < 0.0 for item in parsed):
            raise_yiu_error(
                YIU_PWM_CONTEXT_INVALID,
                f"sample context pwm_matrix[{idx}] for TF {tf_name!r} must be finite and >= 0",
            )
        total = sum(parsed)
        if not math.isfinite(total) or total <= 0.0:
            raise_yiu_error(
                YIU_PWM_CONTEXT_INVALID,
                f"sample context pwm_matrix[{idx}] for TF {tf_name!r} must have positive mass",
            )
        # Sample config_used.yaml rows may be rounded for publication; normalize once at the Yiu boundary.
        rows.append([item / total for item in parsed])
    return rows


def _motif_from_sample_detail(
    *,
    tf_name: str,
    pwm_info: dict[str, Any],
    detail: dict[str, Any],
    source_ref: str,
    index: int,
) -> YiuPwmMotifInstanceV1:
    rows = _rows_from_pwm_info(tf_name, pwm_info)
    start_raw = detail.get("best_start", detail.get("offset", detail.get("start")))
    width_raw = detail.get("width", len(rows))
    strand_raw = str(detail.get("strand", detail.get("reference_strand", ""))).strip()
    if strand_raw.lower() in {"fwd", "+"}:
        reference_strand = "+"
    elif strand_raw.lower() in {"rev", "-"}:
        reference_strand = "-"
    else:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context motif for TF {tf_name!r} is missing a valid strand",
        )
    try:
        start = int(start_raw)
        width = int(width_raw)
    except Exception as exc:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context motif for TF {tf_name!r} has invalid start/width ({exc})",
        )
    if width != len(rows):
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context motif for TF {tf_name!r} width={width} does not match pwm rows={len(rows)}",
        )
    motif_name = str(detail.get("motif_name") or detail.get("motif") or tf_name).strip()
    motif_instance_id = str(
        detail.get("motif_instance_id") or f"{tf_name}:{start}:{start + width}:{reference_strand}:{index}"
    )
    return YiuPwmMotifInstanceV1(
        motif_instance_id=motif_instance_id,
        tf_name=tf_name,
        motif_name=motif_name,
        reference_strand=reference_strand,
        start=start,
        end=start + width,
        probabilities=YiuPwmProbabilities(alphabet=["A", "C", "G", "T"], rows=rows),
        provenance=YiuPwmProvenance(source_kind="sample_context", source_ref=source_ref),
    )


def _load_selected_occurrence_rows(*, sample_workspace_root: Path, elite_id: str) -> list[dict[str, Any]]:
    path = sample_workspace_root / "outputs" / "optimize" / "tables" / "elites_occurrences.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"sample_context occurrence loading requires pandas ({exc})")

    required = {"elite_id", "tf", "occurrence_rank", "start", "end", "strand", "selected"}
    try:
        import pyarrow.parquet as pq  # type: ignore

        columns = set(pq.read_schema(path).names)
    except Exception:
        columns = set(pd.read_parquet(path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample_context occurrence table is missing required columns {missing}: {path}",
        )
    projected = ["elite_id", "tf", "occurrence_rank", "start", "end", "strand", "selected"]
    try:
        frame = pd.read_parquet(path, columns=projected, filters=[("elite_id", "==", elite_id)])
    except Exception:
        frame = pd.read_parquet(path, columns=projected)
        frame = frame.loc[frame["elite_id"].astype(str) == elite_id]
    frame = frame.loc[frame["selected"].astype(bool)]
    if frame.empty:
        return []
    frame = frame.sort_values(["tf", "occurrence_rank", "start", "end", "strand"], kind="stable")
    return frame.to_dict(orient="records")


def _motif_from_occurrence_row(
    *,
    row: dict[str, Any],
    pwm_info: dict[str, Any],
    source_ref: str,
) -> YiuPwmMotifInstanceV1:
    tf_name = str(row.get("tf", "")).strip()
    if not tf_name:
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, "sample context occurrence row is missing tf")
    rows = _rows_from_pwm_info(tf_name, pwm_info)
    try:
        start = int(row.get("start"))
        end = int(row.get("end"))
        occurrence_rank = int(row.get("occurrence_rank"))
    except Exception as exc:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context occurrence row for TF {tf_name!r} has invalid coordinates ({exc})",
        )
    width = end - start
    if width != len(rows):
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context occurrence row for TF {tf_name!r} width={width} does not match pwm rows={len(rows)}",
        )
    strand_raw = str(row.get("strand", "")).strip()
    if strand_raw.lower() in {"fwd", "+"}:
        reference_strand = "+"
    elif strand_raw.lower() in {"rev", "-"}:
        reference_strand = "-"
    else:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample context occurrence row for TF {tf_name!r} is missing a valid strand",
        )
    return YiuPwmMotifInstanceV1(
        motif_instance_id=f"{tf_name}:{start}:{end}:{reference_strand}:{occurrence_rank}",
        tf_name=tf_name,
        motif_name=tf_name,
        reference_strand=reference_strand,
        start=start,
        end=end,
        probabilities=YiuPwmProbabilities(alphabet=["A", "C", "G", "T"], rows=rows),
        provenance=YiuPwmProvenance(source_kind="sample_context", source_ref=source_ref),
    )


def _sample_context_to_model(
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
    payload = _load_yaml_mapping(config_used_path)
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
        for row in _load_selected_occurrence_rows(
            sample_workspace_root=resolved_input.sample_workspace_root,
            elite_id=hit_id,
        ):
            pwm_info = pwms_info.get(str(row.get("tf", "")).strip())
            if not isinstance(pwm_info, dict):
                continue
            motifs.append(
                _motif_from_occurrence_row(
                    row=row,
                    pwm_info=pwm_info,
                    source_ref=str(occurrences_path.resolve()),
                )
            )
    if not motifs:
        per_tf_payload = _parse_per_tf_json(resolved_input.hit_row.get("per_tf_json"))
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
            for index, detail in enumerate(_coerce_detail_entries(tf_name, raw_detail)):
                motifs.append(
                    _motif_from_sample_detail(
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


def _resolve_context_model(
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
        path = _resolve_file_path(pwm_spec.source.path, workspace_root=workspace_root)
        return YiuPwmContextV1.model_validate(_load_yaml_mapping(path))
    if pwm_spec.source.kind == "sample_context":
        return _sample_context_to_model(resolved_input=resolved_input, spec_name=spec_name)
    raise_yiu_error(
        YIU_PWM_CONTEXT_INVALID,
        f"unsupported PWM source kind for active PWM mode: {pwm_spec.source.kind!r}",
    )


def resolve_motif_context(
    *,
    pwm_spec: PwmOptimizationSpec,
    resolved_input: ResolvedInputPayload,
    workspace_root: Path,
    spec_name: str,
) -> NormalizedMotifContext:
    if pwm_spec.mode == "none":
        return NormalizedMotifContext(
            requested_mode="none",
            effective=False,
            source_kind="none",
            fallback_reason=None,
            motifs=[],
        )

    try:
        context_model = _resolve_context_model(
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
