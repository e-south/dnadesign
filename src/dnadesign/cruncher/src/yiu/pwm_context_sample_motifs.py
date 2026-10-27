"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/pwm_context_sample_motifs.py

Motif-materialization helpers for sample-backed YIU PWM context resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from dnadesign.cruncher.yiu.errors import YIU_PWM_CONTEXT_INVALID, raise_yiu_error
from dnadesign.cruncher.yiu.spec_pwm_models import (
    YiuPwmMotifInstanceV1,
    YiuPwmProbabilities,
    YiuPwmProvenance,
)


def coerce_detail_entries(raw: object) -> Iterable[dict[str, Any]]:
    if isinstance(raw, dict):
        yield dict(raw)
        return
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield dict(item)
        return
    return []


def parse_per_tf_json(raw: object) -> dict[str, object]:
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
        # Sample config_used.yaml rows may be rounded for publication; normalize once at the YIU boundary.
        rows.append([item / total for item in parsed])
    return rows


def _resolve_reference_strand(raw: object, *, ctx: str) -> str:
    strand_raw = str(raw or "").strip()
    if strand_raw.lower() in {"fwd", "+"}:
        return "+"
    if strand_raw.lower() in {"rev", "-"}:
        return "-"
    raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"{ctx} is missing a valid strand")


def motif_from_sample_detail(
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
    reference_strand = _resolve_reference_strand(
        detail.get("strand", detail.get("reference_strand")),
        ctx=f"sample context motif for TF {tf_name!r}",
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


def motif_from_occurrence_row(
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
    reference_strand = _resolve_reference_strand(
        row.get("strand"),
        ctx=f"sample context occurrence row for TF {tf_name!r}",
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


__all__ = [
    "coerce_detail_entries",
    "motif_from_occurrence_row",
    "motif_from_sample_detail",
    "parse_per_tf_json",
]
