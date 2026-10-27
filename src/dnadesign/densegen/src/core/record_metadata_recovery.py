"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/record_metadata_recovery.py

Recover DenseGen record metadata from stable source labels when older record.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

_PLAN_POOL_PREFIX = "plan_pool__"
_RECOVERY_FLAG_COLUMN = "densegen__metadata_inferred_from_source"
_MISSING_TEXT = {"", "nan", "none", "null"}


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in _MISSING_TEXT:
        return None
    return text


def infer_densegen_input_name_from_source(source: object) -> str | None:
    return _normalize_optional_text(source)


def infer_densegen_plan_from_source(source: object) -> str | None:
    source_text = _normalize_optional_text(source)
    if source_text is None or not source_text.startswith(_PLAN_POOL_PREFIX):
        return None
    suffix = source_text[len(_PLAN_POOL_PREFIX) :]
    if not suffix:
        return None
    parts = [part for part in suffix.split("__") if part]
    if not parts:
        return None
    normalized_parts = [parts[0]]
    for part in parts[1:]:
        if "=" in part:
            normalized_parts.append(part)
            continue
        if "_" in part:
            key, value = part.split("_", 1)
            key = str(key).strip()
            value = str(value).strip()
            if key and value:
                normalized_parts.append(f"{key}={value}")
                continue
        normalized_parts.append(part)
    return "__".join(normalized_parts)


def recover_densegen_metadata_from_source(
    records_df: pd.DataFrame,
    *,
    source_col: str = "source",
    plan_col: str = "densegen__plan",
    input_col: str = "densegen__input_name",
    recovery_flag_col: str = _RECOVERY_FLAG_COLUMN,
) -> pd.DataFrame:
    if records_df.empty or source_col not in records_df.columns:
        return records_df

    frame = records_df.copy()
    source_series = frame[source_col].map(_normalize_optional_text)
    recovered_mask = pd.Series(False, index=frame.index, dtype=bool)

    if plan_col not in frame.columns:
        frame[plan_col] = None
    if input_col not in frame.columns:
        frame[input_col] = None

    inferred_plan = source_series.map(infer_densegen_plan_from_source)
    inferred_input = source_series.map(infer_densegen_input_name_from_source)

    plan_missing = frame[plan_col].map(_normalize_optional_text).isna()
    input_missing = frame[input_col].map(_normalize_optional_text).isna()

    plan_recovered = plan_missing & inferred_plan.notna()
    input_recovered = input_missing & inferred_input.notna()

    if bool(plan_recovered.any()):
        frame.loc[plan_recovered, plan_col] = inferred_plan.loc[plan_recovered]
        recovered_mask |= plan_recovered
    if bool(input_recovered.any()):
        frame.loc[input_recovered, input_col] = inferred_input.loc[input_recovered]
        recovered_mask |= input_recovered

    if bool(recovered_mask.any()):
        frame[recovery_flag_col] = recovered_mask
    elif recovery_flag_col in frame.columns:
        frame[recovery_flag_col] = frame[recovery_flag_col].fillna(False).astype(bool)
    return frame


__all__ = [
    "infer_densegen_input_name_from_source",
    "infer_densegen_plan_from_source",
    "recover_densegen_metadata_from_source",
]
