"""Active OPAL target contracts for the DenseGen motif QA probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import (
    ACTIVE_LABEL_FAMILY_ID,
    CAMPAIGNS,
    DENSEGEN_PLAN_LOGIC4_COLUMNS,
)
from .label_families import TF_FAMILY_COUNT_COLUMNS

TF_COUNT_OBJECTIVE_COLUMNS = (
    "tf_count__lexA",
    "tf_count__cpxR_plus_baeR",
    "tf_count__lexA_plus_cpxR_plus_baeR",
)

_TF_COUNT_TARGET_BY_CAMPAIGN: dict[str, tuple[int, str, str]] = {
    "cipro": (0, "tf_count__lexA", "LexA count"),
    "ethanol": (1, "tf_count__cpxR_plus_baeR", "CpxR plus BaeR count"),
    "dual": (2, "tf_count__lexA_plus_cpxR_plus_baeR", "LexA plus CpxR plus BaeR count"),
}


@dataclass(frozen=True)
class ActiveTargetSpec:
    """Study-owned description of one OPAL-trainable target surface."""

    label_family_id: str
    campaign_key: str
    target_key: str
    target_channel: str
    target_description: str
    y_space: str
    y_expected_length: int
    label_input_columns: tuple[str, ...]
    transforms_y: Mapping[str, Any]
    objectives: tuple[Mapping[str, Any], ...]
    score_ref: str
    objective_mode: str
    plot_family: str
    channel_labels: tuple[str, ...]
    reference_vector: tuple[float, ...] = ()


def supported_active_label_families() -> tuple[str, ...]:
    return (ACTIVE_LABEL_FAMILY_ID, "tf_family_count")


def active_target_spec(label_family_id: str, campaign_key: str) -> ActiveTargetSpec:
    family = str(label_family_id)
    campaign = str(campaign_key)
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign key: {campaign_key}")
    if family == ACTIVE_LABEL_FAMILY_ID:
        return _plan_logic4_target_spec(campaign)
    if family == "tf_family_count":
        return _tf_count_target_spec(campaign)
    raise ValueError(f"unsupported active label family: {label_family_id}")


def validate_active_label_families(label_family_ids: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(value).strip() for value in label_family_ids if str(value).strip()))
    if not requested:
        raise ValueError("at least one active label family is required")
    supported = set(supported_active_label_families())
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"unsupported active label family id(s): {unknown}")
    return requested


def with_active_target_columns(labels: pd.DataFrame, label_family_id: str) -> pd.DataFrame:
    """Return labels with any study-owned derived active-target columns added."""

    if str(label_family_id) == ACTIVE_LABEL_FAMILY_ID:
        return labels.copy()
    if str(label_family_id) != "tf_family_count":
        raise ValueError(f"unsupported active label family: {label_family_id}")
    missing = sorted(set(TF_FAMILY_COUNT_COLUMNS) - set(labels.columns))
    if missing:
        raise ValueError(f"label frame missing TF-count source column(s): {missing}")
    out = labels.copy()
    lex_a = _finite_numeric(out["tf_family__lexA__count"], column="tf_family__lexA__count")
    cpx_r = _finite_numeric(out["tf_family__cpxR__count"], column="tf_family__cpxR__count")
    bae_r = _finite_numeric(out["tf_family__baeR__count"], column="tf_family__baeR__count")
    out["tf_count__lexA"] = lex_a
    out["tf_count__cpxR_plus_baeR"] = cpx_r + bae_r
    out["tf_count__lexA_plus_cpxR_plus_baeR"] = lex_a + cpx_r + bae_r
    return out


def target_values_for_labels(
    labels: pd.DataFrame,
    *,
    label_family_id: str,
    campaign_key: str,
) -> pd.Series:
    """Return the active target values indexed by candidate id for continuous/count metrics."""

    if "id" not in labels.columns:
        raise ValueError("label frame missing id column")
    spec = active_target_spec(label_family_id, campaign_key)
    if spec.label_family_id != "tf_family_count":
        raise ValueError(f"target_values_for_labels only supports continuous active targets, got {label_family_id}")
    frame = with_active_target_columns(labels, spec.label_family_id)
    if spec.target_channel not in frame.columns:
        raise ValueError(f"active target column missing after derivation: {spec.target_channel}")
    values = _finite_numeric(frame[spec.target_channel], column=spec.target_channel)
    return pd.Series(values, index=frame["id"].astype(str), name=spec.target_channel)


def _plan_logic4_target_spec(campaign_key: str) -> ActiveTargetSpec:
    campaign = CAMPAIGNS[campaign_key]
    target_logic4 = tuple(float(value) for value in campaign["target_logic4"])
    return ActiveTargetSpec(
        label_family_id=ACTIVE_LABEL_FAMILY_ID,
        campaign_key=campaign_key,
        target_key=str(campaign["target_class"]),
        target_channel=str(campaign["target_class"]),
        target_description=f"DenseGen plan-logic4 objective for {campaign_key}",
        y_space="numeric_vector",
        y_expected_length=len(DENSEGEN_PLAN_LOGIC4_COLUMNS),
        label_input_columns=("id", "sequence", *DENSEGEN_PLAN_LOGIC4_COLUMNS),
        transforms_y={
            "name": "vector_from_table_v1",
            "params": {
                "id_column": "id",
                "sequence_column": "sequence",
                "value_columns": list(DENSEGEN_PLAN_LOGIC4_COLUMNS),
            },
        },
        objectives=(
            {
                "name": "vector_target_similarity_v1",
                "params": {
                    "target_vector": list(target_logic4),
                },
            },
        ),
        score_ref="vector_target_similarity_v1/negative_mse",
        objective_mode="maximize",
        plot_family="generic_numeric_vector",
        channel_labels=DENSEGEN_PLAN_LOGIC4_COLUMNS,
        reference_vector=target_logic4,
    )


def _tf_count_target_spec(campaign_key: str) -> ActiveTargetSpec:
    channel_index, channel_name, description = _TF_COUNT_TARGET_BY_CAMPAIGN[campaign_key]
    return ActiveTargetSpec(
        label_family_id="tf_family_count",
        campaign_key=campaign_key,
        target_key=channel_name,
        target_channel=channel_name,
        target_description=description,
        y_space="numeric_vector",
        y_expected_length=len(TF_COUNT_OBJECTIVE_COLUMNS),
        label_input_columns=("id", "sequence", *TF_COUNT_OBJECTIVE_COLUMNS),
        transforms_y={
            "name": "vector_from_table_v1",
            "params": {
                "id_column": "id",
                "sequence_column": "sequence",
                "value_columns": list(TF_COUNT_OBJECTIVE_COLUMNS),
            },
        },
        objectives=(
            {
                "name": "vector_channel_v1",
                "params": {
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "mode": "maximize",
                },
            },
        ),
        score_ref=f"vector_channel_v1/{channel_name}",
        objective_mode="maximize",
        plot_family="generic_numeric_vector",
        channel_labels=TF_COUNT_OBJECTIVE_COLUMNS,
    )


def _finite_numeric(series: pd.Series, *, column: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"label column contains non-finite value(s): {column}")
    return values
