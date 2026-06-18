"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/active_targets.py

Active target declarations for DenseGen TFBS learnability v1 labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from .nulls import TFBS_COUNT_COLUMNS, TFBS_COUNT_FRACTION_COLUMNS, TFBS_PRESENCE_COLUMNS, TFBS_SLOT_EVENT_COLUMNS
from .schema import (
    TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
    TFBS_LEARNABILITY_MINIMUM_TARGET_SET,
    TFBS_LEARNABILITY_SENTINEL_TARGET_SET,
)

TargetKind = Literal["integer_count", "binary_presence", "count_fraction", "slot_family_presence"]


@dataclass(frozen=True)
class TfbsExpectedScalarTargetSpec:
    """Study-owned OPAL target config for one v1 scalar label."""

    label_name: str
    label_family_id: str
    target_kind: TargetKind
    target_description: str
    y_space: str
    y_expected_length: int
    label_input_columns: tuple[str, ...]
    transforms_y: Mapping[str, Any]
    objectives: tuple[Mapping[str, Any], ...]
    score_ref: str
    objective_mode: str
    score_label: str
    score_title_label: str
    score_short_label: str
    score_expression: str
    score_axis: Mapping[str, Any]
    channel_labels: tuple[str, ...]
    interpretation_boundary: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_input_columns"] = list(self.label_input_columns)
        payload["objectives"] = [dict(objective) for objective in self.objectives]
        payload["channel_labels"] = list(self.channel_labels)
        return payload


def tfbs_learnability_active_target_spec(label_name: str) -> TfbsExpectedScalarTargetSpec:
    """Return the strict v1 scalar expected-label target spec for OPAL."""

    label = str(label_name)
    if label not in TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES:
        raise ValueError(f"unsupported TFBS learnability active label: {label_name}")
    target_kind = _target_kind(label)
    family_id = _label_family_id(target_kind)
    target_description = _target_description(label, target_kind)
    score_label = _score_label(target_kind, target_description)
    return TfbsExpectedScalarTargetSpec(
        label_name=label,
        label_family_id=family_id,
        target_kind=target_kind,
        target_description=target_description,
        y_space="numeric_vector",
        y_expected_length=1,
        label_input_columns=("id", "sequence", label),
        transforms_y={
            "name": "vector_from_table_v1",
            "params": {
                "id_column": "id",
                "sequence_column": "sequence",
                "value_columns": [label],
            },
        },
        objectives=(
            {
                "name": "vector_channel_v1",
                "params": {
                    "channel_index": 0,
                    "channel_name": label,
                    "mode": "maximize",
                },
            },
        ),
        score_ref=f"vector_channel_v1/{label}",
        objective_mode="maximize",
        score_label=score_label,
        score_title_label=score_label,
        score_short_label=_score_short_label(label, target_kind),
        score_expression=f"selection_score = E[{label} | X]",
        score_axis=_score_axis(target_kind),
        channel_labels=(target_description,),
        interpretation_boundary=(
            "Synthetic DenseGen construction-label learnability target. "
            "The score is an expected scalar label for ranking, not a calibrated biological probability "
            "or wet-lab phenotype."
        ),
    )


def tfbs_learnability_sentinel_target_specs() -> tuple[TfbsExpectedScalarTargetSpec, ...]:
    """Return the recommended Stage B sentinel target specs."""

    return tuple(tfbs_learnability_active_target_spec(label) for label in TFBS_LEARNABILITY_SENTINEL_TARGET_SET)


def validate_tfbs_learnability_target_set(label_names: tuple[str, ...]) -> tuple[str, ...]:
    """Fail fast unless target labels are unique v1 active labels."""

    requested = tuple(dict.fromkeys(str(label).strip() for label in label_names if str(label).strip()))
    if not requested:
        raise ValueError("at least one TFBS learnability target label is required")
    unknown = sorted(set(requested) - set(TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported TFBS learnability target label(s): {unknown}")
    return requested


def minimum_tfbs_learnability_target_set() -> tuple[str, ...]:
    """Return the v1 minimum production target set from the spec."""

    return TFBS_LEARNABILITY_MINIMUM_TARGET_SET


def _target_kind(label_name: str) -> TargetKind:
    if label_name in TFBS_COUNT_COLUMNS:
        return "integer_count"
    if label_name in TFBS_PRESENCE_COLUMNS:
        return "binary_presence"
    if label_name in TFBS_COUNT_FRACTION_COLUMNS:
        return "count_fraction"
    if label_name in TFBS_SLOT_EVENT_COLUMNS:
        return "slot_family_presence"
    raise ValueError(f"unsupported TFBS learnability active label: {label_name}")


def _label_family_id(target_kind: TargetKind) -> str:
    if target_kind == "integer_count":
        return "tf_family_count"
    if target_kind == "binary_presence":
        return "tf_family_presence"
    if target_kind == "count_fraction":
        return "tf_family_count_fraction"
    if target_kind == "slot_family_presence":
        return "tf_slot_family_presence"
    raise ValueError(f"unsupported target kind: {target_kind}")


def _target_description(label_name: str, target_kind: TargetKind) -> str:
    family = _family_display(label_name)
    if target_kind == "integer_count":
        return f"{family} count"
    if target_kind == "binary_presence":
        return f"{family} present"
    if target_kind == "count_fraction":
        return f"{family} count / 3"
    if target_kind == "slot_family_presence":
        return f"{family} in {_slot_display(label_name)} TFBS slot"
    raise ValueError(f"unsupported target kind: {target_kind}")


def _score_label(target_kind: TargetKind, target_description: str) -> str:
    if target_kind in {"binary_presence", "slot_family_presence"}:
        return f"Predicted P({target_description})"
    if target_kind == "count_fraction":
        return f"Predicted E[{target_description}]"
    return f"Predicted E[{target_description}]"


def _score_short_label(label_name: str, target_kind: TargetKind) -> str:
    if target_kind in {"binary_presence", "slot_family_presence"}:
        return f"predicted P({label_name})"
    return f"predicted E[{label_name}]"


def _score_axis(target_kind: TargetKind) -> dict[str, Any]:
    if target_kind in {"binary_presence", "count_fraction", "slot_family_presence"}:
        return {
            "scale_class": "tfbs_expected_scalar_unit_interval",
            "limits": [0.0, 1.0],
            "include_zero_tick": True,
        }
    return {
        "scale_class": "tfbs_expected_scalar_count",
        "limits": [0.0, 3.0],
        "include_zero_tick": True,
    }


def _family_display(label_name: str) -> str:
    if label_name.startswith("lexA"):
        return "LexA"
    if label_name.startswith("cpxR_or_baeR"):
        return "CpxR or BaeR"
    if label_name.startswith("cpxR"):
        return "CpxR"
    if label_name.startswith("baeR"):
        return "BaeR"
    raise ValueError(f"cannot infer TF family display name from label: {label_name}")


def _slot_display(label_name: str) -> str:
    if label_name.endswith("slot0"):
        return "leftmost"
    if label_name.endswith("slot1"):
        return "middle"
    if label_name.endswith("slot2"):
        return "rightmost"
    raise ValueError(f"cannot infer TFBS slot display name from label: {label_name}")
