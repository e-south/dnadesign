"""Study-owned synthetic label-family contracts for the DenseGen motif QA probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .constants import (
    ACTIVE_LABEL_FAMILY_ID,
    AXIS_CLASS_TO_DENSEGEN_PLAN_CLASS,
    PASSIVE_LABEL_FAMILY_IDS,
    SFXI_INTENSITY_COLUMNS,
    SFXI_STATE_COLUMNS,
)

TF_FAMILIES = ("lexA", "cpxR", "baeR")
TF_FAMILY_COUNT_COLUMNS = tuple(f"tf_family__{family}__count" for family in TF_FAMILIES)
TF_FAMILY_PRESENCE_COLUMNS = tuple(f"tf_family__{family}__presence" for family in TF_FAMILIES)
DENSEGEN_PLAN_CLASS_COLUMN = "densegen_plan_class"


@dataclass(frozen=True)
class LabelFamilySpec:
    """Manifest-ready description of one study-owned synthetic label family."""

    label_family_id: str
    role: str
    target_type: str
    columns: tuple[str, ...]
    source: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload


def label_family_specs() -> tuple[LabelFamilySpec, ...]:
    return (
        LabelFamilySpec(
            label_family_id=ACTIVE_LABEL_FAMILY_ID,
            role="active",
            target_type="sfxi_vec8",
            columns=(*SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS, "vec8", "axis_class"),
            source="densegen__used_tfbs_detail",
            description=(
                "Binary SFXI-compatible vec8 derived from DenseGen TFBS composition; used by OPAL scratch campaigns."
            ),
        ),
        LabelFamilySpec(
            label_family_id="tf_family_presence",
            role="passive_readout",
            target_type="multi_binary",
            columns=TF_FAMILY_PRESENCE_COLUMNS,
            source="densegen__used_tfbs_detail",
            description="Per-candidate LexA/CpxR/BaeR presence indicators derived from TFBS detail.",
        ),
        LabelFamilySpec(
            label_family_id="tf_family_count",
            role="passive_readout",
            target_type="count_vector",
            columns=TF_FAMILY_COUNT_COLUMNS,
            source="densegen__used_tfbs_detail",
            description="Per-candidate LexA/CpxR/BaeR motif counts derived from TFBS detail.",
        ),
        LabelFamilySpec(
            label_family_id="densegen_plan_class",
            role="passive_readout",
            target_type="categorical",
            columns=(DENSEGEN_PLAN_CLASS_COLUMN,),
            source="densegen__used_tfbs_detail",
            description=(
                "Part-derived DenseGen plan-class proxy. The raw densegen__plan string remains audit metadata, "
                "not the label source."
            ),
        ),
    )


def label_family_ids() -> tuple[str, ...]:
    return tuple(spec.label_family_id for spec in label_family_specs())


def passive_label_family_ids() -> tuple[str, ...]:
    return PASSIVE_LABEL_FAMILY_IDS


def densegen_plan_class_from_axis_class(axis_class: str | None) -> str | None:
    if axis_class is None:
        return None
    return AXIS_CLASS_TO_DENSEGEN_PLAN_CLASS.get(str(axis_class))


def tf_family_columns(*, lex_a: int, cpx_r: int, bae_r: int) -> dict[str, int]:
    counts = {
        "tf_family__lexA__count": int(lex_a),
        "tf_family__cpxR__count": int(cpx_r),
        "tf_family__baeR__count": int(bae_r),
    }
    presence = {
        "tf_family__lexA__presence": int(lex_a > 0),
        "tf_family__cpxR__presence": int(cpx_r > 0),
        "tf_family__baeR__presence": int(bae_r > 0),
    }
    return {**counts, **presence}


def label_family_manifest(
    labels: pd.DataFrame | None = None,
    *,
    active_label_family: str = ACTIVE_LABEL_FAMILY_ID,
    passive_label_families: Sequence[str] = PASSIVE_LABEL_FAMILY_IDS,
) -> dict[str, Any]:
    specs = label_family_specs()
    spec_by_id = {spec.label_family_id: spec for spec in specs}
    unknown = sorted({active_label_family, *map(str, passive_label_families)} - set(spec_by_id))
    if unknown:
        raise ValueError(f"unknown label family id(s): {unknown}")
    payload: dict[str, Any] = {
        "schema_version": "stress_ethanol_cipro_growth.densegen_label_families.v1",
        "active_label_family": active_label_family,
        "passive_label_families": list(map(str, passive_label_families)),
        "families": [spec.to_dict() for spec in specs],
    }
    if labels is not None:
        payload["columns_present"] = sorted(set(labels.columns))
        payload["summaries"] = _label_family_summaries(labels, specs)
    return payload


def _label_family_summaries(labels: pd.DataFrame, specs: Sequence[LabelFamilySpec]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for spec in specs:
        present = [column for column in spec.columns if column in labels.columns]
        summary: dict[str, Any] = {
            "columns_present": present,
            "columns_missing": [column for column in spec.columns if column not in labels.columns],
        }
        if spec.label_family_id == "densegen_plan_class" and DENSEGEN_PLAN_CLASS_COLUMN in labels.columns:
            summary["value_counts"] = _value_counts(labels[DENSEGEN_PLAN_CLASS_COLUMN])
        elif spec.label_family_id in {"tf_family_presence", "tf_family_count"}:
            summary["column_sums"] = {
                column: _numeric_sum(labels[column]) for column in present if column in labels.columns
            }
        elif spec.label_family_id == ACTIVE_LABEL_FAMILY_ID and "axis_class" in labels.columns:
            summary["axis_class_counts"] = _value_counts(labels["axis_class"])
        summaries[spec.label_family_id] = summary
    return summaries


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_sum(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def require_label_family_columns(labels: pd.DataFrame, label_family_ids_: Sequence[str]) -> None:
    specs_by_id = {spec.label_family_id: spec for spec in label_family_specs()}
    missing: dict[str, list[str]] = {}
    for label_family_id in label_family_ids_:
        spec = specs_by_id.get(str(label_family_id))
        if spec is None:
            raise ValueError(f"unknown label family id: {label_family_id}")
        absent = [column for column in spec.columns if column not in labels.columns]
        if absent:
            missing[spec.label_family_id] = absent
    if missing:
        raise ValueError(f"label frame missing label-family column(s): {missing}")


def label_family_records() -> list[Mapping[str, Any]]:
    return [spec.to_dict() for spec in label_family_specs()]
