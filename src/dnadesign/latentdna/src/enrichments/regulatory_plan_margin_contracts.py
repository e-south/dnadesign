"""Contracts for native regulator plan-margin enrichment artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa

from ..contracts.errors import ContractViolationError
from .table_contracts import string_values as _string_values

CONTRACT_NAME = "native_regulator_plan_margin_enrichment"
EPS = 1e-8
PLAN_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SUPPORTED_TAIL_MODES = frozenset(
    {
        "margin_top_quantile",
        "margin_top_quantile_nearest_plan_only",
    }
)


@dataclass(frozen=True, slots=True)
class RegulatoryPlanMarginArtifacts:
    """Tables emitted by the native regulator plan-margin builder."""

    scores_table: pa.Table
    tail_membership_table: pa.Table
    enrichment_table: pa.Table
    stats: dict[str, object]


def string_values(value: object, *, field_name: str) -> list[str]:
    return _string_values(value, field_name=field_name, contract_name=CONTRACT_NAME)


def _coerce_plan_order(raw_plan_order: Iterable[object] | None, raw_groups: dict[str, object]) -> list[str]:
    if raw_plan_order is None:
        plan_order = [str(plan).strip() for plan in raw_groups]
    else:
        plan_order = string_values(raw_plan_order, field_name="plan_order")
    if len(plan_order) < 2:
        raise ContractViolationError(f"{CONTRACT_NAME} requires at least two centroid groups")
    duplicates = sorted({plan for plan in plan_order if plan_order.count(plan) > 1})
    if duplicates:
        raise ContractViolationError(f"{CONTRACT_NAME} duplicate plan ids in plan_order: {duplicates[:5]}")
    invalid = [plan for plan in plan_order if not PLAN_ID_PATTERN.match(plan)]
    if invalid:
        raise ContractViolationError(f"{CONTRACT_NAME} plan ids must match {PLAN_ID_PATTERN.pattern}: {invalid[:5]}")
    return plan_order


def coerce_centroid_groups(
    raw_groups: dict[str, object],
    *,
    plan_order: Iterable[object] | None = None,
) -> tuple[list[str], dict[str, set[str]]]:
    if not isinstance(raw_groups, dict):
        raise ContractViolationError(f"{CONTRACT_NAME} centroid_groups must be a mapping")
    normalized_groups: dict[str, object] = {}
    duplicate_keys: list[str] = []
    for key, value in raw_groups.items():
        plan = str(key).strip()
        if plan in normalized_groups:
            duplicate_keys.append(plan)
        normalized_groups[plan] = value
    if duplicate_keys:
        raise ContractViolationError(f"{CONTRACT_NAME} duplicate centroid group keys: {duplicate_keys[:5]}")
    ordered_plans = _coerce_plan_order(plan_order, normalized_groups)
    missing = [plan for plan in ordered_plans if plan not in normalized_groups or not normalized_groups[plan]]
    if missing:
        raise ContractViolationError(f"{CONTRACT_NAME} missing centroid groups: {missing}")
    groups: dict[str, set[str]] = {}
    for plan in ordered_plans:
        values = set(string_values(normalized_groups.get(plan), field_name=f"centroid group {plan!r}"))
        if not values:
            raise ContractViolationError(f"{CONTRACT_NAME} centroid group {plan!r} is empty")
        groups[plan] = values
    unused = sorted(set(normalized_groups).difference(ordered_plans))
    if unused:
        raise ContractViolationError(f"{CONTRACT_NAME} centroid_groups contains groups not in plan_order: {unused[:5]}")
    return ordered_plans, groups


def validate_thresholds(thresholds: Iterable[object]) -> list[float]:
    if isinstance(thresholds, str) or not isinstance(thresholds, Iterable):
        raise ContractViolationError(f"{CONTRACT_NAME} thresholds must be a sequence of numeric fractions")
    try:
        output = [float(value) for value in thresholds]
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(f"{CONTRACT_NAME} thresholds must be numeric fractions") from exc
    if not output:
        raise ContractViolationError(f"{CONTRACT_NAME} requires at least one threshold")
    invalid = [value for value in output if not math.isfinite(value) or value <= 0.0 or value >= 1.0]
    if invalid:
        raise ContractViolationError(f"{CONTRACT_NAME} thresholds must be finite fractions between 0 and 1")
    return output


def validate_tail_modes(tail_modes: Iterable[object]) -> list[str]:
    output = string_values(tail_modes, field_name="tail_modes")
    if not output:
        raise ContractViolationError(f"{CONTRACT_NAME} requires at least one tail_mode")
    unsupported = sorted(set(output).difference(SUPPORTED_TAIL_MODES))
    if unsupported:
        raise ContractViolationError(f"{CONTRACT_NAME} unsupported tail_modes: {unsupported}")
    return list(dict.fromkeys(output))
