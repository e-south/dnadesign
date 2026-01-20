"""Filtering utilities for dashboard views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class NumericRule:
    enabled: bool
    column: str | None
    op: str
    value: float | None = None


SUPPORTED_NUMERIC_OPS = {">=", "<=", "is null", "is not null"}


def build_numeric_rule_exprs(rules: Sequence[NumericRule]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for rule in rules:
        if not rule.enabled:
            continue
        if not rule.column:
            continue
        if rule.op not in SUPPORTED_NUMERIC_OPS:
            raise ValueError(f"Unsupported operator: {rule.op}")
        if rule.op == ">=":
            if rule.value is None:
                continue
            exprs.append(pl.col(rule.column) >= float(rule.value))
        elif rule.op == "<=":
            if rule.value is None:
                continue
            exprs.append(pl.col(rule.column) <= float(rule.value))
        elif rule.op == "is null":
            exprs.append(pl.col(rule.column).is_null())
        elif rule.op == "is not null":
            exprs.append(pl.col(rule.column).is_not_null())
    return exprs


def apply_numeric_rules(df: pl.DataFrame, rules: Sequence[NumericRule]) -> pl.DataFrame:
    exprs = build_numeric_rule_exprs(rules)
    if not exprs:
        return df
    return df.filter(pl.all_horizontal(exprs))
