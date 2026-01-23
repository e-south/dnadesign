"""Hue registry helpers for dashboard plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import polars as pl

from .util import safe_is_numeric


@dataclass(frozen=True)
class HueOption:
    key: str
    label: str
    kind: str  # "numeric" | "categorical"
    dtype: pl.DataType
    category_labels: tuple[str, str] | None = None


@dataclass(frozen=True)
class HueRegistry:
    options: list[HueOption]
    label_map: dict[str, HueOption]

    def labels(self) -> list[str]:
        return list(self.label_map.keys())

    def get(self, label: str | None) -> HueOption | None:
        if label is None:
            return None
        return self.label_map.get(label)


_DENY_PREFIXES = (
    "opal__ledger__",
    "opal__cache__",
    "opal__overlay__y_hat",
    "opal__overlay__y_vec",
)
_DENY_SUBSTRINGS = ("ledger", "cache")


def _is_nested(dtype: pl.DataType) -> bool:
    try:
        return bool(getattr(dtype, "is_nested")())
    except Exception:
        return False


def _has_any_value(df: pl.DataFrame, col: str) -> bool:
    if df.is_empty() or col not in df.columns:
        return False
    try:
        return df.select(pl.col(col).drop_nulls().head(1)).height > 0
    except Exception:
        return False


def _infer_kind(df: pl.DataFrame, col: str, dtype: pl.DataType, *, max_unique: int) -> str | None:
    if safe_is_numeric(dtype):
        return "numeric"
    if dtype == pl.Boolean:
        return "categorical"
    if dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8, pl.Date, pl.Datetime):
        try:
            n_unique = df.select(pl.col(col).drop_nulls().n_unique()).item()
        except Exception:
            return None
        if n_unique == 0:
            return None
        if n_unique <= max_unique:
            return "categorical"
    return None


def _should_skip(name: str, *, denylist: Iterable[str], deny_prefixes: Iterable[str]) -> bool:
    if name in denylist:
        return True
    if any(name.startswith(prefix) for prefix in deny_prefixes):
        return True
    lowered = name.lower()
    if any(token in lowered for token in _DENY_SUBSTRINGS):
        return True
    return False


def build_hue_registry(
    df: pl.DataFrame,
    *,
    preferred: Sequence[HueOption] | None = None,
    include_columns: bool = True,
    denylist: Iterable[str] = (),
    deny_prefixes: Iterable[str] = (),
    max_unique: int = 30,
) -> HueRegistry:
    options: list[HueOption] = []
    label_map: dict[str, HueOption] = {}
    used_keys: set[str] = set()
    deny_prefixes = tuple(deny_prefixes) + _DENY_PREFIXES

    def _add_option(option: HueOption) -> None:
        if option.key in used_keys:
            return
        if option.key not in df.columns:
            return
        dtype = df.schema.get(option.key, pl.Null)
        if _is_nested(dtype) or dtype in (pl.Object, pl.Null):
            return
        if not _has_any_value(df, option.key):
            return
        label = option.label
        if label in label_map:
            label = option.key
        normalized = HueOption(
            key=option.key,
            label=label,
            kind=option.kind,
            dtype=dtype,
            category_labels=option.category_labels,
        )
        options.append(normalized)
        label_map[label] = normalized
        used_keys.add(option.key)

    for option in preferred or []:
        _add_option(option)

    if include_columns:
        for name, dtype in df.schema.items():
            if name.startswith("__"):
                continue
            if name in used_keys:
                continue
            if _should_skip(name, denylist=denylist, deny_prefixes=deny_prefixes):
                continue
            if _is_nested(dtype) or dtype in (pl.Object, pl.Null):
                continue
            if not _has_any_value(df, name):
                continue
            kind = _infer_kind(df, name, dtype, max_unique=max_unique)
            if kind is None:
                continue
            label = name
            if label in label_map:
                continue
            option = HueOption(key=name, label=label, kind=kind, dtype=dtype)
            options.append(option)
            label_map[label] = option

    return HueRegistry(options=options, label_map=label_map)


def default_view_hues() -> list[HueOption]:
    return [
        HueOption(key="opal__view__score", label="Score", kind="numeric", dtype=pl.Float64),
        HueOption(key="opal__view__logic_fidelity", label="Logic fidelity", kind="numeric", dtype=pl.Float64),
        HueOption(key="opal__view__effect_scaled", label="Effect scaled", kind="numeric", dtype=pl.Float64),
        HueOption(key="opal__view__rank", label="Rank", kind="numeric", dtype=pl.Float64),
        HueOption(
            key="opal__view__top_k",
            label="Top-K",
            kind="categorical",
            dtype=pl.Boolean,
            category_labels=("Top-K", "Not Top-K"),
        ),
        HueOption(
            key="opal__view__observed",
            label="Observed (labeled)",
            kind="categorical",
            dtype=pl.Boolean,
            category_labels=("Observed", "Unlabeled"),
        ),
        HueOption(
            key="opal__view__pred_score_unlabeled",
            label="Predicted score (unlabeled)",
            kind="numeric",
            dtype=pl.Float64,
        ),
        HueOption(
            key="opal__view__observed_score",
            label="Observed score",
            kind="numeric",
            dtype=pl.Float64,
        ),
    ]
