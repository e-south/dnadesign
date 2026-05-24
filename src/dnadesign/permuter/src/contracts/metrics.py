"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/contracts/metrics.py

Metric-column contract helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

OBSERVED_PREFIX = "permuter__observed__"
EXPECTED_PREFIX = "permuter__expected__"
LEGACY_METRIC_PREFIX = "permuter__metric__"
INTERACTION_PREFIX = "permuter__interaction__"

_METRIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _clean_metric_id(metric_id: str) -> str:
    mid = str(metric_id or "").strip()
    if not mid:
        raise ValueError("metric_id is required")
    if "__" in mid or not _METRIC_ID_RE.fullmatch(mid):
        raise ValueError(
            "metric_id must be a compact identifier using letters, digits, '.', '-', or '_' "
            f"without double underscores; got {metric_id!r}"
        )
    return mid


def observed_metric_column(metric_id: str) -> str:
    return f"{OBSERVED_PREFIX}{_clean_metric_id(metric_id)}"


def expected_metric_column(metric_id: str) -> str:
    return f"{EXPECTED_PREFIX}{_clean_metric_id(metric_id)}"


def interaction_metric_column(interaction_id: str, metric_id: str) -> str:
    return f"{INTERACTION_PREFIX}{_clean_metric_id(interaction_id)}__{_clean_metric_id(metric_id)}"


def observed_metric_subcolumn(metric_id: str, suffix: str | int) -> str:
    return f"{observed_metric_column(metric_id)}__{_clean_metric_id(str(suffix))}"


def metric_id_from_observed_column(column: str) -> str:
    col = str(column)
    if not col.startswith(OBSERVED_PREFIX):
        raise ValueError(f"Expected observed metric column with prefix {OBSERVED_PREFIX!r}; got {column!r}")
    return _clean_metric_id(col.removeprefix(OBSERVED_PREFIX))


def observed_metric_ids(columns: Iterable[str]) -> list[str]:
    ids: list[str] = []
    for column in columns:
        col = str(column)
        if col.startswith(OBSERVED_PREFIX):
            suffix = col.removeprefix(OBSERVED_PREFIX)
            ids.append(_clean_metric_id(suffix.split("__", 1)[0]))
    return sorted(set(ids))


def reject_legacy_metric_columns(df: pd.DataFrame, *, context: str, expected_observed: str | None = None) -> None:
    legacy = sorted(c for c in df.columns if str(c).startswith(LEGACY_METRIC_PREFIX))
    if legacy:
        expected = expected_observed or f"{OBSERVED_PREFIX}<metric_id>"
        raise ValueError(
            f"{context}: legacy metric columns are not accepted: {legacy}. "
            f"Use canonical observed metric columns named {expected}."
        )
