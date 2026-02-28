"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_state_facade.py

State and tombstone facade helpers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import pyarrow as pa

from .dataset_state import (
    clear_state as dataset_clear_state,
)
from .dataset_state import (
    ensure_ids_exist as dataset_ensure_ids_exist,
)
from .dataset_state import (
    get_state as dataset_get_state,
)
from .dataset_state import (
    restore as dataset_restore,
)
from .dataset_state import (
    set_state as dataset_set_state,
)
from .dataset_state import (
    tombstone as dataset_tombstone,
)


def ensure_dataset_ids_exist(dataset, ids: list[str]) -> None:
    dataset_ensure_ids_exist(dataset, ids)


def tombstone_dataset_rows(
    dataset,
    ids: Sequence[str],
    *,
    reason: str | None,
    deleted_at: str | None,
    allow_missing: bool,
    tombstone_namespace: str,
) -> int:
    return dataset_tombstone(
        dataset,
        ids,
        reason=reason,
        deleted_at=deleted_at,
        allow_missing=allow_missing,
        tombstone_namespace=tombstone_namespace,
    )


def restore_dataset_rows(
    dataset,
    ids: Sequence[str],
    *,
    allow_missing: bool,
    tombstone_namespace: str,
) -> int:
    return dataset_restore(
        dataset,
        ids,
        allow_missing=allow_missing,
        tombstone_namespace=tombstone_namespace,
    )


def set_dataset_state_fields(
    dataset,
    ids: Sequence[str],
    *,
    masked: bool | None,
    qc_status: str | None,
    split: str | None,
    supersedes: str | None,
    lineage: Sequence[str] | str | None,
    allow_missing: bool,
    state_namespace: str,
    state_schema_types: dict[str, pa.DataType],
    state_qc_status_allowed: set[str],
    state_split_allowed: set[str],
) -> int:
    return dataset_set_state(
        dataset,
        ids,
        masked=masked,
        qc_status=qc_status,
        split=split,
        supersedes=supersedes,
        lineage=lineage,
        allow_missing=allow_missing,
        state_namespace=state_namespace,
        state_schema_types=state_schema_types,
        state_qc_status_allowed=state_qc_status_allowed,
        state_split_allowed=state_split_allowed,
    )


def clear_dataset_state_fields(
    dataset,
    ids: Sequence[str],
    *,
    allow_missing: bool,
    state_namespace: str,
    state_schema_types: dict[str, pa.DataType],
) -> int:
    return dataset_clear_state(
        dataset,
        ids,
        allow_missing=allow_missing,
        state_namespace=state_namespace,
        state_schema_types=state_schema_types,
    )


def get_dataset_state_frame(
    dataset,
    ids: Sequence[str],
    *,
    allow_missing: bool,
    state_namespace: str,
) -> pd.DataFrame:
    return dataset_get_state(
        dataset,
        ids,
        allow_missing=allow_missing,
        state_namespace=state_namespace,
    )
