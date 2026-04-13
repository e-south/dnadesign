"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/dataset_overlay_ops.py

Public USR overlay-write surface for cross-tool consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import pandas as pd

from .src.dataset_overlay_ops import (
    _attach_frame_dataset,
    write_overlay_dataset,
    write_overlay_part_dataset,
)


def attach_frame_dataset(
    *,
    dataset,
    incoming: pd.DataFrame,
    namespace: str,
    key: str,
    key_col: str,
    columns: Optional[Iterable[str]] = None,
    allow_overwrite: bool = False,
    allow_missing: bool = False,
    parse_json: bool = True,
    fail_on_non_null_overwrite: bool = False,
    note: str = "",
    actor: Optional[dict] = None,
    event_args: Mapping[str, object] | None = None,
    reserved_namespaces: set[str],
) -> int:
    return _attach_frame_dataset(
        dataset=dataset,
        incoming=incoming,
        namespace=namespace,
        key=key,
        key_col=key_col,
        columns=columns,
        allow_overwrite=allow_overwrite,
        allow_missing=allow_missing,
        parse_json=parse_json,
        fail_on_non_null_overwrite=fail_on_non_null_overwrite,
        note=note,
        actor=actor,
        event_args=event_args,
        reserved_namespaces=reserved_namespaces,
    )


__all__ = [
    "attach_frame_dataset",
    "write_overlay_dataset",
    "write_overlay_part_dataset",
]
