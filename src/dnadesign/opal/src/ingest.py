"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/ingest.py

Ingestion pipeline for Y — opal ingest-y
 - Transform tidy CSV → vector/scalar y
 - Strict completeness & fail-fast checks
 - "Add if missing" with essentials enforced
 - Idempotent for (id, round)
 - Label history append: opal__<slug>__label_hist

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import pandas as pd

from .registries.ingest_transforms import get_ingest_transform
from .utils import OpalError


@dataclass
class IngestPreview:
    num_rows_in_csv: int
    num_unique_ids: int
    num_new_ids: int
    new_ids: List[str]
    will_write_count: int


def run_ingest(
    df_records: pd.DataFrame,
    csv_df: pd.DataFrame,
    *,
    transform_name: str,
    transform_params: dict,
    y_expected_length: int | None,
    setpoint_vector: list[float],
) -> tuple[pd.DataFrame, IngestPreview]:
    """
    Returns (labels_df[id,y], preview)
    """
    tf = get_ingest_transform(transform_name)
    labels = tf(
        csv_df,
        transform_params or {},
        setpoint_vector,
    )
    if not {"id", "y"}.issubset(labels.columns):
        raise OpalError("Ingest transform did not return columns: id,y")

    # coerce y to list if possible
    def _coerce(y):
        if isinstance(y, str):
            try:
                return json.loads(y)
            except Exception:
                pass
        return y

    labels["id"] = labels["id"].astype(str)
    labels["y"] = labels["y"].map(_coerce)

    # expected length (if vector)
    if y_expected_length is not None:
        bad = labels[
            labels["y"].map(
                lambda v: isinstance(v, list) and len(v) != y_expected_length
            )
        ]
        if not bad.empty:
            raise OpalError(
                f"Some y vecs don't match expected length {y_expected_length}: sample ids {bad['id'].head(10).tolist()}"
            )

    ids_in_records = set(df_records["id"].astype(str))
    ids_in_labels = set(labels["id"].astype(str))
    new_ids = sorted(ids_in_labels - ids_in_records)
    prev = IngestPreview(
        num_rows_in_csv=int(len(csv_df)),
        num_unique_ids=int(len(ids_in_labels)),
        num_new_ids=int(len(new_ids)),
        new_ids=new_ids[:20],
        will_write_count=int(len(ids_in_labels)),
    )
    return labels[["id", "y"]].copy(), prev
