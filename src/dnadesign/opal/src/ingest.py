"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/ingest.py

Ingestion pipeline for Y — opal ingest-y
 - Transform CSV → vector/scalar y
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

from .registries.transforms_y import get_ingest_transform
from .utils import OpalError


@dataclass
class IngestPreview:
    num_rows_in_csv: int
    num_unique_sequences: int
    num_with_explicit_id: int
    num_new_sequences: int
    new_sequences: List[str]
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
    Returns:
      labels_df: DataFrame with columns ['id','y'] or ['sequence','y'] (id optional)
      preview:   IngestPreview
    """
    tf = get_ingest_transform(transform_name)
    try:
        out = tf(csv_df, transform_params or {}, setpoint_vector, records_df=df_records)
    except TypeError:
        out = tf(csv_df, transform_params or {}, setpoint_vector)

    labels = out[0] if isinstance(out, tuple) and len(out) == 2 else out
    if not {"y"}.issubset(labels.columns):
        raise OpalError("Ingest transform must return a 'y' column.")

    # coerce y to list if possible
    def _coerce(y):
        if isinstance(y, str):
            try:
                return json.loads(y)
            except Exception:
                pass
        return y

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
                f"Some y vecs don't match expected length {y_expected_length}: sample seqs/ids {bad.head(10).to_dict(orient='records')}"  # noqa
            )

    has_id = "id" in labels.columns
    if has_id:
        labels["id"] = labels["id"].astype(str)

    if "sequence" not in labels.columns and not has_id:
        raise OpalError("Ingest transform returned neither 'id' nor 'sequence'.")

    # Preview by sequences (the resolution key when id is absent)
    seqs_in_labels = (
        set(labels["sequence"].astype(str)) if "sequence" in labels.columns else set()
    )
    seqs_in_records = set(df_records["sequence"].astype(str))
    new_seqs = sorted(seqs_in_labels - seqs_in_records) if seqs_in_labels else []

    prev = IngestPreview(
        num_rows_in_csv=int(len(csv_df)),
        num_unique_sequences=int(len(seqs_in_labels)) if seqs_in_labels else 0,
        num_with_explicit_id=int(labels["id"].notna().sum()) if has_id else 0,
        num_new_sequences=int(len(new_seqs)),
        new_sequences=new_seqs[:20],
        will_write_count=int(len(labels)),
    )
    return labels.copy(), prev
