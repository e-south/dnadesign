"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/ingest.py

Validates and ingests label data into records with transform handling. Produces
ingest reports and contract checks for label integrity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.round_context import PluginCtx
from ..core.utils import OpalError
from ..registries.transforms_y import get_transform_y


@dataclass
class IngestPreview:
    # Input overview
    total_rows_in_csv: int
    rows_with_id: int
    rows_with_sequence: int

    # Resolution counts
    resolved_ids_by_sequence: int
    unknown_sequences: int

    # Vector checks
    y_expected_length: Optional[int]
    y_length_ok: int
    y_length_bad: int

    # What we will write
    y_column_name: str
    duplicate_policy: str
    duplicate_key: str
    duplicates_found: int
    duplicates_dropped: int

    # Notes/warnings
    warnings: List[str]


def _vec_len(v: Any) -> int:
    if isinstance(v, list):
        return len(v)
    if isinstance(v, (np.ndarray,)):
        return int(v.shape[-1])
    return 1


def _apply_transform_via_registry(
    name: str,
    params: Dict[str, Any],
    csv_df: pd.DataFrame,
    *,
    ctx: PluginCtx,
) -> pd.DataFrame:
    """
    Call the registered Y-ingest transform. We tolerate (csv_df, params)
    or (csv_df, **params) call patterns for plugin friendliness.
    """
    fn = get_transform_y(name)
    try:
        return fn(csv_df, params, ctx=ctx)  # required signature
    except OpalError:
        raise
    except TypeError as e:
        raise OpalError(
            f"Y transform '{name}' has an unsupported signature. Expected: fn(df_tidy, params: dict, ctx=PluginCtx)."
        ) from e
    except Exception as e:
        cols = list(map(str, csv_df.columns))
        preview = cols[:12]
        suffix = "..." if len(cols) > 12 else ""
        hint = (
            f"Input columns: {preview}{suffix}. "
            "If your file uses different column names, update transforms_y.params in campaign.yaml "
            "or pass --params to ingest-y."
        )
        raise OpalError(f"Y transform '{name}' failed: {e}. {hint}") from e


def run_ingest(
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    *,
    transform_name: str,
    transform_params: Dict[str, Any],
    y_expected_length: Optional[int],
    y_column_name: str,
    duplicate_policy: str,
    ctx: PluginCtx,
) -> Tuple[pd.DataFrame, IngestPreview]:
    """
    Returns:
      labels_df: DataFrame with at least ['sequence','y'] and, where resolvable, 'id'
      preview:   IngestPreview
    """
    if ctx is None:
        raise OpalError("run_ingest requires a PluginCtx for transform_y.")

    # 1) Transform tidy -> (sequence, y)
    labels = _apply_transform_via_registry(transform_name, transform_params, csv_df, ctx=ctx)

    # 1a) Validate transform output shape and key columns
    if "y" not in labels.columns:
        raise OpalError(
            f"Y transform '{transform_name}' returned no 'y' column. Columns={list(map(str, labels.columns))}."
        )
    if "id" not in labels.columns and "sequence" not in labels.columns:
        raise OpalError(
            f"Y transform '{transform_name}' must return 'id' and/or 'sequence'. "
            f"Columns={list(map(str, labels.columns))}."
        )
    if "id" in labels.columns and labels["id"].isna().any() and "sequence" not in labels.columns:
        raise OpalError(
            f"Y transform '{transform_name}' returned missing ids but no 'sequence' column to resolve them."
        )

    # 1b) Validate Y vectors strictly (no fallbacks)
    for i, v in enumerate(labels["y"].tolist()):
        arr = np.asarray(v, dtype=float).ravel()
        if y_expected_length is not None and arr.size != int(y_expected_length):
            raise OpalError(f"Y length mismatch at row {i}: expected {int(y_expected_length)}, got {int(arr.size)}")
        if not np.all(np.isfinite(arr)):
            raise OpalError(f"Y contains non-finite values at row {i}.")

    # 2) Try to resolve ids by sequence (existing rows only; new rows remain without id for now)
    seq2id = {}
    if "sequence" in records_df.columns and "id" in records_df.columns:
        if records_df["sequence"].duplicated().any():
            if "id" not in labels.columns or labels["id"].isna().any():
                dup = records_df["sequence"][records_df["sequence"].duplicated()].astype(str).unique().tolist()[:10]
                raise OpalError(
                    "records.parquet contains duplicate sequences; ingest-y requires id for all rows to disambiguate "
                    f"(sample={dup})."
                )
        seq2id = records_df.drop_duplicates(subset=["sequence"]).set_index("sequence")["id"].astype(str).to_dict()
    if "id" not in labels.columns:
        labels["id"] = labels["sequence"].map(seq2id)
    elif "sequence" in labels.columns:
        missing_ids = labels["id"].isna()
        if missing_ids.any():
            labels.loc[missing_ids, "id"] = labels.loc[missing_ids, "sequence"].map(seq2id)

    # 3) Duplicate handling (assertive, policy-driven)
    policy = str(duplicate_policy or "error").strip().lower()
    if policy not in {"error", "keep_first", "keep_last"}:
        raise OpalError(
            f"Unknown ingest.duplicate_policy={duplicate_policy!r} (expected: error | keep_first | keep_last)."
        )
    # Build a stable key (prefer id if present; else sequence)
    if "id" in labels.columns and labels["id"].notna().any():
        key = labels["id"].astype("string")
        key = key.fillna(labels["sequence"].astype("string"))
        key_name = "id"
    else:
        key = labels["sequence"].astype("string")
        key_name = "sequence"
    if key.isna().any():
        raise OpalError("Ingest requires id or sequence for every row (found missing values).")
    dup_mask = key.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    dropped = 0
    if dup_count > 0:
        dup_keys = key[dup_mask].astype(str).unique().tolist()[:10]
        if policy == "error":
            raise OpalError(f"Duplicate {key_name} values found (sample={dup_keys}).")
        keep = "first" if policy == "keep_first" else "last"
        before = len(labels)
        labels = labels.loc[~key.duplicated(keep=keep)].copy()
        dropped = before - len(labels)

    # 4) Preview stats
    total = int(len(csv_df))
    if "id" in csv_df.columns:
        rows_with_id = int(csv_df["id"].notna().sum())
    else:
        rows_with_id = 0
    if "sequence" in csv_df.columns:
        rows_with_seq = int(csv_df["sequence"].notna().sum())
    else:
        rows_with_seq = 0

    resolved = int(labels["id"].notna().sum())
    unknown = int((~labels["id"].notna()).sum())

    # vector length checks (full, already validated above)
    y_len_ok = 0
    y_len_bad = 0
    if y_expected_length:
        for v in labels["y"].tolist():
            if _vec_len(v) == y_expected_length:
                y_len_ok += 1
            else:
                y_len_bad += 1

    # Warnings / nudges
    warnings: List[str] = []
    if rows_with_id == 0 and rows_with_seq > 0:
        warnings.append("input missing id column; ids will be resolved by sequence or created for new sequences.")
    if rows_with_id > 0 and rows_with_seq == 0:
        warnings.append("input missing sequence column; all rows must have ids and exist in records.")
    if labels["id"].isna().any() and "sequence" in labels.columns:
        warnings.append(
            f"{unknown} rows missing id; will resolve by sequence or create deterministic ids for new sequences."
        )
    if unknown > 0:
        warnings.append(f"{unknown} sequences not found in records; new rows will be created.")
    if dup_count > 0 and policy != "error":
        warnings.append(f"{dup_count} duplicate keys detected; policy dropped {dropped} row(s).")

    # Logic bounds warning (best-effort): try to read first 4 entries as logic
    try:
        v = np.asarray([yy[:4] for yy in labels["y"].tolist()], dtype=float)
        if (v < -1e-9).any() or (v > 1 + 1e-9).any():
            warnings.append("logic_out_of_bounds_detected")
    except Exception:
        pass

    preview = IngestPreview(
        total_rows_in_csv=total,
        rows_with_id=rows_with_id,
        rows_with_sequence=rows_with_seq,
        resolved_ids_by_sequence=resolved,
        unknown_sequences=unknown,
        y_expected_length=y_expected_length,
        y_length_ok=y_len_ok,
        y_length_bad=y_len_bad,
        y_column_name=y_column_name,
        duplicate_policy=policy,
        duplicate_key=key_name,
        duplicates_found=dup_count,
        duplicates_dropped=dropped,
        warnings=warnings,
    )

    # 5) Return labels (sequence, id?, y) and the preview
    cols = ["sequence", "id", "y"] if "id" in labels.columns else ["sequence", "y"]
    return labels[cols], preview
