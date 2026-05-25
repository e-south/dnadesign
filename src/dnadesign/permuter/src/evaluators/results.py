"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/results.py

Evaluator output normalization contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numbers
import os
from typing import Any

import pandas as pd

from dnadesign.permuter.src.contracts.metrics import (
    observed_metric_column,
    observed_metric_subcolumn,
)


def normalize_scores(scores: Any, *, n: int, metric_id: str) -> dict[str, pd.Series]:
    """
    Accept one record per sequence and return canonical observed metric columns.

    Supported record shapes:
    - scalar number -> permuter__observed__<metric_id>
    - dict[str, number] -> permuter__observed__<metric_id>__<key>
    - fixed-length numeric vector -> scalar subcolumns, except logits-style
      metrics which remain Arrow list columns by default.
    """

    def _err(msg: str) -> ValueError:
        return ValueError(f"Evaluator output for metric '{metric_id}' invalid: {msg}")

    if not isinstance(scores, (list, tuple)):
        raise _err(f"expected a list/tuple of length {n}, got {type(scores).__name__}")
    if len(scores) != n:
        raise _err(f"expected length {n}, got {len(scores)}")

    if n == 0:
        return {observed_metric_column(metric_id): pd.Series([], dtype="float64")}

    first = scores[0]
    if isinstance(first, numbers.Number):
        try:
            ser = pd.Series(
                [float(x) if x is not None else float("nan") for x in scores],
                dtype="float64",
            )
        except Exception as e:
            raise _err(f"unable to coerce scalars to float: {e}") from e
        return {observed_metric_column(metric_id): ser}

    if isinstance(first, dict):
        keys = sorted(first.keys())
        if not keys:
            raise _err("empty dict records")
        for i, rec in enumerate(scores):
            if not isinstance(rec, dict) or set(rec.keys()) != set(keys):
                rec_keys = sorted(getattr(rec, "keys", lambda: [])())
                raise _err(f"record {i} has keys {rec_keys}, expected {keys}")
            for k, v in rec.items():
                if not (v is None or isinstance(v, numbers.Number)):
                    raise _err(f"record {i} key '{k}' has non-numeric value {type(v).__name__}")
        out: dict[str, pd.Series] = {}
        for k in keys:
            ser = pd.Series(
                [float(rec[k]) if rec[k] is not None else float("nan") for rec in scores],
                dtype="float64",
            )
            out[observed_metric_subcolumn(metric_id, k)] = ser
        return out

    if isinstance(first, (list, tuple, pd.Series)) or hasattr(first, "__array__"):
        try:

            def _to_list(x):
                if isinstance(x, (list, tuple, pd.Series)):
                    return list(x)
                tolist = getattr(x, "tolist", None)
                return list(tolist()) if callable(tolist) else list(x)

            seqs = [_to_list(x) for x in scores]
            lens = [len(x) for x in seqs]
        except Exception:
            raise _err("variable records must be sized sequences of numbers")
        if len(set(lens)) != 1:
            raise _err(f"inconsistent inner lengths {sorted(set(lens))}")

        keep_as_list = "logits" in str(metric_id).lower() or os.environ.get(
            "PERMUTER_VECTOR_AS_LIST",
            "",
        ).strip().lower() in {"1", "true", "yes"}
        if keep_as_list:
            cleaned = []
            for i, rec in enumerate(seqs):
                row = []
                for j, v in enumerate(rec):
                    if not (v is None or isinstance(v, numbers.Number)):
                        raise _err(f"record {i}[{j}] has non-numeric value {type(v).__name__}")
                    row.append(float(v) if v is not None else float("nan"))
                cleaned.append(row)
            return {observed_metric_column(metric_id): pd.Series(cleaned, dtype="object")}

        k = lens[0]
        out = {}
        for j in range(k):
            col = []
            for i, rec in enumerate(seqs):
                v = rec[j]
                if not (v is None or isinstance(v, numbers.Number)):
                    raise _err(f"record {i}[{j}] has non-numeric value {type(v).__name__}")
                col.append(float(v) if v is not None else float("nan"))
            out[observed_metric_subcolumn(metric_id, j)] = pd.Series(col, dtype="float64")
        return out

    raise _err(f"unsupported record type: {type(first).__name__}")
