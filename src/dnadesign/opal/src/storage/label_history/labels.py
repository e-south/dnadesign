from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ...core.utils import OpalError
from ._coercion import _coerce_mapping


def append_labels_from_df(
    history: Any,
    df: pd.DataFrame,
    labels: pd.DataFrame,
    r: int,
    *,
    src: str = "ingest_y",
    fail_if_any_existing_labels: bool = True,
    if_exists: str = "fail",
) -> pd.DataFrame:
    lh = history.label_hist_col()
    out = df.copy()
    if lh not in out.columns:
        out[lh] = None

    hist_map: Dict[str, List[Dict[str, Any]]] = {}
    for _id, hist_cell in out[["id", lh]].itertuples(index=False, name=None):
        _id_str = str(_id)
        try:
            hist_map[_id_str] = history.parse_hist_cell_strict(hist_cell)
        except OpalError as exc:
            raise OpalError(f"Malformed label history for id={_id_str}: {exc}") from exc

    new_ids = labels["id"].astype(str).tolist()
    new_ys = labels["y"].tolist()
    for i, _id in enumerate(new_ids):
        cur = hist_map.get(_id, [])
        exists = any(e.get("kind") == "label" and int(e.get("observed_round", -1)) == int(r) for e in cur)
        if exists:
            policy = (if_exists or "fail").strip().lower()
            if policy == "fail" and fail_if_any_existing_labels:
                raise OpalError(f"Label history already has (id={_id}, r={r})")
            if policy == "skip":
                continue
            if policy == "replace":
                cur = [
                    e
                    for e in cur
                    if not (e.get("kind") == "label" and int(e.get("observed_round", e.get("r", -1))) == int(r))
                ]
        y_list = list(map(float, new_ys[i]))
        entry = {
            "kind": "label",
            "observed_round": int(r),
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "y_obs": {
                "value": y_list,
                "dtype": "vector",
                "schema": {"length": int(len(y_list))},
            },
            "src": src,
        }
        cur.append(entry)
        hist_map[_id] = cur

    hist_series = out["id"].astype(str).map(hist_map.get)
    out[lh] = hist_series
    return out


def training_labels_with_round(
    history: Any,
    df: pd.DataFrame,
    as_of_round: int,
    *,
    cumulative_training: bool,
    dedup_policy: str,
) -> pd.DataFrame:
    lh = history.label_hist_col()
    if lh not in df.columns:
        raise OpalError(f"Expected label history column '{lh}' not found in records.parquet. ")
    policy = str(dedup_policy or "").strip().lower()
    if policy not in {"latest_only", "all_rounds", "error_on_duplicate"}:
        raise OpalError(
            f"Unknown label_cross_round_deduplication_policy: {dedup_policy!r} "
            "(expected: latest_only | all_rounds | error_on_duplicate)."
        )
    use_all = bool(cumulative_training)
    recs: List[Tuple[str, List[float], int]] = []
    for _id, hist_cell in df[["id", lh]].itertuples(index=False, name=None):
        _id = str(_id)
        try:
            hist = history.parse_hist_cell_strict(hist_cell)
        except OpalError as exc:
            raise OpalError(f"Malformed label history for id={_id}: {exc}") from exc
        entries = []
        for e in hist:
            if e.get("kind") != "label":
                continue
            rr = int(e.get("observed_round", 9_999_999))
            if use_all:
                if rr <= as_of_round:
                    entries.append(e)
            elif rr == as_of_round:
                entries.append(e)

        if not entries:
            continue

        if policy == "latest_only":
            best = max(entries, key=lambda x: int(x.get("observed_round", -1)))
            recs.append(_label_record(_id, best))
        elif policy == "all_rounds":
            for e in entries:
                recs.append(_label_record(_id, e))
        elif policy == "error_on_duplicate":
            if len(entries) > 1:
                raise OpalError(f"Duplicate labels for id={_id} at multiple rounds (policy=error_on_duplicate).")
            recs.append(_label_record(_id, entries[0]))

    out = pd.DataFrame(recs, columns=["id", "y", "r"])
    return out


def training_labels_from_y(history: Any, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
    out = training_labels_with_round(
        history,
        df,
        as_of_round,
        cumulative_training=True,
        dedup_policy="latest_only",
    )
    return out[["id", "y"]]


def _label_record(_id: str, entry: dict[str, Any]) -> tuple[str, list[float], int]:
    y_wrap = _coerce_mapping(entry.get("y_obs"))
    if y_wrap is None or "value" not in y_wrap:
        raise OpalError(f"Label history y_obs missing/invalid for id={_id}.")
    y = [float(v) for v in y_wrap.get("value", [])]
    return _id, y, int(entry.get("observed_round", -1))


def count_entries(cell: Any) -> int:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return 0
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except Exception:
            return 0
    if isinstance(cell, dict):
        return 1
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
        try:
            return len(cell)
        except Exception:
            return 0
    return 0
