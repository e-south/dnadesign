"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/label_history.py

Label history helpers for records.parquet.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core.utils import OpalError


@dataclass(frozen=True)
class LabelHistory:
    campaign_slug: str

    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    # --------------- normalization / strict parsing ---------------
    @staticmethod
    def normalize_hist_cell(cell: Any) -> List[Dict[str, Any]]:
        """
        Normalize a 'label_hist' cell into a Python List[Dict] where each dict has:
          {r:int, ts:str, src:str, y: List[float]}
        Be permissive on container types to tolerate different Parquet round-trips.
        """

        def _deep_as_py(x: Any) -> Any:
            # pyarrow scalars/arrays/structs
            try:
                if hasattr(x, "as_py"):
                    return x.as_py()
                if hasattr(x, "to_pylist"):
                    return x.to_pylist()
            except Exception:
                pass
            if isinstance(x, np.ndarray):
                return [_deep_as_py(v) for v in x.tolist()]
            if isinstance(x, np.generic):
                return x.item()
            if isinstance(x, pd.Series):
                return [_deep_as_py(v) for v in x.to_list()]
            if isinstance(x, dict):
                return {k: _deep_as_py(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_deep_as_py(v) for v in x]
            return x

        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        if isinstance(cell, str):
            try:
                cell = json.loads(cell)
            except Exception:
                return []

        cell = _deep_as_py(cell)
        if isinstance(cell, dict):
            cell = [cell]
        elif isinstance(cell, tuple):
            cell = list(cell)
        elif not isinstance(cell, list):
            return []
        out: List[Dict[str, Any]] = []
        for e in cell:
            if not isinstance(e, dict):
                continue
            try:
                r = int(e.get("r", -1))
            except Exception:
                continue
            y = e.get("y", [])
            try:
                y_list = [float(v) for v in np.asarray(y, dtype=float).ravel().tolist()]
            except Exception:
                continue
            out.append({"r": r, "ts": e.get("ts"), "src": e.get("src"), "y": y_list})
        return out

    @staticmethod
    def parse_hist_cell_strict(cell: Any) -> List[Dict[str, Any]]:
        """
        Strict validation for label_hist: ensure every entry has required keys, finite values.
        """

        def _deep_as_py(x: Any) -> Any:
            try:
                if hasattr(x, "as_py"):
                    return x.as_py()
                if hasattr(x, "to_pylist"):
                    return x.to_pylist()
            except Exception:
                pass
            if isinstance(x, np.ndarray):
                return [_deep_as_py(v) for v in x.tolist()]
            if isinstance(x, np.generic):
                return x.item()
            if isinstance(x, pd.Series):
                return [_deep_as_py(v) for v in x.to_list()]
            if isinstance(x, dict):
                return {k: _deep_as_py(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_deep_as_py(v) for v in x]
            return x

        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        if isinstance(cell, str):
            try:
                cell = json.loads(cell)
            except Exception as e:
                raise OpalError(f"label_hist JSON parse failed: {e}")

        cell = _deep_as_py(cell)

        if isinstance(cell, dict):
            cell = [cell]
        elif isinstance(cell, tuple):
            cell = list(cell)
        elif not isinstance(cell, list):
            raise OpalError("label_hist cell must be a list or dict.")

        out: List[Dict[str, Any]] = []
        for e in cell:
            if not isinstance(e, dict):
                raise OpalError("label_hist entries must be dicts.")
            if "r" not in e:
                raise OpalError("label_hist entry missing 'r'.")
            try:
                r = int(e["r"])
            except Exception as exc:
                raise OpalError("label_hist entry has non-integer 'r'.") from exc
            y = e.get("y", None)
            y = _deep_as_py(y)
            if not isinstance(y, (list, tuple, np.ndarray)):
                raise OpalError("label_hist entry 'y' must be a list.")
            try:
                y_list = [float(v) for v in y]
            except Exception as exc:
                raise OpalError("label_hist entry 'y' contains non-numeric values.") from exc
            if not np.all(np.isfinite(np.asarray(y_list, dtype=float))):
                raise OpalError("label_hist entry 'y' contains non-finite values.")
            out.append({"r": r, "ts": e.get("ts"), "src": e.get("src"), "y": y_list})
        return out

    # --------------- validation / repair ---------------
    def validate_label_hist(self, df: pd.DataFrame, *, require: bool = True) -> None:
        lh = self.label_hist_col()
        if lh not in df.columns:
            if require:
                raise OpalError(f"Expected label history column '{lh}' not found in records.parquet.")
            return
        bad: List[Dict[str, str]] = []
        for _id, cell in df[["id", lh]].itertuples(index=False, name=None):
            try:
                _ = self.parse_hist_cell_strict(cell)
            except Exception as e:
                bad.append({"id": str(_id), "error": str(e)})
                if len(bad) >= 5:
                    break
        if bad:
            raise OpalError(f"label_hist validation failed (sample={bad}).")

    def repair_label_hist(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
        """
        Coerce label_hist cells into normalized list-of-dicts; drop malformed entries.
        Returns (clean_df, report).
        """
        lh = self.label_hist_col()
        if lh not in df.columns:
            raise OpalError(f"Expected label history column '{lh}' not found in records.parquet.")

        def _count_entries(cell: Any) -> int:
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

        out = df.copy()
        changed_rows = 0
        dropped_total = 0
        for idx, cell in out[lh].items():
            before = _count_entries(cell)
            cleaned = self.normalize_hist_cell(cell)
            after = len(cleaned)
            dropped_total += max(0, before - after)
            if before != after or not isinstance(cell, list):
                changed_rows += 1
            out.at[idx, lh] = cleaned

        report = {"rows_changed": int(changed_rows), "entries_dropped": int(dropped_total)}
        return out, report

    # --------------- append + training labels ---------------
    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        r: int,
        *,
        src: str = "ingest_y",
        fail_if_any_existing_labels: bool = True,
        if_exists: str = "fail",
    ) -> pd.DataFrame:
        lh = self.label_hist_col()
        out = df.copy()
        if lh not in out.columns:
            out[lh] = None

        hist_map: Dict[str, List[Dict[str, Any]]] = {}
        for _id, hist_cell in out[["id", lh]].itertuples(index=False, name=None):
            hist_map[str(_id)] = self.normalize_hist_cell(hist_cell)

        new_ids = labels["id"].astype(str).tolist()
        new_ys = labels["y"].tolist()
        for i, _id in enumerate(new_ids):
            cur = hist_map.get(_id, [])
            exists = any((int(e.get("r", -1)) == int(r)) for e in cur)
            if exists:
                policy = (if_exists or "fail").strip().lower()
                if policy == "fail" and fail_if_any_existing_labels:
                    raise OpalError(f"Label history already has (id={_id}, r={r})")
                elif policy == "skip":
                    continue
                elif policy == "replace":
                    cur = [e for e in cur if int(e.get("r", -1)) != int(r)]
                else:
                    pass
            entry = {
                "r": int(r),
                "ts": pd.Timestamp.utcnow().isoformat(),
                "y": list(map(float, new_ys[i])),
                "src": src,
            }
            cur.append(entry)
            hist_map[_id] = cur

        hist_series = out["id"].astype(str).map(hist_map.get)
        out[lh] = hist_series
        return out

    def training_labels_with_round(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        lh = self.label_hist_col()
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
            hist = self.normalize_hist_cell(hist_cell)
            entries = []
            for e in hist:
                try:
                    rr = int(e.get("r", 9_999_999))
                except Exception:
                    continue
                if use_all:
                    if rr <= as_of_round:
                        entries.append(e)
                else:
                    if rr == as_of_round:
                        entries.append(e)

            if not entries:
                continue

            if policy == "latest_only":
                best = max(entries, key=lambda x: int(x.get("r", -1)))
                y = [float(v) for v in (best.get("y", []) or [])]
                recs.append((_id, y, int(best.get("r", -1))))
            elif policy == "all_rounds":
                for e in entries:
                    y = [float(v) for v in (e.get("y", []) or [])]
                    recs.append((_id, y, int(e.get("r", -1))))
            elif policy == "error_on_duplicate":
                if len(entries) > 1:
                    raise OpalError(f"Duplicate labels for id={_id} at multiple rounds (policy=error_on_duplicate).")
                e = entries[0]
                y = [float(v) for v in (e.get("y", []) or [])]
                recs.append((_id, y, int(e.get("r", -1))))

        out = pd.DataFrame(recs, columns=["id", "y", "r"])
        return out

    def training_labels_from_y(self, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
        out = self.training_labels_with_round(
            df,
            as_of_round,
            cumulative_training=True,
            dedup_policy="latest_only",
        )
        return out[["id", "y"]]
