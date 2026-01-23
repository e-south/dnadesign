# ABOUTME: Stores and validates per-record label/prediction history for OPAL campaigns.
# ABOUTME: Enforces schema contracts for label history entries and training extraction.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/label_history.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..core.utils import OpalError

try:
    import polars as pl
except Exception:  # pragma: no cover - optional for non-dashboard contexts
    pl = None


def _deep_as_py(x: Any) -> Any:
    try:
        if hasattr(x, "as_py"):
            return x.as_py()
        if hasattr(x, "to_pylist"):
            return x.to_pylist()
    except Exception:
        pass
    if pl is not None and isinstance(x, pl.Series):
        return [_deep_as_py(v) for v in x.to_list()]
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


def _coerce_mapping(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _coerce_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        val = str(value).strip()
    except Exception:
        return None
    return val or None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _coerce_float_list(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.tolist()


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "t", "1", "yes"}:
            return True
        if val in {"false", "f", "0", "no"}:
            return False
    return None


def _normalize_value_wrapper(value: Any, *, require_numeric: bool) -> dict | None:
    wrapper = _coerce_mapping(value)
    if wrapper is None:
        return None
    if "value" not in wrapper:
        return None
    dtype_val = _coerce_non_empty_str(wrapper.get("dtype"))
    if dtype_val is None:
        return None
    schema_val = wrapper.get("schema")
    schema_map = _coerce_mapping(schema_val) if schema_val is not None else None
    raw_value = _deep_as_py(wrapper.get("value"))
    if require_numeric:
        numeric = _coerce_float_list(raw_value)
        if numeric is None:
            return None
        raw_value = numeric
    out = {"value": raw_value, "dtype": dtype_val}
    if schema_map is not None:
        out["schema"] = schema_map
    return out


def _normalize_label_entry(entry_map: Mapping[str, Any]) -> dict | None:
    round_val = entry_map.get("observed_round", entry_map.get("r", entry_map.get("round")))
    r_int = _coerce_int(round_val)
    if r_int is None:
        return None
    y_wrap = _normalize_value_wrapper(entry_map.get("y_obs"), require_numeric=True)
    if y_wrap is None:
        return None
    return {
        "kind": "label",
        "observed_round": r_int,
        "ts": entry_map.get("ts"),
        "src": entry_map.get("src"),
        "y_obs": y_wrap,
    }


def _normalize_pred_entry(entry_map: Mapping[str, Any]) -> dict | None:
    round_val = entry_map.get("as_of_round", entry_map.get("r", entry_map.get("round")))
    r_int = _coerce_int(round_val)
    if r_int is None:
        return None
    run_id = entry_map.get("run_id")
    if run_id is None:
        return None
    y_pred_wrap = _normalize_value_wrapper(entry_map.get("y_pred"), require_numeric=False)
    if y_pred_wrap is None:
        return None
    y_space = _coerce_non_empty_str(entry_map.get("y_space"))
    if y_space is None:
        return None
    pred: dict[str, Any] = {
        "kind": "pred",
        "as_of_round": r_int,
        "run_id": str(run_id),
        "ts": entry_map.get("ts"),
        "y_pred": y_pred_wrap,
        "y_space": y_space,
    }
    objective = entry_map.get("objective")
    if isinstance(objective, Mapping):
        pred["objective"] = dict(objective)
    metrics = entry_map.get("metrics")
    if isinstance(metrics, Mapping):
        pred["metrics"] = dict(metrics)
    selection = entry_map.get("selection")
    if isinstance(selection, Mapping):
        pred["selection"] = dict(selection)
    return pred


def _record_parse_error(
    *,
    errors: list[dict],
    row_id: str,
    message: str,
    sample: Any | None = None,
) -> None:
    if len(errors) >= 5:
        return
    errors.append(
        {
            "id": row_id,
            "error": message,
            "sample": (repr(sample)[:240] if sample is not None else None),
        }
    )


def parse_label_hist_cell_for_dashboard(
    cell: Any,
    *,
    row_id: str,
    y_col_name: str,
    errors: list[dict],
) -> List[Dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    entries = LabelHistory.normalize_hist_cell(cell)
    if not entries:
        try:
            entries = LabelHistory.parse_hist_cell_strict(cell)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist parse failed: {exc}",
                sample=cell,
            )
            return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "label":
            continue
        r_val = entry.get("observed_round")
        if r_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry missing observed_round",
                sample=entry,
            )
            continue
        try:
            r_int = int(r_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist entry round is not int: {exc}",
                sample=entry,
            )
            continue
        y_wrap = _coerce_mapping(entry.get("y_obs"))
        if y_wrap is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry missing y_obs wrapper",
                sample=entry,
            )
            continue
        if "value" not in y_wrap:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry y_obs missing value",
                sample=y_wrap,
            )
            continue
        y_val = _deep_as_py(y_wrap.get("value"))
        try:
            y_list = [float(v) for v in np.asarray(y_val, dtype=float).ravel().tolist()]
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist entry 'y' not numeric: {exc}",
                sample=y_val,
            )
            continue
        out.append(
            {
                "observed_round": r_int,
                "label_src": entry.get("src"),
                "label_ts": entry.get("ts"),
                y_col_name: y_list,
            }
        )
    return out


def parse_pred_hist_cell_for_dashboard(
    cell: Any,
    *,
    row_id: str,
    errors: list[dict],
) -> List[Dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    entries = LabelHistory.normalize_hist_cell(cell)
    if not entries:
        try:
            entries = LabelHistory.parse_hist_cell_strict(cell)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist parse failed: {exc}",
                sample=cell,
            )
            return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "pred":
            continue
        as_of_round = entry.get("as_of_round")
        run_id = entry.get("run_id")
        if as_of_round is None or run_id is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing as_of_round or run_id",
                sample=entry,
            )
            continue
        try:
            round_int = int(as_of_round)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry as_of_round not int: {exc}",
                sample=entry,
            )
            continue
        y_wrap = _coerce_mapping(entry.get("y_pred"))
        if y_wrap is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing y_pred wrapper",
                sample=entry,
            )
            continue
        if "value" not in y_wrap:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry y_pred missing value",
                sample=y_wrap,
            )
            continue

        pred_value = _deep_as_py(y_wrap.get("value"))
        pred_dtype = y_wrap.get("dtype")
        pred_y_hat = None
        try:
            pred_y_hat = [float(v) for v in np.asarray(pred_value, dtype=float).ravel().tolist()]
        except Exception:
            pred_y_hat = None
        try:
            pred_value_json = json.dumps(pred_value, ensure_ascii=True)
        except Exception:
            pred_value_json = repr(pred_value)

        metrics = _coerce_mapping(entry.get("metrics") or {})
        selection = _coerce_mapping(entry.get("selection") or {})
        score_val = metrics.get("score") if metrics is not None else None
        if score_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing metrics.score",
                sample=entry,
            )
            continue
        try:
            score_val = float(score_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry metrics.score not float: {exc}",
                sample=entry,
            )
            continue

        rank_val = selection.get("rank") if selection is not None else None
        top_k_val = selection.get("top_k") if selection is not None else None
        if rank_val is None or top_k_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing selection.rank/top_k",
                sample=entry,
            )
            continue
        try:
            rank_val = int(rank_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry selection.rank not int: {exc}",
                sample=entry,
            )
            continue
        top_k_val = bool(top_k_val)

        objective = _coerce_mapping(entry.get("objective") or {})
        out.append(
            {
                "as_of_round": round_int,
                "run_id": str(run_id),
                "pred_ts": entry.get("ts"),
                "pred_y_hat": pred_y_hat,
                "pred_y_value": pred_value_json,
                "pred_y_dtype": str(pred_dtype) if pred_dtype is not None else None,
                "pred_score": score_val,
                "pred_logic_fidelity": metrics.get("logic_fidelity") if metrics else None,
                "pred_effect_scaled": metrics.get("effect_scaled") if metrics else None,
                "pred_effect_raw": metrics.get("effect_raw") if metrics else None,
                "pred_rank": rank_val,
                "pred_top_k": top_k_val,
                "pred_objective_name": objective.get("name") if objective else None,
                "pred_objective_params": objective.get("params") if objective else None,
            }
        )
    return out


@dataclass(frozen=True)
class LabelHistory:
    campaign_slug: str

    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    # --------------- normalization / strict parsing ---------------
    @staticmethod
    def normalize_hist_cell(cell: Any) -> List[Dict[str, Any]]:
        """
        Normalize a 'label_hist' cell into a Python List[Dict] in the current schema:
          - label entry: {kind:'label', observed_round:int, ts:str, src:str,
            y_obs:{value,dtype,schema?}}
          - pred entry:  {kind:'pred', as_of_round:int, run_id:str, ts:str,
            y_pred:{value,dtype,schema?}, y_space:str, ...}
        Be permissive on container types to tolerate different Parquet round-trips.
        """
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
            if isinstance(e, str):
                try:
                    e = json.loads(e)
                except Exception:
                    continue
            entry_map = _coerce_mapping(_deep_as_py(e))
            if entry_map is None:
                continue
            kind = entry_map.get("kind")
            if kind is None:
                normalized = _normalize_label_entry(entry_map)
                if normalized is not None:
                    out.append(normalized)
                continue
            kind_str = str(kind).strip().lower()
            if kind_str == "label":
                normalized = _normalize_label_entry(entry_map)
                if normalized is not None:
                    out.append(normalized)
            elif kind_str == "pred":
                normalized = _normalize_pred_entry(entry_map)
                if normalized is not None:
                    out.append(normalized)
        return out

    @staticmethod
    def parse_hist_cell_strict(cell: Any) -> List[Dict[str, Any]]:
        """
        Strict validation for label_hist: ensure every entry has required keys, finite values.
        """
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
            if isinstance(e, str):
                try:
                    e = json.loads(e)
                except Exception as exc:
                    raise OpalError(f"label_hist entry JSON parse failed: {exc}") from exc
            entry_map = _coerce_mapping(_deep_as_py(e))
            if entry_map is None:
                raise OpalError("label_hist entries must be dicts.")

            kind = entry_map.get("kind")
            if kind is None or str(kind).strip().lower() == "label":
                normalized = _normalize_label_entry(entry_map)
                if normalized is None:
                    raise OpalError("label_hist label entry missing required keys.")
                out.append(normalized)
                continue

            kind_str = str(kind).strip().lower()
            if kind_str != "pred":
                raise OpalError(f"label_hist entry has unknown kind: {kind_str!r}")

            normalized = _normalize_pred_entry(entry_map)
            if normalized is None:
                raise OpalError("label_hist pred entry missing required keys.")

            y_space_val = normalized.get("y_space")
            if not isinstance(y_space_val, str) or not y_space_val.strip():
                raise OpalError("label_hist pred entry y_space must be a non-empty string.")

            objective = entry_map.get("objective")
            if not isinstance(objective, Mapping):
                raise OpalError("label_hist pred entry missing objective mapping.")
            if not objective.get("name"):
                raise OpalError("label_hist pred entry objective missing name.")
            params_val = objective.get("params")
            if params_val is not None and not isinstance(params_val, Mapping):
                raise OpalError("label_hist pred entry objective.params must be a mapping or null.")

            metrics = entry_map.get("metrics")
            if not isinstance(metrics, Mapping):
                raise OpalError("label_hist pred entry missing metrics mapping.")
            if "score" not in metrics:
                raise OpalError("label_hist pred entry metrics missing score.")
            score_val = _coerce_float(metrics.get("score"))
            if score_val is None:
                raise OpalError("label_hist pred entry metrics.score must be finite.")

            for key in ("logic_fidelity", "effect_scaled", "effect_raw"):
                if key in metrics:
                    if _coerce_float(metrics.get(key)) is None:
                        raise OpalError(f"label_hist pred entry metrics.{key} must be finite.")

            selection = entry_map.get("selection")
            if not isinstance(selection, Mapping):
                raise OpalError("label_hist pred entry missing selection mapping.")
            rank_val = selection.get("rank")
            if rank_val is None or _coerce_int(rank_val) is None:
                raise OpalError("label_hist pred entry selection.rank must be an int.")
            top_k_val = _coerce_bool(selection.get("top_k"))
            if top_k_val is None:
                raise OpalError("label_hist pred entry selection.top_k must be a bool.")

            normalized["objective"] = dict(objective)
            normalized["metrics"] = dict(metrics)
            normalized["selection"] = dict(selection)
            out.append(normalized)
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

        report = {
            "rows_changed": int(changed_rows),
            "entries_dropped": int(dropped_total),
        }
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
            exists = any((e.get("kind") == "label" and int(e.get("observed_round", -1)) == int(r)) for e in cur)
            if exists:
                policy = (if_exists or "fail").strip().lower()
                if policy == "fail" and fail_if_any_existing_labels:
                    raise OpalError(f"Label history already has (id={_id}, r={r})")
                elif policy == "skip":
                    continue
                elif policy == "replace":
                    cur = [
                        e
                        for e in cur
                        if not (e.get("kind") == "label" and int(e.get("observed_round", e.get("r", -1))) == int(r))
                    ]
                else:
                    pass
            y_list = list(map(float, new_ys[i]))
            entry = {
                "kind": "label",
                "observed_round": int(r),
                "ts": pd.Timestamp.utcnow().isoformat(),
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
                    if e.get("kind") != "label":
                        continue
                    rr = int(e.get("observed_round", 9_999_999))
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
                best = max(entries, key=lambda x: int(x.get("observed_round", -1)))
                y_wrap = _normalize_value_wrapper(best.get("y_obs"), require_numeric=True)
                if y_wrap is None:
                    raise OpalError(f"Label history y_obs missing/invalid for id={_id}.")
                y = [float(v) for v in (y_wrap.get("value") or [])]
                recs.append((_id, y, int(best.get("observed_round", -1))))
            elif policy == "all_rounds":
                for e in entries:
                    y_wrap = _normalize_value_wrapper(e.get("y_obs"), require_numeric=True)
                    if y_wrap is None:
                        raise OpalError(f"Label history y_obs missing/invalid for id={_id}.")
                    y = [float(v) for v in (y_wrap.get("value") or [])]
                    recs.append((_id, y, int(e.get("observed_round", -1))))
            elif policy == "error_on_duplicate":
                if len(entries) > 1:
                    raise OpalError(f"Duplicate labels for id={_id} at multiple rounds (policy=error_on_duplicate).")
                e = entries[0]
                y_wrap = _normalize_value_wrapper(e.get("y_obs"), require_numeric=True)
                if y_wrap is None:
                    raise OpalError(f"Label history y_obs missing/invalid for id={_id}.")
                y = [float(v) for v in (y_wrap.get("value") or [])]
                recs.append((_id, y, int(e.get("observed_round", -1))))

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

    # --------------- prediction writebacks ---------------
    def append_predictions_from_arrays(
        self,
        df: pd.DataFrame,
        *,
        ids: List[str],
        y_hat: np.ndarray,
        as_of_round: int,
        run_id: str,
        objective: Dict[str, Any],
        metrics_by_name: Dict[str, List[float]],
        selection_rank: np.ndarray,
        selection_top_k: np.ndarray,
        ts: str | None = None,
    ) -> pd.DataFrame:
        objective = _coerce_mapping(_deep_as_py(objective))
        if objective is None:
            raise OpalError("append_predictions_from_arrays requires objective mapping.")
        if not objective.get("name"):
            raise OpalError("append_predictions_from_arrays requires objective.name.")
        params_val = objective.get("params")
        if params_val is not None and not isinstance(params_val, Mapping):
            raise OpalError("append_predictions_from_arrays requires objective.params mapping or null.")
        if isinstance(params_val, Mapping):
            params_val = _coerce_mapping(_deep_as_py(params_val))
            if not params_val:
                params_val = None
        objective = {**objective, "params": params_val}
        lh = self.label_hist_col()
        out = df.copy()
        if lh not in out.columns:
            out[lh] = None
        if y_hat.shape[0] != len(ids):
            raise OpalError("append_predictions_from_arrays length mismatch: ids vs y_hat")
        if selection_rank.shape[0] != len(ids) or selection_top_k.shape[0] != len(ids):
            raise OpalError("append_predictions_from_arrays length mismatch: selection arrays")
        if "score" not in (metrics_by_name or {}):
            raise OpalError("append_predictions_from_arrays requires metrics_by_name['score'].")

        for key, values in (metrics_by_name or {}).items():
            if len(values) != len(ids):
                raise OpalError(f"append_predictions_from_arrays metrics length mismatch for '{key}'")

        y_hat_arr = np.asarray(y_hat, dtype=float)
        if not np.all(np.isfinite(y_hat_arr)):
            raise OpalError("append_predictions_from_arrays received non-finite y_hat values.")

        metrics_by_name = metrics_by_name or {}
        for key, values in metrics_by_name.items():
            arr = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise OpalError(f"append_predictions_from_arrays received non-finite metrics for '{key}'.")

        ranks_arr = np.asarray(selection_rank)
        selected_arr = np.asarray(selection_top_k)
        ts_val = ts or pd.Timestamp.utcnow().isoformat()

        hist_map: Dict[str, List[Dict[str, Any]]] = {}
        for _id, hist_cell in out[["id", lh]].itertuples(index=False, name=None):
            hist_map[str(_id)] = self.normalize_hist_cell(hist_cell)

        for i, _id in enumerate(ids):
            _id = str(_id)
            cur = hist_map.get(_id, [])
            cur = [
                e for e in cur if not (e.get("kind") == "pred" and int(e.get("as_of_round", -1)) == int(as_of_round))
            ]

            rank_val = _coerce_int(ranks_arr[i])
            if rank_val is None:
                raise OpalError(f"Prediction selection rank invalid for id={_id}.")
            top_k_val = _coerce_bool(selected_arr[i])
            if top_k_val is None:
                raise OpalError(f"Prediction selection top_k invalid for id={_id}.")

            metrics_entry: dict[str, Any] = {}
            for key, values in metrics_by_name.items():
                val = _coerce_float(values[i])
                if val is None:
                    raise OpalError(f"Prediction metric '{key}' invalid for id={_id}.")
                metrics_entry[key] = val

            y_vec = y_hat_arr[i, :].tolist()
            entry = {
                "kind": "pred",
                "as_of_round": int(as_of_round),
                "run_id": str(run_id),
                "ts": ts_val,
                "y_pred": {
                    "value": y_vec,
                    "dtype": "vector",
                    "schema": {"length": int(len(y_vec))},
                },
                "y_space": "objective",
                "objective": dict(objective or {}),
                "metrics": metrics_entry,
                "selection": {"rank": rank_val, "top_k": bool(top_k_val)},
            }
            cur.append(entry)
            hist_map[_id] = cur

        out[lh] = out["id"].astype(str).map(hist_map.get)
        return out
