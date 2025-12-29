"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/data_access.py

RecordsStore abstracts reading/writing the records Parquet (USR or local),
schema validation, label history caches, candidate universe, and the single
representation (X) transform path.

This file implements the caches:
  - opal__<slug>__label_hist: list<struct{ r:int, ts:str, y:list<double>, src:str }>
  - opal__<slug>__latest_as_of_round: int
  - opal__<slug>__latest_pred_scalar: double

No predictions are stored here otherwise; canonical predictions live in the
ledger under outputs/ledger.* (predictions/runs/labels).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .registries.transforms_x import get_transform_x
from .utils import OpalError

ESSENTIAL_COLS = [
    "id",
    "bio_type",
    "sequence",
    "alphabet",
]


@dataclass
class RecordsStore:
    kind: str  # "usr" | "local"
    records_path: Path
    campaign_slug: str
    x_col: str
    y_col: str
    x_transform_name: str
    x_transform_params: Dict[str, Any]

    # --------------- basic IO ---------------
    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        return pd.read_parquet(self.records_path)

    def save_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.records_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.records_path)

    def assert_unique_ids(self, df: pd.DataFrame) -> None:
        if "id" not in df.columns:
            raise OpalError("records.parquet is missing required column 'id'.")
        ids = df["id"].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].unique().tolist()
            preview = dup[:10]
            raise OpalError(f"records.parquet contains duplicate ids (sample={preview}).")

    @staticmethod
    def deterministic_id_from_sequence(seq: str) -> str:
        import hashlib

        return "s" + hashlib.sha1(str(seq).encode("utf-8")).hexdigest()[:16]

    # --------------- cache column names ---------------
    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    def latest_as_of_round_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_as_of_round"

    def latest_pred_scalar_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_scalar"

    def upsert_current_y_column(self, df: pd.DataFrame, labels_resolved: pd.DataFrame, y_col_name: str) -> pd.DataFrame:
        """
        Ensure the configured y-column holds the latest label for the affected ids.
        'labels_resolved' must have columns ['id','y'] with concrete ids (no NaN).
        """
        out = df.copy()
        if y_col_name not in out.columns:
            out[y_col_name] = None

        id_to_y = dict(zip(labels_resolved["id"].astype(str), labels_resolved["y"]))
        mapped = out["id"].astype(str).map(id_to_y)  # list-or-NaN per row
        mask = mapped.notna()
        out.loc[mask, y_col_name] = mapped[mask]
        return out

    # --------------- label history ops ---------------
    @staticmethod
    def _normalize_hist_cell(cell: Any) -> List[Dict[str, Any]]:
        """
        Normalize a 'label_hist' cell into a Python List[Dict] where each dict has:
          {r:int, ts:str, src:str, y: List[float]}
        Be permissive on container types to tolerate different Parquet round-trips.
        Accepted inputs:
          - list / tuple of dicts
          - numpy.ndarray (object) containing dicts or lists
          - pandas.Series of dicts
          - a single dict (treated as 1-element list)
          - JSON string
          - pyarrow scalars/arrays (via .as_py() / .to_pylist())
        Any malformed entry is dropped; empty/unknown → [] (assertive default).
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
            # numpy containers / scalars
            if isinstance(x, np.ndarray):
                return [_deep_as_py(v) for v in x.tolist()]
            if isinstance(x, np.generic):
                return x.item()
            # pandas Series
            if isinstance(x, pd.Series):
                return [_deep_as_py(v) for v in x.to_list()]
            # plain python
            if isinstance(x, dict):
                return {k: _deep_as_py(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_deep_as_py(v) for v in x]
            return x

        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []

        # JSON string?
        if isinstance(cell, str):
            try:
                cell = json.loads(cell)
            except Exception:
                return []

        # Convert any nested structures deeply to Python
        cell = _deep_as_py(cell)

        # Normalize top-level to a list
        if isinstance(cell, dict):
            cell = [cell]
        elif isinstance(cell, tuple):
            cell = list(cell)
        elif not isinstance(cell, list):
            # e.g., still some exotic scalar → give up assertively
            return []

        out: List[Dict[str, Any]] = []
        for e in cell:
            if not isinstance(e, dict):
                continue
            # Coerce r to int (or drop if missing)
            r = e.get("r", None)
            try:
                e["r"] = int(r) if r is not None else None
            except Exception:
                e["r"] = None
            # Coerce y to list[float]
            y = e.get("y", [])
            y = _deep_as_py(y)
            try:
                e["y"] = [float(v) for v in (y or [])]
            except Exception:
                e["y"] = []
            # Keep only minimally valid entries
            if e["r"] is not None and isinstance(e.get("y", None), list):
                out.append(e)
        return out

    @staticmethod
    def _parse_hist_cell_strict(cell: Any) -> List[Dict[str, Any]]:
        """
        Strict parser for label history cells. Raises on malformed entries.
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

    def validate_label_hist(self, df: pd.DataFrame, *, require: bool = True) -> None:
        """
        Strictly validate label_hist cells. Raises on malformed entries.
        If require=False, missing label_hist column is allowed.
        """
        lh = self.label_hist_col()
        if lh not in df.columns:
            if require:
                raise OpalError(f"Expected label history column '{lh}' not found in records.parquet.")
            return
        bad: List[Dict[str, str]] = []
        for _id, cell in df[["id", lh]].itertuples(index=False, name=None):
            try:
                _ = self._parse_hist_cell_strict(cell)
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
            cleaned = self._normalize_hist_cell(cell)
            after = len(cleaned)
            dropped_total += max(0, before - after)
            if before != after or cleaned != cell:
                changed_rows += 1
            out.at[idx, lh] = cleaned

        report = {"rows_changed": int(changed_rows), "entries_dropped": int(dropped_total)}
        return out, report

    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,  # must have columns: id, y
        r: int,
        *,
        src: str = "ingest_y",
        fail_if_any_existing_labels: bool = True,
        if_exists: str = "fail",  # 'fail' | 'skip' | 'replace'
    ) -> pd.DataFrame:
        """
        Append to label history with immutable semantics.
        One new entry per (id, r). Fails if (id, r) already present and
        fail_if_any_existing_labels is True.
        """
        lh = self.label_hist_col()
        out = df.copy()
        if lh not in out.columns:
            out[lh] = None

        # Build map id -> list(hist entries)
        hist_map: Dict[str, List[Dict[str, Any]]] = {}
        for _id, hist_cell in out[["id", lh]].itertuples(index=False, name=None):
            hist_map[str(_id)] = self._normalize_hist_cell(hist_cell)

        new_ids = labels["id"].astype(str).tolist()
        new_ys = labels["y"].tolist()
        for i, _id in enumerate(new_ids):
            cur = hist_map.get(_id, [])
            # Duplicate guard / policy
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

        # materialize back to column
        hist_series = out["id"].astype(str).map(hist_map.get)
        out[lh] = hist_series
        return out

    def training_labels_from_y(self, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
        """
        Compute effective training labels at or before 'as_of_round' from label_hist cache.
        Returns a frame with columns: id, y
        """
        out = self.training_labels_with_round(
            df,
            as_of_round,
            cumulative_training=True,
            dedup_policy="latest_only",
        )
        return out[["id", "y"]]

    def training_labels_with_round(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        """
        Like training_labels_from_y, but also returns the observed round for the effective label.
        Returns: DataFrame columns: id, y, r
        """
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
            hist = self._normalize_hist_cell(hist_cell)
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
                r = int(best.get("r", as_of_round))
                recs.append((_id, y, r))
            elif policy == "all_rounds":
                for e in entries:
                    y = [float(v) for v in (e.get("y", []) or [])]
                    r = int(e.get("r", as_of_round))
                    recs.append((_id, y, r))
            else:  # error_on_duplicate
                if len(entries) > 1:
                    raise OpalError(
                        f"Duplicate labels for id={_id} within training scope "
                        f"(policy=error_on_duplicate, as_of_round={as_of_round})."
                    )
                e = entries[0]
                y = [float(v) for v in (e.get("y", []) or [])]
                r = int(e.get("r", as_of_round))
                recs.append((_id, y, r))

        return pd.DataFrame(recs, columns=["id", "y", "r"])

    # --------------- candidate universe & transforms ---------------
    def candidate_universe(self, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
        """
        Return a DataFrame with at least 'id' and 'sequence' for all rows with X present.
        We do not exclude labeled rows from scoring; selection policy can decide.
        """
        if self.x_col not in df.columns:
            raise OpalError(f"Candidate universe requires X column '{self.x_col}'.")
        keep = df[self.x_col].notna()
        cols = ["id", "sequence", self.x_col]
        cols = [c for c in cols if c in df.columns]
        return df.loc[keep, cols].copy()

    def transform_matrix(self, df: pd.DataFrame, ids: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build X matrix for given ids using configured transform_x plugin.
        Returns (X, id_order)
        """
        id_list = [str(i) for i in ids]
        if len(id_list) != len(set(id_list)):
            raise OpalError("transform_matrix received duplicate ids.")
        if "id" not in df.columns:
            raise OpalError("records.parquet is missing required column 'id'.")
        if self.x_col not in df.columns:
            raise OpalError(f"Missing X column '{self.x_col}'.")
        self.assert_unique_ids(df)

        df_idx = df.copy()
        df_idx["id"] = df_idx["id"].astype(str)
        df_idx = df_idx.set_index("id", drop=False)

        missing = [i for i in id_list if i not in df_idx.index]
        if missing:
            preview = missing[:10]
            raise OpalError(f"Missing ids in records.parquet for transform_matrix (sample={preview}).")

        series = df_idx.loc[id_list, self.x_col]
        null_mask = series.isna()
        if null_mask.any():
            bad_ids = series[null_mask].index.tolist()[:10]
            raise OpalError(f"X column '{self.x_col}' is null for ids (sample={bad_ids}).")

        tx = get_transform_x(self.x_transform_name, self.x_transform_params)
        X = tx(series)
        if X.shape[0] != len(id_list):
            raise OpalError(f"transform_x[{self.x_transform_name}] returned {X.shape[0]} rows for {len(id_list)} ids.")
        return np.asarray(X), id_list

    # --------------- ensure rows exist for ingest ---------------
    def ensure_rows_exist(
        self,
        df: pd.DataFrame,
        labels_df: pd.DataFrame,
        csv_df: pd.DataFrame,
        *,
        required_cols: List[str],
        conflict_policy: str,
    ) -> pd.DataFrame:
        """
        If labels_df contains sequences that are not in df, create new rows with essentials from csv_df.
        When new rows are created, required columns are copied from the CSV (strict).
        """
        out = df.copy()
        self.assert_unique_ids(out)

        policy = str(conflict_policy or "").strip().lower()
        if policy not in {"error", "skip", "replace"}:
            raise OpalError(
                f"Unknown conflict_policy_on_duplicate_ids: {conflict_policy!r} (expected: error | skip | replace)."
            )

        # Require essentials to exist in records if safety says so
        need_cols = set(ESSENTIAL_COLS + ["id"])
        missing = [c for c in need_cols if c not in out.columns]
        if missing:
            raise OpalError(f"records.parquet missing required columns: {missing}")

        # Required columns to materialize new rows from CSV
        required_cols = [str(c) for c in (required_cols or [])]
        # Build CSV lookup by sequence and/or id (must be unique)
        csv_by_seq = {}
        csv_by_id = {}
        if "sequence" in csv_df.columns:
            if csv_df["sequence"].duplicated().any():
                dup = csv_df["sequence"][csv_df["sequence"].duplicated()].unique().tolist()[:10]
                raise OpalError(f"CSV contains duplicate sequences (sample={dup}).")
            csv_by_seq = csv_df.set_index("sequence").to_dict(orient="index")
        if "id" in csv_df.columns:
            if csv_df["id"].duplicated().any():
                dup = csv_df["id"][csv_df["id"].duplicated()].unique().tolist()[:10]
                raise OpalError(f"CSV contains duplicate ids (sample={dup}).")
            csv_by_id = csv_df.set_index("id").to_dict(orient="index")

        have_id_col = "id" in labels_df.columns
        have_seq_col = "sequence" in labels_df.columns
        if not have_id_col and not have_seq_col:
            return out  # nothing to do

        known_ids = set(out["id"].astype(str).tolist()) if "id" in out.columns else set()
        seq_to_id = out[["sequence", "id"]].dropna().astype(str).drop_duplicates().set_index("sequence")["id"].to_dict()

        rows_to_add: List[Dict[str, Any]] = []

        def _csv_row_for(id_val: str | None, seq_val: str | None) -> Dict[str, Any]:
            if id_val is not None and id_val in csv_by_id:
                return csv_by_id[id_val]
            if seq_val is not None and seq_val in csv_by_seq:
                return csv_by_seq[seq_val]
            raise OpalError(
                f"CSV missing required row for id={id_val!r} sequence={seq_val!r} needed to create new records."
            )

        def _check_conflict(existing_row: pd.Series, csv_row: Dict[str, Any]) -> Dict[str, Any]:
            mismatches = {}
            for c in ("sequence", "bio_type", "alphabet"):
                if c in csv_row and c in existing_row:
                    v_existing = existing_row[c]
                    v_csv = csv_row[c]
                    if pd.notna(v_csv) and pd.notna(v_existing) and str(v_csv) != str(v_existing):
                        mismatches[c] = (v_existing, v_csv)
            if mismatches:
                if policy == "error":
                    raise OpalError(f"Duplicate id conflict for {existing_row['id']}: {mismatches}")
            return mismatches

        # a) rows WITH a real id → ensure that id exists (attach sequence if provided)
        if have_id_col:
            rows_with_id = labels_df.loc[labels_df["id"].notna()]
            for _id, seq in rows_with_id[["id", "sequence"]].itertuples(index=False, name=None):
                _id = str(_id)
                seq_val = None if not have_seq_col else (None if pd.isna(seq) else str(seq))
                if _id in known_ids:
                    row = out.loc[out["id"].astype(str) == _id].iloc[0]
                    csv_row = _csv_row_for(_id, seq_val) if (csv_by_id or csv_by_seq) else {}
                    mismatches = _check_conflict(row, csv_row)
                    if mismatches and policy == "replace":
                        for col, (_, newv) in mismatches.items():
                            out.loc[out["id"].astype(str) == _id, col] = newv
                    continue
                # new id → require sequence + essentials
                if seq_val is None:
                    raise OpalError(f"Cannot create new id={_id} without sequence.")
                csv_row = _csv_row_for(_id, seq_val)
                new_row = {"id": _id, "sequence": seq_val}
                for c in required_cols:
                    if c not in csv_row:
                        raise OpalError(f"CSV missing required column '{c}' for new id={_id}.")
                    new_row[c] = csv_row[c]
                rows_to_add.append(new_row)
        # b) rows WITHOUT id but WITH sequence → create or reuse by sequence
        if have_seq_col:
            rows_no_id = labels_df.loc[~labels_df["id"].notna()] if have_id_col else labels_df
            for seq in rows_no_id["sequence"].dropna().astype(str).tolist():
                if seq not in seq_to_id:
                    csv_row = _csv_row_for(None, seq)
                    new_row = {"id": self.deterministic_id_from_sequence(seq), "sequence": seq}
                    for c in required_cols:
                        if c not in csv_row:
                            raise OpalError(f"CSV missing required column '{c}' for new sequence={seq}.")
                        new_row[c] = csv_row[c]
                    rows_to_add.append(new_row)

        if rows_to_add:
            new_df = pd.DataFrame(rows_to_add)
            for c in need_cols:
                if c not in new_df.columns:
                    new_df[c] = None
            out = pd.concat([out, new_df], ignore_index=True)

        return out

    # --------------- update ergonomic caches ---------------
    def update_latest_cache(
        self,
        df: pd.DataFrame,
        *,
        slug: str,
        latest_as_of_round: int,
        latest_scalar_by_id: Dict[str, float],
        require_columns_present: bool,
    ) -> pd.DataFrame:
        out = df.copy()
        col_r = self.latest_as_of_round_col()
        col_s = self.latest_pred_scalar_col()
        if require_columns_present:
            missing = [c for c in (col_r, col_s) if c not in out.columns]
            if missing:
                raise OpalError(f"Required cache columns missing: {missing}")
        else:
            if col_r not in out.columns:
                out[col_r] = None
            if col_s not in out.columns:
                out[col_s] = None
        # Validate incoming values are finite — fail fast with context
        import numpy as _np
        import pandas as _pd

        incoming = _pd.Series(latest_scalar_by_id, dtype="float64")
        non_finite = ~_np.isfinite(incoming.to_numpy())
        if non_finite.any():
            bad = incoming[non_finite]
            # preview up to 15 offenders
            preview = [{"id": str(k), "value": (None if _pd.isna(v) else float(v))} for k, v in bad.head(15).items()]
            raise OpalError(
                "update_latest_cache received non‑finite values for opal__{slug}__latest_pred_scalar "
                "({n} offender(s)). Sample: {pv}".format(slug=slug, n=int(non_finite.sum()), pv=preview)
            )
        # map assignments (all finite)
        id_series = out["id"].astype(str)
        mapped = id_series.map(incoming.to_dict())
        mask_new = mapped.notna()
        out.loc[mask_new, col_s] = mapped[mask_new]
        out.loc[mask_new, col_r] = int(latest_as_of_round)
        return out

    # --------------- labeled ids ≤ round ---------------
    def labeled_id_set_leq_round(self, df: pd.DataFrame, as_of_round: int) -> set[str]:
        lh = self.label_hist_col()
        s: set[str] = set()
        if lh not in df.columns:
            return s
        for _id, hist_cell in df[["id", lh]].itertuples(index=False, name=None):
            _id = str(_id)
            for e in self._normalize_hist_cell(hist_cell):
                try:
                    if int(e.get("r", 9_999_999)) <= as_of_round:
                        s.add(_id)
                        break
                except Exception:
                    continue
        return s

    def labeled_id_set_any_round(self, df: pd.DataFrame) -> set[str]:
        lh = self.label_hist_col()
        s: set[str] = set()
        if lh not in df.columns:
            return s
        for _id, hist_cell in df[["id", lh]].itertuples(index=False, name=None):
            _id = str(_id)
            for e in self._normalize_hist_cell(hist_cell):
                if e.get("r", None) is not None:
                    s.add(_id)
                    break
        return s
