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

    # --------------- cache column names ---------------
    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    def latest_as_of_round_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_as_of_round"

    def latest_pred_scalar_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_scalar"

    def upsert_current_y_column(
        self, df: pd.DataFrame, labels_resolved: pd.DataFrame, y_col_name: str
    ) -> pd.DataFrame:
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

    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,  # must have columns: id, y
        r: int,
        *,
        src: str = "ingest_y",
        fail_if_any_existing_labels: bool = True,
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
            # Duplicate guard
            if any((int(e.get("r", -1)) == int(r)) for e in cur):
                if fail_if_any_existing_labels:
                    raise OpalError(f"Label history already has (id={_id}, r={r})")
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

    def training_labels_from_y(
        self, df: pd.DataFrame, as_of_round: int
    ) -> pd.DataFrame:
        """
        Compute effective training labels at or before 'as_of_round' from label_hist cache.
        Returns a frame with columns: id, y
        """
        lh = self.label_hist_col()
        if lh not in df.columns:
            raise OpalError(
                f"Expected label history column '{lh}' not found in records.parquet. "
            )
        recs: List[Tuple[str, List[float]]] = []
        for _id, hist_cell in df[["id", lh]].itertuples(index=False, name=None):
            _id = str(_id)
            hist = self._normalize_hist_cell(hist_cell)
            # choose latest entry with r <= as_of_round
            best = None
            for e in hist:
                try:
                    rr = int(e.get("r", 9_999_999))
                    if rr <= as_of_round and (
                        best is None or rr > int(best.get("r", -1))
                    ):
                        best = e
                except Exception:
                    continue
            if best is not None:
                recs.append((_id, [float(v) for v in best.get("y", [])]))
        import os
        import sys

        if not recs and os.getenv("OPAL_DEBUG_LABELS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            non_empty = df[lh].dropna()
            print(
                f"[opal.debug] label_hist={lh} non_empty_rows={len(non_empty)} as_of_round={as_of_round}",
                file=sys.stderr,
            )
            for i, (_, cell) in enumerate(non_empty.iloc[:3].items()):
                print(
                    f"[opal.debug] sample_cell_{i} type={type(cell)} preview={str(cell)[:160]}",
                    file=sys.stderr,
                )
        return pd.DataFrame(recs, columns=["id", "y"])

    def training_labels_with_round(
        self, df: pd.DataFrame, as_of_round: int
    ) -> pd.DataFrame:
        """
        Like training_labels_from_y, but also returns the observed round for the effective label.
        Returns: DataFrame columns: id, y, r
        """
        lh = self.label_hist_col()
        if lh not in df.columns:
            raise OpalError(
                f"Expected label history column '{lh}' not found in records.parquet. "
            )
        recs: List[Tuple[str, List[float], int]] = []
        for _id, hist_cell in df[["id", lh]].itertuples(index=False, name=None):
            _id = str(_id)
            hist = self._normalize_hist_cell(hist_cell)
            best = None
            for e in hist:
                try:
                    rr = int(e.get("r", 9_999_999))
                    if rr <= as_of_round and (
                        best is None or rr > int(best.get("r", -1))
                    ):
                        best = e
                except Exception:
                    continue
            if best is not None:
                y = [float(v) for v in (best.get("y", []) or [])]
                r = int(best.get("r", as_of_round))
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

    def transform_matrix(
        self, df: pd.DataFrame, ids: Iterable[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build X matrix for given ids using configured transform_x plugin.
        Returns (X, id_order)
        """
        id_set = set(map(str, ids))
        cols = ["id", self.x_col]
        subset = df.loc[df["id"].astype(str).isin(id_set), cols].copy()
        subset = subset.dropna(subset=[self.x_col])
        subset = subset.sort_values("id")
        tx = get_transform_x(self.x_transform_name, self.x_transform_params)
        X = tx(subset[self.x_col])
        return np.asarray(X), subset["id"].astype(str).tolist()

    # --------------- ensure rows exist for ingest ---------------
    def ensure_rows_exist(
        self, df: pd.DataFrame, labels_df: pd.DataFrame, csv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        If labels_df contains sequences that are not in df, create new rows with essentials from csv_df.
        We do not write X here; X should arrive via normal data pipelines.
        """
        out = df.copy()
        need_cols = set(ESSENTIAL_COLS + ["id"])
        for c in need_cols:
            if c not in out.columns:
                out[c] = None

        have_id_col = "id" in labels_df.columns
        have_seq_col = "sequence" in labels_df.columns
        if not have_id_col and not have_seq_col:
            return out  # nothing to do

        known_ids = (
            set(out["id"].astype(str).tolist()) if "id" in out.columns else set()
        )
        seq_to_id = (
            out[["sequence", "id"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .set_index("sequence")["id"]
            .to_dict()
            if "sequence" in out.columns and "id" in out.columns
            else {}
        )

        def _gen_id(seq: str) -> str:
            # simple deterministic hash-based id for reproducibility when creating by sequence
            import hashlib

            return "s" + hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]

        rows_to_add: List[Dict[str, Any]] = []
        # a) rows WITH a real id → ensure that id exists (attach sequence if provided)
        if have_id_col:
            rows_with_id = labels_df.loc[labels_df["id"].notna()]
            for _id, seq in rows_with_id[["id", "sequence"]].itertuples(
                index=False, name=None
            ):
                _id = str(_id)
                if _id not in known_ids:
                    rows_to_add.append(
                        {
                            "id": _id,
                            "sequence": (
                                None
                                if not have_seq_col
                                else (None if pd.isna(seq) else str(seq))
                            ),
                        }
                    )
        # b) rows WITHOUT id but WITH sequence → create or reuse by sequence
        if have_seq_col:
            rows_no_id = (
                labels_df.loc[~labels_df["id"].notna()] if have_id_col else labels_df
            )
            for seq in rows_no_id["sequence"].dropna().astype(str).tolist():
                if seq not in seq_to_id:
                    rows_to_add.append({"id": _gen_id(seq), "sequence": seq})

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
    ) -> pd.DataFrame:
        out = df.copy()
        col_r = self.latest_as_of_round_col()
        col_s = self.latest_pred_scalar_col()
        if col_r not in out.columns:
            out[col_r] = None
        if col_s not in out.columns:
            out[col_s] = None

        out[col_r] = int(latest_as_of_round)
        # Validate incoming values are finite — fail fast with context
        import numpy as _np
        import pandas as _pd

        incoming = _pd.Series(latest_scalar_by_id, dtype="float64")
        non_finite = ~_np.isfinite(incoming.to_numpy())
        if non_finite.any():
            bad = incoming[non_finite]
            # preview up to 15 offenders
            preview = [
                {"id": str(k), "value": (None if _pd.isna(v) else float(v))}
                for k, v in bad.head(15).items()
            ]
            raise OpalError(
                "update_latest_cache received non‑finite values for opal__{slug}__latest_pred_scalar "
                "({n} offender(s)). Sample: {pv}".format(
                    slug=slug, n=int(non_finite.sum()), pv=preview
                )
            )
        # map assignments (all finite)
        id_series = out["id"].astype(str)
        mapped = id_series.map(incoming.to_dict())
        mask_new = mapped.notna()
        out.loc[mask_new, col_s] = mapped[mask_new]
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
