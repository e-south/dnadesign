"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/data_access.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..core.round_context import PluginCtx
from ..core.utils import OpalError
from ..registries.transforms_x import get_transform_x
from .label_history import LabelHistory
from .records_io import RecordsIO

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
    _io: RecordsIO = field(init=False, repr=False)
    _label_hist: LabelHistory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._io = RecordsIO(self.records_path)
        self._label_hist = LabelHistory(self.campaign_slug)

    # --------------- basic IO ---------------
    def load(self) -> pd.DataFrame:
        return self._io.load()

    def save_atomic(self, df: pd.DataFrame) -> None:
        self._io.save_atomic(df)

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

    # --------------- label history column ---------------
    def label_hist_col(self) -> str:
        return self._label_hist.label_hist_col()

    def ensure_label_hist_column(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        lh = self.label_hist_col()
        if lh in df.columns:
            return df, []
        out = df.copy()
        out[lh] = None
        return out, [lh]

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
        Normalize a 'label_hist' cell into a Python List[Dict] in the current schema.
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
        return LabelHistory.normalize_hist_cell(cell)

    @staticmethod
    def _parse_hist_cell_strict(cell: Any) -> List[Dict[str, Any]]:
        """
        Strict parser for label history cells. Raises on malformed entries.
        """
        return LabelHistory.parse_hist_cell_strict(cell)

    def validate_label_hist(self, df: pd.DataFrame, *, require: bool = True) -> None:
        self._label_hist.validate_label_hist(df, require=require)

    def repair_label_hist(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
        return self._label_hist.repair_label_hist(df)

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
        return self._label_hist.append_labels_from_df(
            df,
            labels,
            r,
            src=src,
            fail_if_any_existing_labels=fail_if_any_existing_labels,
            if_exists=if_exists,
        )

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
        return self._label_hist.append_predictions_from_arrays(
            df,
            ids=ids,
            y_hat=y_hat,
            as_of_round=as_of_round,
            run_id=run_id,
            objective=objective,
            metrics_by_name=metrics_by_name,
            selection_rank=selection_rank,
            selection_top_k=selection_top_k,
            ts=ts,
        )

    def training_labels_from_y(self, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
        return self._label_hist.training_labels_from_y(df, as_of_round)

    def training_labels_with_round(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        return self._label_hist.training_labels_with_round(
            df,
            as_of_round,
            cumulative_training=cumulative_training,
            dedup_policy=dedup_policy,
        )

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

    def transform_matrix(self, df: pd.DataFrame, ids: Iterable[str], *, ctx: PluginCtx) -> Tuple[np.ndarray, List[str]]:
        """
        Build X matrix for given ids using configured transform_x plugin.
        Returns (X, id_order)
        """
        if ctx is None:
            raise OpalError("transform_matrix requires a PluginCtx for transform_x.")
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
        X = tx(series, ctx=ctx)
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
                f"CSV missing required row for id={id_val!r} sequence={seq_val!r} needed to create new records. "
                "Ensure the input file includes the sequence/id and required columns for new rows."
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
                if seq_val is not None and seq_val in seq_to_id and seq_to_id[seq_val] != _id:
                    raise OpalError(
                        f"Sequence already exists for id={seq_to_id[seq_val]!r}; "
                        f"cannot create new id={_id!r} for sequence={seq_val!r}."
                    )
                # new id → require sequence + essentials
                if seq_val is None:
                    raise OpalError(
                        f"Cannot create new id={_id} without sequence. "
                        "Provide a sequence column or pre-create this id in records.parquet."
                    )
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
                    new_id = self.deterministic_id_from_sequence(seq)
                    if new_id in known_ids:
                        raise OpalError(
                            f"Deterministic id collision for sequence={seq!r}. "
                            f"Generated id={new_id!r} already exists in records."
                        )
                    new_row = {
                        "id": new_id,
                        "sequence": seq,
                    }
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
                    if e.get("kind") != "label":
                        continue
                    if int(e.get("observed_round", 9_999_999)) <= as_of_round:
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
                if e.get("kind") == "label" and e.get("observed_round", None) is not None:
                    s.add(_id)
                    break
        return s
