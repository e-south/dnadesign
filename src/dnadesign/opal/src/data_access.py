"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/data_access.py

RecordsStore abstracts reading/writing the records Parquet (USR or local),
validating essential schema, label history, candidate universe, and the single
representation transform (X) path.

- Adds append_predictions_history(...), keeping predictions in append-only history.
- Provides ensure_rows_exist / append_labels_from_df used by ingest.
- Finishes candidate_universe and exposes labeled_id_set_leq_round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .registries.transforms_x import get_transform_x
from .utils import ExitCodes, OpalError, ensure_dir, now_iso

ESSENTIAL_COLS = [
    "id",
    "bio_type",
    "sequence",
    "alphabet",
    "length",
    "source",
    "created_at",
]


def _infer_bio(sequence: str) -> tuple[str, str]:
    """Heuristically infer (bio_type, alphabet) from the sequence."""
    s = (sequence or "").upper()
    if re.fullmatch(r"[ACGT]+", s):
        return "dna", "dna_4"
    return "protein", "protein_20"


def _stable_id_from_sequence(sequence: str) -> str:
    """Generate a stable hex id from the sequence."""
    return hashlib.sha256((sequence or "").encode("utf-8")).hexdigest()


def _is_null_like(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


@dataclass
class RecordsStore:
    kind: str  # "usr" or "local"
    records_path: Path
    campaign_slug: str
    x_col: str
    y_col: str
    x_transform_name: str
    x_transform_params: Dict[str, Any]

    # ---------- IO ----------
    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(
                f"records.parquet not found: {self.records_path}", ExitCodes.NOT_FOUND
            )
        return pd.read_parquet(self.records_path)

    def save_atomic(self, df: pd.DataFrame) -> None:
        """
        Write Parquet atomically and normalize datetime columns to avoid
        mixed dtypes with pyarrow.
        """
        ensure_dir(self.records_path.parent)
        tmp = self.records_path.with_suffix(".tmp.parquet")
        df2 = df.copy()

        # Normalize 'created_at' to UTC timestamps
        if "created_at" in df2.columns:
            ca = pd.to_datetime(df2["created_at"], errors="coerce", utc=True)
            # retry with common integer units if needed
            mask = ca.isna() & df2["created_at"].notna()
            if mask.any():
                ca.loc[mask] = pd.to_datetime(
                    df2.loc[mask, "created_at"], errors="coerce", utc=True, unit="ms"
                )
                mask = ca.isna() & df2["created_at"].notna()
            if mask.any():
                ca.loc[mask] = pd.to_datetime(
                    df2.loc[mask, "created_at"], errors="coerce", utc=True, unit="s"
                )
            df2["created_at"] = ca

        df2.to_parquet(tmp, index=False)
        tmp.replace(self.records_path)

    def schema_validate_essential(self, df: pd.DataFrame) -> None:
        missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
        if missing:
            raise OpalError(
                f"Missing essential columns: {missing}", ExitCodes.CONTRACT_VIOLATION
            )

    # ---------- Label history helpers ----------
    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    @staticmethod
    def _normalize_hist_cell(v) -> list[dict]:
        """
        Label history cells may be Python lists (preferred) or JSON strings.
        Always return list-of-dicts (possibly empty).
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                vv = json.loads(v)
                return vv if isinstance(vv, list) else []
            except Exception:
                return []
        return []

    @staticmethod
    def _labeled_leq_round(hist_cell, k: int) -> bool:
        """True if ANY history entry has r <= k (treat r==-1 as eligible)."""
        try:
            for e in RecordsStore._normalize_hist_cell(hist_cell):
                r = int(e.get("r", 9_999_999))
                t = e.get("t", "label")
                if t == "label" and r <= int(k):
                    return True
        except Exception:
            return False
        return False

    # ---------- Cells parsing ----------
    @staticmethod
    def _parse_y_cell_to_vec(v) -> list[float] | None:
        """
        Parse a Y cell into list[float]. Returns None if invalid.
        Accepts: list/tuple/ndarray/Series or JSON-like strings. Scalars -> [scalar].
        """
        if v is None:
            return None
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            try:
                arr = np.asarray(v, dtype=float).ravel()
            except Exception:
                return None
            if not np.all(np.isfinite(arr)):
                return None
            return [float(x) for x in arr.tolist()]
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            try:
                vv = json.loads(s)
                return RecordsStore._parse_y_cell_to_vec(vv)
            except Exception:
                pass
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                parts = [] if inner == "" else [p.strip() for p in inner.split(",")]
                try:
                    arr = np.asarray([float(p) for p in parts], dtype=float)
                except Exception:
                    return None
                if not np.all(np.isfinite(arr)):
                    return None
                return [float(x) for x in arr.tolist()]
            try:
                f = float(s)
                if np.isfinite(f):
                    return [float(f)]
            except Exception:
                return None
            return None
        try:
            f = float(v)
            if np.isfinite(f):
                return [float(f)]
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_x_cell_to_vec(v) -> list[float] | None:
        """Normalize X cell to list[float]."""
        if v is None:
            return None
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).ravel()
            if arr.ndim != 1:
                return None
            if not np.all(np.isfinite(arr)):
                return None
            return [float(x) for x in arr.tolist()]
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    vv = json.loads(s)
                    return RecordsStore._parse_x_cell_to_vec(vv)
                except Exception:
                    return None
            try:
                f = float(s)
                return [float(f)] if np.isfinite(f) else None
            except Exception:
                return None
        try:
            f = float(v)
            return [float(f)] if np.isfinite(f) else None
        except Exception:
            return None

    # ---------- Training labels (Y-column is the source of truth) ----------
    def training_labels_from_y(self, df: pd.DataFrame, round_k: int) -> pd.DataFrame:
        """
        Return rows eligible for training at round_k with columns ['id','y'].
        Eligibility:
          • X present & non-null AND
          • Y present & parses to finite vector AND
          • (no history) OR (history empty) OR (ANY history event 'label' has r <= round_k)
            (r==-1 allowed and treated as eligible for all rounds)
        """
        if self.x_col not in df.columns or self.y_col not in df.columns:
            return pd.DataFrame(columns=["id", "y"])

        lh = self.label_hist_col()
        have_h = lh in df.columns

        rows: list[dict] = []
        for _, row in df.iterrows():
            vx = row.get(self.x_col, None)
            if _is_null_like(vx):
                continue
            y_vec = self._parse_y_cell_to_vec(row.get(self.y_col, None))
            if y_vec is None:
                continue
            eligible = (
                (not have_h)
                or (len(self._normalize_hist_cell(row.get(lh))) == 0)
                or self._labeled_leq_round(row.get(lh), round_k)
            )
            if eligible:
                rows.append({"id": str(row["id"]), "y": y_vec})
        return pd.DataFrame(rows, columns=["id", "y"])

    def labeled_id_set_leq_round(self, df: pd.DataFrame, round_k: int) -> set[str]:
        """Set of IDs that have at least one label history entry with r <= round_k."""
        lh = self.label_hist_col()
        if lh not in df.columns:
            return set()
        out: set[str] = set()
        for _, row in df.iterrows():
            if self._labeled_leq_round(row.get(lh), round_k):
                out.add(str(row["id"]))
        return out

    # ---------- Candidate universe ----------
    def candidate_universe(self, df: pd.DataFrame, round_k: int) -> pd.DataFrame:
        """
        Return rows with X present and NOT labeled ≤ round_k.
        """
        lh = self.label_hist_col()
        have_h = lh in df.columns

        mask_x = (
            df[self.x_col].apply(lambda v: not _is_null_like(v))
            if self.x_col in df.columns
            else pd.Series(False, index=df.index)
        )
        if have_h:
            mask_not_labeled = df[lh].apply(
                lambda hist: not self._labeled_leq_round(hist, round_k)
            )
        else:
            mask_not_labeled = pd.Series(True, index=df.index)

        out = df.loc[mask_x & mask_not_labeled, ["id", "sequence"]].copy()
        return out.reset_index(drop=True)

    # ---------- X transform ----------
    def transform_matrix(
        self, df: pd.DataFrame, ids: Iterable[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract X vectors (after transform_x) for the given ids.
        Returns (X_matrix, id_list_in_order). Raises OpalError on any transform issue.
        """
        ids = [str(x) for x in ids]
        if not ids:
            return np.zeros((0, 0), dtype=float), []

        # Subset and keep lookup by id
        df_id_map = (
            df.set_index(df["id"].astype(str), drop=False)
            if "id" in df.columns
            else None
        )
        if df_id_map is None:
            raise OpalError("Records table missing 'id' column.")

        tx = get_transform_x(self.x_transform_name)
        rows: list[np.ndarray] = []
        id_list: list[str] = []

        for rid in ids:
            if rid not in df_id_map.index:
                raise OpalError(f"ID not found in records: {rid}")
            row = df_id_map.loc[rid]

            v = row.get(self.x_col, None)
            if _is_null_like(v):
                raise OpalError(
                    f"Missing X value in column '{self.x_col}' for id={rid}"
                )
            # Apply transform_x plugin
            try:
                x_vec = tx(v, params=self.x_transform_params)
            except Exception as e:
                raise OpalError(
                    f"X transform '{self.x_transform_name}' failed for id={rid}: {e}"
                ) from e

            # Validate 1-D finite vector
            try:
                arr = np.asarray(x_vec, dtype=float).ravel()
            except Exception as e:
                raise OpalError(
                    f"X transform '{self.x_transform_name}' returned non-numeric for id={rid}"
                ) from e
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise OpalError(
                    f"X transform '{self.x_transform_name}' produced invalid vector for id={rid} (empty or NaN/Inf)"
                )
            rows.append(arr)
            id_list.append(rid)

        # Enforce constant width and give a pinpoint error otherwise
        lengths = {r.shape[0] for r in rows}
        if len(lengths) != 1:
            dim_counts: Dict[int, List[str]] = {}
            for rid, arr in zip(id_list, rows):
                dim_counts.setdefault(arr.shape[0], []).append(str(rid))
            detail = ", ".join(
                f"d={d} (n={len(vs)}; e.g. {vs[0]})"
                for d, vs in sorted(dim_counts.items())
            )
            raise OpalError(
                f"X vectors must have a constant dimension across rows; got {detail}"
            )

        X = np.vstack(rows).astype(float, copy=False)
        return X, id_list

    # ---------- Ingest helpers ----------
    def ensure_rows_exist(
        self, df: pd.DataFrame, labels_df: pd.DataFrame, csv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        When ingesting labels: if labels_df has sequences not present in df,
        create minimal rows (essentials) with a stable id.
        """
        new = df.copy()
        seq_col = "sequence"
        if seq_col not in labels_df.columns:
            return new  # nothing to create
        have = set(new["sequence"].astype(str)) if "sequence" in new.columns else set()
        need = [s for s in labels_df[seq_col].astype(str).tolist() if s not in have]
        if not need:
            return new

        essentials: list[dict] = []
        for s in need:
            bio, alpha = _infer_bio(s)
            rid = _stable_id_from_sequence(s)
            essentials.append(
                {
                    "id": rid,
                    "bio_type": bio,
                    "sequence": s,
                    "alphabet": alpha,
                    "length": len(s or ""),
                    "source": "ingest",
                    "created_at": now_iso(),
                }
            )

        new_rows = pd.DataFrame(essentials)
        # Create missing columns in df for essentials
        for c in ESSENTIAL_COLS:
            if c not in new.columns:
                new[c] = pd.Series(dtype=new_rows[c].dtype)
        # Also ensure label history column exists
        lh = self.label_hist_col()
        if lh not in new.columns:
            new[lh] = None

        new = pd.concat([new, new_rows], ignore_index=True)
        return new

    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels_id_y: pd.DataFrame,
        r: int,
        *,
        fail_if_any_existing_labels: bool = True,
    ) -> pd.DataFrame:
        """
        Append label rows (id, y) to label history, and write current Y column.
        """
        if not {"id", "y"}.issubset(set(labels_id_y.columns)):
            raise OpalError("append_labels_from_df expects columns ['id','y']")

        new = df.copy()
        lh = self.label_hist_col()
        if lh not in new.columns:
            new[lh] = None

        # Check conflicts per-(id, r)
        if fail_if_any_existing_labels:
            id_set = set(labels_id_y["id"].astype(str))
            for _, row in new.iterrows():
                if str(row["id"]) in id_set:
                    for e in self._normalize_hist_cell(row.get(lh)):
                        if e.get("t", "label") == "label" and int(
                            e.get("r", -999)
                        ) == int(r):
                            raise OpalError(
                                f"Label history already has a label for id={row['id']} at r={r}."
                            )

        # Build map id -> y
        id2y = {}
        for i, y in labels_id_y[["id", "y"]].to_records(index=False):
            if isinstance(y, (list, tuple, np.ndarray)):
                vec = list(np.asarray(y, dtype=float).ravel())
            else:
                vec = [float(y)]
            if not np.all(np.isfinite(vec)):
                raise OpalError(f"Non-finite values in label for id={i}")
            id2y[str(i)] = vec

        # Update Y column and history
        for idx, row in new.iterrows():
            sid = str(row["id"])
            if sid not in id2y:
                continue
            yv = id2y[sid]
            new.at[idx, self.y_col] = yv
            hist = self._normalize_hist_cell(row.get(lh))
            hist.append({"t": "label", "r": int(r), "y": yv, "ts": now_iso()})
            new.at[idx, lh] = hist

        return new

    # ---------- Predictions history ----------
    def append_predictions_history(
        self,
        df: pd.DataFrame,
        preds_df: pd.DataFrame,
        r: int,
        y_expected_length: int | None = None,
    ) -> pd.DataFrame:
        """
        Append prediction entries to the same label history column (single source of truth).
        """
        required = {"id", "yhat", "unc_mean_std", "score", "rank", "selected"}
        if not required.issubset(set(preds_df.columns)):
            missing = required - set(preds_df.columns)
            raise OpalError(f"append_predictions_history missing columns: {missing}")

        new = df.copy()
        lh = self.label_hist_col()
        if lh not in new.columns:
            new[lh] = None

        pmap = {
            str(rw["id"]): {
                "yhat": (
                    list(rw["yhat"])
                    if isinstance(rw["yhat"], (list, tuple, np.ndarray))
                    else rw["yhat"]
                ),
                "unc": (
                    float(rw["unc_mean_std"]) if pd.notna(rw["unc_mean_std"]) else None
                ),
                "score": float(rw["score"]) if pd.notna(rw["score"]) else None,
                "rank": int(rw["rank"]) if pd.notna(rw["rank"]) else None,
                "selected": bool(rw["selected"]),
            }
            for _, rw in preds_df.iterrows()
        }

        for idx, row in new.iterrows():
            sid = str(row["id"])
            if sid not in pmap:
                continue
            entry = pmap[sid]
            hist = self._normalize_hist_cell(row.get(lh))
            hist.append(
                {
                    "t": "pred",
                    "r": int(r),
                    "yhat": entry["yhat"],
                    "unc": entry["unc"],
                    "score": entry["score"],
                    "rank": entry["rank"],
                    "selected": entry["selected"],
                    "ts": now_iso(),
                }
            )
            new.at[idx, lh] = hist

        return new
