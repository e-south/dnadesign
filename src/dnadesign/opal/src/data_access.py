"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/data_access.py

RecordsStore abstracts reading/writing the records Parquet (USR or local),
validating essential schema, and applying stateless transforms to the
representation column (X). It also:

- Manages the per-campaign opal__<slug>__label_hist append-only column.
- Enforces immutability of history per (id, round) unless explicitly overridden.
- Builds the candidate universe (has X, not yet labeled ≤ round k).
- Checks uniformity of bio_type and alphabet for training/candidates.
- Provides fixed-dimension X matrices (np.float32) for model calls.

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

from .registries.rep_transforms import get_rep_transform
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


@dataclass
class RecordsStore:
    kind: str  # "usr" or "local"
    records_path: Path
    campaign_slug: str
    x_col: str
    y_col: str
    rep_transform_name: str
    rep_transform_params: Dict[str, Any]

    # ---------- IO ----------
    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(
                f"records.parquet not found: {self.records_path}", ExitCodes.NOT_FOUND
            )
        return pd.read_parquet(self.records_path)

    def save_atomic(self, df: pd.DataFrame) -> None:
        ensure_dir(self.records_path.parent)
        tmp = self.records_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.records_path)

    def schema_validate_essential(self, df: pd.DataFrame) -> None:
        missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
        if missing:
            raise OpalError(
                f"Missing essential columns: {missing}", ExitCodes.CONTRACT_VIOLATION
            )

    # ---------- Label history ----------
    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels_df: pd.DataFrame,  # must contain id, y (y can be float or list[float])
        r: int,
    ) -> pd.DataFrame:
        """
        Append labels (id, y) for round r. If id is new, caller must have added the row with essentials+X.
        This method enforces per-(id, round) immutability.
        """
        if not {"id", "y"}.issubset(labels_df.columns):
            raise OpalError("labels_df must contain columns: id, y")

        out = df.copy()
        lh = self.label_hist_col()
        if lh not in out.columns:
            out[lh] = [[] for _ in range(len(out))]

        id_to_idx = {i: idx for idx, i in enumerate(out["id"].astype(str).tolist())}

        for rec in labels_df.to_dict(orient="records"):
            _id = str(rec["id"])
            y = rec["y"]
            now = now_iso()

            if _id not in id_to_idx:
                raise OpalError(
                    f"Unknown id in append_labels_from_df: {_id}. Add row with essentials+X first."
                )
            idx = id_to_idx[_id]

            # write Y column (accept float or list[float])
            out.at[idx, self.y_col] = y

            # append history, enforce immutability per round
            hist = list(out.at[idx, lh] or [])
            for ev in hist:
                if int(ev.get("r", -1)) == int(r):
                    # idempotent if equal, else refuse
                    if ev.get("y") == y:
                        break
                    if isinstance(ev.get("y"), str) and json.dumps(y) == ev.get("y"):
                        break
                    if isinstance(y, str) and json.dumps(ev.get("y")) == y:
                        break
                    raise OpalError(
                        f"Label history conflict for id {_id} at r={r}; refusing to change history."
                    )
            else:
                hist.append({"r": int(r), "y": y, "ts": now})
                out.at[idx, lh] = hist

        return out

    # ---------- Representation matrix ----------
    def transform_matrix(
        self, df: pd.DataFrame, ids: List[str]
    ) -> Tuple[np.ndarray, int]:
        ids = [str(i) for i in ids]
        sub = df.loc[df["id"].astype(str).isin(ids), self.x_col]
        # Keep the original order of ids as provided
        idx_order = df.index[df["id"].astype(str).isin(ids)]
        sub = sub.reindex(idx_order)
        transform = get_rep_transform(
            self.rep_transform_name, self.rep_transform_params
        )
        X, d = transform.transform(sub)
        if X.shape[0] != len(ids):
            raise OpalError("transform_matrix row count mismatch.")
        return X, d

    # ---------- Effective labels ----------
    def effective_labels_latest_only(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Returns DataFrame with id and effective y for rounds <= k,
        choosing the label with the highest r <= k for each id.
        """
        lh = self.label_hist_col()
        out = []
        for rid, hist in df[["id", lh]].itertuples(index=False):
            if not hist or not isinstance(hist, list):
                continue
            elig = [h for h in hist if int(h.get("r", -1)) <= int(k)]
            if not elig:
                continue
            hbest = max(elig, key=lambda h: int(h.get("r", -1)))
            out.append({"id": str(rid), "y": hbest["y"], "src_round": int(hbest["r"])})
        return pd.DataFrame(out)

    # ---------- Candidate universe ----------
    def candidate_universe(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Candidates are rows with x_col present and NOT labeled in rounds <= k.
        """
        if self.x_col not in df.columns:
            raise OpalError(f"Representation column not found: {self.x_col}")

        df = df.copy()
        lh = self.label_hist_col()
        if lh not in df.columns:
            df[lh] = [[] for _ in range(len(df))]

        def has_labeled_le_k(hist) -> bool:
            if not isinstance(hist, list):
                return False
            return any(int(h.get("r", -1)) <= int(k) for h in hist)

        has_x = df[self.x_col].notna()
        labeled = df[lh].map(has_labeled_le_k)
        return df[has_x & (~labeled)][["id", "sequence", self.x_col]]

    # ---------- Uniformity checks ----------
    def check_biotype_alphabet_uniformity(
        self, df: pd.DataFrame, ids: Iterable[str]
    ) -> None:
        ids = set([str(i) for i in ids])
        subset = df.loc[
            df["id"].astype(str).isin(list(ids)), ["bio_type", "alphabet", "id"]
        ]
        if subset.empty:
            return
        bvals = set(subset["bio_type"].dropna().unique().tolist())
        avals = set(subset["alphabet"].dropna().unique().tolist())
        if len(bvals) != 1 or len(avals) != 1:
            sample = subset.head(10).to_dict(orient="records")
            raise OpalError(
                f"Mixed bio_type/alphabet detected among selected rows. "
                f"bio_type={sorted(bvals)}, alphabet={sorted(avals)}; sample={sample}",
                ExitCodes.CONTRACT_VIOLATION,
            )

    # ---------- Ensure new rows for new IDs ----------
    def ensure_rows_exist(
        self, df: pd.DataFrame, ids: List[str], csv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For any id in `ids` not present in df, add a row using essentials pulled from `csv_df`.
        Requires columns: id (or design_id), sequence, bio_type, alphabet, and the configured X column.
        """
        out = df.copy()
        existing = set(out["id"].astype(str))
        needed = [str(i) for i in ids if str(i) not in existing]
        if not needed:
            return out

        # normalize CSV id column
        csv = csv_df.copy()
        id_col = (
            "id"
            if "id" in csv.columns
            else ("design_id" if "design_id" in csv.columns else None)
        )
        if id_col is None:
            raise OpalError(
                "CSV must contain 'id' or 'design_id' to add new rows.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if self.x_col not in csv.columns:
            raise OpalError(
                f"CSV missing representation column '{self.x_col}' required for new rows.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        for col in ("sequence", "bio_type", "alphabet"):
            if col not in csv.columns:
                raise OpalError(
                    f"CSV missing essential column '{col}' for new rows.",
                    ExitCodes.CONTRACT_VIOLATION,
                )

        csv[id_col] = csv[id_col].astype(str)
        by_id = {r[id_col]: r for r in csv.to_dict(orient="records")}
        rows = []
        for _id in needed:
            if _id not in by_id:
                raise OpalError(
                    f"New id '{_id}' not found in CSV to create essentials+X."
                )
            rec = by_id[_id]
            seq = str(rec["sequence"])
            row = {
                "id": _id,
                "bio_type": rec["bio_type"],
                "sequence": seq,
                "alphabet": rec["alphabet"],
                "length": int(len(seq)),
                "source": "opal_ingest",
                "created_at": now_iso(),
                self.x_col: rec[self.x_col],
            }
            rows.append(row)

        # ensure all essential columns in df to align
        for c in ESSENTIAL_COLS:
            if c not in out.columns:
                out[c] = pd.NA
        if self.y_col not in out.columns:
            out[self.y_col] = pd.NA
        lh = self.label_hist_col()
        if lh not in out.columns:
            out[lh] = [[] for _ in range(len(out))]

        out = pd.concat([out, pd.DataFrame(rows)], axis=0, ignore_index=True)
        return out
