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

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .registries.transforms_x import get_rep_transform
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
    s = sequence.upper()
    if re.fullmatch(r"[ACGT]+", s or ""):
        return "dna", "dna_4"
    return "protein", "protein_20"


def _stable_id_from_sequence(sequence: str) -> str:
    """Generate a stable hex id from the sequence."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


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
        # Normalize datetime-like columns to avoid mixed object/int issues with pyarrow
        df2 = df.copy()
        if "created_at" in df2.columns:
            # Try parse as ISO/strings first; then fallback to ms and s integers
            ca = pd.to_datetime(df2["created_at"], errors="coerce", utc=True)
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
        else:
            df2 = df
        df2.to_parquet(tmp, index=False)
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
        *,
        fail_if_any_existing_labels: bool = False,
    ) -> pd.DataFrame:
        """
        Append labels (id, y) for round r. If id is new, caller must have added the row with essentials+X (if any).
        Enforces per-(id, round) immutability and optional "already-labeled" fail-fast.
        """
        if not {"id", "y"}.issubset(labels_df.columns):
            raise OpalError("labels_df must contain columns: id, y")

        out = df.copy()
        lh = self.label_hist_col()
        if lh not in out.columns:
            out[lh] = [[] for _ in range(len(out))]
        else:
            # Normalize pre-existing column: treat NaN/None/str as empty list (or parsed list)
            def _norm_hist(v):
                if isinstance(v, list):
                    return v
                if isinstance(v, str):
                    try:
                        import json as _json

                        vv = _json.loads(v)
                        return vv if isinstance(vv, list) else []
                    except Exception:
                        return []
                return []

            out[lh] = out[lh].map(_norm_hist)

        id_to_idx = {i: idx for idx, i in enumerate(out["id"].astype(str).tolist())}

        for rec in labels_df.to_dict(orient="records"):
            _id = str(rec["id"])
            y = rec["y"]
            now = now_iso()

            if _id not in id_to_idx:
                raise OpalError(
                    f"Unknown id in append_labels_from_df: {_id}. Add row first."
                )
            idx = id_to_idx[_id]

            # optional "already labeled anywhere" fail-fast
            cell = out.at[idx, lh]
            hist = list(cell) if isinstance(cell, list) else []
            if fail_if_any_existing_labels and len(hist) > 0:
                raise OpalError(
                    f"Sequence/ID already has labels in this campaign (id={_id}); refusing to add more."
                )

            # write Y column (accept float or list[float])
            out.at[idx, self.y_col] = y

            # append history, enforce immutability per round
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
            if not isinstance(hist, list) or len(hist) == 0:
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

    # ---------- Ensure rows exist / resolve by sequence ----------
    def ensure_rows_exist(
        self, df: pd.DataFrame, labels_df: pd.DataFrame, csv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resolve IDs and ensure rows exist for label writes. Behavior:
          • If labels_df includes 'id': each id must already exist, OR be creatable by finding its sequence in csv_df.
          • If labels_df lacks 'id': resolve by 'sequence':
               - If sequence exists in df → use that row's id
               - Else → create a NEW row with a generated id and essential columns
        Never infers id names from non-'id' columns.
        """
        out = df.copy()

        # Ensure essential columns exist
        for c in ESSENTIAL_COLS:
            if c not in out.columns:
                out[c] = pd.NA
        if self.y_col not in out.columns:
            out[self.y_col] = pd.NA
        lh = self.label_hist_col()
        if lh not in out.columns:
            out[lh] = [[] for _ in range(len(out))]
        else:
            out[lh] = out[lh].map(lambda v: v if isinstance(v, list) else [])

        # Build maps
        id_set = set(out["id"].astype(str))
        seq_to_id: Dict[str, str] = (
            out[["sequence", "id"]]
            .astype(str)
            .dropna()
            .drop_duplicates()
            .set_index("sequence")["id"]
            .to_dict()
        )

        # Normalize CSV helper (only for new rows when we need extra fields)
        csv = csv_df.copy()
        if "sequence" not in csv.columns:
            raise OpalError("CSV must contain 'sequence' when resolving by sequence.")
        csv["sequence"] = csv["sequence"].astype(str)
        by_seq = {r["sequence"]: r for r in csv.to_dict(orient="records")}

        # Ensure labels_df has sequence column to resolve when id is missing
        has_id = "id" in labels_df.columns
        if has_id:
            labels_df["id"] = labels_df["id"].astype(str)
        if "sequence" in labels_df.columns:
            labels_df["sequence"] = labels_df["sequence"].astype(str)

        resolved_records = []
        for rec in labels_df.to_dict(orient="records"):
            seq = rec.get("sequence")
            lab_id = rec.get("id")

            if lab_id is not None:
                lab_id = str(lab_id)
                if lab_id in id_set:
                    # nothing to add; ensure sequence filled if missing
                    resolved_records.append({"id": lab_id, "y": rec["y"]})
                    continue
                # id not found — try resolving by sequence in the CSV
                if not seq:
                    raise OpalError(
                        f"Incoming id '{lab_id}' not found and no 'sequence' provided to create a new row."
                    )
                # fallthrough to creation path using sequence

            if not seq:
                raise OpalError("CSV ingest requires 'sequence' when 'id' is absent.")

            if seq in seq_to_id:
                resolved_records.append({"id": seq_to_id[seq], "y": rec["y"]})
                continue

            # New sequence → create a row
            if seq not in by_seq:
                raise OpalError(
                    f"Sequence '{seq}' not present in the CSV to source essentials for a new row."
                )
            src = by_seq[seq]
            bio_type, alphabet = _infer_bio(seq)
            new_id = _stable_id_from_sequence(seq)
            row = {
                "id": new_id,
                "bio_type": src.get("bio_type", bio_type),
                "sequence": seq,
                "alphabet": src.get("alphabet", alphabet),
                "length": int(len(seq)),
                "source": "opal_ingest",
                # tz-aware timestamp; save_atomic keeps dtype consistent across the column
                "created_at": pd.Timestamp.now(tz="UTC"),
                self.x_col: src.get(self.x_col, pd.NA),
            }
            out = pd.concat([out, pd.DataFrame([row])], axis=0, ignore_index=True)
            id_set.add(new_id)
            seq_to_id[seq] = new_id
            resolved_records.append({"id": new_id, "y": rec["y"]})

        # Replace labels_df ids in-place for downstream calls
        labels_df.drop(
            columns=[c for c in ["id", "sequence"] if c in labels_df.columns],
            inplace=True,
            errors="ignore",
        )
        labels_df["id"] = [r["id"] for r in resolved_records]
        labels_df["y"] = [r["y"] for r in resolved_records]
        return out
