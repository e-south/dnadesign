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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .transforms.registry import get_transform
from .utils import (
    ExitCodes,
    OpalError,
    compute_records_sha256,
    ensure_dir,
    now_iso,
    usr_compatible_id,
)

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
    transform_name: str
    transform_params: Dict[str, Any]

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

    def append_labels(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        r: int,
        allow_overwrite_meta: bool = False,
    ) -> pd.DataFrame:
        """
        labels must have: id, y [, sequence, bio_type, alphabet, ...]
        - if id not found -> require essential columns + x_col present
        - else validate optional sequence match (if provided)
        - write y to y_col, append {r,y,ts} to label_hist
        """
        lab_required = ["id", "y"]
        missing = [c for c in lab_required if c not in labels.columns]
        if missing:
            raise OpalError(
                f"Labels missing required columns: {missing}", ExitCodes.BAD_ARGS
            )

        df = df.copy()
        lh = self.label_hist_col()
        if lh not in df.columns:
            df[lh] = [[] for _ in range(len(df))]

        # map id -> row index
        id_to_idx = {i: idx for idx, i in enumerate(df["id"].tolist())}

        added_rows: List[dict] = []
        to_update_idx: List[int] = []

        for rec in labels.to_dict(orient="records"):
            _id = rec["id"]
            y = float(rec["y"])
            now = now_iso()

            if _id in id_to_idx:
                idx = id_to_idx[_id]
                # optional integrity checks
                if "sequence" in rec and isinstance(rec["sequence"], str):
                    if str(df.at[idx, "sequence"]).upper() != rec["sequence"].upper():
                        raise OpalError(
                            f"Sequence mismatch for id {_id}; refusing to append label."
                        )
                # write y
                df.at[idx, self.y_col] = y
                # append history (immutability per r)
                hist = list(df.at[idx, lh] or [])
                conflict = [h for h in hist if int(h.get("r", -1)) == int(r)]
                if conflict and any(
                    abs(float(h.get("y")) - y) > 1e-12 for h in conflict
                ):
                    raise OpalError(
                        f"Label history conflict for id {_id} at r={r}; refusing to change history."
                    )
                hist.append({"r": int(r), "y": y, "ts": now})
                df.at[idx, lh] = hist
                to_update_idx.append(idx)
            else:
                # require essential cols to add a new row
                essentials_missing = [
                    c
                    for c in ["sequence", "bio_type", "alphabet"]
                    if c not in rec or pd.isna(rec[c])
                ]
                if essentials_missing:
                    raise OpalError(
                        f"New id {_id} missing required fields {essentials_missing} to add row. "
                        f"Provide: sequence,bio_type,alphabet and {self.x_col}.",
                        ExitCodes.CONTRACT_VIOLATION,
                    )
                if self.x_col not in rec or rec[self.x_col] is None:
                    raise OpalError(
                        f"New id {_id} missing {self.x_col} (representation) for labeled import.",
                        ExitCodes.CONTRACT_VIOLATION,
                    )
                seq = str(rec["sequence"])
                biotype = str(rec["bio_type"])
                alphabet = str(rec["alphabet"])
                length = int(len(seq))
                new_id = _id or usr_compatible_id(biotype, seq)
                added_rows.append(
                    {
                        "id": new_id,
                        "bio_type": biotype,
                        "sequence": seq,
                        "alphabet": alphabet,
                        "length": length,
                        "source": f"opal_labels_import_r{r}",
                        "created_at": now,
                        self.y_col: y,
                        self.x_col: rec[self.x_col],
                        lh: [{"r": int(r), "y": y, "ts": now}],
                    }
                )

        if added_rows:
            df = pd.concat([df, pd.DataFrame(added_rows)], ignore_index=True)

        return df

    # ---------- Representation coercion ----------
    def transform_matrix(
        self, df: pd.DataFrame, ids: List[str]
    ) -> Tuple[np.ndarray, int]:
        sub = df.loc[df["id"].isin(ids), self.x_col]
        # maintain order
        sub = sub.reindex(df.index[df["id"].isin(ids)])
        transform = get_transform(self.transform_name, self.transform_params)
        X, d = transform.transform(sub)
        return X, d

    # ---------- Effective labels ----------
    def effective_labels_latest_only(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Returns DataFrame with id and effective y for rounds <= k,
        choosing the label with the highest r <= k for each id.
        """
        lh = self.label_hist_col()
        out = []
        for row in df[["id", lh]].itertuples(index=False):
            rid, hist = row
            if not hist:
                continue
            if not isinstance(hist, list):
                continue
            elig = [h for h in hist if int(h.get("r", -1)) <= int(k)]
            if not elig:
                continue
            hbest = max(elig, key=lambda h: int(h.get("r", -1)))
            out.append(
                {"id": rid, "y": float(hbest["y"]), "src_round": int(hbest["r"])}
            )
        return pd.DataFrame(out)

    def candidate_universe(
        self, df: pd.DataFrame, k: int, allow_repeats_until_labeled: bool = True
    ) -> pd.DataFrame:
        """
        Candidates are rows with x_col present and NOT labeled in rounds <= k.
        If previously selected but unlabeled, they remain eligible (since eligibility is defined by label history only).
        """
        if self.x_col not in df.columns:
            raise OpalError(
                f"Representation column not found: {self.x_col}",
                ExitCodes.CONTRACT_VIOLATION,
            )

        df = df.copy()
        lh = self.label_hist_col()
        df[lh] = df.get(lh, [[]])

        def has_labeled_le_k(hist) -> bool:
            if not isinstance(hist, list):
                return False
            return any(int(h.get("r", -1)) <= int(k) for h in hist)

        has_x = df[self.x_col].notna()
        labeled = df[lh].map(has_labeled_le_k)
        return df[has_x & (~labeled)]

    # ---------- Preflight helpers ----------
    def check_biotype_alphabet_uniformity(
        self, df: pd.DataFrame, ids: Iterable[str]
    ) -> None:
        subset = df.loc[df["id"].isin(list(ids)), ["bio_type", "alphabet", "id"]]
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

    # ---------- Misc ----------
    def records_sha256(self) -> str:
        return compute_records_sha256(self.records_path)
