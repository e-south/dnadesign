"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/preflight.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.utils import OpalError
from ..storage.data_access import ESSENTIAL_COLS, RecordsStore


@dataclass
class PreflightReport:
    x_dim: Optional[int] = None
    manual_attach_ids: List[str] = None
    manual_attach_count: int = 0
    backfill: Dict[str, int] | None = None
    warnings: List[str] | None = None


def _vector_len(v) -> Optional[int]:
    try:
        arr = np.asarray(v, dtype=float).ravel()
        return int(arr.size)
    except Exception:
        return None


def preflight_run(
    store: RecordsStore,
    df: pd.DataFrame,
    as_of_round: int,
    fail_on_mixed_biotype_or_alphabet: bool,
    auto_backfill: bool = True,
) -> PreflightReport:
    rep = PreflightReport(
        manual_attach_ids=[],
        manual_attach_count=0,
        backfill={"checked": 0, "backfilled": 0},
        warnings=[],
    )
    # essentials present
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise OpalError(f"records.parquet missing essentials: {missing}")

    if store.x_col not in df.columns:
        raise OpalError(f"Missing X column '{store.x_col}' in records.parquet")

    # X dimension probe (via first non-null row)
    probe = df[store.x_col].dropna().head(1)
    rep.x_dim = _vector_len(probe.iloc[0]) if not probe.empty else None

    # Manual attach detection:
    # Current y_col non-null AND not present in label_hist at ANY round → record as manual_attach at current as_of_round
    lh = store.label_hist_col()
    if lh not in df.columns:
        df[lh] = None
    ycol = store.y_col
    if ycol in df.columns:
        to_attach: List[Tuple[str, List[float]]] = []
        for _, row in df.iterrows():
            _id = str(row["id"])
            y = row[ycol]
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue
            hist = store._normalize_hist_cell(row.get(lh))
            has_label = any(e.get("kind") == "label" for e in hist)
            if not has_label:
                try:
                    vec = [float(x) for x in np.asarray(y, dtype=float).ravel().tolist()]
                except Exception:
                    continue
                to_attach.append((_id, vec))

        if to_attach and auto_backfill:
            rep.backfill["checked"] = len(to_attach)
            # append to history (src="manual_attach")
            ids = [t[0] for t in to_attach]
            vecs = [t[1] for t in to_attach]
            df2 = store.append_labels_from_df(
                df,
                labels=pd.DataFrame({"id": ids, "y": vecs}),
                r=as_of_round,
                src="manual_attach",
                fail_if_any_existing_labels=False,
            )
            store.save_atomic(df2)
            rep.backfill["backfilled"] = len(to_attach)
            rep.manual_attach_ids = ids
            rep.manual_attach_count = len(ids)
        elif to_attach and not auto_backfill:
            rep.manual_attach_ids = [t[0] for t in to_attach]
            rep.manual_attach_count = len(to_attach)
            rep.warnings.append("manual_labels_present_without_label_hist")

    # Optional consistency check
    if fail_on_mixed_biotype_or_alphabet:
        # mixed values indicate likely dataset mis-merge
        for col in ("bio_type", "alphabet"):
            if col in df.columns:
                vals = sorted(set(str(v) for v in df[col].dropna().unique().tolist()))
                if len(vals) > 1:
                    raise OpalError(f"Preflight error: mixed {col} detected: {vals}")

    return rep
