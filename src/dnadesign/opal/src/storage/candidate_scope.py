"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/candidate_scope.py

Candidate-scope helpers for OPAL runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.types import CandidateScope
from ..core.utils import OpalError


def load_candidate_scope_ids(scope: CandidateScope) -> list[str]:
    if str(scope.kind) != "id_list":
        raise OpalError(f"Unsupported candidate_scope.kind: {scope.kind!r}")
    path = Path(scope.path)
    if not path.exists():
        raise OpalError(f"candidate_scope id-list file not found: {path}")
    id_column = str(scope.id_column or "id")
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path, columns=[id_column])
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, usecols=[id_column])
        else:
            raise OpalError(f"candidate_scope path must be .parquet, .pq, or .csv: {path}")
    except OpalError:
        raise
    except Exception as exc:
        raise OpalError(f"Failed to read candidate_scope id-list file {path}: {exc}") from exc
    if id_column not in frame.columns:
        raise OpalError(f"candidate_scope id-list file missing column {id_column!r}: {path}")
    if frame[id_column].isna().any():
        raise OpalError(f"candidate_scope id-list file contains null ids: {path}")
    ids = [str(value).strip() for value in frame[id_column].tolist()]
    if any(not value for value in ids):
        raise OpalError(f"candidate_scope id-list file contains blank ids: {path}")
    duplicates = pd.Series(ids, dtype="string").loc[lambda series: series.duplicated()].drop_duplicates().tolist()
    if duplicates:
        raise OpalError(f"candidate_scope id-list file contains duplicate ids (sample={duplicates[:10]}): {path}")
    if not ids:
        raise OpalError(f"candidate_scope id-list file contains no ids: {path}")
    return ids


def apply_candidate_scope(frame: pd.DataFrame, scope: CandidateScope | None) -> pd.DataFrame:
    if scope is None:
        return frame
    if "id" not in frame.columns:
        raise OpalError("candidate_scope requires candidate frame column 'id'.")
    ids = load_candidate_scope_ids(scope)
    wanted = set(ids)
    out = frame.loc[frame["id"].astype(str).isin(wanted)].copy()
    found = set(out["id"].astype(str).tolist())
    missing = sorted(wanted - found)
    if missing:
        raise OpalError(f"candidate_scope references ids missing from records.parquet (sample={missing[:10]}).")
    if out.empty:
        raise OpalError("candidate_scope produced an empty candidate universe.")
    order = {candidate_id: index for index, candidate_id in enumerate(ids)}
    out["__candidate_scope_order__"] = out["id"].astype(str).map(order)
    out = out.sort_values("__candidate_scope_order__").drop(columns=["__candidate_scope_order__"])
    return out.reset_index(drop=True)
