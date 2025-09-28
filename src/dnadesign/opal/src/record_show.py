"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/record_show.py

Per-record reporting utilities used by the CLI command `record-show`.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import OpalError


def _label_hist_col(slug: str) -> str:
    """Single source of truth for the label-history column name."""
    return f"opal__{slug}__label_hist"


def _is_null_like(v: Any) -> bool:
    # Avoid importing numpy: handle common nulls and pandas NA
    if v is None:
        return True
    try:
        # pd.isna is robust across numpy/pandas scalars and python None
        return bool(pd.isna(v))
    except Exception:
        return False


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    # JSON-encoded string?
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return obj
                return [obj]
            except Exception:
                return []
    # dict-like single entry
    if isinstance(v, dict):
        return [v]
    return []


def _normalize_hist_cell(cell: Any) -> List[Dict[str, Any]]:
    """
    Normalize the "label history" cell into a list[dict].

    The store writes append-only snapshots per round with keys typically like:
      - labels: {"r": int, "y": list[float]}
      - preds : {"r": int, "yhat": list[float], "unc_mean_std": float, "score": float,
                 "rank": int, "selected": bool}

    We are defensive here because history can evolve over time.
    """
    if _is_null_like(cell):
        return []
    items = _as_list(cell)
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            # tolerate stringly-typed entries that failed json.loads above
            continue
        # Make a shallow copy and ensure 'r' is int when present
        e = dict(it)
        if "r" in e:
            try:
                e["r"] = int(e["r"])
            except Exception:
                # drop bad 'r' so it sorts last
                e.pop("r", None)
        out.append(e)
    # sort by round if present
    out.sort(key=lambda d: (d.get("r", 9_999_999),))
    return out


def _select_row_by_id_or_sequence(
    df: pd.DataFrame, id_: Optional[str], sequence: Optional[str]
) -> Tuple[int, Dict[str, Any]]:
    if id_:
        mask = df["id"].astype(str) == str(id_)
        if not mask.any():
            raise OpalError(f"id not found: {id_}")
        idx = int(df.index[mask][0])
        return idx, {"id": str(df.at[idx, "id"])}
    if sequence:
        if "sequence" not in df.columns:
            raise OpalError("records table has no 'sequence' column.")
        # exact match (string)
        mask = df["sequence"].astype(str) == str(sequence)
        if not mask.any():
            # try case-insensitive match as a helpful fallback
            seqs = df["sequence"].astype(str)
            mask = seqs.str.lower() == str(sequence).lower()
        if not mask.any():
            raise OpalError(f"sequence not found: {sequence}")
        # if multiple match (dup sequences), pick the first deterministically and signal
        matches = df.index[mask].tolist()
        idx = int(matches[0])
        info = {"id": str(df.at[idx, "id"]), "sequence_match_count": len(matches)}
        return idx, info
    raise OpalError("Provide either id or sequence.")


def build_record_report(
    df: pd.DataFrame,
    campaign_slug: str,
    id_: Optional[str] = None,
    sequence: Optional[str] = None,
    with_sequence: bool = False,
) -> Dict[str, Any]:
    """
    Create a compact per-record report with ground truth & history, and per-round
    prediction/selection snapshots (when present).

    Parameters
    ----------
    df : DataFrame
        The records table (must include 'id'; 'sequence' is optional if selecting by id).
    campaign_slug : str
        Campaign namespace slug used to resolve the label-history column name.
    id_ : Optional[str]
        Record id selector.
    sequence : Optional[str]
        Sequence selector (used when `id_` is None).
    with_sequence : bool
        If True, include the actual sequence string in the report.

    Returns
    -------
    Dict[str, Any]
        A plain dict safe for JSON or simple text rendering.
    """
    if "id" not in df.columns:
        raise OpalError("records table missing required column: 'id'")

    idx, sel_info = _select_row_by_id_or_sequence(df, id_, sequence)
    rec_id = str(df.at[idx, "id"])
    seq_val = (
        str(df.at[idx, "sequence"])
        if ("sequence" in df.columns and with_sequence)
        else None
    )

    # Label-history parsing
    lh_col = _label_hist_col(campaign_slug)
    hist_entries: List[Dict[str, Any]] = (
        _normalize_hist_cell(df.at[idx, lh_col]) if lh_col in df.columns else []
    )

    # Partition by round; keep the latest entry per round for compactness
    per_round: Dict[int, Dict[str, Any]] = {}
    rounds_labeled = 0
    rounds_pred = 0
    for e in hist_entries:
        r = int(e.get("r", -1))
        if r < 0:
            # unscoped entry; keep it in a special bucket
            per_round.setdefault(-1, {}).update(e)
            continue
        # we prefer later entries in case the same round has multiple snapshots
        per_round[r] = {**per_round.get(r, {}), **e}
        if "y" in e:
            rounds_labeled += 1
        if "yhat" in e or "score" in e:
            rounds_pred += 1

    latest_round = max([rk for rk in per_round.keys() if rk >= 0], default=None)

    # Present a compact, human-friendly view
    rounds_card = []
    for rk in sorted([r for r in per_round.keys() if r >= 0]):
        entry = per_round[rk]
        rounds_card.append(
            {
                "r": rk,
                "label_present": "y" in entry,
                "pred_present": ("yhat" in entry) or ("score" in entry),
                "score": entry.get("score"),
                "rank": entry.get("rank"),
                "selected": entry.get("selected"),
                "unc_mean_std": entry.get("unc_mean_std"),
            }
        )

    report: Dict[str, Any] = {
        "id": rec_id,
        "campaign": campaign_slug,
        "label_history_column": lh_col if lh_col in df.columns else None,
        "history_entries_total": len(hist_entries),
        "latest_round_seen": latest_round,
        "rounds_with_labels": rounds_labeled,
        "rounds_with_predictions": rounds_pred,
        "rounds": rounds_card,
    }
    report.update(sel_info)
    if seq_val is not None:
        report["sequence"] = seq_val

    return report
