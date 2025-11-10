"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/selection.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _normalize_mut_tag(wt: str, pos: int, alt: str) -> str:
    return f"{str(wt).upper()}{int(pos)}{str(alt).upper()}"


def select_elite_aa_events(
    df: pd.DataFrame, metric_col: str, cfg: Dict
) -> List[Tuple[int, str, str, float]]:
    """
    Select Top‑K single AA events with strict ruleouts.

    Returns a list of (pos:int, wt:str, alt:str, score:float) sorted by score desc.
    """
    required_cols = [
        "permuter__round",
        "permuter__aa_pos",
        "permuter__aa_wt",
        "permuter__aa_alt",
        metric_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"combine_aa: input dataset missing columns: {missing}. "
            f"Expected singles with {required_cols}"
        )

    singles = df[df["permuter__round"] == 1].copy()
    singles = singles[
        singles["permuter__aa_pos"].notna()
        & singles["permuter__aa_wt"].notna()
        & singles["permuter__aa_alt"].notna()
        & singles[metric_col].notna()
    ][["permuter__aa_pos", "permuter__aa_wt", "permuter__aa_alt", metric_col]]

    if singles.empty:
        raise RuntimeError("combine_aa: no round‑1 single AA events found")

    # Canonical types
    singles = singles.assign(
        _pos=singles["permuter__aa_pos"].astype(int),
        _wt=singles["permuter__aa_wt"].astype(str).str.upper(),
        _alt=singles["permuter__aa_alt"].astype(str).str.upper(),
        _score=singles[metric_col].astype(float),
    )[["_pos", "_wt", "_alt", "_score"]]

    # Average duplicates (same (pos, wt, alt))
    singles = singles.groupby(["_pos", "_wt", "_alt"], as_index=False)["_score"].mean()

    # Apply selection rules
    sel = (cfg or {}).get("select", {})
    top_global = int(sel.get("top_global", 100))
    min_delta = sel.get("min_delta", None)
    allowed_positions = set(int(x) for x in sel.get("allowed_positions", []) or [])
    exclude_positions = set(int(x) for x in sel.get("exclude_positions", []) or [])
    exclude_mutations = set(
        str(x).strip().upper() for x in (sel.get("exclude_mutations", []) or [])
    )

    singles = singles.sort_values("_score", ascending=False, kind="mergesort")
    if top_global > 0:
        singles = singles.head(top_global)

    if min_delta is not None:
        singles = singles[singles["_score"] >= float(min_delta)]

    if allowed_positions:
        singles = singles[singles["_pos"].isin(allowed_positions)]

    if exclude_positions:
        singles = singles[~singles["_pos"].isin(exclude_positions)]

    if exclude_mutations:
        singles = singles[
            ~singles.apply(
                lambda r: _normalize_mut_tag(r["_wt"], r["_pos"], r["_alt"])
                in exclude_mutations,
                axis=1,
            )
        ]

    if singles.empty:
        raise RuntimeError("combine_aa: selection produced an empty elite set")

    out: List[Tuple[int, str, str, float]] = [
        (int(r["_pos"]), str(r["_wt"]), str(r["_alt"]), float(r["_score"]))
        for _, r in singles.iterrows()
    ]
    return out
