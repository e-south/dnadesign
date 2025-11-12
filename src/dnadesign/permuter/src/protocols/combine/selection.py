"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/selection.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import math

import pandas as pd
import logging
from pathlib import Path

_LOG = logging.getLogger("permuter.protocol.combine_aa")

def _normalize_mut_tag(wt: str, pos: int, alt: str) -> str:
    return f"{str(wt).upper()}{int(pos)}{str(alt).upper()}"


def select_elite_aa_events(
    df: pd.DataFrame, metric_col: str, cfg: Dict
) -> List[Tuple[int, str, str, float]]:
    """
    Select single-AA events for combination with strict ruleouts — assertively.
    Modes (cfg.select.mode):
      • "global" (default): top_global by score across all events.
      • "per_position_best": pick the best alternative *per position*, then keep the
        best `top_global` positions by score.

    Guards:
      • By default, disallow negative winners (cfg.select.disallow_negative_best=True).
        If triggered, we hard-stop with guidance (no silent fallbacks).

    Returns: list of (pos:int, wt:str, alt:str, score:float) sorted by score↓.
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
    mode = str(sel.get("mode", "global")).strip().lower()
    if mode not in {"global", "per_position_best"}:
        raise ValueError(
            "combine_aa.select.mode must be 'global' or 'per_position_best'"
        )
    top_global = int(sel.get("top_global", 100))
    min_delta = sel.get("min_delta", None)
    allowed_positions = set(int(x) for x in sel.get("allowed_positions", []) or [])
    exclude_positions = set(int(x) for x in sel.get("exclude_positions", []) or [])
    exclude_mutations = set(
        str(x).strip().upper() for x in (sel.get("exclude_mutations", []) or [])
    )
    disallow_negative = bool(sel.get("disallow_negative_best", True))
    emit_table = bool(sel.get("emit_per_position_table", True))

    singles = singles.sort_values("_score", ascending=False, kind="mergesort")
    if top_global > 0:
        singles = singles.head(top_global)
        
    singles = singles.sort_values("_score", ascending=False, kind="mergesort")

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

    # ----- Mode logic -----
    if mode == "global":
        chosen = singles if top_global <= 0 else singles.head(top_global)
        msg_mode = "global(top_global)"
    else:
        # pick the best alternative per position, then limit by top_global
        idx = singles.groupby("_pos")["_score"].idxmax()
        winners = singles.loc[idx].sort_values("_score", ascending=False, kind="mergesort")
        chosen = winners if top_global <= 0 else winners.head(top_global)
        msg_mode = "per_position_best→top_global"

    n_positions = int(singles["_pos"].nunique())
    n_chosen = int(len(chosen))
    n_neg = int((chosen["_score"] < 0).sum())
    _LOG.info(
        "[select] mode=%s • positions_scanned=%d • selected=%d%s",
        msg_mode,
        n_positions,
        n_chosen,
        (f" • negatives={n_neg}" if n_neg else ""),
    )

    if n_neg and disallow_negative:
        # Show the first few offending rows to guide the user
        bad = chosen[chosen["_score"] < 0].head(10)
        examples = "  ".join(
            f"{r['_wt']}{int(r['_pos'])}{r['_alt']}:{r['_score']:+.3f}"
            for _, r in bad.iterrows()
        )
        raise ValueError(
            "combine_aa: negative singles among selected Top-K. "
            f"Examples: {examples}\n"
            "Guidance: decrease select.top_global, set select.min_delta≥0, "
            "or explicitly set select.disallow_negative_best=false."
        )

    if emit_table:
        # Pretty, compact “selected” table: metric-aware, 10 rows per column.
        # Entry form: R31G  +0.017   (no redundant 'pos=' text)
        metric_id = str(metric_col).split("permuter__metric__", 1)[-1].lstrip("_")
        lines = [f"{r['_wt']}{int(r['_pos'])}{r['_alt']}  {r['_score']:+.3f}"
                 for _, r in chosen.iterrows()]
        # 10 rows per column
        rows_per_col = 10
        n = len(lines)
        ncol = max(1, math.ceil(n / rows_per_col))
        cols = [lines[i*rows_per_col:(i+1)*rows_per_col] for i in range(ncol)]
        # pad columns to equal height
        for i in range(ncol):
            while len(cols[i]) < rows_per_col:
                cols[i].append("")
        # column widths for clean padding
        widths = [max(len(s) for s in col) for col in cols]
        rendered = []
        for r in range(rows_per_col):
            row_cells = []
            for c in range(ncol):
                cell = cols[c][r]
                pad = widths[c] if c < ncol - 1 else len(cell)
                row_cells.append(cell.ljust(pad + (2 if c < ncol - 1 else 0)))
            line = "  " + "".join(row_cells).rstrip()
            if line.strip():
                rendered.append(line)
        header = f"[select] selected (metric={metric_id}):"
        body = ("\n".join(rendered)) if rendered else "  —"
        _LOG.info("%s\n%s", header, body)

    # Persist the chosen singles as a CSV artifact when an artifact_dir is provided.
    try:
        art_dir = (Path(str((cfg or {}).get("_artifact_dir"))).expanduser().resolve()
                   if isinstance(cfg, dict) and "_artifact_dir" in cfg else None)
        if art_dir:
            art_dir.mkdir(parents=True, exist_ok=True)
            metric_id = str(metric_col).split("permuter__metric__", 1)[-1].lstrip("_")
            chosen_out = chosen.copy()
            chosen_out = chosen_out.rename(columns={"_pos": "pos", "_wt": "wt", "_alt": "alt", "_score": "score"})
            chosen_out["canon"] = chosen_out.apply(lambda r: f"{r['wt']}{int(r['pos'])}{r['alt']}", axis=1)
            out_csv = art_dir / f"COMBINE_AA__ELITE_SELECTION__{metric_id}.csv"
            chosen_out[["canon", "wt", "pos", "alt", "score"]].to_csv(out_csv, index=False)
            _LOG.info("[select] wrote selection table → %s", out_csv)
    except Exception as _e:
        _LOG.debug("selection artifact write skipped: %s", _e)

    out: List[Tuple[int, str, str, float]] = [
        (int(r["_pos"]), str(r["_wt"]), str(r["_alt"]), float(r["_score"]))
        for _, r in chosen.iterrows()
    ]
    return out
