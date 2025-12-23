"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/mutation_summary.py

Given a dataset (DataFrame) and a metric id, compute the per-mutation
effects (Δ vs reference if available; for LLR this equals the value),
extract the Top +K and Bottom -K events, and write a tidy CSV next to
the dataset's plots directory (i.e., in the dataset directory itself).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_LOG = logging.getLogger("permuter.summary.aa_mutation")

_RE_AA_EDIT_STRUCT = re.compile(
    r"\baa\b.*?\bpos\s*=\s*(?P<pos>\d+)\b.*?\bwt\s*=\s*(?P<wt>[A-Z\*])\b.*?\balt\s*=\s*(?P<alt>[A-Z\*])\b",
    flags=re.IGNORECASE,
)


def _parse_any_aa_edit(token: str) -> Optional[tuple[int, str, str]]:
    s = str(token).strip()
    m = _RE_AA_EDIT_STRUCT.search(s)
    if m:
        return int(m.group("pos")), m.group("wt").upper(), m.group("alt").upper()
    # Compact AA notation — only if not nucleotide→nucleotide
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha() and s[1:-1].isdigit():
        frm, to = s[0].upper(), s[-1].upper()
        if frm in "ACGT" and to in "ACGT":
            return None
        return int(s[1:-1]), frm, to
    return None


def _has_aa_signals(df: pd.DataFrame) -> bool:
    if {"permuter__aa_pos", "permuter__aa_alt"}.issubset(df.columns):
        return True
    mods = df.get("permuter__modifications")
    if isinstance(mods, pd.Series):
        for m in mods.dropna():
            seq = list(m) if isinstance(m, (list, tuple)) else []
            for t in seq:
                if _parse_any_aa_edit(str(t)):
                    return True
    return False


def _series_for_metric(df: pd.DataFrame, metric_id: Optional[str]) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError("metric_id is required (expects a column permuter__metric__<id>)")
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        # fallback: exact suffix match
        cand = [c for c in df.columns if c.startswith("permuter__metric__") and c.split("__", 2)[-1] == str(metric_id)]
        if len(cand) == 1:
            col = cand[0]
        else:
            raise RuntimeError(f"Metric column not found: {col}")
    return df[col].astype("float64"), str(metric_id)


def _looks_like_llr(mid: str) -> bool:
    s = (mid or "").lower()
    return ("llr" in s) or ("ratio" in s)


def _ref_value_round1_seed(df: pd.DataFrame, ycol: str) -> Optional[float]:
    # round-1, modifications == [] → seed. If missing, None (for LLR we'll treat Δ=y).
    try:
        for r in df.to_dict("records"):
            if (
                int(r.get("permuter__round", 0)) == 1
                and isinstance(r.get("permuter__modifications"), list)
                and len(r["permuter__modifications"]) == 0
            ):
                v = r.get(ycol)
                return float(v) if pd.notna(v) else None
    except Exception:
        pass
    return None


def _extract_aa_events(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """Return tidy rows (wt, pos, to_res, value). Duplicates averaged later."""
    recs: List[dict] = []
    for _, row in df.iterrows():
        val = float(row[ycol])
        # Prefer structured
        if (
            "permuter__aa_pos" in row
            and pd.notna(row["permuter__aa_pos"])
            and "permuter__aa_wt" in row
            and pd.notna(row["permuter__aa_wt"])
            and "permuter__aa_alt" in row
            and pd.notna(row["permuter__aa_alt"])
        ):
            try:
                recs.append(
                    {
                        "wt": str(row["permuter__aa_wt"]).upper(),
                        "pos": int(row["permuter__aa_pos"]),
                        "to_res": str(row["permuter__aa_alt"]).upper(),
                        "value": val,
                    }
                )
            except Exception:
                pass
        # Parse tokens as a fallback
        mods = row.get("permuter__modifications", [])
        if isinstance(mods, (list, tuple)):
            for tok in mods:
                p = _parse_any_aa_edit(str(tok))
                if p:
                    pos, wt, to = p
                    recs.append({"wt": wt, "pos": int(pos), "to_res": to, "value": val})
    if not recs:
        raise RuntimeError("No amino-acid edits recognized for summary.")
    return pd.DataFrame(recs).drop_duplicates()


def compute_aa_llr_summary(all_df: pd.DataFrame, metric_id: str, *, top_k: int = 20) -> pd.DataFrame:
    """
    Core computation (no I/O). Returns tidy DF with:
      ['direction','rank','canon','wt','pos','to_res','delta','metric_id','job','ref']
    'direction' ∈ {'top','bottom'}, rank 1..K in each direction.
    """
    if "sequence" not in all_df.columns or "permuter__round" not in all_df.columns:
        raise RuntimeError("Dataset missing required columns: sequence / permuter__round")

    if not _has_aa_signals(all_df):
        # Caller decides whether to skip silently; we throw to be explicit here.
        raise RuntimeError("No amino-acid signals present (permuter__aa_* or AA tokens).")

    y, _ = _series_for_metric(all_df, metric_id)
    df = all_df.assign(_y=y).dropna(subset=["_y"]).copy()
    ycol = "_y"

    # Reference (seed) when present; for LLR this is usually 0.
    ref_value = _ref_value_round1_seed(df, ycol)
    events = _extract_aa_events(df, ycol)
    # Δ := value - ref_value  (LLR: ref_value≈0 → Δ≈value)
    delta = events["value"] - (float(ref_value) if ref_value is not None else 0.0)
    events = events.assign(delta=delta).drop(columns=["value"])

    # Average duplicates (same AA event in multiple rows)
    ev_mean = events.groupby(["wt", "pos", "to_res"], as_index=False)["delta"].mean()
    ev_mean["canon"] = ev_mean.apply(lambda r: f"{r['wt']}{int(r['pos'])}{r['to_res']}", axis=1)

    # Rank
    ev_sorted = ev_mean.sort_values("delta", ascending=False, kind="mergesort").reset_index(drop=True)
    k = int(min(top_k, len(ev_sorted)))
    top = ev_sorted.head(k).copy()
    bot = ev_sorted.sort_values("delta", ascending=True, kind="mergesort").head(k).copy()
    top.insert(0, "direction", "top")
    bot.insert(0, "direction", "bottom")
    top.insert(1, "rank", range(1, len(top) + 1))
    bot.insert(1, "rank", range(1, len(bot) + 1))

    job = str(df.get("permuter__job", pd.Series(["job"])).iloc[0])
    ref = str(df.get("permuter__ref", pd.Series(["ref"])).iloc[0])
    for frame in (top, bot):
        frame["metric_id"] = str(metric_id)
        frame["job"] = job
        frame["ref"] = ref

    out = pd.concat([top, bot], ignore_index=True)
    # Tidy column order
    return out[
        [
            "direction",
            "rank",
            "canon",
            "wt",
            "pos",
            "to_res",
            "delta",
            "metric_id",
            "job",
            "ref",
        ]
    ]


def emit_aa_mutation_llr_summary(
    all_df: pd.DataFrame,
    *,
    dataset_dir: Path,
    metric_id: str,
    top_k: int = 20,
    strict_llr_only: bool = True,
) -> Optional[Path]:
    """
    Entry point for the pipeline: compute + write CSV. Returns path or None.
    Skips (returns None) when not applicable (no AA signals or non-LLR metric).
    """
    # quick gate: only for LLR by default
    if strict_llr_only and not _looks_like_llr(metric_id):
        _LOG.info(
            "AA mutation summary skipped: metric_id=%r does not look like LLR.",
            metric_id,
        )
        return None
    try:
        tidy = compute_aa_llr_summary(all_df, metric_id=metric_id, top_k=top_k)
    except RuntimeError as e:
        _LOG.info("AA mutation summary skipped: %s", e)
        return None
    # Ensure dataset_dir exists
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = dataset_dir / f"AA_MUTATION_SUMMARY__{metric_id}.csv"
    tidy.to_csv(out, index=False)
    _LOG.info("Wrote AA mutation summary → %s", out)
    return out
