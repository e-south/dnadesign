"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/opal/join.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


def _ledger_paths(campaign_dir: Path) -> dict:
    # We expect OPAL 'outputs' folder with ledger files
    base = Path(campaign_dir) / "outputs"
    return {
        "index": base / "ledger.index.parquet",
        "predictions": base / "ledger.predictions",
        "labels": base / "ledger.labels",
        "runs": base / "ledger.runs",
    }


OPAL_CAMPAIGNS_ENV = "DNADESIGN_OPAL_CAMPAIGNS_ROOT"


def _is_campaign_dir(d: Path) -> bool:
    """
    A campaign dir is the parent of 'outputs' and should contain known ledgers under outputs/.
    """
    out = d / "outputs"
    return out.exists() and (
        (out / "ledger.predictions").exists()
        or (out / "ledger.index.parquet").exists()
        or (out / "ledger.runs").exists()
    )


def resolve_campaign_dir(spec: str | Path) -> Path:
    """
    Resolve an OPAL campaign directory from a user-provided spec.

    Accepted forms:
      - Absolute or relative path to the campaign directory
      - Path to '.../outputs' or one of its ledger subdirs (normalized to the campaign dir)
      - Plain campaign name:
          1) $DNADESIGN_OPAL_CAMPAIGNS_ROOT/<name>
          2) <repo>/dnadesign/opal/campaigns/<name>

    Returns:
      Path to the canonical campaign directory (the parent of 'outputs').
    Raises:
      FileNotFoundError with a clear, enumerated list of attempted paths.
    """
    tried: list[Path] = []

    p = Path(spec).expanduser()
    # 1) Treat as path; normalize if pointing at outputs/ or a ledger subdir
    direct_candidates: list[Path] = [p]
    if p.exists() and p.is_dir():
        # If user points at .../outputs, normalize up one; if .../outputs/ledger.*, up two
        if p.name == "outputs":
            direct_candidates.append(p.parent)
        elif p.parent.name == "outputs":
            direct_candidates.append(p.parents[1])
    else:
        # Also consider parent and parents[1] in case the spec points inside outputs that doesn't exist from CWD
        direct_candidates += [p.parent, p.parents[1] if len(p.parents) >= 2 else p]

    for cand in direct_candidates:
        tried.append(cand)
        if cand.exists() and _is_campaign_dir(cand):
            return cand

    # 2) Environment override (campaign name only)
    name = Path(spec).name
    env_root = os.environ.get(OPAL_CAMPAIGNS_ENV)
    if env_root:
        cand = Path(env_root).expanduser() / name
        tried.append(cand)
        if cand.exists() and _is_campaign_dir(cand):
            return cand

    # 3) Installed repo campaigns dir
    repo_root = Path(__file__).resolve().parents[3]  # .../dnadesign
    cand = repo_root / "opal" / "campaigns" / name
    tried.append(cand)
    if cand.exists() and _is_campaign_dir(cand):
        return cand

    # Nothing matched — assemble a precise, helpful error
    lines = [
        f"OPAL campaign directory not found for spec '{spec}'.",
        "Paths tried:",
        *[f"  - {t}" for t in tried],
        (
            "Hint: pass a campaign *name* (e.g. 'my_campaign'), an absolute path to the "
            "campaign directory, or set DNADESIGN_OPAL_CAMPAIGNS_ROOT to your campaigns root."
        ),
    ]
    raise FileNotFoundError("\n".join(lines))


def _resolve_run_slice(
    campaign_dir: Path, run_selector: str | None, as_of_round: Optional[int]
) -> tuple[Optional[str], Optional[int]]:
    """
    Normalize the requested OPAL slice.
    Returns (run_id or None, as_of_round or None).
    Accepted selectors:
      - 'latest'
      - 'round:<n>'
      - 'run_id:<rid>'
    If both run_selector and as_of_round encode a round, raise a ValueError.
    """
    sel = (run_selector or "latest").strip().lower()
    paths = _ledger_paths(campaign_dir)

    # Prefer a concise 'runs' ledger if available, otherwise fallback to predictions
    def _runs_df():
        p = paths["runs"]
        return (
            _read_parquet_parts(p, columns=["run_id", "as_of_round"])
            if p.exists()
            else None
        )

    runs = _runs_df()

    if sel.startswith("round:") and as_of_round is not None:
        raise ValueError(
            "Provide either run_selector='round:<n>' OR as_of_round, not both."
        )

    if sel.startswith("run_id:"):
        rid = sel.split(":", 1)[1]
        ao = as_of_round
        if ao is None:
            base = (
                runs
                if runs is not None
                else _read_parquet_parts(
                    paths["predictions"], columns=["run_id", "as_of_round"]
                )
            )
            m = base[base["run_id"].astype(str) == str(rid)]
            ao = (
                int(m["as_of_round"].dropna().astype(int).max())
                if not m.empty
                else None
            )
        return str(rid), (int(ao) if ao is not None else None)

    if sel.startswith("round:"):
        return None, int(sel.split(":", 1)[1])

    if as_of_round is not None:
        return None, int(as_of_round)

    # latest
    base = (
        runs
        if runs is not None
        else _read_parquet_parts(
            paths["predictions"], columns=["run_id", "as_of_round"]
        )
    )
    if base.empty:
        raise FileNotFoundError("No runs/predictions found to resolve 'latest'.")
    ao = int(base["as_of_round"].dropna().astype(int).max())
    cand = base[base["as_of_round"].astype(int) == ao]
    rid = (
        str(cand["run_id"].dropna().astype(str).max())
        if "run_id" in cand.columns
        else None
    )
    return rid, ao


def _read_parquet_parts(
    d: Path, columns: list[str] | None = None, newest_only: bool = False
) -> pd.DataFrame:
    """Read a directory of parquet parts (or a single parquet). By default, concatenate all parts."""
    if not d.exists():
        raise FileNotFoundError(f"Path not found: {d}")
    if d.is_dir():
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {d}")
        if newest_only:
            files = [max(files, key=lambda p: p.stat().st_mtime)]
        dfs = [
            pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
            for p in files
        ]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(d, columns=columns) if columns else pd.read_parquet(d)


def _list_parquet_files(d: Path) -> list[Path]:
    """Return the concrete parquet file(s) discovered at a ledger path."""
    if not d.exists():
        return []
    if d.is_dir():
        return sorted(d.glob("*.parquet"))
    return [d]


def select_ids(
    campaign_dir: Path,
    run_selector: str = "latest",
    scope: str = "scored_pool",
    ids_path: Path | None = None,
) -> set[str]:
    paths = _ledger_paths(campaign_dir)
    if scope == "custom":
        if not ids_path:
            raise ValueError("--ids is required for scope=custom.")
        df = (
            pd.read_parquet(ids_path)
            if str(ids_path).endswith(".parquet")
            else pd.read_csv(ids_path)
        )
        col = "id" if "id" in df.columns else df.columns[0]
        return set(map(str, df[col].tolist()))
    # For now, fallback to predictions ledger for scored_pool
    preds = (
        _read_parquet_parts(paths["predictions"], columns=["run_id", "id"])
        if Path(paths["predictions"]).exists()
        else None
    )

    if preds is None:
        raise FileNotFoundError("OPAL predictions ledger not found.")
    if run_selector == "latest" and "run_id" in preds.columns:
        rid = preds["run_id"].iloc[-1]
        sub = preds[preds["run_id"] == rid]
    elif run_selector.startswith("round:") and "run_id" in preds.columns:
        # naive mapping: pick last run_id (approximation)
        sub = preds
    else:
        sub = preds
    if scope == "scored_pool":
        return set(map(str, sub["id"].unique().tolist()))
    if scope == "selected_top_k":
        if "selected" not in sub.columns:
            raise KeyError("'selected' column not found in predictions ledger.")
        return set(
            map(str, sub.loc[sub["selected"].astype(bool), "id"].unique().tolist())
        )
    raise ValueError(f"Unsupported ids scope: {scope}")


def _ensure_left_has_id_column_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure merges by 'id' are unambiguous:
      • There must be an 'id' column.
      • There must not be an index level also named 'id'.
    Returns a new DataFrame if normalization is needed, otherwise the original.
    """
    left = df
    # Case A: index named 'id' AND 'id' column exists → drop index name (keep column)
    if left.index.name == "id" and "id" in left.columns:
        left = left.reset_index(drop=True)
    # Case B: index named 'id' but no 'id' column → materialize the column
    elif left.index.name == "id" and "id" not in left.columns:
        left = left.reset_index()
    # Case C: neither index named 'id' nor 'id' column present → this is an API misuse
    elif "id" not in left.columns:
        raise KeyError(
            "Left table must contain an 'id' column for OPAL joins. "
            "Ensure your dataset provides 'id'."
        )
    return left


def join_fields(
    df: pd.DataFrame,
    campaign_dir: Path,
    run_selector: str,
    fields: list[str],
    as_of_round: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    paths = _ledger_paths(campaign_dir)

    if not Path(paths["predictions"]).exists():
        raise FileNotFoundError("OPAL predictions ledger not found.")
    # Discover concrete parquet parts and log them up front (assertive observability)
    parts = _list_parquet_files(paths["predictions"])
    if log_fn:
        names = ", ".join(p.name for p in parts[:12]) + (
            " ..." if len(parts) > 12 else ""
        )
        log_fn(
            f"OPAL: predictions source={paths['predictions']} "
            f"({len(parts)} part file(s)): {names}"
        )
    preds = _read_parquet_parts(paths["predictions"])
    preds["id"] = preds["id"].astype(str)

    # Resolve the slice deterministically
    rid, ao = _resolve_run_slice(campaign_dir, run_selector, as_of_round)
    if ao is not None and "as_of_round" in preds.columns:
        preds = preds[preds["as_of_round"].astype(int) == int(ao)]
        if preds.empty:
            raise FileNotFoundError(f"No predictions for as_of_round={ao}.")
    if rid is not None and "run_id" in preds.columns:
        preds = preds[preds["run_id"].astype(str) == str(rid)]
        if preds.empty:
            raise FileNotFoundError(f"No predictions for run_id={rid}.")
    if log_fn:
        log_fn(
            "OPAL: resolved slice -> "
            f"run_selector='{run_selector}', run_id={rid}, as_of_round={ao}; "
            f"loaded_rows={len(preds)}; join_fields={fields}"
        )
    # Assert every requested field exists in this slice
    missing = [f for f in fields if f not in preds.columns]
    if missing:
        avail = sorted(
            [c for c in preds.columns if c.startswith(("obj__", "pred__", "sel__"))]
        )
        raise KeyError(
            f"Fields not present in OPAL predictions: {missing}. "
            f"Available in this slice: {avail[:50]}{' ...' if len(avail) > 50 else ''}"
        )
    cols = ["id"] + list(fields)
    join_df = preds[cols].drop_duplicates("id", keep="last")
    # Normalize left side for an unambiguous merge and enforce string ids
    left = _ensure_left_has_id_column_only(df).copy()
    left["id"] = left["id"].astype(str)
    join_df["id"] = join_df["id"].astype(str)
    return left.merge(join_df, on="id", how="left")


def list_available_fields(
    campaign_dir: Path,
    run_selector: str = "latest",
    as_of_round: Optional[int] = None,
) -> list[str]:
    """Helper for discoverability and actionable error messages."""
    paths = _ledger_paths(campaign_dir)
    if not Path(paths["predictions"]).exists():
        return []
    preds = _read_parquet_parts(paths["predictions"])
    rid, ao = _resolve_run_slice(campaign_dir, run_selector, as_of_round)
    if ao is not None and "as_of_round" in preds.columns:
        preds = preds[preds["as_of_round"].astype(int) == int(ao)]
    if rid is not None and "run_id" in preds.columns:
        preds = preds[preds["run_id"].astype(str) == str(rid)]
    return sorted(
        [c for c in preds.columns if c.startswith(("obj__", "pred__", "sel__"))]
    )
