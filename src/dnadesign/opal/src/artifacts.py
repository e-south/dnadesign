"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/artifacts.py

Round artifacts helpers and canonical events append.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .ledger import LedgerWriter
from .utils import OpalError, ensure_dir, file_sha256
from .workspace import CampaignWorkspace


def round_dir(workdir: Path, round_index: int) -> Path:
    d = workdir / "outputs" / f"round_{round_index}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ArtifactPaths:
    model: Path
    selection_csv: Path
    round_log_jsonl: Path
    round_ctx_json: Path
    objective_meta_json: Path


def write_selection_csv(path: Path, df_selected: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    df_selected.to_csv(path, index=False)
    return file_sha256(path)


def write_feature_importance_csv(path: Path, df: pd.DataFrame) -> str:
    """
    Persist per-feature importances. Expected columns:
      - feature_index (int)
      - importance    (float; should sum to 1.0)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return file_sha256(path)


def append_round_log_event(path: Path, event: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def write_round_ctx(path: Path, ctx: dict) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(ctx, indent=2))
    return file_sha256(path)


def write_objective_meta(path: Path, meta: Dict[str, Any]) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(meta, indent=2))
    return file_sha256(path)


def write_model_meta(path: Path, meta: Dict[str, Any]) -> str:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(meta, indent=2))
    return file_sha256(path)


def write_labels_used_parquet(path: Path, df: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return file_sha256(path)


def events_path(workdir: Path) -> Path:
    """
    Historically pointed at outputs/ledger.index.parquet.
    We keep this helper to compute the *outputs/* root via `.parent`,
    but the thin index is no longer written by default.
    """
    d = workdir / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ledger.index.parquet"


def _append_parquet(path: Path, df: pd.DataFrame) -> None:
    """
    Append rows by **atomic concatenate-write** (no version-specific flags).
    Steps:
      • if file missing → write new file
      • else → read existing, validate schema, concat, write tmp, atomic replace
    """
    ensure_dir(path.parent)

    new_tbl = pa.Table.from_pandas(df, preserve_index=False)

    if not path.exists():
        pq.write_table(new_tbl, path)
        return

    try:
        old_tbl = pq.read_table(path)
    except Exception as e:
        raise RuntimeError(f"[ledger.index] failed to read existing Parquet file at {path!s}") from e

    # Assert schema compatibility (strict & explicit, easy to change later)
    if list(old_tbl.schema.names) != list(new_tbl.schema.names):
        raise RuntimeError(
            "Ledger index schema mismatch.\n"
            f"  existing: {list(old_tbl.schema.names)}\n"
            f"  new     : {list(new_tbl.schema.names)}\n"
            "Refusing to merge. You may delete the index to rebuild it, "
            "or run a migration step."
        )

    # NOTE: promote=True is the correct flag in recent pyarrow versions.
    out_tbl = pa.concat_tables([old_tbl, new_tbl], promote=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(out_tbl, tmp)
    tmp.replace(path)


def _append_parquet_dedupe(path: Path, df: pd.DataFrame, *, key: Sequence[str]) -> None:
    """
    Append with de-duplication on `key` and atomic replace.

    This is intended for small/skinny sinks (e.g., runs=1 row per run; labels << predictions),
    so we keep the implementation simple and assertive by materializing in pandas.
    """
    ensure_dir(path.parent)

    if not path.exists():
        df.to_parquet(path, index=False)
        return

    existing = pd.read_parquet(path)
    # Align column order strictly; assert mismatches
    if list(existing.columns) != list(df.columns):
        raise RuntimeError(
            f"Ledger sink schema mismatch.\n  existing: {list(existing.columns)}\n  new     : {list(df.columns)}"
        )
    out = pd.concat([existing, df], ignore_index=True)
    out = out.drop_duplicates(subset=list(key), keep="last")
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def append_events(index_path: Path, df: pd.DataFrame, *, write_index: bool = False) -> str:
    """
    Deprecated shim: prefer LedgerWriter directly. This wrapper enforces strict
    ledger semantics and never writes the legacy thin index.
    """
    if write_index:
        raise OpalError("ledger.index.parquet is deprecated and no longer written.")
    if "event" not in df.columns:
        raise ValueError("append_events requires an 'event' column.")

    workdir = index_path.parent.parent
    ws = CampaignWorkspace(config_path=index_path, workdir=workdir)
    writer = LedgerWriter(ws)
    for ev, sub in df.groupby("event", dropna=False):
        ev_name = str(ev) if pd.notna(ev) else "unknown"
        if ev_name == "run_pred":
            writer.append_run_pred(sub)
        elif ev_name == "run_meta":
            writer.append_run_meta(sub)
        elif ev_name == "label":
            writer.append_label(sub)
        else:
            raise OpalError(f"Unknown ledger event kind: {ev_name!r}")
    return str(index_path)
