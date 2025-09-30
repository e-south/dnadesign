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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import LEDGER_SCHEMA_VERSION
from .utils import ensure_dir, file_sha256


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


def events_path(workdir: Path) -> Path:
    d = workdir / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ledger.index.parquet"


def _append_parquet(path: Path, df: pd.DataFrame) -> None:
    """
    Append rows efficiently as a new row group.
    Schema must be stable; caller ensures column set/order.
    """
    ensure_dir(path.parent)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    if not path.exists():
        pq.write_table(table, path)  # creates file with schema
        return
    # append new row group; fails fast if schema mismatches (assertive)
    pq.write_table(table, path, append=True)


def append_events(index_path: Path, df: pd.DataFrame) -> str:
    """
    Canonical append for the **ledger** (append-only):
      • Typed sinks written as **part files** under outputs/ledger.* directories.
      • Thin **index** written to outputs/ledger.index.parquet for quick scans.

    Enforces per-sink allow-lists to prevent schema pollution.
    """
    ensure_dir(index_path.parent)

    # ---- Allow-lists (Ledger v1.1) ----
    ALLOW: dict[str, set[str]] = {
        "run_pred": {
            "event",
            "run_id",
            "as_of_round",
            "id",
            "sequence",
            "pred__y_dim",
            "pred__y_hat_model",
            "pred__y_obj_scalar",
            "sel__rank_competition",
            "sel__is_selected",
            # row-level objective diagnostics only
            "obj__logic_fidelity",
            "obj__effect_raw",
            "obj__effect_scaled",
            "obj__clip_lo_mask",
            "obj__clip_hi_mask",
        },
        "run_meta": {
            "event",
            "run_id",
            "as_of_round",
            "model__name",
            "model__params",
            "training__y_ops",
            "x_transform__name",
            "x_transform__params",
            "y_ingest__name",
            "y_ingest__params",
            "objective__name",
            "objective__params",
            "objective__summary_stats",
            "objective__denom_value",
            "objective__denom_percentile",
            "selection__name",
            "selection__params",
            "selection__score_field",
            "selection__objective",
            "selection__tie_handling",
            "stats__n_train",
            "stats__n_scored",
            "stats__unc_mean_sd_targets",
            "artifacts",
            "pred__preview",
            "schema__version",
            "opal__version",
        },
        "label": {
            "event",
            "observed_round",
            "id",
            "sequence",
            "y_obs",
            "src",
            "note",
        },
    }

    TARGET_DIR = {
        "run_pred": index_path.parent / "ledger.predictions",
        "run_meta": index_path.parent / "ledger.runs",
        "label": index_path.parent / "ledger.labels",
    }

    def _write_part(target_dir: Path, subdf: pd.DataFrame) -> None:
        ensure_dir(target_dir)
        tbl = pa.Table.from_pandas(subdf, preserve_index=False)
        out = target_dir / f"part-{uuid4().hex}.parquet"
        pq.write_table(tbl, out)

    # ---- 1) Typed sinks ----
    if "event" not in df.columns:
        raise ValueError("append_events requires an 'event' column.")

    for ev, sub in df.groupby("event", dropna=False):
        ev_name = str(ev) if pd.notna(ev) else "unknown"
        if ev_name not in ALLOW:
            raise ValueError(f"Unknown ledger event kind: {ev_name!r}")
        allow = ALLOW[ev_name]
        unknown = [c for c in sub.columns if c not in allow]
        if unknown:
            # Assertive by default: drop; strict in debug
            if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                raise ValueError(
                    f"[ledger:{ev_name}] refusing to append unknown columns {sorted(unknown)} "
                    f"(schema version {LEDGER_SCHEMA_VERSION})."
                )
            sub = sub[[c for c in sub.columns if c in allow]].copy()
        _write_part(TARGET_DIR[ev_name], sub)

    # ---- 2) Thin index for convenience ----
    INDEX_COLS = [
        "event",
        "run_id",
        "as_of_round",
        "id",
        "sequence",
        "pred__y_dim",
        "pred__y_obj_scalar",
        "sel__rank_competition",
        "sel__is_selected",
    ]

    if True:  # keep schema stable; fill missing with None
        idx = {}
        n = len(df)
        for c in INDEX_COLS:
            if c in df.columns:
                idx[c] = df[c]
            else:
                idx[c] = [None] * n
        idx_df = pd.DataFrame(idx, columns=INDEX_COLS)
        _append_parquet(index_path, idx_df)
