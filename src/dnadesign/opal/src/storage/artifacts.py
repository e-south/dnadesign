"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/artifacts.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..core.utils import ensure_dir, file_sha256
from .parquet_io import write_parquet_df


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


def write_selection_parquet(path: Path, df_selected: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    write_parquet_df(path, df_selected, index=False)
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
    write_parquet_df(path, df, index=False)
    return file_sha256(path)
