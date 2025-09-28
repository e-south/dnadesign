"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/artifacts.py

Round artifacts helpers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import file_sha256


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
    df_selected.to_csv(path, index=False)
    return file_sha256(path)


def append_round_log_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def write_round_ctx(path: Path, ctx: dict) -> str:
    Path(path).write_text(json.dumps(ctx, indent=2))
    return file_sha256(path)


def write_objective_meta(path: Path, meta: Dict[str, Any]) -> str:
    Path(path).write_text(json.dumps(meta, indent=2))
    return file_sha256(path)


def events_path(workdir: Path) -> Path:
    d = workdir / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "events.parquet"


def append_events(path: Path, df: pd.DataFrame) -> str:
    # create if missing; append otherwise
    if not path.exists():
        df.to_parquet(path, index=False)
    else:
        existing = pd.read_parquet(path)
        out = pd.concat([existing, df], ignore_index=True)
        out.to_parquet(path, index=False)
    return file_sha256(path)
