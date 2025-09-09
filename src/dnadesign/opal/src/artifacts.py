"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/artifacts.py

Defines where round outputs live and how they're written:

- model.joblib (frozen model),
- selection_top_k.csv (lab handoff),
- feature_importance.csv,
- round_model_metrics.json.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import ensure_dir, file_sha256, write_json


@dataclass
class ArtifactPaths:
    model: Path
    selection_csv: Path
    feature_importance_csv: Path
    metrics_json: Path
    round_log_jsonl: Path


def round_dir(workdir: Path, k: int) -> Path:
    return workdir / "outputs" / f"round_{k}"


def write_selection_csv(path: Path, selected_df: pd.DataFrame) -> str:
    ensure_dir(path.parent)
    cols = ["rank_competition", "id", "sequence", "y_pred"]
    selected_df[cols].to_csv(path, index=False)
    return file_sha256(path)


def write_feature_importance(path: Path, importances: np.ndarray | None) -> str:
    ensure_dir(path.parent)
    if importances is None:
        df = pd.DataFrame(
            {"feature_index": [], "feature_importance": [], "feature_rank": []}
        )
    else:
        order = np.argsort(-importances)
        df = pd.DataFrame(
            {
                "feature_index": order.astype(int),
                "feature_importance": importances[order],
                "feature_rank": np.arange(1, len(order) + 1, dtype=int),
            }
        )
    df.to_csv(path, index=False)
    return file_sha256(path)


def write_round_metrics(path: Path, metrics: Dict[str, Any]) -> str:
    write_json(path, metrics)
    return file_sha256(path)
