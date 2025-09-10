"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/artifacts.py

Round artifacts helpers.

- model.joblib (frozen model),
- selection_top_k.csv (lab handoff),
- feature_importance.csv,
- predictions_with_uncertainty.csv,
- round_model_metrics.json.

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
    feature_importance_csv: Path
    metrics_json: Path
    round_log_jsonl: Path
    preds_with_uncertainty_csv: Path


def write_selection_csv(path: Path, df_selected: pd.DataFrame) -> str:
    df_selected.to_csv(path, index=False)
    return file_sha256(path)


def write_feature_importance(path: Path, importances) -> str:
    if importances is None:
        df = pd.DataFrame({"feature_index": [], "feature_importance": []})
    else:
        df = pd.DataFrame(
            {
                "feature_index": range(len(importances)),
                "feature_importance": importances,
            }
        )
    df.to_csv(path, index=False)
    return file_sha256(path)


def write_round_metrics(path: Path, metrics: Dict[str, Any]) -> str:
    Path(path).write_text(json.dumps(metrics, indent=2))
    return file_sha256(path)


def write_predictions_with_uncertainty(path: Path, df: pd.DataFrame) -> str:
    df.to_csv(path, index=False)
    return file_sha256(path)
