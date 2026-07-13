"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/evaluation/prediction_ledger.py

Run-scoped OPAL prediction ledger helpers for the DenseGen axis probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.opal import read_campaign_selection_view_predictions


def read_probe_predictions(config_path: Path, *, round_selector: str | None = "latest") -> pd.DataFrame:
    return read_campaign_selection_view_predictions(
        config_path,
        selection_view_id="primary",
        columns=[
            "run_id",
            "as_of_round",
            "id",
            "pred__y_hat_model",
            "view__selection_score",
            "view__rank_competition",
            "view__is_selected",
        ],
        round_selector=round_selector,
        require_run_id=True,
    )


def prediction_id_problems(predictions: pd.DataFrame, eval_ids: set[str], *, run_key: str) -> list[str]:
    prediction_ids = predictions["id"].astype(str)
    problems: list[str] = []
    duplicate_ids = prediction_ids.loc[prediction_ids.duplicated()].drop_duplicates().sort_values().head(5).tolist()
    if duplicate_ids:
        problems.append(f"duplicate prediction id(s) for scored run {run_key}: {', '.join(duplicate_ids)}")
    prediction_id_set = set(prediction_ids.tolist())
    missing_eval_ids = sorted(eval_ids - prediction_id_set)
    if missing_eval_ids:
        preview = ", ".join(missing_eval_ids[:5])
        suffix = "" if len(missing_eval_ids) <= 5 else f", ... ({len(missing_eval_ids)} total)"
        problems.append(f"missing eval id(s) for scored run {run_key}: {preview}{suffix}")
    return problems
