# ABOUTME: Test helpers for building OPAL campaign fixtures and ledgers.
# ABOUTME: Provides small utilities for CLI workflow tests.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/_cli_helpers.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from dnadesign.opal.src import LEDGER_SCHEMA_VERSION
from dnadesign.opal.src import __version__ as OPAL_VERSION
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.state import CampaignState, RoundEntry
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionEmit,
    build_run_meta_event,
    build_run_pred_events,
)


def write_records(
    path: Path,
    *,
    include_opal_cols: bool = False,
    slug: str = "demo",
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3]],
        }
    )
    if include_opal_cols:
        df[f"opal__{slug}__label_hist"] = [[], []]
        df["Y"] = [None, None]
    df.to_parquet(path, index=False)
    return df


def write_campaign_yaml(
    path: Path,
    *,
    workdir: Path,
    records_path: Path,
    plots: Optional[List[Dict[str, Any]]] = None,
    slug: str = "demo",
    transforms_y_name: str = "sfxi_vec8_from_table_v1",
    transforms_y_params: Optional[Dict[str, Any]] = None,
    objective_name: str = "sfxi_v1",
    objective_params: Optional[Dict[str, Any]] = None,
    y_expected_length: int = 8,
    model_name: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    selection_name: str = "top_n",
    selection_params: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
) -> None:
    if transforms_y_params is None:
        transforms_y_params = {}
    if objective_params is None:
        objective_params = {"setpoint_vector": [0, 0, 0, 1]}
    if model_params is None:
        model_params = {"n_estimators": 5, "random_state": 0, "oob_score": False}
    score_channel = "sfxi" if objective_name == "sfxi_v1" else "scalar"
    default_selection_params: Dict[str, Any] = {
        "top_k": 1,
        "score_ref": f"{objective_name}/{score_channel}",
        "objective_mode": "maximize",
        "tie_handling": "competition_rank",
    }
    if selection_name == "expected_improvement":
        default_selection_params["uncertainty_ref"] = default_selection_params["score_ref"]
    merged_selection_params = dict(default_selection_params)
    if selection_params is not None:
        merged_selection_params.update(selection_params)
    if selection_name == "expected_improvement" and "uncertainty_ref" not in merged_selection_params:
        merged_selection_params["uncertainty_ref"] = str(merged_selection_params["score_ref"])
    cfg: Dict[str, Any] = {
        "campaign": {"name": "Demo", "slug": slug, "workdir": str(workdir)},
        "data": {
            "location": {"kind": "local", "path": str(records_path)},
            "x_column_name": "X",
            "y_column_name": "Y",
            "y_expected_length": int(y_expected_length),
        },
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {"name": transforms_y_name, "params": transforms_y_params},
        "model": {"name": model_name, "params": model_params},
        "objectives": [{"name": objective_name, "params": objective_params}],
        "selection": {"name": selection_name, "params": merged_selection_params},
    }
    if plots is not None:
        cfg["plots"] = plots
    if safety is not None:
        cfg["safety"] = safety
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_round_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"ts": "2025-01-01T00:00:00+00:00", "stage": "start"},
        {"ts": "2025-01-01T00:00:05+00:00", "stage": "fit_start"},
        {"ts": "2025-01-01T00:00:06+00:00", "stage": "fit"},
        {"ts": "2025-01-01T00:00:07+00:00", "stage": "predict_batch", "rows": 2},
        {"ts": "2025-01-01T00:00:08+00:00", "stage": "done"},
    ]
    path.write_text("\n".join([json.dumps(line) for line in lines]))


def write_state(
    workdir: Path,
    *,
    records_path: Path,
    run_id: str,
    round_index: int = 0,
) -> None:
    round_dir = workdir / "outputs" / "rounds" / f"round_{int(round_index)}"
    round_dir.mkdir(parents=True, exist_ok=True)
    round_log = round_dir / "logs" / "round.log.jsonl"
    if not round_log.exists():
        write_round_log(round_log)
    st = CampaignState(
        campaign_slug="demo",
        campaign_name="Demo",
        workdir=str(workdir),
        data_location={
            "kind": "local",
            "path": str(records_path),
            "records_path": str(records_path),
        },
        x_column_name="X",
        y_column_name="Y",
    )
    st.add_round(
        RoundEntry(
            run_id=run_id,
            round_index=int(round_index),
            round_name=f"round_{int(round_index)}",
            round_dir=str(round_dir),
            labels_used_rounds=[int(round_index)],
            number_of_training_examples_used_in_round=2,
            number_of_candidates_scored_in_round=2,
            selection_top_k_requested=1,
            selection_top_k_effective_after_ties=1,
            model={
                "type": "random_forest",
                "params": {},
                "artifact_path": str(round_dir / "model" / "model.joblib"),
            },
            metrics={},
            durations_sec={},
            seeds={},
            artifacts={"round_log_jsonl": str(round_log)},
            writebacks={},
            warnings=[],
        )
    )
    st.save(workdir / "state.json")


def write_ledger(workdir: Path, *, run_id: str, round_index: int = 0) -> None:
    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    writer = LedgerWriter(ws)

    y_hat = np.array([[0.1], [0.2]])
    y_obj = np.array([0.1, 0.2])
    sel_emit = SelectionEmit(
        ranks_competition=np.array([1, 2]),
        selected_bool=np.array([True, False]),
    )
    run_pred = build_run_pred_events(
        run_id=run_id,
        as_of_round=int(round_index),
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=y_hat,
        selected_score=y_obj,
        selected_score_ref="sfxi_v1/sfxi",
        y_dim=1,
        obj_diagnostics={"logic_fidelity": np.array([1.0, 0.5])},
        sel_emit=sel_emit,
    )

    run_meta = build_run_meta_event(
        run_id=run_id,
        as_of_round=int(round_index),
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="sfxi_vec8_from_table_v1",
        y_ingest_transform_params={},
        objective_name="sfxi_v1",
        objective_params={"setpoint_vector": [0, 0, 0, 1]},
        objective_defs=[{"name": "sfxi_v1", "score_channels": ["sfxi_v1/sfxi"], "uncertainty_channels": []}],
        selection_name="top_n",
        selection_params={"top_k": 1, "score_ref": "sfxi_v1/sfxi"},
        selection_score_ref="sfxi_v1/sfxi",
        selection_uncertainty_ref=None,
        selection_objective_mode="maximize",
        sel_tie_handling="competition_rank",
        stats_n_train=2,
        stats_n_scored=2,
        unc_mean_sd=None,
        pred_rows_df=run_pred,
        artifact_paths_and_hashes={},
        objective_summary_stats={
            "score_min": 0.1,
            "score_median": 0.2,
            "score_max": 0.3,
        },
    )
    # Ensure required schema/version fields are present (build_run_meta_event already includes them)
    run_meta["schema__version"] = LEDGER_SCHEMA_VERSION
    run_meta["opal__version"] = OPAL_VERSION

    writer.append_run_pred(run_pred)
    writer.append_run_meta(run_meta)


def write_ledger_labels(workdir: Path, *, round_index: int = 0) -> None:
    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    writer = LedgerWriter(ws)
    labels = pd.DataFrame(
        {
            "event": ["label"],
            "observed_round": [int(round_index)],
            "id": ["a"],
            "y_obs": [[0.1]],
            "src": ["test"],
        }
    )
    writer.append_label(labels)
