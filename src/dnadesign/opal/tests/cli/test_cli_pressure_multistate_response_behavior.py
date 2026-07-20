"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_pressure_multistate_response_behavior.py

Synthetic full-path pressure test for Multistate Response Behavior campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.opal.api.multistate_response_behavior import score_multistate_response_behavior
from dnadesign.opal.src.analysis.campaign import CampaignAnalysis
from dnadesign.opal.src.analysis.notebook_components import build_notebook_visual_surface_model
from dnadesign.opal.src.analysis.notebook_components.layered_scatter import (
    build_notebook_layered_scatter_contract,
    filter_notebook_layered_scatter_rows,
)
from dnadesign.opal.src.analysis.notebook_components.visual_hierarchy import notebook_visual_group
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.reporting.notebook import build_notebook_view_model
from dnadesign.opal.src.reporting.selection_set import load_selection_batch
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml

CHANNELS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
OBJECTIVE_PARAMS = {
    "state_ids": ["00", "10", "01", "11"],
    "target_mask": [0, 1, 0, 1],
    "softmin_scale": 1.0,
}


def _write_records(path: Path) -> tuple[list[str], list[str]]:
    ids = [f"candidate-{index}" for index in range(6)]
    sequences = ["ACGT" + ("A" * index) + ("C" * (5 - index)) for index in range(6)]
    flat_x = [value for index in range(6) for value in (index / 10.0, (5 - index) / 10.0)]
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "sequence": pa.array(sequences, type=pa.string()),
                "bio_type": pa.array(["dna"] * len(ids), type=pa.string()),
                "alphabet": pa.array(["dna_4"] * len(ids), type=pa.string()),
                "X": pa.FixedSizeListArray.from_arrays(pa.array(flat_x, type=pa.float32()), 2),
            }
        ),
        path,
    )
    return ids, sequences


def _write_labels(path: Path, *, ids: list[str], sequences: list[str]) -> None:
    values = np.asarray(
        [
            [0.0, 2.0, 0.2, 1.2, -1.0, 1.0, -0.5, 0.8],
            [0.4, 0.8, 1.8, 1.3, -0.3, 0.1, 0.9, 0.7],
        ],
        dtype=float,
    )
    frame = pd.DataFrame({"id": ids[:2], "sequence": sequences[:2]})
    for index, column in enumerate(CHANNELS):
        frame[column] = values[:, index]
    frame.to_csv(path, index=False)


def _assert_ok(result) -> None:
    assert result.exit_code == 0, result.stdout


def test_multistate_behavior_real_round_plot_and_notebook_contract(tmp_path: Path) -> None:
    """Run the smallest real two-view behavior campaign without study fixtures."""

    workdir = tmp_path / "campaign"
    workdir.mkdir()
    records_path = workdir / "records.parquet"
    ids, sequences = _write_records(records_path)
    campaign_path = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign_path,
        workdir=workdir,
        records_path=records_path,
        plots=[
            {
                "name": "behavior_frontier",
                "kind": "multistate_response_behavior_frontier",
                "round_selector": "latest",
                "output": {"save_data": True},
            },
            {
                "name": "behavior_decomposition",
                "kind": "multistate_response_behavior_selected_decomposition",
                "round_selector": "latest",
                "output": {"save_data": True},
            },
        ],
        transforms_y_name="vector_from_table_v1",
        transforms_y_params={"id_column": "id", "value_columns": list(CHANNELS)},
        objective_name="multistate_response_behavior_v1",
        objective_params=OBJECTIVE_PARAMS,
        y_expected_length=8,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={
            "top_k": 1,
            "score_ref": "behavior_score",
            "tie_handling": "ordinal",
            "require_exact_top_k": True,
        },
    )
    payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    view_a = payload["selection_views"][0]
    view_a["id"] = "view-a"
    view_b = deepcopy(view_a)
    view_b["id"] = "view-b"
    # The identical targets intentionally collide at raw top one. The runtime
    # allocator must advance view-b to the next sequence-unique candidate.
    payload["selection_views"] = [view_a, view_b]
    payload["selection_batch"] = {
        "deduplicate_by": "sequence",
        "expected_unique_count": 2,
        "allocation": {
            "strategy": "round_robin_next_best_unallocated",
            "view_priority": ["view-a", "view-b"],
        },
    }
    campaign_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    labels_path = workdir / "labels.csv"
    _write_labels(labels_path, ids=ids, sequences=sequences)
    runner = CliRunner()
    app = _build()
    _assert_ok(runner.invoke(app, ["--no-color", "init", "-c", str(campaign_path)]))
    _assert_ok(runner.invoke(app, ["--no-color", "validate", "-c", str(campaign_path)]))
    _assert_ok(
        runner.invoke(
            app,
            [
                "--no-color",
                "ingest-y",
                "-c",
                str(campaign_path),
                "--round",
                "0",
                "--csv",
                str(labels_path),
                "--apply",
            ],
        )
    )
    _assert_ok(runner.invoke(app, ["--no-color", "run", "-c", str(campaign_path), "--round", "0"]))

    analysis = CampaignAnalysis.from_config_path(campaign_path)
    runs = analysis.read_runs()
    assert runs.height == 1
    run_id = str(runs.get_column("run_id")[0])
    selected_ids_by_view: dict[str, set[str]] = {}
    for view_id in ("view-a", "view-b"):
        predicted = analysis.read_selection_view_predictions(
            selection_view_id=view_id,
            round_selector=0,
            run_id=run_id,
        )
        y_hat = np.asarray(predicted.get_column("pred__y_hat_model").to_list(), dtype=float)
        assert y_hat.shape == (4, 8)
        replayed = score_multistate_response_behavior(y_hat, **OBJECTIVE_PARAMS)
        np.testing.assert_allclose(
            predicted.get_column("view__score").to_numpy(),
            replayed.behavior_score,
            rtol=1e-10,
            atol=1e-12,
        )
        selected = predicted.filter(predicted.get_column("view__is_selected"))
        assert selected.height == 1
        selected_ids_by_view[view_id] = set(selected.get_column("id").to_list())
    assert selected_ids_by_view["view-a"].isdisjoint(selected_ids_by_view["view-b"])

    batch = load_selection_batch(campaign_path, round_selector=0, run_id=run_id)
    assert batch["allocation_strategy"] == "round_robin_next_best_unallocated"
    assert batch["unique_count"] == 2
    assert batch["verification"]["status"] == "pass"
    assert batch["verification"]["allocation_trace_digest"]["status"] == "pass"

    contracts = []
    for view_id in ("view-a", "view-b"):
        _assert_ok(
            runner.invoke(
                app,
                [
                    "--no-color",
                    "plot",
                    "-c",
                    str(campaign_path),
                    "--view",
                    view_id,
                    "--run-id",
                    run_id,
                ],
            )
        )
        view_model = build_notebook_view_model(campaign_path, round_selector="latest", run_id=run_id)
        surface = build_notebook_visual_surface_model(view_model, selection_view_id=view_id)
        choices_by_kind = {item["kind"]: item for item in surface["choices"]}
        assert choices_by_kind["multistate_response_behavior_selected_decomposition"]["manifest"]["status"] == "written"
        choice = choices_by_kind["multistate_response_behavior_frontier"]
        assert notebook_visual_group(choice)[0].key == "decision"
        assert choice["manifest"]["selection_view_id"] == view_id
        assert choice["manifest"]["status"] == "written"
        assert choice["manifest"]["freshness"]["status"] == "fresh"
        contract = build_notebook_layered_scatter_contract(choice)
        assert contract is not None
        assert contract["adapter"] == "layered_scatter_v1"
        assert contract["runtime"]["reference_lines"] == {"x": [], "y": []}
        assert "not feasibility" in contract["runtime"]["color_scale"]["context"]
        assert contract["view"]["x_column"] == "response_family_score"
        assert contract["view"]["y_column"] == "on_signal_family_score"
        assert contract["view"]["color_column"] == "off_signal_suppression_family_score"
        assert contract["observed_batches"]
        tidy = pd.read_csv(contract["tidy_path"])
        assert set(tidy["record_kind"]) == {"prediction", "observed_label"}
        filtered = filter_notebook_layered_scatter_rows(
            tidy,
            contract=contract,
            state={
                "show_prediction_pool": True,
                "show_selected": True,
                "observed_batches": [item["id"] for item in contract["observed_batches"]],
                "label_scope": "both",
            },
        )
        assert filtered.attrs["effective_label_scope"] == "both"
        assert filtered.attrs["annotate_row_positions"]
        assert filtered.loc[filtered["record_kind"].eq("prediction"), "selected"].any()
        assert filtered["record_kind"].eq("observed_label").any()
        contracts.append(contract)
    assert contracts[0]["key"] == contracts[1]["key"]
