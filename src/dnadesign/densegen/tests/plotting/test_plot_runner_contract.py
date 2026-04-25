"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_runner_contract.py

Focused contract tests for the DenseGen plot runner.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.viz import plotting as plotting_module
from dnadesign.densegen.src.viz.plotting import run_plots_from_config


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              format: png
              default: ["placement_occupancy_map"]
              options:
                placement_occupancy_map:
                  scope: auto
                tfbs_concentration_profile:
                  scope: auto
              video:
                enabled: true
                mode: all_plans_round_robin_single_video
                sampling:
                  stride: 1
                  max_source_rows: 10
                  max_snapshots: 10
                playback:
                  target_duration_sec: 5
                  fps: 8
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_optional_stage_b_rerender_preserves_existing_core_outputs(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n", encoding="utf-8")

    loaded = load_config(cfg_path)
    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "sequence": ["ACGTACGTAA"],
            "source": ["demo"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
            "densegen__used_tfbs_detail": [[{"regulator": "TF_A", "sequence": "AAAA", "offset": 2}]],
        }
    )
    composition_df = pd.DataFrame(
        {
            "solution_id": ["sol-1"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
            "offset": [2],
            "length": [4],
            "end": [6],
        }
    )
    library_builds_df = pd.DataFrame({"library_index": [1], "library_hash": ["hash1"]})
    library_members_df = pd.DataFrame(
        {
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "library_index": [1],
            "library_hash": ["hash1"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
        }
    )

    monkeypatch.setattr(
        plotting_module,
        "load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "parquet:outputs/tables/records.parquet"),
    )
    monkeypatch.setattr(plotting_module, "_load_composition", lambda *_args, **_kwargs: composition_df.copy())
    monkeypatch.setattr(
        plotting_module,
        "_maybe_load_libraries",
        lambda *_args, **_kwargs: (library_builds_df.copy(), library_members_df.copy()),
    )
    monkeypatch.setattr(
        plotting_module,
        "_load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"sequence_length": 10}}},
    )

    def _fake_placement(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        **_kwargs,
    ) -> list[Path]:
        assert list(dense_arrays_df["id"]) == ["sol-1"]
        assert list(composition_df["solution_id"]) == ["sol-1"]
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placement", encoding="utf-8")
        return [target]

    def _fake_tfbs(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        library_members_df: pd.DataFrame | None = None,
        **_kwargs,
    ) -> list[Path]:
        assert list(composition_df["tfbs"]) == ["AAAA"]
        assert library_members_df is not None
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "tfbs_usage.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("tfbs", encoding="utf-8")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["placement_occupancy_map"], "fn", _fake_placement)
    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["tfbs_concentration_profile"], "fn", _fake_tfbs)

    run_plots_from_config(loaded.root, cfg_path, only="placement_occupancy_map")
    placement_path = run_root / "outputs" / "plots" / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
    assert placement_path.exists()

    run_plots_from_config(loaded.root, cfg_path, only="tfbs_concentration_profile")

    assert placement_path.exists()
    tfbs_path = run_root / "outputs" / "plots" / "stage_b" / "demo_plan" / "demo_input" / "tfbs_usage.png"
    assert tfbs_path.exists()

    payload = json.loads((run_root / "outputs" / "plots" / "current_inventory.json").read_text(encoding="utf-8"))
    paths = {str(item.get("path") or "") for item in payload.get("plots", [])}
    assert "stage_b/demo_plan/demo_input/occupancy.png" in paths
    assert "stage_b/demo_plan/demo_input/tfbs_usage.png" in paths
