"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_stage_a_overrides.py

CLI coverage for Stage-A override propagation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.densegen.src.adapters.sources.pwm_sampling import build_log_odds
from dnadesign.densegen.src.cli.main import _apply_stage_a_overrides
from dnadesign.densegen.src.config import load_config


def _write_pwm_artifact(path: Path, *, motif_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    payload = {
        "schema_version": "1.0",
        "producer": "test",
        "motif_id": motif_id,
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": background,
        "probabilities": matrix,
        "log_odds": build_log_odds(matrix, background),
    }
    path.write_text(json.dumps(payload))


def _write_config(tmp_path: Path) -> Path:
    _write_pwm_artifact(tmp_path / "inputs" / "M1.json", motif_id="M1")
    _write_pwm_artifact(tmp_path / "inputs" / "M2.json", motif_id="M2")
    cfg = {
        "densegen": {
            "schema_version": "2.9",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo_pwm",
                    "type": "pwm_artifact_set",
                    "paths": ["inputs/M1.json", "inputs/M2.json"],
                    "sampling": {
                        "strategy": "stochastic",
                        "n_sites": 2,
                        "mining": {
                            "batch_size": 10,
                            "budget": {"mode": "fixed_candidates", "candidates": 30},
                        },
                    },
                    "overrides_by_motif_id": {
                        "M2": {
                            "n_sites": 1,
                            "mining": {
                                "batch_size": 5,
                                "budget": {"mode": "fixed_candidates", "candidates": 15},
                            },
                        }
                    },
                }
            ],
            "output": {
                "targets": ["parquet"],
                "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                "parquet": {"path": "outputs/tables/records.parquet"},
            },
            "generation": {
                "sequence_length": 30,
                "plan": [
                    {
                        "name": "default",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo_pwm"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "logging": {"log_dir": "outputs/logs"},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_apply_stage_a_overrides_updates_motif_override_mining_budget(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path).root.densegen

    input_cfg = cfg.inputs[0]
    assert input_cfg.sampling.mining.budget.max_seconds is None
    assert input_cfg.overrides_by_motif_id["M2"].mining.budget.max_seconds is None

    _apply_stage_a_overrides(
        cfg,
        selected={"demo_pwm"},
        n_sites=7,
        batch_size=13,
        max_seconds=4.5,
    )

    input_cfg = cfg.inputs[0]
    assert input_cfg.sampling.n_sites == 7
    assert input_cfg.sampling.mining.batch_size == 13
    assert input_cfg.sampling.mining.budget.max_seconds == pytest.approx(4.5)

    override = input_cfg.overrides_by_motif_id["M2"]
    assert override.n_sites == 7
    assert override.mining.batch_size == 13
    assert override.mining.budget.max_seconds == pytest.approx(4.5)
