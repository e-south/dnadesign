"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_compute_config.py

Validate compute + sequence length schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dnadesign.cruncher.config.load import load_config


def _base_config() -> dict:
    return {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": ".cruncher", "pwm_source": "matrix"},
            "sample": {
                "seed": 7,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 3},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "filter": {"min_per_tf_norm": 0.0, "require_all_tfs": True, "pwm_sum_min": 0.0},
                    "select": {"alpha": 0.85, "pool_size": "auto"},
                },
                "moves": {"profile": "balanced"},
                "output": {
                    "save_trace": False,
                    "save_sequences": True,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
            },
        }
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_compute_config_loads(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config())
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.sequence_length == 12
    assert cfg.sample.budget.tune == 1
    assert cfg.sample.budget.draws == 3


@pytest.mark.parametrize("value", [0, -1])
def test_draws_requires_positive(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["budget"]["draws"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.budget.draws"):
        load_config(config_path)


@pytest.mark.parametrize("value", [-1])
def test_tune_requires_non_negative(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["budget"]["tune"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="sample.budget.tune"):
        load_config(config_path)


def test_analysis_particle_trajectory_fields_load(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["analysis"] = {
        "run_selector": "latest",
        "trajectory_plot_style": "particles",
        "trajectory_identity_mode": "particle",
        "trajectory_particle_alpha_min": 0.2,
        "trajectory_particle_alpha_max": 0.9,
        "trajectory_slot_overlay": False,
    }
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    assert cfg.analysis is not None
    assert cfg.analysis.trajectory_plot_style == "particles"
    assert cfg.analysis.trajectory_identity_mode == "particle"
