"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_pt_adaptation_config.py

Validates schema support for optimizer cooling and move/proposal tuning blocks.

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
                "sequence_length": 30,
                "budget": {"tune": 10, "draws": 20},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "filter": {"min_per_tf_norm": 0.0, "require_all_tfs": True, "pwm_sum_min": 0.0},
                    "select": {"alpha": 0.85, "pool_size": "auto"},
                },
                "moves": {
                    "profile": "balanced",
                    "overrides": {
                        "move_probs": {"S": 0.5, "B": 0.2, "M": 0.2, "L": 0.0, "W": 0.0, "I": 0.1},
                        "adaptive_weights": {"enabled": True, "window": 25},
                        "proposal_adapt": {"enabled": True, "window": 25},
                    },
                },
                "optimizer": {
                    "kind": "gibbs_anneal",
                    "chains": 4,
                    "cooling": {
                        "kind": "linear",
                        "beta": None,
                        "beta_start": 0.25,
                        "beta_end": 1.50,
                    },
                },
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


def test_optimizer_and_move_adaptation_blocks_load(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config())
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.optimizer.kind == "gibbs_anneal"
    assert cfg.sample.optimizer.chains == 4
    assert cfg.sample.optimizer.cooling.kind == "linear"
    assert cfg.sample.optimizer.cooling.beta_start == pytest.approx(0.25)
    assert cfg.sample.optimizer.cooling.beta_end == pytest.approx(1.50)
    assert cfg.sample.moves.overrides is not None
    assert cfg.sample.moves.overrides.adaptive_weights is not None
    assert cfg.sample.moves.overrides.proposal_adapt is not None


def test_piecewise_cooling_sweeps_must_be_strictly_increasing(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["optimizer"] = {
        "kind": "gibbs_anneal",
        "chains": 2,
        "cooling": {
            "kind": "piecewise",
            "beta": None,
            "beta_start": None,
            "beta_end": None,
            "stages": [
                {"sweeps": 10, "beta": 0.4},
                {"sweeps": 10, "beta": 0.9},
            ],
        },
    }
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="strictly increasing"):
        load_config(config_path)
