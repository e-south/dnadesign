"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_pt_adaptation_config.py

Validates schema support for adaptive PT and move/proposal tuning blocks.

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
                "pt": {
                    "n_temps": 4,
                    "temp_max": 20.0,
                    "adapt": {
                        "enabled": True,
                        "target_swap": 0.25,
                        "window": 50,
                        "k": 0.5,
                        "min_scale": 0.25,
                        "max_scale": 4.0,
                        "strict": True,
                        "saturation_windows": 5,
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


def test_pt_and_move_adaptation_blocks_load(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config())
    cfg = load_config(config_path)
    assert cfg.sample is not None
    assert cfg.sample.pt.adapt.strict is True
    assert cfg.sample.moves.overrides is not None
    assert cfg.sample.moves.overrides.adaptive_weights is not None
    assert cfg.sample.moves.overrides.proposal_adapt is not None


def test_pt_adapt_saturation_windows_must_be_positive(tmp_path: Path) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["pt"]["adapt"]["saturation_windows"] = 0
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="saturation_windows"):
        load_config(config_path)
