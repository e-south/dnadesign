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
            "out_dir": "runs",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {"catalog_root": ".cruncher", "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "mode": "sample",
                "rng": {"seed": 7, "deterministic": True},
                "sequence_length": 12,
                "compute": {"total_sweeps": 4, "adapt_sweep_frac": 0.25},
                "init": {"kind": "random", "pad_with": "background"},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr"},
                "elites": {"k": 1, "min_per_tf_norm": 0.0, "mmr_alpha": 0.85},
                "moves": {"profile": "balanced"},
                "output": {"trace": {"save": False}, "save_sequences": True},
                "ui": {"progress_bar": False, "progress_every": 0},
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
    assert cfg.sample.compute.total_sweeps == 4
    assert cfg.sample.compute.adapt_sweep_frac == 0.25


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_adapt_sweep_frac_bounds(tmp_path: Path, value: float) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["compute"]["adapt_sweep_frac"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="adapt_sweep_frac"):
        load_config(config_path)


@pytest.mark.parametrize("value", [0, -1])
def test_total_sweeps_requires_positive(tmp_path: Path, value: int) -> None:
    payload = _base_config()
    payload["cruncher"]["sample"]["compute"]["total_sweeps"] = value
    config_path = _write_config(tmp_path, payload)
    with pytest.raises(ValidationError, match="total_sweeps"):
        load_config(config_path)
