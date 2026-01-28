"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_config_migration.py

Config migration coverage for deprecated Stage-A sampling keys.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from dnadesign.densegen.src.config import load_config


def test_legacy_sampling_keys_rewritten(tmp_path: Path, caplog) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.7",
                    "run": {"id": "demo", "root": "."},
                    "inputs": [
                        {
                            "name": "demo_pwm",
                            "type": "pwm_meme",
                            "path": "inputs.meme",
                            "sampling": {
                                "strategy": "stochastic",
                                "n_sites": 2,
                                "oversample_factor": 3,
                                "scoring_backend": "fimo",
                                "dedupe_by": "core",
                                "length_policy": "range",
                                "length_range": [5, 6],
                                "trim_window_length": 4,
                                "trim_window_strategy": "max_info",
                            },
                        }
                    ],
                    "output": {
                        "targets": ["parquet"],
                        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                        "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
                    },
                    "generation": {
                        "sequence_length": 10,
                        "quota": 1,
                        "plan": [{"name": "default", "quota": 1}],
                    },
                    "solver": {"backend": "CBC", "strategy": "iterate"},
                    "logging": {"log_dir": "outputs/logs"},
                }
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        loaded = load_config(cfg_path)
    assert "Deprecated DenseGen config keys rewritten" in caplog.text
    sampling = loaded.root.densegen.inputs[0].sampling
    assert sampling.mining.budget.mode == "fixed_candidates"
    assert sampling.mining.budget.candidates == 6
    assert sampling.uniqueness.key == "core"
    assert sampling.length.policy == "range"
    assert list(sampling.length.range or []) == [5, 6]
    assert sampling.trimming.window_length == 4
