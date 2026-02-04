"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_auto_opt_override_strict.py

Validate strict handling of --no-auto-opt overrides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.sample_workflow import run_sample
from dnadesign.cruncher.config.load import load_config


def test_no_auto_opt_override_requires_explicit_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {
                "catalog_root": str(tmp_path / ".cruncher"),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
                "min_sites_for_pwm": 1,
                "allow_low_sites": True,
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "mode": "optimize",
                "rng": {"seed": 11, "deterministic": True},
                "budget": {"draws": 1, "tune": 1, "restarts": 1},
                "init": {"kind": "random", "length": 6, "pad_with": "background"},
                "objective": {"bidirectional": True, "score_scale": "llr"},
                "elites": {"k": 1, "min_hamming": 0, "filters": {"pwm_sum_min": 0.0}},
                "moves": {"profile": "balanced", "overrides": {"move_probs": {"S": 1.0, "B": 0.0, "M": 0.0}}},
                "optimizer": {"name": "auto"},
                "optimizers": {"gibbs": {"beta_schedule": {"kind": "fixed", "beta": 1.0}, "apply_during": "tune"}},
                "auto_opt": {"enabled": True},
                "output": {"trace": {"save": False}, "save_sequences": False},
                "ui": {"progress_bar": False, "progress_every": 0},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    monkeypatch.setattr("dnadesign.cruncher.app.sample_workflow.ensure_mpl_cache", lambda *_: None)
    monkeypatch.setattr("dnadesign.cruncher.app.sample_workflow._lockmap_for", lambda *_: json.loads("{}"))
    monkeypatch.setattr("dnadesign.cruncher.app.sample_workflow.target_statuses", lambda **_: [])

    with pytest.raises(ValueError, match="sample.optimizer.name='auto'"):
        run_sample(cfg, config_path, auto_opt_override=False)
