"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_removed_config_keys.py

Validates removed config keys fail fast.

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
                "budget": {"draws": 2, "tune": 1, "restarts": 1},
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr"},
                "elites": {"k": 1, "filters": {"pwm_sum_min": 0.0}, "selection": {"policy": "mmr"}},
                "moves": {"profile": "balanced"},
                "optimizer": {"name": "pt"},
                "optimizers": {"pt": {"beta_ladder": {"kind": "geometric", "betas": [1.0, 0.5]}, "swap_prob": 0.1}},
                "auto_opt": {"enabled": False},
                "output": {"trace": {"save": False}, "save_sequences": True},
                "ui": {"progress_bar": False, "progress_every": 0},
            },
        }
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.mark.parametrize(
    "path,value",
    [
        (("cruncher", "sample", "output", "trim"), {"enabled": True}),
        (("cruncher", "sample", "output", "polish"), {"enabled": True}),
        (("cruncher", "sample", "auto_opt", "length"), {"enabled": True}),
        (("cruncher", "sample", "auto_opt", "pt_ladder_sizes"), [2]),
        (("cruncher", "sample", "auto_opt", "beta_ladder_scales"), [1.0]),
        (("cruncher", "sample", "auto_opt", "beta_schedule_scales"), [1.0]),
        (("cruncher", "sample", "auto_opt", "gibbs_move_probs"), [{"S": 1.0}]),
        (("cruncher", "sample", "auto_opt", "allow_trim_polish_in_pilots"), True),
        (("cruncher", "sample", "auto_opt", "prefer_simpler_if_close"), True),
        (("cruncher", "sample", "auto_opt", "tolerance"), {"score": 0.1}),
        (("cruncher", "sample", "auto_opt", "policy", "scorecard", "metric"), "elites_mmr"),
        (("cruncher", "sample", "auto_opt", "policy", "scorecard", "top_k"), 10),
        (("cruncher", "sample", "elites", "min_hamming"), 0),
        (("cruncher", "sample", "elites", "dsDNA_canonicalize"), True),
        (("cruncher", "sample", "elites", "dsDNA_hamming"), True),
        (("cruncher", "sample", "elites", "selection", "distance"), {"kind": "sequence_hamming"}),
        (("cruncher", "sample", "elites", "selection", "relevance_norm"), "percentile"),
    ],
)
def test_removed_keys_are_rejected(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    config = _base_config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_gibbs_optimizer_name_is_rejected(tmp_path: Path) -> None:
    config = _base_config()
    config["cruncher"]["sample"]["optimizer"]["name"] = "gibbs"
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError):
        load_config(config_path)
