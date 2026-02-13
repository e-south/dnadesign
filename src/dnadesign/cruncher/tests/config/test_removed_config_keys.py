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
                    "select": {"diversity": 0.0, "pool_size": "auto"},
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


@pytest.mark.parametrize(
    "path,value",
    [
        (("cruncher", "motif_store"), {"catalog_root": ".cruncher", "pwm_source": "matrix"}),
        (("cruncher", "parse"), {"plot": {"logo": False}}),
        (("cruncher", "sample", "output", "trim"), {"enabled": True}),
        (("cruncher", "sample", "output", "polish"), {"enabled": True}),
        (("cruncher", "sample", "optimizer"), {"name": "pt"}),
        (("cruncher", "sample", "optimizers"), {"pt": {"swap_prob": 0.1}}),
        (("cruncher", "sample", "auto_opt"), {"enabled": True}),
        (("cruncher", "sample", "init", "length"), 12),
        (("cruncher", "sample", "auto_opt", "length"), {"enabled": True}),
        (("cruncher", "sample", "auto_opt", "pt_ladder_sizes"), [2]),
        (("cruncher", "sample", "auto_opt", "beta_ladder_scales"), [1.0]),
        (("cruncher", "sample", "auto_opt", "beta_schedule_scales"), [1.0]),
        (("cruncher", "sample", "auto_opt", "gibbs_move_probs"), [{"S": 1.0}]),
        (("cruncher", "sample", "auto_opt", "allow_trim_polish_in_pilots"), True),
        (("cruncher", "sample", "auto_opt", "prefer_simpler_if_close"), True),
        (("cruncher", "sample", "auto_opt", "tolerance"), {"score": 0.1}),
        (("cruncher", "sample", "auto_opt", "policy", "scorecard", "metric"), "elites_mmr"),
        (("cruncher", "sample", "auto_opt", "policy", "scorecard", "k"), 10),
        (("cruncher", "sample", "auto_opt", "policy", "scorecard", "top_k"), 10),
        (("cruncher", "sample", "elites", "min_hamming"), 0),
        (("cruncher", "sample", "elites", "dsDNA_canonicalize"), True),
        (("cruncher", "sample", "elites", "dsDNA_hamming"), True),
        (("cruncher", "sample", "elites", "selection"), {"policy": "mmr"}),
        (("cruncher", "sample", "elites", "filters"), {"min_per_tf_norm": 0.5}),
        (("cruncher", "sample", "elites", "selection", "distance"), {"kind": "sequence_hamming"}),
        (("cruncher", "sample", "elites", "selection", "relevance_norm"), "percentile"),
        (("cruncher", "sample", "elites", "selection", "min_distance"), 0.1),
        (("cruncher", "sample", "elites", "filter", "require_all_tfs"), True),
        (("cruncher", "sample", "elites", "select", "policy"), "mmr"),
        (("cruncher", "sample", "elites", "select", "diversity_metric"), "tfbs_core_weighted_hamming"),
        (("cruncher", "sample", "elites", "select", "alpha"), 0.8),
        (("cruncher", "sample", "elites", "select", "relevance"), "joint_score"),
        (("cruncher", "sample", "elites", "select", "distance_metric"), "hybrid"),
        (("cruncher", "sample", "elites", "select", "constraint_policy"), "strict"),
        (("cruncher", "sample", "elites", "select", "min_hamming_bp"), 2),
        (("cruncher", "sample", "elites", "select", "min_core_hamming_bp"), 2),
        (("cruncher", "sample", "elites", "select", "relax_step_bp"), 1),
        (("cruncher", "sample", "elites", "select", "relax_min_bp"), 0),
        (("cruncher", "sample", "elites", "select", "pool_strategy"), "stratified"),
        (("cruncher", "sample", "budget", "restarts"), 2),
    ],
)
def test_removed_keys_are_rejected(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    config = _base_config()
    node = config
    for key in path[:-1]:
        next_node = node.get(key)
        if not isinstance(next_node, dict):
            next_node = {}
            node[key] = next_node
        node = next_node
    node[path[-1]] = value
    config_path = _write_config(tmp_path, config)

    with pytest.raises(ValidationError) as exc:
        load_config(config_path)

    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())
