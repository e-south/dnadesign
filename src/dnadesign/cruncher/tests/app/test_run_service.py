"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_run_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dnadesign.cruncher.app.run_service import (
    drop_run_index_entries,
    get_run,
    load_run_index,
    save_run_index,
)
from dnadesign.cruncher.artifacts.layout import manifest_path
from dnadesign.cruncher.config.load import load_config


def test_get_run_accepts_path(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {
                "catalog_root": "cache_root",
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
                "min_sites_for_pwm": 2,
                "allow_low_sites": False,
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "mode": "sample",
                "rng": {"seed": 1, "deterministic": True},
                "sequence_length": 6,
                "compute": {"total_sweeps": 2, "adapt_sweep_frac": 0.5},
                "init": {"kind": "random", "pad_with": "background"},
                "objective": {"bidirectional": True, "score_scale": "llr"},
                "elites": {"k": 1, "min_per_tf_norm": None, "mmr_alpha": 0.85},
                "moves": {
                    "profile": "balanced",
                    "overrides": {
                        "block_len_range": [2, 2],
                        "multi_k_range": [2, 2],
                        "slide_max_shift": 1,
                        "swap_len_range": [2, 2],
                        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0},
                    },
                },
                "output": {"trace": {"save": False}, "save_sequences": True},
                "ui": {"progress_bar": False, "progress_every": 0},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    run_name = "20250101_000000_abcd12"
    run_dir = tmp_path / "results" / "sample" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": ["meta/config_used.yaml"],
    }
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest))

    run = get_run(cfg, config_path, str(run_dir))
    assert run.name == run_name
    assert run.run_dir == run_dir

    rel_path = run_dir.relative_to(config_path.parent)
    run_rel = get_run(cfg, config_path, str(rel_path))
    assert run_rel.run_dir == run_dir


def test_drop_run_index_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}")
    payload = {
        "run_a": {"stage": "sample", "run_dir": str(tmp_path / "run_a")},
        "run_b": {"stage": "sample", "run_dir": str(tmp_path / "run_b")},
    }
    save_run_index(config_path, payload, catalog_root=".cruncher")
    removed = drop_run_index_entries(config_path, ["run_b"], catalog_root=".cruncher")
    assert removed == 1
    index = load_run_index(config_path, catalog_root=".cruncher")
    assert "run_b" not in index
    assert "run_a" in index
