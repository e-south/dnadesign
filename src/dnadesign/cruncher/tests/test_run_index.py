"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_run_index.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.run_service import list_runs, rebuild_run_index
from dnadesign.cruncher.utils.run_layout import manifest_path, status_path


def test_run_index_rebuild(tmp_path: Path) -> None:
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
                "bidirectional": True,
                "seed": 1,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 6, "pad_with": "background"},
                "draws": 1,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.0,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    run_name = "set1_lexA_20250101_000000_abcd12"
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
    status_file = status_path(run_dir)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps({"stage": "sample", "status": "completed", "started_at": created_at}))

    index_path = rebuild_run_index(cfg, config_path)
    assert index_path == tmp_path / "cache_root" / "run_index.json"
    payload = json.loads(index_path.read_text())
    assert run_name in payload

    runs = list_runs(cfg, config_path)
    assert runs
    assert runs[0].name == run_name
    assert runs[0].status == "completed"
