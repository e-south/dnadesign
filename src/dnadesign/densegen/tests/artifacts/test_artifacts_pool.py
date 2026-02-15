"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/artifacts/test_artifacts_pool.py

Stage-A pool artifact tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact
from dnadesign.densegen.src.core.pipeline import default_deps


def test_build_pool_artifact_binding_sites(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAAA\nTF2,CCCC\n")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.9",
                    "run": {"id": "demo", "root": "."},
                    "inputs": [
                        {
                            "name": "demo input",
                            "type": "binding_sites",
                            "path": str(csv_path),
                            "format": "csv",
                        }
                    ],
                    "output": {
                        "targets": ["parquet"],
                        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                        "parquet": {"path": "outputs/tables/records.parquet"},
                    },
                    "generation": {
                        "sequence_length": 10,
                        "plan": [
                            {
                                "name": "default",
                                "quota": 1,
                                "sampling": {"include_inputs": ["demo input"]},
                                "regulator_constraints": {
                                    "groups": [
                                        {
                                            "name": "all",
                                            "members": ["TF1", "TF2"],
                                            "min_required": 1,
                                        }
                                    ]
                                },
                            }
                        ],
                    },
                    "solver": {"backend": "CBC", "strategy": "iterate"},
                    "runtime": {
                        "round_robin": False,
                        "arrays_generated_before_resample": 10,
                        "min_count_per_tf": 0,
                        "max_duplicate_solutions": 5,
                        "stall_seconds_before_resample": 10,
                        "stall_warning_every_seconds": 10,
                        "max_consecutive_failures": 25,
                        "max_seconds_per_plan": 0,
                        "max_failed_solutions": 0,
                        "checkpoint_every": 0,
                        "leaderboard_every": 50,
                    },
                    "logging": {"log_dir": "outputs/logs", "level": "INFO"},
                    "postprocess": {"pad": {"mode": "off"}},
                }
            }
        )
    )

    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    out_dir = tmp_path / "outputs" / "pools"
    outputs_root = tmp_path / "outputs"
    artifact, pool_data = build_pool_artifact(
        cfg=cfg,
        cfg_path=cfg_path,
        deps=default_deps(),
        rng=np.random.default_rng(0),
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    assert artifact.manifest_path.exists()
    entry = artifact.entry_for("demo input")
    assert " " not in entry.pool_path.name
    pool = pool_data["demo input"]
    assert pool.df is not None
    assert "tfbs_id" in pool.df.columns
    assert "motif_id" in pool.df.columns
    assert "tfbs_core" in pool.df.columns
