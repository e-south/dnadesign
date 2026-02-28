"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/artifacts/test_artifacts_pool.py

Stage-A pool artifact tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact, load_pool_data
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
                                "sequences": 1,
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
                        "max_accepted_per_library": 10,
                        "min_count_per_tf": 0,
                        "max_duplicate_solutions": 5,
                        "no_progress_seconds_before_resample": 10,
                        "max_consecutive_no_progress_resamples": 25,
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


def test_build_pool_artifact_selected_inputs_ignore_unselected_missing_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAAA\n")
    missing_path = tmp_path / "missing.csv"
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
                        },
                        {
                            "name": "unused input",
                            "type": "binding_sites",
                            "path": str(missing_path),
                            "format": "csv",
                        },
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
                                "sequences": 1,
                                "sampling": {"include_inputs": ["demo input"]},
                                "regulator_constraints": {
                                    "groups": [
                                        {
                                            "name": "all",
                                            "members": ["TF1"],
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
                        "max_accepted_per_library": 10,
                        "min_count_per_tf": 0,
                        "max_duplicate_solutions": 5,
                        "no_progress_seconds_before_resample": 10,
                        "max_consecutive_no_progress_resamples": 25,
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
        overwrite=True,
        selected_inputs={"demo input"},
    )

    assert artifact.manifest_path.exists()
    assert set(pool_data) == {"demo input"}
    assert set(artifact.inputs) == {"demo input"}


def test_load_pool_data_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs" / "pools"
    out_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"tf": ["TF1"], "tfbs": ["AAAA"], "motif_id": ["m1"], "tfbs_id": ["id1"]}).to_parquet(
        outside, index=False
    )
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "../../../outside.parquet",
                "rows": 1,
                "columns": ["tf", "tfbs", "motif_id", "tfbs_id"],
                "pool_mode": "tfbs",
            }
        ],
    }
    (out_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match="must be a relative path without '\\.\\.'"):
        load_pool_data(out_dir)
