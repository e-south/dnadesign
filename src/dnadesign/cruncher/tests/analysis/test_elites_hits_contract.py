"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_elites_hits_contract.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.cruncher.app.analyze_workflow import run_analyze
from dnadesign.cruncher.artifacts.layout import config_used_path, elites_path, manifest_path, sequences_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.hashing import sha256_path


def _write_config(tmp_path: Path, run_name: str) -> Path:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "outputs",
                "regulator_sets": [["lexA", "cpxR"]],
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "sample": {
                "seed": 7,
                "sequence_length": 12,
                "budget": {"tune": 0, "draws": 1},
            },
            "analysis": {
                "run_selector": "explicit",
                "runs": [run_name],
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_minimal_run(tmp_path: Path, run_name: str, *, optimizer_kind: str = "gibbs_anneal") -> Path:
    run_dir = tmp_path / "outputs" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            "pwms_info": {
                "lexA": {"pwm_matrix": pwm_matrix},
                "cpxR": {"pwm_matrix": pwm_matrix},
            },
            "active_regulator_set": {"index": 1, "tfs": ["lexA", "cpxR"]},
        }
    }
    config_used_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    config_used_path(run_dir).write_text(yaml.safe_dump(config_used))

    manifest = {
        "stage": "sample",
        "run_dir": str(run_dir.resolve()),
        "draws": 1,
        "adapt_sweeps": 0,
        "top_k": 2,
        "optimizer": {"kind": optimizer_kind},
        "optimizer_stats": {"beta_ladder_final": [1.0]},
        "objective": {"bidirectional": True},
        "motif_store": {"catalog_root": str((tmp_path / ".cruncher").resolve())},
    }
    manifest_path(run_dir).write_text(json.dumps(manifest, indent=2))

    seq_df = pd.DataFrame(
        {
            "sequence": ["AAAAAAAAAAAA"],
            "score_lexA": [0.2],
            "score_cpxR": [0.3],
            "min_norm": [0.2],
            "phase": ["draw"],
        }
    )
    sequences_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(sequences_path(run_dir), engine="pyarrow", index=False)

    elite_df = pd.DataFrame(
        {
            "id": ["elite-1"],
            "sequence": ["AAAAAAAAAAAA"],
            "rank": [1],
            "score_lexA": [0.2],
            "score_cpxR": [0.3],
            "norm_lexA": [0.2],
            "norm_cpxR": [0.3],
            "min_norm": [0.2],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elite_df.to_parquet(elites_path(run_dir), engine="pyarrow", index=False)

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    _ = sha256_path(lock_path)

    return run_dir


def test_analyze_requires_elites_hits_parquet(tmp_path: Path) -> None:
    run_name = "sample_test"
    config_path = _write_config(tmp_path, run_name)
    _write_minimal_run(tmp_path, run_name)
    cfg = load_config(config_path)

    with pytest.raises(FileNotFoundError, match="elites_hits"):
        run_analyze(cfg, config_path, runs_override=[run_name])


def test_analyze_rejects_non_gibbs_optimizer_kind(tmp_path: Path) -> None:
    run_name = "sample_pt"
    config_path = _write_config(tmp_path, run_name)
    _write_minimal_run(tmp_path, run_name, optimizer_kind="pt")
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="optimizer kind.*gibbs_anneal"):
        run_analyze(cfg, config_path, runs_override=[run_name])
