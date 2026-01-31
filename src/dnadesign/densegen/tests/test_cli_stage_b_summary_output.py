"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_stage_b_summary_output.py

CLI tests for Stage-B build-libraries output summary (no per-library rows).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact
from dnadesign.densegen.src.core.pipeline import default_deps


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.8",
                    "run": {"id": "demo", "root": "."},
                    "inputs": [
                        {
                            "name": "demo_input",
                            "type": "binding_sites",
                            "path": "inputs.csv",
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
                        "sampling": {
                            "pool_strategy": "subsample",
                            "library_size": 2,
                            "subsample_over_length_budget_by": 0,
                            "library_sampling_strategy": "coverage_weighted",
                            "cover_all_regulators": True,
                            "unique_binding_sites": True,
                            "unique_binding_cores": True,
                        },
                        "plan": [{"name": "demo_plan", "quota": 1}],
                    },
                    "solver": {"backend": "CBC", "strategy": "iterate"},
                    "logging": {"log_dir": "outputs/logs", "level": "INFO"},
                    "postprocess": {"pad": {"mode": "off"}},
                }
            }
        )
    )


def test_stage_b_build_libraries_summary_output(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAAA\nTF2,CCCC\n")

    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    out_dir = tmp_path / "outputs" / "pools"
    outputs_root = tmp_path / "outputs"
    build_pool_artifact(
        cfg=cfg,
        cfg_path=cfg_path,
        deps=default_deps(),
        rng=np.random.default_rng(0),
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["stage-b", "build-libraries", "-c", str(cfg_path), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert "Stage-B libraries" in result.output
    header_lines = [line for line in result.output.splitlines() if "input" in line and "plan" in line]
    assert header_lines
    for line in header_lines:
        assert "build" not in line.lower()
