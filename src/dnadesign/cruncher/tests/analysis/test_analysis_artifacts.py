"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_analysis_artifacts.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.cruncher.app.analyze_workflow import run_analyze
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_path,
    manifest_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.hashing import sha256_path


def _sample_block(*, save_trace: bool, top_k: int, draws: int = 2, tune: int = 1) -> dict:
    return {
        "mode": "sample",
        "rng": {"seed": 7, "deterministic": True},
        "budget": {"draws": draws, "tune": tune, "restarts": 1},
        "early_stop": {"enabled": True, "patience": 10, "min_delta": 0.01},
        "init": {"kind": "random", "length": 12, "pad_with": "background"},
        "objective": {
            "bidirectional": True,
            "score_scale": "normalized-llr",
            "scoring": {"pwm_pseudocounts": 0.1, "log_odds_clip": None},
        },
        "elites": {"k": top_k, "filters": {"pwm_sum_min": 0.0}},
        "moves": {
            "profile": "balanced",
            "overrides": {
                "block_len_range": [2, 2],
                "multi_k_range": [2, 2],
                "slide_max_shift": 1,
                "swap_len_range": [2, 2],
                "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
            },
        },
        "optimizer": {"name": "pt"},
        "optimizers": {"pt": {"beta_ladder": {"kind": "geometric", "betas": [1.0, 0.5]}}},
        "auto_opt": {"enabled": False},
        "output": {"trace": {"save": save_trace}, "save_sequences": True},
        "ui": {"progress_bar": False, "progress_every": 0},
    }


def _make_sample_run_dir(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / "results" / "sample" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_analyze_creates_analysis_run_and_manifest_updates(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=True, top_k=2),
            "analysis": {
                "runs": ["sample_test"],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_test")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    # minimal config_used.yaml with pwms_info
    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    # sequences.parquet
    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    # trace.nc
    import arviz as az

    idata = az.from_dict(posterior={"score": np.random.randn(1, 4)})
    trace_file = trace_path(run_dir)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, trace_file)

    # elites parquet
    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    # run_manifest.json
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    run_dir_str = str(run_dir.resolve())
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "{run_dir_str}",
  "config_path": "{config_path.resolve()}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "analysis_used.yaml").exists()
    assert (analysis_dir / "summary.json").exists()
    assert (analysis_dir / "manifest.json").exists()
    assert (analysis_dir / "manifest.json").exists()
    assert (analysis_dir / "score_summary.parquet").exists()
    assert (analysis_dir / "joint_metrics.parquet").exists()
    assert (analysis_dir / "diagnostics.json").exists()

    table_manifest = json.loads((analysis_dir / "table_manifest.json").read_text())
    keys = {entry.get("key") for entry in table_manifest.get("tables", [])}
    assert "joint_metrics" in keys
    assert "diagnostics" in keys
    assert "per_pwm" not in keys

    manifest = yaml.safe_load(manifest_path(run_dir).read_text())
    artifacts = manifest.get("artifacts", [])
    assert artifacts

    summary_before = json.loads((analysis_dir / "summary.json").read_text())
    analysis_runs_repeat = run_analyze(cfg, config_path)
    assert analysis_runs_repeat
    summary_after = json.loads((analysis_dir / "summary.json").read_text())
    assert summary_before.get("analysis_id") == summary_after.get("analysis_id")


def test_analyze_defaults_to_latest_when_runs_empty(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=True, top_k=2),
            "analysis": {
                "runs": [],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_latest")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    import arviz as az

    idata = az.from_dict(posterior={"score": np.random.randn(1, 4)})
    trace_file = trace_path(run_dir)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, trace_file)

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    run_dir_str = str(run_dir.resolve())
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "{run_dir_str}",
  "config_path": "{config_path.resolve()}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs


def test_analyze_pairgrid_plot(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR", "fur"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=2),
            "analysis": {
                "runs": ["sample_pairgrid"],
                "extra_plots": True,
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "pairgrid": True,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_pairgrid")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {
                "lexA": {"pwm_matrix": pwm_matrix},
                "cpxR": {"pwm_matrix": pwm_matrix},
                "fur": {"pwm_matrix": pwm_matrix},
            },
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
            "score_fur": [0.7, 0.95],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
            "score_fur": [0.7],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_pairgrid",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "score__pairgrid.png").exists()


def test_analyze_pairgrid_single_tf(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_pairgrid_single"],
                "extra_plots": True,
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "pairgrid": True,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_pairgrid_single")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {
                "lexA": {"pwm_matrix": pwm_matrix},
            },
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [1.0],
            "score_lexA": [1.0],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_pairgrid_single",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "score__pairgrid.png").exists()


def test_analyze_without_trace_when_no_trace_plots(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_no_trace"],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": True,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_no_trace")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    lock_sha = sha256_path(lock_path)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_no_trace",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "analysis_used.yaml").exists()
    assert (analysis_dir / "summary.json").exists()
    assert (analysis_dir / "score_summary.parquet").exists()
    assert (analysis_dir / "joint_metrics.parquet").exists()


def test_analyze_missing_trace_with_mcmc_diagnostics(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_no_trace_diag"],
                "mcmc_diagnostics": True,
                "plots": {
                    "trace": True,
                    "autocorr": True,
                    "convergence": True,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_no_trace_diag")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_no_trace_diag",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    with pytest.raises(FileNotFoundError, match="trace.nc"):
        run_analyze(cfg, config_path)


def test_analyze_auto_tf_pair_selection(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_autopick"],
                "plots": {
                    "pair_pwm": True,
                    "parallel_pwm": False,
                    "scatter_pwm": False,
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_autopick")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0],
            "draw": [0, 1, 2, 3],
            "phase": ["draw", "draw", "draw", "draw"],
            "sequence": ["ACGTACGTACGT"] * 4,
            "score_lexA": [0.0, 0.1, 0.0, 0.1],
            "score_cpxR": [0.4, 0.3, 0.4, 0.3],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [0.1],
            "score_cpxR": [0.3],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_autopick",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    used_payload = yaml.safe_load((analysis_dir / "analysis_used.yaml").read_text())
    assert "analysis_autopicks" in used_payload
    assert used_payload["analysis_autopicks"]["tf_pair"] == ["lexA", "cpxR"]
    assert (analysis_dir / "pwm__pair_scores.png").exists()


def test_analyze_prunes_stale_analysis_artifacts_when_not_archiving(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_stale"],
                "tf_pair": ["lexA", "cpxR"],
                "archive": False,
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_stale")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    analysis_root = run_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "summary.json").write_text(json.dumps({"analysis_id": "old-analysis"}))
    stale_plot = analysis_root / "score__box.png"
    stale_plot.write_text("stale")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_stale",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": ["analysis/score__box.png"]
}}"""
    )

    cfg = load_config(config_path)
    run_analyze(cfg, config_path)

    manifest = yaml.safe_load(manifest_path(run_dir).read_text())
    artifacts = manifest.get("artifacts", [])
    assert "analysis/score__box.png" not in {a.get("path") if isinstance(a, dict) else a for a in artifacts}


def test_analyze_fails_on_lockfile_mismatch(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(save_trace=False, top_k=1),
            "analysis": {
                "runs": ["sample_bad_lock"],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": False,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": True,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "llr",
                "scatter_style": "edges",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_bad_lock")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")

    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {"lexA": {"pwm_matrix": pwm_matrix}, "cpxR": {"pwm_matrix": pwm_matrix}},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
            "score_cpxR": [0.8, 0.9],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_bad_lock",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "badsha",
  "artifacts": []
}}"""
    )

    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="Lockfile checksum mismatch"):
        run_analyze(cfg, config_path)
