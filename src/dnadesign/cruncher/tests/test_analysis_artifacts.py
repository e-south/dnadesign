"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_analysis_artifacts.py

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

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.workflows.analyze_workflow import run_analyze


def test_analyze_creates_analysis_run_and_manifest_updates(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": True,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 2,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
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

    run_dir = tmp_path / "results" / "sample_test"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

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
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

    # trace.nc
    import arviz as az

    idata = az.from_dict(posterior={"score": np.random.randn(1, 4)})
    az.to_netcdf(idata, run_dir / "trace.nc")

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
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    # run_manifest.json
    (run_dir / "run_manifest.json").write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_test",
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
    assert (analysis_dir / "tables" / "score_summary.csv").exists()
    assert (analysis_dir / "tables" / "joint_metrics.csv").exists()

    table_manifest = json.loads((analysis_dir / "table_manifest.json").read_text())
    keys = {entry.get("key") for entry in table_manifest.get("tables", [])}
    assert "joint_metrics" in keys

    manifest = yaml.safe_load((run_dir / "run_manifest.json").read_text())
    artifacts = manifest.get("artifacts", [])
    assert artifacts


def test_analyze_pairgrid_plot(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR", "fur"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 2,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
            "analysis": {
                "runs": ["sample_pairgrid"],
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

    run_dir = tmp_path / "results" / "sample_pairgrid"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

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
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

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
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    (run_dir / "run_manifest.json").write_text(
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
    assert (analysis_dir / "plots" / "score__pairgrid.png").exists()


def test_analyze_pairgrid_single_tf(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
            "analysis": {
                "runs": ["sample_pairgrid_single"],
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

    run_dir = tmp_path / "results" / "sample_pairgrid_single"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            "score_lexA": [1.0, 1.2],
        }
    )
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [1.0],
            "score_lexA": [1.0],
        }
    )
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    (run_dir / "run_manifest.json").write_text(
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
    assert (analysis_dir / "plots" / "score__pairgrid.png").exists()


def test_analyze_without_trace_when_no_trace_plots(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
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

    run_dir = tmp_path / "results" / "sample_no_trace"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

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
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    lock_sha = sha256_path(lock_path)
    (run_dir / "run_manifest.json").write_text(
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
    assert (analysis_dir / "tables" / "score_summary.csv").exists()
    assert (analysis_dir / "tables" / "joint_metrics.csv").exists()


def test_analyze_prunes_stale_analysis_artifacts_when_not_archiving(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
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

    run_dir = tmp_path / "results" / "sample_stale"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

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
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    analysis_root = run_dir / "analysis"
    plots_dir = analysis_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (analysis_root / "summary.json").write_text(json.dumps({"analysis_id": "old-analysis"}))
    stale_plot = plots_dir / "score__box.png"
    stale_plot.write_text("stale")

    (run_dir / "run_manifest.json").write_text(
        f"""{{
  "stage": "sample",
  "run_dir": "sample_stale",
  "config_path": "{config_path}",
  "lockfile_path": "{lock_path.resolve()}",
  "lockfile_sha256": "{lock_sha}",
  "artifacts": ["analysis/plots/score__box.png"]
}}"""
    )

    cfg = load_config(config_path)
    run_analyze(cfg, config_path)

    manifest = yaml.safe_load((run_dir / "run_manifest.json").read_text())
    artifacts = manifest.get("artifacts", [])
    assert "analysis/plots/score__box.png" not in {a.get("path") if isinstance(a, dict) else a for a in artifacts}


def test_analyze_fails_on_lockfile_mismatch(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
                "source_preference": [],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
                "pwm_sum_threshold": 0.0,
            },
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

    run_dir = tmp_path / "results" / "sample_bad_lock"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used))

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
    seq_df.to_parquet(run_dir / "sequences.parquet", engine="fastparquet")

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [2.0],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_df.to_parquet(run_dir / "elites.parquet", engine="fastparquet")

    (run_dir / "run_manifest.json").write_text(
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
