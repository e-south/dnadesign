"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_analysis_validation.py

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
from pydantic import ValidationError

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.utils.run_layout import elites_path, manifest_path, sequences_path
from dnadesign.cruncher.workflows.analyze.per_pwm import gather_per_pwm_scores
from dnadesign.cruncher.workflows.analyze.plots.diagnostics import make_pair_idata
from dnadesign.cruncher.workflows.analyze.plots.scatter import _normalize_threshold_points, plot_scatter
from dnadesign.cruncher.workflows.analyze.plots.summary import write_elite_topk
from dnadesign.cruncher.workflows.analyze_workflow import _get_git_commit


def test_gather_per_pwm_scores_rejects_invalid_sequence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGTN"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    with pytest.raises(ValueError, match="invalid base"):
        gather_per_pwm_scores(
            run_dir,
            change_threshold=0.1,
            pwms=pwms,
            bidirectional=False,
            scale="llr",
            out_path=run_dir / "out.csv",
        )


def test_gather_per_pwm_scores_rejects_non_positive_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGT"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    with pytest.raises(ValueError, match="change_threshold must be > 0"):
        gather_per_pwm_scores(
            run_dir,
            change_threshold=0.0,
            pwms=pwms,
            bidirectional=False,
            scale="llr",
            out_path=run_dir / "out.csv",
        )


def test_gather_per_pwm_scores_single_draw_not_duplicated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGTACGT"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    out_path = run_dir / "out.csv"
    gather_per_pwm_scores(
        run_dir,
        change_threshold=0.1,
        pwms=pwms,
        bidirectional=False,
        scale="llr",
        out_path=out_path,
    )

    out_df = pd.read_csv(out_path)
    assert len(out_df) == 1
    assert out_df.loc[0, "chain"] == 0
    assert out_df.loc[0, "draw"] == 0


def test_make_pair_idata_requires_consistent_draws(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0, 0, 1],
            "draw": [0, 1, 0],
            "phase": ["draw", "draw", "draw"],
            "score_lexA": [1.0, 1.1, 0.9],
            "score_cpxR": [0.8, 0.85, 0.75],
        }
    )
    seq_path = sequences_path(sample_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    with pytest.raises(ValueError, match="Inconsistent draws"):
        make_pair_idata(sample_dir, ("lexA", "cpxR"))


def test_get_git_commit_handles_gitdir_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "nested" / "dir"
    nested.mkdir(parents=True)

    git_dir = tmp_path / "actual_git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text("abc123deadbeef\n")

    (repo / ".git").write_text(f"gitdir: {git_dir}\n")

    assert _get_git_commit(nested) == "abc123deadbeef"


def test_write_elite_topk_requires_sequence(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "rank": [1],
            "score_lexA": [1.0],
        }
    )
    out_path = tmp_path / "topk.csv"
    with pytest.raises(ValueError, match="sequence"):
        write_elite_topk(df, ["lexA"], out_path, top_k=1)


def test_scatter_thresholds_requires_llr(tmp_path: Path) -> None:
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
                "runs": ["sample_thresholds"],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": True,
                    "pair_pwm": False,
                    "parallel_pwm": False,
                    "score_hist": False,
                    "score_box": False,
                    "correlation_heatmap": False,
                    "parallel_coords": False,
                },
                "scatter_scale": "z",
                "scatter_style": "thresholds",
                "subsampling_epsilon": 10.0,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValidationError, match="scatter_style='thresholds' requires scatter_scale='llr'"):
        load_config(config_path)


def test_plot_scatter_requires_score_columns(tmp_path: Path) -> None:
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
                "runs": [],
                "tf_pair": ["lexA", "cpxR"],
                "plots": {
                    "trace": False,
                    "autocorr": False,
                    "convergence": False,
                    "scatter_pwm": True,
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
    cfg = load_config(config_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"sequence_length": 12}))

    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "score_lexA": [1.0],
            "score_cpxR": [0.8],
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    per_pwm_path = run_dir / "per_pwm.csv"
    pd.DataFrame({"chain": [0], "draw": [0]}).to_csv(per_pwm_path, index=False)

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix), "cpxR": PWM(name="cpxR", matrix=pwm_matrix)}

    with pytest.raises(ValueError, match="per-PWM table missing required score columns"):
        plot_scatter(
            run_dir,
            pwms,
            cfg,
            tf_pair=("lexA", "cpxR"),
            per_pwm_path=per_pwm_path,
            out_dir=run_dir / "plots",
            bidirectional=True,
            pwm_sum_threshold=0.0,
            annotation="",
        )


def test_normalize_threshold_points_scales_values() -> None:
    points = [(2.0, 6.0, "a"), (1.0, 3.0, "b")]
    normalized = _normalize_threshold_points(points, cons_x=2.0, cons_y=3.0)
    assert normalized == [(1.0, 2.0, "a"), (0.5, 1.0, "b")]
