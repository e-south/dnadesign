"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_analysis_artifacts.py

Validates analysis artifact contracts, plot manifests, and failure modes.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.cruncher.app import analyze_workflow
from dnadesign.cruncher.app.analyze_workflow import _load_elites_meta, run_analyze
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_path,
    elites_yaml_path,
    manifest_path,
    random_baseline_hits_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.hashing import sha256_path


def _sample_block(*, save_trace: bool, top_k: int, draws: int = 2, tune: int = 1) -> dict:
    return {
        "seed": 7,
        "sequence_length": 12,
        "budget": {"tune": tune, "draws": draws},
        "elites": {"k": top_k},
        "output": {"save_sequences": True, "save_trace": save_trace},
    }


def test_load_elites_meta_requires_file(tmp_path: Path) -> None:
    missing = tmp_path / "elites.yaml"
    with pytest.raises(FileNotFoundError, match="Missing elites metadata YAML"):
        _load_elites_meta(missing)


def test_load_elites_meta_requires_mapping(tmp_path: Path) -> None:
    invalid = tmp_path / "elites.yaml"
    invalid.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        _load_elites_meta(invalid)


def _make_sample_run_dir(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / "results" / "sample" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _base_config(
    *,
    catalog_root: Path,
    regulator_sets: list[list[str]],
    sample: dict,
    analysis: dict,
) -> dict:
    return {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": regulator_sets},
            "catalog": {"root": str(catalog_root), "source_preference": [], "pwm_source": "matrix"},
            "sample": sample,
            "analysis": analysis,
        }
    }


def _write_basic_run_artifacts(
    *,
    run_dir: Path,
    config: dict,
    config_path: Path,
    lock_path: Path,
    lock_sha: str,
    tf_names: list[str],
    include_trace: bool,
    top_k: int,
    draws: int = 2,
    tune: int = 1,
) -> None:
    pwm_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    config_used = {
        "cruncher": {
            **config["cruncher"],
            "pwms_info": {tf: {"pwm_matrix": pwm_matrix} for tf in tf_names},
            "active_regulator_set": {"tfs": tf_names},
        }
    }
    cfg_path = config_used_path(run_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_used))

    scores = {f"score_{tf}": [1.0 + 0.05 * idx, 0.9 + 0.05 * idx] for idx, tf in enumerate(tf_names)}
    seq_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "draw": [0, 1],
            "phase": ["draw", "draw"],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA"],
            **scores,
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_parquet(seq_path, engine="fastparquet")

    baseline_scores = {
        f"score_{tf}": [0.1 + 0.02 * idx, 0.2 + 0.02 * idx, 0.15 + 0.02 * idx] for idx, tf in enumerate(tf_names)
    }
    baseline_df = pd.DataFrame(
        {
            "baseline_id": [0, 1, 2],
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA", "AACCGGTTAACC"],
            "canonical_sequence": ["ACGTACGTACGT", "TGCATGCATGCA", "AACCGGTTAACC"],
            "baseline_seed": 7,
            "baseline_n": 3,
            "seed": 7,
            "n_samples": 3,
            "sequence_length": 12,
            "length": 12,
            "score_scale": "normalized-llr",
            "bidirectional": True,
            "bg_model": "uniform",
            "bg_a": 0.25,
            "bg_c": 0.25,
            "bg_g": 0.25,
            "bg_t": 0.25,
            **baseline_scores,
        }
    )
    baseline_path = random_baseline_path(run_dir)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(baseline_path, engine="fastparquet")

    baseline_hits_rows = []
    for baseline_id, seq in enumerate(baseline_df["sequence"]):
        for tf_idx, tf_name in enumerate(tf_names):
            baseline_hits_rows.append(
                {
                    "baseline_id": baseline_id,
                    "tf": tf_name,
                    "best_start": 0,
                    "best_core_offset": 0,
                    "best_strand": "+",
                    "best_window_seq": seq[:4],
                    "best_core_seq": seq[:4],
                    "best_score_raw": 1.0 + 0.1 * tf_idx,
                    "best_score_scaled": 1.0 + 0.1 * tf_idx,
                    "best_score_norm": 0.9,
                    "tiebreak_rule": "leftmost",
                    "pwm_ref": "demo",
                    "pwm_hash": "hash",
                    "pwm_width": 4,
                    "core_width": 4,
                    "core_def_hash": "core",
                }
            )
    baseline_hits_df = pd.DataFrame(baseline_hits_rows)
    baseline_hits_path = random_baseline_hits_path(run_dir)
    baseline_hits_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_hits_df.to_parquet(baseline_hits_path, engine="fastparquet")

    elite_scores = {f"score_{tf}": [1.0 + 0.05 * idx] for idx, tf in enumerate(tf_names)}
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1"],
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "norm_sum": [float(len(tf_names))],
            **elite_scores,
        }
    )
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")
    elites_yaml_path(run_dir).write_text(
        yaml.safe_dump(
            {
                "n_elites": int(len(elites_df)),
                "mmr_alpha": 0.85,
                "pool_size": max(top_k, len(elites_df)),
            }
        )
    )

    hits_rows = []
    for idx, tf in enumerate(tf_names):
        hits_rows.append(
            {
                "elite_id": "elite-1",
                "tf": tf,
                "rank": 1,
                "chain": 0,
                "draw_idx": 0,
                "best_start": idx * 4,
                "best_core_offset": idx * 4,
                "best_strand": "+",
                "best_window_seq": "ACGT",
                "best_core_seq": "ACGT",
                "best_score_raw": 1.0,
                "best_score_scaled": 1.0,
                "best_score_norm": 1.0,
                "tiebreak_rule": "max_leftmost_plus",
                "pwm_ref": f"demo:{tf}",
                "pwm_hash": "sha256",
                "pwm_width": 4,
                "core_width": 4,
                "core_def_hash": "corehash",
            }
        )
    hits_df = pd.DataFrame(hits_rows)
    hits_path = elites_hits_path(run_dir)
    hits_path.parent.mkdir(parents=True, exist_ok=True)
    hits_df.to_parquet(hits_path, engine="fastparquet")

    if include_trace:
        import arviz as az

        idata = az.from_dict(posterior={"score": np.random.randn(1, 4)})
        trace_file = trace_path(run_dir)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        az.to_netcdf(idata, trace_file)

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
  "draws": {draws},
  "adapt_sweeps": {tune},
  "top_k": {top_k},
  "objective": {{"bidirectional": true, "score_scale": "normalized-llr"}},
  "optimizer_stats": {{"beta_ladder_final": [1.0]}},
  "artifacts": []
}}"""
    )


def test_analyze_creates_analysis_run_and_manifest_updates(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=True, top_k=2),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_test"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_test")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=True,
        top_k=2,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "analysis_used.yaml").exists()
    assert (analysis_dir / "summary.json").exists()
    assert (analysis_dir / "manifest.json").exists()
    assert (analysis_dir / "manifest.json").exists()
    assert (analysis_dir / "table__scores__summary.parquet").exists()
    assert (analysis_dir / "table__metrics__joint.parquet").exists()
    assert (analysis_dir / "table__diagnostics__summary.json").exists()
    assert (analysis_dir / "table__opt__trajectory_points.parquet").exists()

    table_manifest = json.loads((analysis_dir / "table_manifest.json").read_text())
    keys = {entry.get("key") for entry in table_manifest.get("tables", [])}
    assert "metrics_joint" in keys
    assert "diagnostics_summary" in keys

    manifest = yaml.safe_load(manifest_path(run_dir).read_text())
    artifacts = manifest.get("artifacts", [])
    assert artifacts

    analysis_runs_repeat = run_analyze(cfg, config_path)
    assert analysis_runs_repeat
    summary_after = json.loads((analysis_dir / "summary.json").read_text())
    assert summary_after.get("analysis_id")


def test_analyze_fails_fast_when_run_lock_exists(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_locked"],
            "pairwise": "off",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 1000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_locked")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
    )

    analyze_lock = run_dir / ".analysis_tmp"
    analyze_lock.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    with pytest.raises(RuntimeError, match="Analyze already in progress"):
        run_analyze(cfg, config_path)


def test_analyze_without_trace_does_not_import_arviz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_no_trace"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_no_trace")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    real_import = builtins.__import__

    def _guarded_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "arviz":
            raise AssertionError("run_analyze imported arviz despite trace being absent.")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs


def test_analyze_fails_on_hits_schema_mismatch(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_bad_hits"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_bad_hits")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    hits_path = elites_hits_path(run_dir)
    hits_df = pd.read_parquet(hits_path, engine="fastparquet")
    hits_df = hits_df.drop(columns=["best_score_raw"])
    hits_df.to_parquet(hits_path, engine="fastparquet")

    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="elites_hits.parquet missing required columns"):
        run_analyze(cfg, config_path)


def test_analyze_fails_when_random_baseline_hits_missing(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_missing_baseline_hits"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_missing_baseline_hits")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    random_baseline_hits_path(run_dir).unlink()

    cfg = load_config(config_path)
    with pytest.raises(FileNotFoundError, match="random baseline hits"):
        run_analyze(cfg, config_path)


def test_analyze_fails_when_baseline_seed_invalid(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_invalid_baseline_seed"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_invalid_baseline_seed")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )
    baseline_path = random_baseline_path(run_dir)
    baseline_df = pd.read_parquet(baseline_path, engine="fastparquet")
    baseline_df["baseline_seed"] = baseline_df["baseline_seed"].astype(float)
    baseline_df.loc[0, "baseline_seed"] = np.nan
    baseline_df.to_parquet(baseline_path, engine="fastparquet")

    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="baseline_seed"):
        run_analyze(cfg, config_path)


def test_analyze_restores_previous_analysis_when_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_restore_previous"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_restore_previous")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(json.dumps({"analysis_id": "old-analysis"}))
    (analysis_dir / "old_marker.txt").write_text("preserve")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure in analyze")

    monkeypatch.setattr(analyze_workflow, "build_trajectory_points", _boom)
    cfg = load_config(config_path)

    with pytest.raises(RuntimeError, match="synthetic failure in analyze"):
        run_analyze(cfg, config_path)

    restored_summary = json.loads((analysis_dir / "summary.json").read_text())
    assert restored_summary["analysis_id"] == "old-analysis"
    assert (analysis_dir / "old_marker.txt").read_text() == "preserve"
    assert not (run_dir / ".analysis_tmp").exists()
    assert not list(run_dir.glob(".analysis_prev_*"))


def test_analyze_defaults_to_latest_when_runs_empty(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=True, top_k=2),
        analysis={
            "run_selector": "latest",
            "runs": [],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_latest")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=True,
        top_k=2,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs


def test_analyze_opt_trajectory_multi_tf(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR", "fur"]],
        sample=_sample_block(save_trace=False, top_k=2),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_pairgrid"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_pairgrid")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR", "fur"],
        include_trace=False,
        top_k=2,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "plot__run__summary.png").exists()
    assert (analysis_dir / "plot__opt__trajectory.png").exists()


def test_analyze_opt_trajectory_single_tf(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_pairgrid_single"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_pairgrid_single")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs

    analysis_dir = analysis_runs[0]
    assert (analysis_dir / "plot__run__summary.png").exists()
    assert (analysis_dir / "plot__opt__trajectory.png").exists()


def test_analyze_without_trace_when_no_trace_plots(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_no_trace"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_no_trace")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    plot_manifest = json.loads((analysis_dir / "plot_manifest.json").read_text())
    health = next(entry for entry in plot_manifest.get("plots", []) if entry.get("key") == "health_panel")
    assert health.get("generated") is False
    assert (health.get("skip_reason") or "") == "trace disabled"


def test_analyze_missing_trace_with_mcmc_diagnostics(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_no_trace_diag"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_no_trace_diag")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    diagnostics = json.loads((analysis_dir / "table__diagnostics__summary.json").read_text())
    trace_metrics = diagnostics.get("metrics", {}).get("trace", {})
    assert trace_metrics.get("rhat") is None


def test_analyze_auto_tf_pair_selection(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=2),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_autopick"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_autopick")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=2,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    used_payload = yaml.safe_load((analysis_dir / "analysis_used.yaml").read_text())
    assert used_payload["analysis"]["pairwise"] == "auto"
    assert used_payload["tf_pair_selected"] == ["lexA", "cpxR"]

    plot_manifest = json.loads((analysis_dir / "plot_manifest.json").read_text())
    overlap = next(entry for entry in plot_manifest.get("plots", []) if entry.get("key") == "overlap_panel")
    assert overlap.get("generated") is False
    assert "elites_count < 2" in str(overlap.get("skip_reason"))


def test_analyze_prunes_stale_analysis_artifacts_when_not_archiving(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_stale"],
            "pairwise": ["lexA", "cpxR"],
            "archive": False,
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_stale")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    analysis_root = run_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "summary.json").write_text(json.dumps({"analysis_id": "old-analysis"}))
    stale_plot = analysis_root / "score__box.png"
    stale_plot.write_text("stale")

    manifest_file = manifest_path(run_dir)
    manifest_payload = json.loads(manifest_file.read_text())
    manifest_payload["artifacts"] = ["analysis/score__box.png"]
    manifest_file.write_text(json.dumps(manifest_payload))

    cfg = load_config(config_path)
    run_analyze(cfg, config_path)

    manifest = yaml.safe_load(manifest_path(run_dir).read_text())
    artifacts = manifest.get("artifacts", [])
    assert "analysis/score__box.png" not in {a.get("path") if isinstance(a, dict) else a for a in artifacts}


def test_analyze_fails_on_lockfile_mismatch(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_bad_lock"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_bad_lock")

    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    manifest_file = manifest_path(run_dir)
    manifest_payload = json.loads(manifest_file.read_text())
    manifest_payload["lockfile_sha256"] = "badsha"
    manifest_file.write_text(json.dumps(manifest_payload))

    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="Lockfile checksum mismatch"):
        run_analyze(cfg, config_path)


def test_analyze_plot_manifest_single_tf_overlap_skip_and_trace_skip(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_single_tf_skip"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_single_tf_skip")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]

    plot_manifest = json.loads((analysis_dir / "plot_manifest.json").read_text())
    plots_by_key = {entry.get("key"): entry for entry in plot_manifest.get("plots", [])}
    assert set(plots_by_key) == {
        "run_summary",
        "opt_trajectory",
        "elites_nn_distance",
        "overlap_panel",
        "health_panel",
    }
    assert plots_by_key["run_summary"]["generated"] is True
    assert plots_by_key["opt_trajectory"]["generated"] is True
    assert plots_by_key["elites_nn_distance"]["generated"] is True
    assert plots_by_key["overlap_panel"]["generated"] is False
    assert "n_tf < 2" in str(plots_by_key["overlap_panel"].get("skip_reason"))
    assert plots_by_key["health_panel"]["generated"] is False
    assert plots_by_key["health_panel"].get("skip_reason") == "trace disabled"


def test_analyze_plot_manifest_skips_overlap_when_elites_count_lt_two(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_overlap_skip"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_overlap_skip")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]

    plot_manifest = json.loads((analysis_dir / "plot_manifest.json").read_text())
    plots_by_key = {entry.get("key"): entry for entry in plot_manifest.get("plots", [])}
    assert plots_by_key["overlap_panel"]["generated"] is False
    assert "elites_count < 2" in str(plots_by_key["overlap_panel"].get("skip_reason"))


def test_analyze_fails_when_required_plot_generation_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=2),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_plot_error"],
            "pairwise": "auto",
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_plot_error")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)

    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=2,
        draws=2,
        tune=1,
    )

    def _explode(*args, **kwargs) -> None:
        raise RuntimeError("intentional run summary error")

    monkeypatch.setattr("dnadesign.cruncher.analysis.plots.run_summary.plot_run_summary", _explode)

    cfg = load_config(config_path)
    with pytest.raises(RuntimeError, match="run summary error"):
        run_analyze(cfg, config_path)


def test_analyze_accepts_run_directory_path_override(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = _base_config(
        catalog_root=catalog_root,
        regulator_sets=[["lexA", "cpxR"]],
        sample=_sample_block(save_trace=False, top_k=1),
        analysis={
            "run_selector": "explicit",
            "runs": ["sample_path_override"],
            "pairwise": ["lexA", "cpxR"],
            "plot_format": "png",
            "plot_dpi": 72,
            "table_format": "parquet",
            "max_points": 2000,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = _make_sample_run_dir(tmp_path, "sample_path_override")
    lock_dir = tmp_path / ".cruncher" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "config.lock.json"
    lock_path.write_text("{}")
    lock_sha = sha256_path(lock_path)
    _write_basic_run_artifacts(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        lock_sha=lock_sha,
        tf_names=["lexA", "cpxR"],
        include_trace=False,
        top_k=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path, runs_override=[str(run_dir.resolve())])
    assert len(analysis_runs) == 1
    assert analysis_runs[0] == run_dir / "analysis"
