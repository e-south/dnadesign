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
    elites_hits_path,
    elites_path,
    manifest_path,
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
            "sequence": ["ACGTACGTACGT", "TGCATGCATGCA", "AACCGGTTAACC"],
            "canonical_sequence": ["ACGTACGTACGT", "TGCATGCATGCA", "AACCGGTTAACC"],
            **baseline_scores,
        }
    )
    baseline_path = random_baseline_path(run_dir)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(baseline_path, engine="fastparquet")

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

    hits_rows = []
    for idx, tf in enumerate(tf_names):
        hits_rows.append(
            {
                "elite_id": "elite-1",
                "tf": tf,
                "best_start": idx * 4,
                "best_strand": "+",
                "best_core_seq": "ACGT",
                "pwm_width": 4,
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
    diag = next(entry for entry in plot_manifest.get("plots", []) if entry.get("key") == "diag_panel")
    assert diag.get("generated") is False
    assert "Trace not available" in (diag.get("skip_reason") or "")


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
        sample=_sample_block(save_trace=False, top_k=1),
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
        top_k=1,
        draws=2,
        tune=1,
    )

    cfg = load_config(config_path)
    analysis_runs = run_analyze(cfg, config_path)
    assert analysis_runs
    analysis_dir = analysis_runs[0]
    used_payload = yaml.safe_load((analysis_dir / "analysis_used.yaml").read_text())
    assert used_payload["analysis"]["pairwise"] == "auto"
    assert (analysis_dir / "plot__overlap__panel.png").exists()


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
