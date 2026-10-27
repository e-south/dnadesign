"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py

Cluster runtime import-boundary tests for USR public APIs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign import cluster


def _runtime_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cluster_runtime_does_not_import_usr_internal_paths() -> None:
    disallowed = "dnadesign.usr.src."
    violations: list[str] = []
    for path in sorted(_runtime_root().rglob("*.py")):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if disallowed in text:
            violations.append(str(path))
    assert not violations, f"Found disallowed USR internal imports in cluster runtime: {violations}"


def test_cluster_public_surface_exposes_workspace_execution_api() -> None:
    assert callable(cluster.run_fit)
    assert callable(cluster.run_umap)
    assert callable(cluster.run_analyze)
    assert callable(cluster.run_sweep)
    assert callable(cluster.list_runs)
    assert callable(cluster.run_fit_workspace)
    assert callable(cluster.run_umap_workspace)
    assert callable(cluster.run_analyze_workspace)
    assert callable(cluster.run_sweep_workspace)
    assert callable(cluster.list_workspace_runs)


def test_cluster_public_root_import_stays_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    probe = (
        "import json, sys; "
        "import dnadesign.cluster as _cluster; "
        "print(json.dumps({"
        "'execution': any(name.startswith('dnadesign.cluster.src.execution') for name in sys.modules), "
        "'pandas': any(name.startswith('pandas') for name in sys.modules), "
        "'numpy': any(name.startswith('numpy') for name in sys.modules)"
        "}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"execution": false, "numpy": false, "pandas": false}'


def test_cluster_public_api_does_not_shell_back_into_cli() -> None:
    api_path = _runtime_root() / "api.py"
    text = api_path.read_text(encoding="utf-8")

    assert "CliRunner" not in text
    assert "typer.testing" not in text
    assert ".src.cli.app" not in text


def _register_fast_public_api_method() -> str:
    method_id = "toy_public_api"

    def resolve_params(preset=None, raw_params=None):
        raw = dict(raw_params or {})
        return {
            "neighbors": int(raw.get("neighbors", 2)),
            "resolution": float(raw.get("resolution", 0.2)),
        }

    def fit(X, **_):
        return np.arange(X.shape[0], dtype=int) % 2

    def sweep(X, *, method_params, res_min, res_max, step, seeds, out_dir):
        (out_dir / "toy_sweep.json").write_text(
            json.dumps(
                {
                    "rows": int(X.shape[0]),
                    "params": dict(method_params),
                    "res_min": float(res_min),
                    "res_max": float(res_max),
                    "step": float(step),
                    "seeds": [int(seed) for seed in seeds],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    cluster.register_method(
        cluster.ClusteringMethod(
            method_id=method_id,
            display_name="Toy public API",
            default_run_prefix=method_id,
            fit_param_names=frozenset({"neighbors", "resolution"}),
            resolve_fit_params=resolve_params,
            fit=fit,
            slug_params=lambda params: {
                "n": int(params["neighbors"]),
                "r": float(params["resolution"]),
            },
            operations={"resolution_sweep": sweep},
        ),
        replace=True,
    )
    return method_id


def test_cluster_public_api_executes_workspace_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.umap.compute as umap_compute_mod

    method_id = _register_fast_public_api_method()
    monkeypatch.setattr(
        umap_compute_mod,
        "compute",
        lambda X, neighbors, min_dist, metric, seed: np.column_stack(
            (np.arange(X.shape[0], dtype=np.float32), np.arange(X.shape[0], dtype=np.float32) + 1.0)
        ),
    )

    records_path = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "x1": [0.0, 0.1, 10.0, 10.1, 20.0, 20.2],
            "x2": [0.1, 0.2, 10.2, 10.3, 20.1, 20.3],
            "source": ["A", "A", "B", "B", "C", "C"],
        }
    ).to_csv(records_path, index=False)

    workspace_dir = cluster.init_workspace(workspace_id="api_ws", root=tmp_path)
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: 1
input:
  file: "../records.csv"

fit:
  name: "api_ws"
  key_col: "id"
  x_cols: "x1,x2"
  method: "{method_id}"
  method_params:
    neighbors: 2
    resolution: 0.2
  write: true
  allow_overwrite: true
  inplace: true

umap:
  name: "api_ws"
  key_col: "id"
  x_cols: "x1,x2"
  neighbors: 2
  min_dist: 0.1
  metric: "euclidean"
  random_state: 42
  attach_coords: true
  write: true
  allow_overwrite: true
  inplace: true
  plot:
    enabled: false
    dims: [4, 4]
    alpha: 0.7
    size: 10.0

analyze:
  cluster_col: "cluster__api_ws"
  group_by: "source"
  composition: true
  plots: false
""".format(method_id=method_id).strip()
        + "\n",
        encoding="utf-8",
    )

    fit_result = cluster.run_fit_workspace(workspace_dir)
    umap_result = cluster.run_umap_workspace(workspace_dir)
    analyze_result = cluster.run_analyze_workspace(workspace_dir)
    sweep_result = cluster.run_sweep_workspace(
        workspace_dir,
        overrides={
            "method": method_id,
            "method_params": {"neighbors": 2},
            "res_min": 0.1,
            "res_max": 0.1,
            "step": 0.1,
            "seeds": "1",
        },
    )

    runs = cluster.list_workspace_runs(workspace_dir)

    assert fit_result.artifact_path.is_dir()
    assert umap_result.artifact_path.is_dir()
    assert analyze_result.artifact_path.is_dir()
    assert sweep_result.artifact_path.is_dir()
    assert (sweep_result.artifact_path / "toy_sweep.json").is_file()
    assert {"fit", "umap", "analysis", "sweep"} <= set(runs["kind"].tolist())


def test_cluster_public_api_executes_ad_hoc_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.umap.compute as umap_compute_mod

    method_id = _register_fast_public_api_method()
    monkeypatch.setattr(
        umap_compute_mod,
        "compute",
        lambda X, neighbors, min_dist, metric, seed: np.column_stack(
            (np.arange(X.shape[0], dtype=np.float32), np.arange(X.shape[0], dtype=np.float32) + 1.0)
        ),
    )

    records_path = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "x1": [0.0, 0.1, 10.0, 10.1, 20.0, 20.2],
            "x2": [0.1, 0.2, 10.2, 10.3, 20.1, 20.3],
            "source": ["A", "A", "B", "B", "C", "C"],
        }
    ).to_csv(records_path, index=False)
    results_root = tmp_path / "results"

    fit_result = cluster.run_fit(
        results_root=results_root,
        file=records_path,
        name="adhoc_ws",
        key_col="id",
        x_cols=("x1", "x2"),
        method=method_id,
        method_params={"neighbors": 2, "resolution": 0.2},
        write=True,
        allow_overwrite=True,
        inplace=True,
    )
    umap_result = cluster.run_umap(
        results_root=results_root,
        file=records_path,
        name="adhoc_ws",
        key_col="id",
        x_cols=("x1", "x2"),
        neighbors=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        attach_coords=True,
        write=True,
        allow_overwrite=True,
        inplace=True,
        render_plots=False,
        plot={"dims": [4, 4], "alpha": 0.7, "size": 10.0},
    )
    analyze_result = cluster.run_analyze(
        results_root=results_root,
        file=records_path,
        cluster_col="cluster__adhoc_ws",
        group_by="source",
        composition=True,
        plots=False,
    )
    sweep_result = cluster.run_sweep(
        results_root=results_root,
        file=records_path,
        key_col="id",
        x_cols=("x1", "x2"),
        method=method_id,
        method_params={"neighbors": 2},
        res_min=0.1,
        res_max=0.1,
        step=0.1,
        seeds=(1,),
    )

    runs = cluster.list_runs(results_root)

    assert fit_result.artifact_path.is_dir()
    assert umap_result.artifact_path.is_dir()
    assert analyze_result.artifact_path.is_dir()
    assert sweep_result.artifact_path.is_dir()
    assert (sweep_result.artifact_path / "toy_sweep.json").is_file()
    assert {"fit", "umap", "analysis", "sweep"} <= set(runs["kind"].tolist())


def test_cluster_public_api_fit_is_quiet_without_cli_console(tmp_path: Path, capsys) -> None:
    records_path = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x1": [0.0, 0.1, 10.0, 10.1],
            "x2": [0.1, 0.2, 10.2, 10.3],
        }
    ).to_csv(records_path, index=False)
    results_root = tmp_path / "results"

    cluster.register_method(
        cluster.ClusteringMethod(
            method_id="toy_quiet",
            display_name="Toy quiet",
            default_run_prefix="toy_quiet",
            fit_param_names=frozenset(),
            resolve_fit_params=lambda preset=None, raw_params=None: {},
            fit=lambda X, **_: np.zeros(X.shape[0], dtype=int),
            slug_params=lambda params: {},
            operations={},
        )
    )

    result = cluster.run_fit(
        results_root=results_root,
        file=records_path,
        name="quiet_fit",
        key_col="id",
        x_cols=("x1", "x2"),
        method="toy_quiet",
        write=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert result.artifact_path.is_dir()


def test_cluster_public_api_umap_can_skip_plot_rendering(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.cluster.src.umap.compute as umap_compute_mod

    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x1": [0.0, 0.1, 10.0, 10.1],
            "x2": [0.1, 0.2, 10.2, 10.3],
        }
    ).to_parquet(records_path, index=False)
    monkeypatch.setattr(
        umap_compute_mod,
        "compute",
        lambda X, neighbors, min_dist, metric, seed: np.column_stack(
            (np.arange(X.shape[0], dtype=np.float32), np.arange(X.shape[0], dtype=np.float32) + 1.0)
        ),
    )

    result = cluster.run_umap(
        results_root=tmp_path / "results",
        file=records_path,
        name="public_noplot",
        key_col="id",
        x_cols=("x1", "x2"),
        neighbors=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        color_by=("numeric:x1",),
        render_plots=False,
    )

    assert result.artifact_path.is_dir()
    assert result.run_record is not None
    assert pd.isna(result.run_record["plot_paths"])
