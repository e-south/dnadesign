"""
Memory preflight workflow tests for latentdna.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.services import memory_service
from dnadesign.latentdna.src.views import reduce as reduce_module
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

_RUNNER = CliRunner()


def _write_parquet_source(path: Path) -> None:
    rows = [
        {"id": "r1", "subject_id": "s1", "embedding": [0.0, 1.0, 0.0]},
        {"id": "r2", "subject_id": "s2", "embedding": [1.0, 0.0, 2.0]},
        {"id": "r3", "subject_id": "s3", "embedding": [2.0, 1.0, 1.0]},
        {"id": "r4", "subject_id": "s4", "embedding": [3.0, 4.0, 0.0]},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(workspace_dir: Path, source_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "memory_preflight_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "demo": {
                        "kind": "parquet",
                        "path": source_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": []},
                "views": {
                    "z_demo": {
                        "source": "demo",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
                "recipes": {
                    "reduce_recipe": {
                        "steps": [
                            {
                                "id": "reduce_demo",
                                "op": "view.reduce",
                                "params": {
                                    "view": "z_demo",
                                    "run_id": "z_demo_pca",
                                    "dims": 2,
                                    "reduced_view_id": "z_demo_pc2",
                                },
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _prepare_workspace(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source_path = tmp_path / "inputs" / "demo.parquet"
    _write_parquet_source(source_path)
    _write_workspace_config(workspace_dir, source_path)
    materialize = _RUNNER.invoke(
        app,
        ["view", "materialize", "z_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize.exit_code == 0, materialize.stdout
    return workspace_dir


def test_view_reduce_memory_preflight_requires_explicit_override(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z_demo_pca",
            "--dims",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 20
    assert "--allow-memory-overage" in result.stdout


def test_view_reduce_memory_preflight_override_records_attention(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z_demo_pca",
            "--dims",
            "2",
            "--reduced-view-id",
            "z_demo_pc2",
            "--allow-memory-overage",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "attention"
    assert payload["warnings"]
    assert payload["metrics"]["memory_preflight"]["state"] == "blocked"

    reducer_manifest = json.loads(
        (workspace_dir / "outputs" / "reducers" / "z_demo_pca" / "manifest.json").read_text(encoding="utf-8")
    )
    assert reducer_manifest["status"] == "attention"
    assert reducer_manifest["warnings"]
    assert reducer_manifest["params"]["memory_preflight"]["state"] == "blocked"
    assert reducer_manifest["params"]["pca_method"] == "dense_svd"


def test_view_reduce_supports_randomized_svd_path(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    monkeypatch.setattr(reduce_module, "select_pca_method", lambda **_: "randomized_svd")

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z_demo_pca",
            "--dims",
            "2",
            "--reduced-view-id",
            "z_demo_pc2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    reducer_manifest = json.loads(
        (workspace_dir / "outputs" / "reducers" / "z_demo_pca" / "manifest.json").read_text(encoding="utf-8")
    )
    assert reducer_manifest["params"]["pca_method"] == "randomized_svd"
    assert (workspace_dir / "outputs" / "reduced_views" / "z_demo_pc2" / "matrix.npy").is_file()


def test_view_reduce_records_truncated_variance_ratio_against_total_variance(tmp_path: Path) -> None:
    workspace_dir = _prepare_workspace(tmp_path)

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z_demo_pca",
            "--dims",
            "2",
            "--reduced-view-id",
            "z_demo_pc2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    summary = json.loads((workspace_dir / "outputs" / "reducers" / "z_demo_pca" / "summary.json").read_text("utf-8"))
    ratios = [float(value) for value in summary["explained_variance_ratio"]]
    assert len(ratios) == 2
    assert 0.0 < sum(ratios) < 1.0


def test_reduce_preflight_uses_randomized_svd_estimate_for_large_views(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    context = load_workspace_config(workspace_dir)
    monkeypatch.setattr(memory_service, "_view_metadata", lambda *_, **__: (157164, 8192, "float32", 4))
    monkeypatch.setattr(memory_service, "_scope_rows", lambda *_, **__: (157164, []))
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 16 * 1024**3)

    preflight = memory_service.evaluate_reduce_preflight(
        context,
        view_id="z_demo",
        dims=32,
        sample_id=None,
        alignment_id=None,
        reduced_view_id="z_demo_pc32",
    )

    assert preflight.algorithm == "pca_randomized_svd"
    assert preflight.state == "ok"
    assert any("randomized-SVD PCA" in note for note in preflight.notes)


def test_recipe_run_progress_emits_memory_warning_events(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "recipe",
            "run",
            "reduce_recipe",
            "--workspace",
            workspace_dir.as_posix(),
            "--allow-memory-overage",
            "--progress",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    event_lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert any(line.get("event_type") == "warning" for line in event_lines)
    assert any(
        line.get("event_type") == "command_result" and line["result"]["status"] == "attention" for line in event_lines
    )

    run_id = next(
        line["result"]["run_id"]
        for line in event_lines
        if line.get("event_type") == "command_result" and isinstance(line.get("result"), dict)
    )
    events_path = workspace_dir / "outputs" / "runs" / run_id / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event["event_type"] == "warning" for event in events)
