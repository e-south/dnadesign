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


def _prepare_unmaterialized_workspace(tmp_path: Path) -> Path:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source_path = tmp_path / "inputs" / "demo.parquet"
    _write_parquet_source(source_path)
    _write_workspace_config(workspace_dir, source_path)
    return workspace_dir


def _append_materialize_recipe(workspace_dir: Path) -> None:
    config_path = workspace_dir / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.setdefault("recipes", {})["materialize_recipe"] = {
        "steps": [
            {
                "id": "materialize_demo",
                "op": "view.materialize",
                "params": {"view": "z_demo"},
            }
        ]
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_view_materialize_memory_preflight_requires_explicit_override(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_unmaterialized_workspace(tmp_path)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "materialize",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )

    assert result.exit_code == 20
    assert "--allow-memory-overage" in result.stdout
    staging_root = workspace_dir / "outputs" / "runs" / "_staging" / "views"
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_view_materialize_memory_preflight_override_records_attention(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_unmaterialized_workspace(tmp_path)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "view",
            "materialize",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--allow-memory-overage",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "attention"
    assert payload["warnings"]
    assert payload["metrics"]["memory_preflight"]["state"] == "blocked"

    manifest_path = workspace_dir / "outputs" / "views" / "z_demo" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "attention"
    assert manifest["warnings"]
    assert manifest["params"]["memory_preflight"]["state"] == "blocked"


def test_view_materialize_memory_preflight_accounts_for_resident_memmap_pages(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_unmaterialized_workspace(tmp_path)
    context = load_workspace_config(workspace_dir)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(
        memory_service,
        "inspect_source_schema",
        lambda resolved: {"row_count": 157164, "columns": ["id"]},
    )
    monkeypatch.setattr(memory_service, "_source_vector_dims", lambda resolved, vector_column: 8192)

    preflight = memory_service.evaluate_materialize_preflight(context, view_id="z_demo")

    expected_batch_bytes = 2048 * 8192 * 4
    expected_output_bytes = 157164 * 8192 * 4
    expected_peak = int(expected_output_bytes * 2.25) + (expected_batch_bytes * 2)
    assert preflight.estimated_peak_bytes == expected_peak
    assert preflight.state == "warning"
    assert "resident" in " ".join(preflight.notes)


def test_recipe_run_passes_memory_override_to_view_materialize_step(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_unmaterialized_workspace(tmp_path)
    _append_materialize_recipe(workspace_dir)
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)

    result = _RUNNER.invoke(
        app,
        [
            "recipe",
            "run",
            "materialize_recipe",
            "--workspace",
            workspace_dir.as_posix(),
            "--allow-memory-overage",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "attention"
    assert any(summary["status"] == "attention" for summary in payload["metrics"]["step_results"])


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


def test_projection_fit_memory_preflight_blocks_before_staging(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "demo_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z_demo",
            "--strategy",
            "all",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout

    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 1)
    result = _RUNNER.invoke(
        app,
        [
            "projection",
            "fit",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--sample",
            "demo_sample",
            "--run-id",
            "z_demo_umap",
            "--json",
        ],
    )

    assert result.exit_code == 20
    assert "--allow-memory-overage" in result.stdout
    staging_root = workspace_dir / "outputs" / "runs" / "_staging" / "projections"
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_projection_fit_memory_preflight_accounts_for_full_population_all_sample(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "demo_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z_demo",
            "--strategy",
            "all",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout

    monkeypatch.setattr(memory_service, "_view_metadata", lambda *args, **kwargs: (157164, 8192, "float32", 4))
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(memory_service, "_row_count", lambda *args, **kwargs: 157164)

    context = load_workspace_config(workspace_dir)
    preflight = memory_service.evaluate_projection_preflight(context, view_id="z_demo", sample_id="demo_sample")

    base_bytes = 157164 * 8192 * 4
    graph_bytes = 157164 * 15 * (8 + 4) * 2
    coords_bytes = 157164 * 2 * 4
    expected = base_bytes + max(int(base_bytes * 0.75), 1024**3) + graph_bytes + coords_bytes
    assert preflight.estimated_peak_bytes == expected
    assert preflight.state == "warning"
    assert "reuses the full source view directly" in preflight.notes[0]


def test_projection_fit_memory_preflight_warning_preserves_ok_status(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = _prepare_workspace(tmp_path)
    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "demo_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z_demo",
            "--strategy",
            "all",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout

    monkeypatch.setattr(memory_service, "_view_metadata", lambda *args, **kwargs: (157164, 8192, "float32", 4))
    monkeypatch.setattr(memory_service, "system_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(memory_service, "_row_count", lambda *args, **kwargs: 157164)

    result = _RUNNER.invoke(
        app,
        [
            "projection",
            "fit",
            "z_demo",
            "--workspace",
            workspace_dir.as_posix(),
            "--sample",
            "demo_sample",
            "--run-id",
            "z_demo_umap",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["warnings"]
    assert payload["metrics"]["memory_preflight"]["state"] == "warning"

    projection_manifest = json.loads(
        (workspace_dir / "outputs" / "projections" / "z_demo_umap" / "manifest.json").read_text(encoding="utf-8")
    )
    assert projection_manifest["status"] == "ok"
    assert projection_manifest["warnings"]
    assert projection_manifest["params"]["memory_preflight"]["state"] == "warning"


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
