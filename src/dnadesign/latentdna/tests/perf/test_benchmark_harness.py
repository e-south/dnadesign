"""
Fixture-scale benchmark harness smoke tests for latentdna.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.services.alignment_service import build_alignment
from dnadesign.latentdna.src.services.deliverable_service import run_deliverable
from dnadesign.latentdna.src.services.distance_service import score_distance
from dnadesign.latentdna.src.services.export_service import export_matrix
from dnadesign.latentdna.src.services.neighbors_service import fit_neighbors
from dnadesign.latentdna.src.services.projection_service import fit_projection
from dnadesign.latentdna.src.services.sample_service import build_sample
from dnadesign.latentdna.src.services.view_service import derive_view, materialize_view, reduce_view


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(workspace_dir: Path, *, anchor_path: Path, context_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": workspace_dir.name, "output_root": "./outputs/latentdna"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "parquet",
                        "path": anchor_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "ctx1k": {
                        "kind": "parquet",
                        "path": context_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["usr_label__primary", "densegen__plan"]},
                "alignments": {
                    "anchor_ctx_20b": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    }
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor"},
                        "coordinate_space_id": "demo_20b_anchor_space",
                        "tags": {"model": "20b", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context"},
                        "coordinate_space_id": "demo_20b_anchor_space",
                        "tags": {"model": "20b", "context": "template_1kb"},
                        "role": "primary",
                    },
                    "delta20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_anchor",
                            "right": "z20_60",
                            "alignment": "anchor_ctx_20b",
                        },
                        "coordinate_space_id": "demo_20b_anchor_space",
                        "tags": {"operation": "difference"},
                        "role": "primary",
                    },
                },
                "scalars": {"delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}}},
                "landmarks": {
                    "spy_p": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "centroid"},
                    },
                    "sul_ap": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "sulAp"},
                        "representation": {"mode": "centroid"},
                    },
                },
                "exports": {
                    "x2_primary_20b": {
                        "row_basis": "anchor_ctx_20b",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z20_60_pc",
                                "source": "z20_60_pc02",
                                "feature_prefix": "z20_60",
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "delta20_pc",
                                "source": "delta20_pc02",
                                "feature_prefix": "delta20",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "landmark_distances",
                                "source": "primary_landmark_distances",
                                "columns": ["d_spy_p", "d_sul_ap"],
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "delta20_scalars",
                                "source": "delta20_norm",
                                "columns": ["delta20_norm"],
                            },
                        ],
                    }
                },
                "recipes": {
                    "atlas_2x2_recipe": {
                        "steps": [
                            {"id": "materialize_anchor", "op": "view.materialize", "params": {"view": "z20_60"}},
                            {
                                "id": "materialize_context",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_anchor"},
                            },
                            {
                                "id": "sample_anchor",
                                "op": "sample.build",
                                "depends_on": ["materialize_anchor"],
                                "params": {"sample": "atlas_anchor_sample", "view": "z20_60", "strategy": "all"},
                            },
                            {
                                "id": "sample_context",
                                "op": "sample.build",
                                "depends_on": ["materialize_context"],
                                "params": {
                                    "sample": "atlas_context_sample",
                                    "view": "z20_1k_anchor",
                                    "strategy": "all",
                                },
                            },
                            {
                                "id": "project_anchor",
                                "op": "projection.fit",
                                "depends_on": ["sample_anchor"],
                                "params": {
                                    "projection_id": "umap_z20_60",
                                    "view": "z20_60",
                                    "sample": "atlas_anchor_sample",
                                    "seed": 17,
                                },
                            },
                            {
                                "id": "project_context",
                                "op": "projection.fit",
                                "depends_on": ["sample_context"],
                                "params": {
                                    "projection_id": "umap_z20_1k_anchor",
                                    "view": "z20_1k_anchor",
                                    "sample": "atlas_context_sample",
                                    "seed": 17,
                                },
                            },
                            {
                                "id": "render_atlas",
                                "op": "plot.render",
                                "depends_on": ["project_anchor", "project_context"],
                                "params": {
                                    "plot_id": "atlas_2x2_main",
                                    "kind": "projection_grid",
                                    "projections": ["umap_z20_60", "umap_z20_1k_anchor"],
                                },
                            },
                        ]
                    }
                },
                "deliverables": {
                    "atlas_2x2_intermediate": {
                        "kind": "projection_grid",
                        "description": "Fixture-scale atlas deliverable.",
                        "recipe": "atlas_2x2_recipe",
                        "requires": {"views": ["z20_60", "z20_1k_anchor"]},
                        "outputs": {
                            "samples": ["atlas_anchor_sample", "atlas_context_sample"],
                            "projections": ["umap_z20_60", "umap_z20_1k_anchor"],
                            "plots": ["atlas_2x2_main"],
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _build_workspace(tmp_path: Path, name: str) -> Path:
    workspace_dir = tmp_path / name
    workspace_dir.mkdir()
    anchor_path = tmp_path / f"{name}_inputs" / "anchor60.parquet"
    context_path = tmp_path / f"{name}_inputs" / "ctx1k.parquet"
    _write_parquet(
        anchor_path,
        [
            {
                "id": "anchor_01",
                "subject_id": "subject_01",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 1.0, 0.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.0, 0.0, 1.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.0, 1.0, 1.0],
            },
        ],
    )
    _write_parquet(
        context_path,
        [
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "template_1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [1.0, 0.0, 1.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "template_1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [1.0, 2.0, 0.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "template_1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [5.0, 1.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "template_1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [6.0, 1.0, 2.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, anchor_path=anchor_path, context_path=context_path)
    return workspace_dir


def _artifact_size_bytes(paths: list[str]) -> int:
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            total += path.stat().st_size
            continue
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return total


def _record(
    name: str,
    *,
    rows: int,
    outputs: list[str],
    correctness: dict[str, object],
    started_at: float,
) -> dict[str, object]:
    wall_seconds = max(time.perf_counter() - started_at, 1e-9)
    return {
        "benchmark_name": name,
        "wall_seconds": wall_seconds,
        "throughput_rows_per_second": rows / wall_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifact_size_bytes": _artifact_size_bytes(outputs),
        "correctness": correctness,
    }


def _bench_view_materialize(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_view_materialize")
    started_at = time.perf_counter()
    result = materialize_view(workspace_dir, "z20_60")
    return _record(
        "bench_view_materialize",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "rows": result.metrics["rows"]},
        started_at=started_at,
    )


def _bench_delta_build(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_delta_build")
    materialize_view(workspace_dir, "z20_60")
    materialize_view(workspace_dir, "z20_1k_anchor")
    build_alignment(workspace_dir, "anchor_ctx_20b")
    started_at = time.perf_counter()
    result = derive_view(workspace_dir, "delta20")
    return _record(
        "bench_delta_build",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "dims": result.metrics["dims"]},
        started_at=started_at,
    )


def _bench_neighbors_fit(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_neighbors_fit")
    materialize_view(workspace_dir, "z20_60")
    build_sample(
        workspace_dir,
        "atlas_sample",
        view_id="z20_60",
        strategy="all",
        n=None,
        group_column=None,
        seed=17,
    )
    started_at = time.perf_counter()
    result = fit_neighbors(
        workspace_dir,
        "z20_60_knn",
        view_id="z20_60",
        k=2,
        metric="cosine",
        backend="exact",
        sample_id="atlas_sample",
        alignment_id=None,
        seed=17,
    )
    return _record(
        "bench_neighbors_fit",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "backend": result.metrics["backend"]},
        started_at=started_at,
    )


def _bench_projection_fit(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_projection_fit")
    materialize_view(workspace_dir, "z20_60")
    build_sample(
        workspace_dir,
        "atlas_sample",
        view_id="z20_60",
        strategy="all",
        n=None,
        group_column=None,
        seed=17,
    )
    started_at = time.perf_counter()
    result = fit_projection(
        workspace_dir,
        "z20_60",
        projection_id="umap_z20_60",
        sample_id="atlas_sample",
        metric="cosine",
        seed=17,
    )
    return _record(
        "bench_projection_fit",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "rows": result.metrics["rows"]},
        started_at=started_at,
    )


def _bench_distance_score(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_distance_score")
    materialize_view(workspace_dir, "z20_60")
    started_at = time.perf_counter()
    result = score_distance(
        workspace_dir,
        "primary_landmark_distances",
        view_id="z20_60",
        landmark_ids=["spy_p", "sul_ap"],
        metric="cosine",
    )
    return _record(
        "bench_distance_score",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "columns": result.metrics["columns"]},
        started_at=started_at,
    )


def _bench_export_x2(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_export_x2")
    materialize_view(workspace_dir, "z20_60")
    materialize_view(workspace_dir, "z20_1k_anchor")
    build_alignment(workspace_dir, "anchor_ctx_20b")
    derive_view(workspace_dir, "delta20")
    score_distance(
        workspace_dir,
        "primary_landmark_distances",
        view_id="z20_60",
        landmark_ids=["spy_p", "sul_ap"],
        metric="cosine",
    )
    from dnadesign.latentdna.src.services.scalar_service import derive_scalar

    derive_scalar(workspace_dir, "delta20_norm")
    reduce_view(
        workspace_dir,
        "z20_60",
        reducer_id="z20_60_pca",
        dims=2,
        sample_id=None,
        alignment_id=None,
        reduced_view_id="z20_60_pc02",
    )
    reduce_view(
        workspace_dir,
        "delta20",
        reducer_id="delta20_pca",
        dims=2,
        sample_id=None,
        alignment_id=None,
        reduced_view_id="delta20_pc02",
    )
    started_at = time.perf_counter()
    result = export_matrix(workspace_dir, "x2_primary_20b")
    return _record(
        "bench_export_x2",
        rows=4,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "features": result.metrics["features"]},
        started_at=started_at,
    )


def _bench_deliverable_atlas_2x2(tmp_path: Path) -> dict[str, object]:
    workspace_dir = _build_workspace(tmp_path, "bench_deliverable_atlas_2x2")
    started_at = time.perf_counter()
    result = run_deliverable(workspace_dir, "atlas_2x2_intermediate")
    return _record(
        "bench_deliverable_atlas_2x2",
        rows=8,
        outputs=result.outputs,
        correctness={"artifact_kind": result.artifact_kind, "outputs": result.metrics["outputs"]},
        started_at=started_at,
    )


BENCHMARK_RUNNERS = {
    "bench_view_materialize": _bench_view_materialize,
    "bench_delta_build": _bench_delta_build,
    "bench_neighbors_fit": _bench_neighbors_fit,
    "bench_projection_fit": _bench_projection_fit,
    "bench_distance_score": _bench_distance_score,
    "bench_export_x2": _bench_export_x2,
    "bench_deliverable_atlas_2x2": _bench_deliverable_atlas_2x2,
}


def test_benchmark_harness_registers_required_slices() -> None:
    assert sorted(BENCHMARK_RUNNERS) == [
        "bench_deliverable_atlas_2x2",
        "bench_delta_build",
        "bench_distance_score",
        "bench_export_x2",
        "bench_neighbors_fit",
        "bench_projection_fit",
        "bench_view_materialize",
    ]


def test_benchmark_harness_emits_required_metrics(tmp_path: Path) -> None:
    records = [runner(tmp_path) for runner in BENCHMARK_RUNNERS.values()]
    output_path = tmp_path / "benchmark_results.json"
    output_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [entry["benchmark_name"] for entry in payload] == list(BENCHMARK_RUNNERS)
    for entry in payload:
        assert entry["wall_seconds"] > 0
        assert entry["throughput_rows_per_second"] > 0
        assert entry["peak_rss_kib"] > 0
        assert entry["artifact_size_bytes"] > 0
        assert entry["correctness"]
