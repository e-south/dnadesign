"""
Freshness-aware contract tests for latentdna artifacts and deliverables.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.services.deliverable_service import deliverable_status
from dnadesign.latentdna.src.services.freshness_service import FreshnessCache
from dnadesign.latentdna.src.services.run_service import list_runs
from dnadesign.latentdna.src.sources import provenance as provenance_module
from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr import Dataset
from dnadesign.usr.src.datasets.demo.mock import MockSpec, create_mock_dataset

_RUNNER = CliRunner()


def _build_overlay_backed_workspace(tmp_path: Path) -> tuple[Path, Path]:
    usr_root = tmp_path / "usr_root"
    register_test_namespace(
        usr_root,
        namespace="mock",
        columns_spec="mock__x_representation:list<float32>,mock__label_vec8:list<float32>",
    )
    register_test_namespace(
        usr_root,
        namespace="infer",
        columns_spec="infer__x_representation:list<float32>",
    )
    register_test_namespace(
        usr_root,
        namespace="densegen",
        columns_spec="densegen__plan:string",
    )
    create_mock_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        MockSpec(n=4, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )

    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    ids = dataset.head(n=4, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__x_representation": pa.array(
                    [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )
    dataset.write_overlay_part(
        "densegen",
        pa.table({"id": ids, "densegen__plan": ["plan_a", "plan_a", "plan_b", "plan_b"]}),
        key="id",
    )

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_workspace", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_anchor_set",
                        "record_key": "id",
                        "subject_key": "id",
                    }
                },
                "metadata": {"include": ["densegen__plan"]},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "infer__x_representation"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
                "recipes": {
                    "materialize_only": {
                        "steps": [
                            {
                                "id": "materialize_view",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            }
                        ]
                    }
                },
                "deliverables": {
                    "view_bundle": {
                        "recipe": "materialize_only",
                        "title": "View bundle",
                        "section": "freshness",
                        "question": "Does the materialized view respond to overlay changes?",
                        "summary": "One materialized view used to exercise freshness propagation.",
                        "requires": {"sources": ["anchor60"], "views": ["z20_60"], "recipes": ["materialize_only"]},
                        "outputs": {"views": ["z20_60"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace_dir, usr_root


def _build_projection_bundle_workspace(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_root"
    register_test_namespace(
        usr_root,
        namespace="mock",
        columns_spec="mock__x_representation:list<float32>,mock__label_vec8:list<float32>",
    )
    create_mock_dataset(
        usr_root,
        "promoter/demo_projection_set",
        MockSpec(n=6, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "projection_workspace", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "euclidean",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_projection_set",
                        "record_key": "id",
                        "subject_key": "id",
                    }
                },
                "metadata": {"include": []},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "mock__x_representation"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
                "recipes": {
                    "projection_bundle_recipe": {
                        "steps": [
                            {
                                "id": "materialize_view",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            },
                            {
                                "id": "build_sample",
                                "op": "sample.build",
                                "depends_on": ["materialize_view"],
                                "params": {
                                    "sample_id": "all_rows",
                                    "view": "z20_60",
                                    "strategy": "all",
                                    "seed": 17,
                                },
                            },
                            {
                                "id": "fit_projection",
                                "op": "projection.fit",
                                "depends_on": ["build_sample"],
                                "params": {
                                    "projection_id": "umap_z20_60",
                                    "view": "z20_60",
                                    "sample": "all_rows",
                                    "metric": "euclidean",
                                    "seed": 17,
                                },
                            },
                        ]
                    }
                },
                "deliverables": {
                    "projection_bundle": {
                        "recipe": "projection_bundle_recipe",
                        "title": "Projection bundle",
                        "section": "freshness",
                        "question": "Does the projection bundle stay fresh when upstream sample or view changes?",
                        "summary": "One source-backed projection bundle used to verify freshness hashing.",
                        "requires": {
                            "sources": ["anchor60"],
                            "views": ["z20_60"],
                            "recipes": ["projection_bundle_recipe"],
                        },
                        "outputs": {
                            "views": ["z20_60"],
                            "samples": ["all_rows"],
                            "projections": ["umap_z20_60"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace_dir


def test_overlay_change_marks_view_run_and_deliverable_attention(tmp_path: Path) -> None:
    workspace_dir, usr_root = _build_overlay_backed_workspace(tmp_path)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    before_runs = _RUNNER.invoke(app, ["runs", "list", "--workspace", workspace_dir.as_posix(), "--json"])
    assert before_runs.exit_code == 0, before_runs.stdout
    before_runs_payload = json.loads(before_runs.stdout)
    assert before_runs_payload["runs"][0]["status"] == "ok"

    before_status = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert before_status.exit_code == 0, before_status.stdout
    assert json.loads(before_status.stdout)["status"] == "ok"

    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    ids = dataset.head(n=4, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__x_representation": pa.array(
                    [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5], [1.6, 1.7]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )

    after_runs = _RUNNER.invoke(app, ["runs", "list", "--workspace", workspace_dir.as_posix(), "--json"])
    assert after_runs.exit_code == 0, after_runs.stdout
    after_runs_payload = json.loads(after_runs.stdout)
    assert after_runs_payload["runs"][0]["status"] == "attention"
    assert "freshness" in after_runs_payload["runs"][0]["reason"]

    after_status = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert after_status.exit_code == 0, after_status.stdout
    after_status_payload = json.loads(after_status.stdout)
    assert after_status_payload["status"] == "attention"
    outputs = {entry["name"]: entry for entry in after_status_payload["outputs"]}
    assert outputs["view:z20_60"]["status"] == "attention"
    assert "stale" in (outputs["view:z20_60"]["reason"] or "")


def test_view_manifest_records_overlay_part_provenance(tmp_path: Path) -> None:
    workspace_dir, _ = _build_overlay_backed_workspace(tmp_path)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    manifest_path = workspace_dir / "outputs" / "views" / "z20_60" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["source_provenance"]

    infer_part_entries = [
        entry for entry in provenance if entry.get("role") == "overlay_part" and entry.get("namespace") == "infer"
    ]
    densegen_part_entries = [
        entry for entry in provenance if entry.get("role") == "overlay_part" and entry.get("namespace") == "densegen"
    ]

    assert infer_part_entries
    assert densegen_part_entries


def test_dependency_only_overlay_column_drift_marks_view_attention(tmp_path: Path) -> None:
    workspace_dir, usr_root = _build_overlay_backed_workspace(tmp_path)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["metadata"] = {
        "include": ["generation_plan_copy"],
        "derivations": {
            "generation_plan_copy": {
                "kind": "copy",
                "source": "densegen__plan",
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    status_before = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_before.exit_code == 0, status_before.stdout
    assert json.loads(status_before.stdout)["status"] == "ok"

    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    ids = dataset.head(n=4, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "densegen",
        pa.table(
            {
                "id": ids,
                "densegen__plan": ["plan_z", "plan_z", "plan_y", "plan_y"],
            }
        ),
        key="id",
    )

    status_after = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_after.exit_code == 0, status_after.stdout
    status_after_payload = json.loads(status_after.stdout)
    assert status_after_payload["status"] == "attention"
    outputs = {entry["name"]: entry for entry in status_after_payload["outputs"]}
    assert outputs["view:z20_60"]["status"] == "attention"
    assert "stale" in str(outputs["view:z20_60"]["reason"] or "")


def test_view_vector_column_drift_marks_view_attention(tmp_path: Path) -> None:
    workspace_dir, _ = _build_overlay_backed_workspace(tmp_path)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["views"]["z20_60"]["vector"]["name"] = "mock__label_vec8"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    status_payload = deliverable_status(workspace_dir, "view_bundle")
    assert status_payload.status == "attention"
    outputs = {entry.name: entry for entry in status_payload.outputs}
    assert outputs["view:z20_60"].status == "attention"
    assert "vector_column" in str(outputs["view:z20_60"].reason or "")


def test_view_manifest_records_overlay_ledger_when_explicit_contract_enabled(tmp_path: Path) -> None:
    workspace_dir, usr_root = _build_overlay_backed_workspace(tmp_path)
    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    infer_ledger = dataset.write_overlay_digest_ledger("infer")
    densegen_ledger = dataset.write_overlay_digest_ledger("densegen")

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    manifest_path = workspace_dir / "outputs" / "views" / "z20_60" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["source_provenance"]

    overlay_entries = [entry for entry in provenance if entry.get("role") == "overlay"]
    ledger_entries = [entry for entry in provenance if entry.get("role") == "overlay_ledger"]
    overlay_part_entries = [entry for entry in provenance if entry.get("role") == "overlay_part"]

    assert overlay_entries
    assert all(entry.get("digest_mode") == "inventory" for entry in overlay_entries)
    assert {Path(str(entry["path"])).resolve() for entry in ledger_entries} == {
        infer_ledger.resolve(),
        densegen_ledger.resolve(),
    }
    assert all(
        entry.get("digest_mode") == provenance_module.OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE for entry in ledger_entries
    )
    assert overlay_part_entries == []


def test_deliverable_status_uses_overlay_ledger_contract_without_raw_path_hashes_when_ledgers_are_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir, usr_root = _build_overlay_backed_workspace(tmp_path)
    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    dataset.write_overlay_digest_ledger("infer")
    dataset.write_overlay_digest_ledger("densegen")

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    original_sha256_path = provenance_module.sha256_path
    hash_counts: Counter[str] = Counter()

    def counting_sha256_path(path: Path) -> str:
        hash_counts[Path(path).as_posix()] += 1
        return original_sha256_path(path)

    monkeypatch.setattr(provenance_module, "sha256_path", counting_sha256_path)

    status = deliverable_status(workspace_dir, "view_bundle")

    assert status.status == "ok"
    assert any(path.endswith("/records.parquet") for path in hash_counts)
    assert not any(path.endswith("/digest_ledger.json") for path in hash_counts)
    assert not any("/_derived/" in path and path.endswith(".parquet") for path in hash_counts)


def test_recipe_run_rebuilds_stale_view_after_overlay_change(tmp_path: Path) -> None:
    workspace_dir, usr_root = _build_overlay_backed_workspace(tmp_path)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    dataset = Dataset(usr_root, "promoter/demo_anchor_set")
    ids = dataset.head(n=4, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__x_representation": pa.array(
                    [[2.0, 2.1], [2.2, 2.3], [2.4, 2.5], [2.6, 2.7]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )

    stale_status = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert stale_status.exit_code == 0, stale_status.stdout
    assert json.loads(stale_status.stdout)["status"] == "attention"

    rerun_result = _RUNNER.invoke(
        app,
        ["recipe", "run", "materialize_only", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert rerun_result.exit_code == 0, rerun_result.stdout
    rerun_payload = json.loads(rerun_result.stdout)
    assert rerun_payload["metrics"]["executed_steps"] == 1
    assert rerun_payload["metrics"]["skipped_steps"] == 0

    refreshed_status = _RUNNER.invoke(
        app,
        ["deliverable", "status", "view_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert refreshed_status.exit_code == 0, refreshed_status.stdout
    assert json.loads(refreshed_status.stdout)["status"] == "ok"


def test_deliverable_status_hashes_shared_freshness_paths_once_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir = _build_projection_bundle_workspace(tmp_path)

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "projection_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    original_sha256_path = provenance_module.sha256_path
    hash_counts: Counter[str] = Counter()

    def counting_sha256_path(path: Path) -> str:
        hash_counts[Path(path).as_posix()] += 1
        return original_sha256_path(path)

    monkeypatch.setattr(provenance_module, "sha256_path", counting_sha256_path)

    status = deliverable_status(workspace_dir, "projection_bundle")
    assert status.status == "ok"
    assert hash_counts
    assert max(hash_counts.values()) == 1


def test_deliverable_status_reuses_shared_freshness_cache_across_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir = _build_projection_bundle_workspace(tmp_path)

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "projection_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    original_sha256_path = provenance_module.sha256_path
    hash_counts: Counter[str] = Counter()

    def counting_sha256_path(path: Path) -> str:
        hash_counts[Path(path).as_posix()] += 1
        return original_sha256_path(path)

    monkeypatch.setattr(provenance_module, "sha256_path", counting_sha256_path)

    freshness_cache = FreshnessCache()
    first = deliverable_status(workspace_dir, "projection_bundle", freshness_cache=freshness_cache)
    second = deliverable_status(workspace_dir, "projection_bundle", freshness_cache=freshness_cache)

    assert first.status == "ok"
    assert second.status == "ok"
    assert hash_counts
    assert max(hash_counts.values()) == 1


def test_runs_list_hashes_shared_freshness_paths_once_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir = _build_projection_bundle_workspace(tmp_path)

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "projection_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    original_sha256_path = provenance_module.sha256_path
    hash_counts: Counter[str] = Counter()

    def counting_sha256_path(path: Path) -> str:
        hash_counts[Path(path).as_posix()] += 1
        return original_sha256_path(path)

    monkeypatch.setattr(provenance_module, "sha256_path", counting_sha256_path)

    runs_payload = list_runs(workspace_dir)
    assert any(run["artifact_id"] == "umap_z20_60" for run in runs_payload["runs"])
    assert hash_counts
    assert max(hash_counts.values()) == 1
