"""Mutation contracts for locks, audit retention and failed reduction cleanup."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dnadesign.latentdna.src.contracts.errors import OperationLockError
from dnadesign.latentdna.src.runs import recorder
from dnadesign.latentdna.src.services import operation_lock_service as locks
from dnadesign.latentdna.src.services import view_service


def test_lock_handoff_preserves_exclusive_ownership(tmp_path, monkeypatch):
    original = locks.fcntl.flock
    next_owner = None
    handed_off = False

    def handoff(fd, operation):
        nonlocal next_owner, handed_off
        original(fd, operation)
        if operation == locks.fcntl.LOCK_UN and not handed_off:
            handed_off = True
            next_owner = locks.acquire_workspace_operation_lock(tmp_path, operation="test", owner_id="second")
            next_owner.__enter__()

    monkeypatch.setattr(locks.fcntl, "flock", handoff)
    try:
        with locks.acquire_workspace_operation_lock(tmp_path, operation="test", owner_id="first"):
            with pytest.raises(OperationLockError):
                with locks.acquire_workspace_operation_lock(tmp_path, operation="test", owner_id="blocked"):
                    pass
        with pytest.raises(OperationLockError):
            with locks.acquire_workspace_operation_lock(tmp_path, operation="test", owner_id="third"):
                pass
    finally:
        if next_owner is not None:
            next_owner.__exit__(None, None, None)
    with locks.acquire_workspace_operation_lock(tmp_path, operation="test", owner_id="last"):
        pass


def test_same_time_audits_all_survive(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "datetime", SimpleNamespace(now=lambda _: datetime(2026, 1, 1, tzinfo=UTC)))

    def record(i):
        return recorder.record_audit(tmp_path, payload={"event": i}, command="view reduce", artifact_id="result")

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(record, range(24)))
    assert len(set(paths)) == 24
    assert sorted(json.loads(path.read_text())["event"] for path in paths) == list(range(24))


def test_invalid_reduction_leaves_no_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(view_service, "load_workspace_config", lambda _: SimpleNamespace(output_root=tmp_path))

    def fail(*args, **kwargs):
        raise ValueError("invalid reduction")

    monkeypatch.setattr(view_service, "evaluate_reduce_preflight", fail)
    with pytest.raises(ValueError, match="invalid reduction"):
        view_service.reduce_view(
            tmp_path, "missing", reducer_id="pca", dims=2, sample_id=None, alignment_id=None, reduced_view_id="reduced"
        )
    assert not list(tmp_path.rglob("*_staging*"))
    assert not list(tmp_path.rglob("pca_*"))


@pytest.fixture
def workspace(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    root = tmp_path / "workspace"
    root.mkdir()
    pq.write_table(
        pa.table(
            {"id": [f"seq_{i}" for i in range(12)], "embedding": np.random.default_rng(5).normal(size=(12, 4)).tolist()}
        ),
        root / "rows.parquet",
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "defaults": {},
                "workspace": {"id": "example", "output_root": "./outputs"},
                "sources": {
                    "records": {"kind": "parquet", "path": "rows.parquet", "record_key": "id", "subject_key": "id"}
                },
                "views": {
                    "embeddings": {
                        "source": "records",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "example",
                    }
                },
            }
        )
    )
    view_service.materialize_view(root, "embeddings")
    return root


def test_cli_rejects_absolute_output_and_accepts_local_id(workspace, tmp_path):
    from typer.testing import CliRunner

    from dnadesign.latentdna.src.cli import app

    runner = CliRunner()
    outside = tmp_path / "outside" / "escaped"
    options = ["--view", "embeddings", "--k", "3", "--backend", "exact", "--quiet", "--workspace", str(workspace)]
    rejected = runner.invoke(app, ["neighbors", "fit", str(outside), *options])
    assert rejected.exit_code != 0
    assert "artifact id" in rejected.output
    assert not outside.parent.exists()
    accepted = runner.invoke(app, ["neighbors", "fit", "local", *options])
    assert accepted.exit_code == 0, accepted.output
    assert (workspace / "outputs/neighbors/local/manifest.json").is_file()


@pytest.mark.parametrize("failure", ["second_allocation", "manifest"])
def test_reduction_cleans_staging_after_late_failure(workspace, monkeypatch, failure):
    if failure == "manifest":

        def fail(*args, **kwargs):
            raise OSError("injected manifest failure")

        monkeypatch.setattr(view_service, "write_manifest", fail)
    else:
        original = view_service.stage_artifact_dir

        def fail(parent, identifier):
            if identifier == "reduced":
                raise OSError("injected allocation failure")
            return original(parent, identifier)

        monkeypatch.setattr(view_service, "stage_artifact_dir", fail)
    with pytest.raises(OSError, match="injected"):
        view_service.reduce_view(
            workspace,
            "embeddings",
            reducer_id="pca",
            dims=2,
            sample_id=None,
            alignment_id=None,
            reduced_view_id="reduced",
        )
    staging = workspace / "outputs/runs/_staging"
    assert not list(staging.glob("*/*"))
    assert not (workspace / "outputs/reducers/pca").exists()
    assert not (workspace / "outputs/reduced_views/reduced").exists()
