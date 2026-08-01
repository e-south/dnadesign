"""Failure and concurrency contracts for create-only overlay publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.usr import Dataset, SchemaError
from dnadesign.usr.src.datasets.overlay import write as dataset_overlay_write_module


def _make_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    dataset = Dataset(root, "demo")
    dataset.init(source="test")
    dataset.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
        ],
        source="test",
    )
    return dataset


def _overlay_input(dataset: Dataset) -> pa.Table:
    target_id = dataset.head(1)["id"].iloc[0]
    return pa.table({"id": [target_id], "mock__score": [1.0]})


def test_create_overlay_is_atomic_and_create_once(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)

    def create() -> int | type[Exception]:
        try:
            return dataset.create_overlay("mock", table, key="id")
        except Exception as exc:  # noqa: BLE001 - concurrency result is asserted below
            return type(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: create(), range(2)))

    assert results.count(1) == 1
    assert results.count(FileExistsError) == 1
    assert len(list((dataset.dir / "_derived/mock").glob("part-*.parquet"))) == 1


def test_create_overlay_stages_before_exposing_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    real_write_table = dataset_overlay_write_module.pq.write_table
    staged_parents: list[Path] = []

    def inspect_write(table, path, *args, **kwargs):
        staged_parents.append(Path(path).parent)
        assert not final.exists()
        assert Path(path).parent != final
        return real_write_table(table, path, *args, **kwargs)

    monkeypatch.setattr(dataset_overlay_write_module.pq, "write_table", inspect_write)

    assert dataset.create_overlay("mock", table, key="id") == 1
    assert final.is_dir()
    assert len(list(final.glob("part-*.parquet"))) == 1
    assert staged_parents and all(not parent.exists() for parent in staged_parents)


def test_create_overlay_cleans_failed_stage_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    real_write_table = dataset_overlay_write_module.pq.write_table
    staged_parents: list[Path] = []
    attempts = 0

    def fail_once(table, path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        staged_parents.append(Path(path).parent)
        if attempts == 1:
            raise OSError("injected parquet write failure")
        return real_write_table(table, path, *args, **kwargs)

    monkeypatch.setattr(dataset_overlay_write_module.pq, "write_table", fail_once)

    with pytest.raises(OSError, match="injected parquet write failure"):
        dataset.create_overlay("mock", table, key="id")
    assert not final.exists()
    assert staged_parents and all(not parent.exists() for parent in staged_parents)

    assert dataset.create_overlay("mock", table, key="id") == 1
    assert len(list(final.glob("part-*.parquet"))) == 1


def test_create_overlay_verifies_stage_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    target_id = table["id"][0].as_py()
    final = dataset.dir / "_derived/mock"
    real_read_table = dataset_overlay_write_module.pq.read_table
    staged_parents: list[Path] = []

    def corrupt_read(path, *_args, **_kwargs):
        staged_parents.append(Path(path).parent)
        return pa.table({"id": [target_id], "mock__score": [2.0]})

    monkeypatch.setattr(dataset_overlay_write_module.pq, "read_table", corrupt_read)
    with pytest.raises(SchemaError, match="Staged overlay verification failed"):
        dataset.create_overlay("mock", table, key="id")
    assert not final.exists()
    assert staged_parents and all(not parent.exists() for parent in staged_parents)

    monkeypatch.setattr(dataset_overlay_write_module.pq, "read_table", real_read_table)
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_create_overlay_cleans_stage_on_cooperative_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    real_write_table = dataset_overlay_write_module.pq.write_table
    staged_parents: list[Path] = []

    def terminate(_table, path, *_args, **_kwargs):
        staged_parents.append(Path(path).parent)
        raise SystemExit("injected cooperative termination")

    monkeypatch.setattr(dataset_overlay_write_module.pq, "write_table", terminate)
    with pytest.raises(SystemExit, match="injected cooperative termination"):
        dataset.create_overlay("mock", table, key="id")
    assert not final.exists()
    assert staged_parents and all(not parent.exists() for parent in staged_parents)

    monkeypatch.setattr(dataset_overlay_write_module.pq, "write_table", real_write_table)
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_create_overlay_preserves_competing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    sentinel = final / "competing.txt"
    real_publish = dataset_overlay_write_module.CreateOnlyDirectoryPublication.publish

    def race_publish(publication, *, required_manifest: str) -> None:
        final.mkdir(parents=True)
        sentinel.write_text("competitor\n", encoding="utf-8")
        real_publish(publication, required_manifest=required_manifest)

    monkeypatch.setattr(
        dataset_overlay_write_module.CreateOnlyDirectoryPublication,
        "publish",
        race_publish,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        dataset.create_overlay("mock", table, key="id")
    assert sentinel.read_text(encoding="utf-8") == "competitor\n"
    assert not list(final.glob("part-*.parquet"))
    assert not list(final.parent.glob(".mock.staging-*"))


def test_create_overlay_rejects_reserved_event_args_before_publication(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"

    with pytest.raises(SchemaError, match="cannot override reserved key 'namespace'"):
        dataset.create_overlay("mock", table, key="id", event_args={"namespace": "spoofed"})

    assert not final.exists()
    assert dataset.create_overlay("mock", table, key="id") == 1
