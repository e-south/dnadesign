"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/overlay/test_create_only_publication.py

Test failure and concurrency contracts for create-only overlay publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pyarrow as pa
import pytest

from dnadesign.artifacts import PublicationError
from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.usr import Dataset, SchemaError
from dnadesign.usr.src.datasets.overlay import write as dataset_overlay_write_module
from dnadesign.usr.src.events import EventAppendFailure, EventAppendState
from dnadesign.usr.src.events import append as event_append_module
from dnadesign.usr.src.events import recording as event_recording_module


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


@pytest.mark.parametrize(
    ("invalid_metadata", "error_type", "error_match"),
    [
        ({"actor": {}}, ValueError, "actor.tool must be a non-empty string"),
        ({"event_args": {"source_context": object()}}, TypeError, "not JSON serializable"),
    ],
    ids=("invalid-actor", "non-serializable-event-args"),
)
def test_create_overlay_rejects_invalid_event_metadata_before_publication(
    tmp_path: Path,
    invalid_metadata: dict[str, object],
    error_type: type[Exception],
    error_match: str,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    events_before = dataset.events_path.read_bytes()

    with pytest.raises(error_type, match=error_match):
        dataset.create_overlay("mock", table, key="id", **invalid_metadata)

    assert not final.exists()
    assert dataset.events_path.read_bytes() == events_before
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_create_overlay_rolls_back_publication_when_event_recording_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    events_before = dataset.events_path.read_bytes()
    record_event = dataset._record_event

    def fail_event(*_args, **_kwargs) -> None:
        raise OSError("injected event write failure")

    monkeypatch.setattr(dataset, "_record_event", fail_event)
    with pytest.raises(OSError, match="injected event write failure"):
        dataset.create_overlay("mock", table, key="id")

    assert not final.exists()
    assert dataset.events_path.read_bytes() == events_before

    monkeypatch.setattr(dataset, "_record_event", record_event)
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_create_overlay_restores_partial_event_append_before_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    events_before = dataset.events_path.read_bytes()
    write_all = event_append_module._write_all

    def write_part_then_fail(descriptor: int, payload: bytes) -> None:
        event_append_module.os.write(descriptor, payload[: max(1, len(payload) // 2)])
        raise OSError("injected partial event append")

    monkeypatch.setattr(event_append_module, "_write_all", write_part_then_fail)
    with pytest.raises(EventAppendFailure, match="restored") as exc_info:
        dataset.create_overlay("mock", table, key="id")

    assert exc_info.value.state is EventAppendState.RESTORED
    assert not final.exists()
    assert dataset.events_path.read_bytes() == events_before

    monkeypatch.setattr(event_append_module, "_write_all", write_all)
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_create_overlay_preserves_publication_after_committed_event_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    events_before = dataset.events_path.read_bytes()
    close_descriptor = event_append_module._close_descriptor

    def close_then_fail(descriptor: int) -> None:
        close_descriptor(descriptor)
        raise OSError("injected close failure after durable append")

    monkeypatch.setattr(event_append_module, "_close_descriptor", close_then_fail)
    with pytest.raises(EventAppendFailure, match="committed") as exc_info:
        dataset.create_overlay("mock", table, key="id")

    assert exc_info.value.state is EventAppendState.COMMITTED
    assert final.is_dir()
    assert len(list(final.glob("part-*.parquet"))) == 1
    assert dataset.events_path.read_bytes().startswith(events_before)
    assert dataset.events_path.read_bytes() != events_before


def test_create_overlay_preserves_publication_when_event_restore_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"

    def write_part_then_fail(descriptor: int, payload: bytes) -> None:
        event_append_module.os.write(descriptor, payload[: max(1, len(payload) // 2)])
        raise OSError("injected partial event append")

    monkeypatch.setattr(event_append_module, "_write_all", write_part_then_fail)
    monkeypatch.setattr(event_append_module, "_restore_prior_length", lambda *_args: False)
    with pytest.raises(EventAppendFailure, match="indeterminate") as exc_info:
        dataset.create_overlay("mock", table, key="id")

    assert exc_info.value.state is EventAppendState.INDETERMINATE
    assert final.is_dir()
    assert len(list(final.glob("part-*.parquet"))) == 1


def test_create_overlay_fails_loudly_when_live_event_log_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    displaced_events = dataset.dir / ".events.displaced.log"
    events_before = dataset.events_path.read_bytes()
    write_all = event_append_module._write_all

    def write_then_replace(descriptor: int, payload: bytes) -> None:
        write_all(descriptor, payload)
        dataset.events_path.rename(displaced_events)
        dataset.events_path.write_bytes(events_before)

    monkeypatch.setattr(event_append_module, "_write_all", write_then_replace)
    with pytest.raises(EventAppendFailure, match="indeterminate") as exc_info:
        dataset.create_overlay("mock", table, key="id")

    assert exc_info.value.state is EventAppendState.INDETERMINATE
    assert final.is_dir()
    assert dataset.events_path.read_bytes() == events_before
    assert b'"action":"write_overlay_part"' in displaced_events.read_bytes()


def test_create_overlay_opens_event_log_only_after_stable_sidecar_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    displaced_events = dataset.dir / ".events.displaced.log"
    events_before = dataset.events_path.read_bytes()
    reached_event_recording = Event()
    record_event = dataset._record_event

    def signal_then_record(*args, **kwargs) -> None:
        reached_event_recording.set()
        record_event(*args, **kwargs)

    monkeypatch.setattr(dataset, "_record_event", signal_then_record)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with event_append_module.event_log_lock(dataset.events_path):
            future = executor.submit(dataset.create_overlay, "mock", table, key="id")
            assert reached_event_recording.wait(timeout=5)
            assert final.is_dir()
            dataset.events_path.rename(displaced_events)
            dataset.events_path.write_bytes(events_before)
        assert future.result(timeout=5) == 1

    assert displaced_events.read_bytes() == events_before
    live_events = dataset.events_path.read_bytes()
    assert live_events.startswith(events_before)
    assert b'"action":"write_overlay_part"' in live_events


def test_create_overlay_rolls_back_when_event_fingerprinting_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    events_before = dataset.events_path.read_bytes()
    fingerprint_parquet = event_recording_module.fingerprint_parquet

    def fail_fingerprint(_path: Path):
        raise OSError("injected event fingerprint failure")

    monkeypatch.setattr(event_recording_module, "fingerprint_parquet", fail_fingerprint)
    with pytest.raises(OSError, match="injected event fingerprint failure"):
        dataset.create_overlay("mock", table, key="id")

    assert not final.exists()
    assert dataset.events_path.read_bytes() == events_before

    monkeypatch.setattr(event_recording_module, "fingerprint_parquet", fingerprint_parquet)
    assert dataset.create_overlay("mock", table, key="id") == 1


def test_event_failure_rollback_preserves_a_swapped_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    table = _overlay_input(dataset)
    final = dataset.dir / "_derived/mock"
    displaced = dataset.dir / "_derived/displaced"
    sentinel = final / "keep.txt"

    def swap_then_fail(*_args, **_kwargs) -> None:
        final.rename(displaced)
        final.mkdir()
        sentinel.write_text("keep\n", encoding="utf-8")
        raise EventAppendFailure(dataset.events_path, state=EventAppendState.RESTORED)

    monkeypatch.setattr(dataset, "_record_event", swap_then_fail)
    with pytest.raises(PublicationError, match="could not be rolled back safely"):
        dataset.create_overlay("mock", table, key="id")

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert len(list(displaced.glob("part-*.parquet"))) == 1
