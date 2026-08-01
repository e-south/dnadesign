"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/test_sync_locking.py

Ensure sync operations take the dataset write lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import fcntl
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.devtools.tests.support.usr import ensure_registry
from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.contracts import REQUIRED_COLUMNS, VerificationError
from dnadesign.usr.src.events import append_event_line
from dnadesign.usr.src.events.append import append_event_payload, encode_event_line
from dnadesign.usr.src.sync.remote.remote import RemoteDatasetStat, RemotePrimaryStat
from dnadesign.usr.src.sync.remote.sidecars import local_sidecar_state


def _write_min_parquet(path: Path) -> None:
    schema = pa.schema(REQUIRED_COLUMNS)
    arrays = [pa.array(["x"]) if f.type == pa.string() else pa.array([1], type=f.type) for f in schema]
    tbl = pa.Table.from_arrays(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


class DummyRemote:
    def __init__(self, remote_template: Path | None = None):
        self.remote_template = remote_template
        self.pushed_file: Path | None = None
        self.remote_lock_calls = 0

    def _stat_from_file(self, path: Path) -> RemoteDatasetStat:
        size = int(path.stat().st_size)
        pf = pq.ParquetFile(str(path))
        rows = pf.metadata.num_rows
        cols = pf.metadata.num_columns
        return RemoteDatasetStat(
            primary=RemotePrimaryStat(True, size, None, rows, cols, "0"),
            meta_mtime=None,
            events_lines=0,
            snapshot_names=[],
        )

    def stat_dataset(
        self, _dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False
    ) -> RemoteDatasetStat:
        if self.pushed_file is not None:
            return self._stat_from_file(self.pushed_file)
        if self.remote_template is not None:
            return self._stat_from_file(self.remote_template)
        return RemoteDatasetStat(
            primary=RemotePrimaryStat(False, None, None, None, None, None),
            meta_mtime=None,
            events_lines=0,
            snapshot_names=[],
        )

    def pull_to_local(self, _dataset: str, dest: Path, **_kwargs) -> None:
        if self.remote_template is None:
            raise AssertionError("remote_template required for pull")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        _write_min_parquet(dest / "records.parquet")

    def push_from_local(self, _dataset: str, src: Path, **_kwargs) -> None:
        self.pushed_file = Path(src) / "records.parquet"

    def dataset_transfer_lock(self, _dataset: str):
        @contextmanager
        def _ctx():
            self.remote_lock_calls += 1
            yield

        return _ctx()


def _event_lock_is_held(event_path: Path) -> bool:
    lock_path = Path(event_path).parent / ".events.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def test_execute_pull_uses_lock(tmp_path: Path, monkeypatch) -> None:
    ensure_registry(tmp_path)
    remote_file = tmp_path / "remote" / "records.parquet"
    _write_min_parquet(remote_file)
    remote = DummyRemote(remote_file)

    def _remote_factory(_cfg):
        return remote

    lock_called = {"value": False}

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "SSHRemote", _remote_factory)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)

    opts = sync_module.SyncOptions(verify="size")
    sync_module.execute_pull(tmp_path, "demo", "remote", opts)
    assert lock_called["value"] is True
    assert remote.remote_lock_calls == 1


def test_full_pull_rejects_local_event_append_during_download(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    dataset = Dataset(root, "demo")
    dataset.init(source="unit-test")
    dataset.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"},
        ],
        source="unit-test",
    )
    records_before = dataset.records_path.read_bytes()
    events_before = dataset.events_path.read_bytes()

    remote_file = tmp_path / "remote" / "records.parquet"
    _write_min_parquet(remote_file)
    staged_pull_ready = Event()
    release_pull = Event()

    class BlockingRemote(DummyRemote):
        def pull_to_local(self, dataset_name: str, dest: Path, **kwargs) -> None:
            super().pull_to_local(dataset_name, dest, **kwargs)
            staged_pull_ready.set()
            assert release_pull.wait(timeout=5)

    remote = BlockingRemote(remote_file)
    monkeypatch.setattr(sync_module, "SSHRemote", lambda _cfg: remote)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())

    with ThreadPoolExecutor(max_workers=1) as executor:
        pull = executor.submit(
            sync_module.execute_pull,
            root,
            "demo",
            "remote",
            sync_module.SyncOptions(verify="size"),
        )
        try:
            assert staged_pull_ready.wait(timeout=5)
            append_event_line(dataset.events_path, '{"event":"local-concurrent"}')
        finally:
            release_pull.set()
        with pytest.raises(VerificationError, match="event log changed while the full pull was staged"):
            pull.result(timeout=5)

    assert dataset.records_path.read_bytes() == records_before
    events_after = dataset.events_path.read_bytes()
    assert events_after.startswith(events_before)
    assert events_after.endswith(b'{"event":"local-concurrent"}\n')
    assert not list(root.glob(".usr-pull-demo-*"))


def test_execute_push_uses_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit-test",
            }
        ],
        source="unit-test",
    )

    remote = DummyRemote()

    def _remote_factory(_cfg):
        return remote

    lock_called = {"value": False}

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "SSHRemote", _remote_factory)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)
    monkeypatch.setattr(sync_module, "_verify_after_push", lambda *_args, **_kwargs: None)

    opts = sync_module.SyncOptions(verify="size")
    sync_module.execute_push(root, "demo", "remote", opts)
    assert lock_called["value"] is True
    assert remote.remote_lock_calls == 1


@pytest.mark.parametrize(("primary_only", "expected_event_scans"), [(False, [True]), (True, [])])
def test_noop_push_scans_events_only_for_a_locked_full_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_only: bool,
    expected_event_scans: list[bool],
) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    dataset = Dataset(root, "demo")
    dataset.init(source="unit-test")
    dataset.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit-test",
            }
        ],
        source="unit-test",
    )

    class MirrorRemote(DummyRemote):
        def __init__(self) -> None:
            super().__init__()
            self.push_calls = 0

        def stat_dataset(
            self,
            _dataset: str,
            *,
            verify: str = "auto",
            include_derived_hashes: bool = False,
        ) -> RemoteDatasetStat:
            state = local_sidecar_state(dataset.dir, include_derived_hashes=include_derived_hashes)
            primary = self._stat_from_file(dataset.records_path).primary
            return RemoteDatasetStat(
                primary=primary,
                meta_mtime=state.meta_mtime,
                events_lines=state.events_lines,
                snapshot_names=list(state.snapshot_names),
                derived_files=list(state.derived_files),
                derived_hashes=dict(state.derived_hashes),
                aux_files=list(state.aux_files),
                aux_hashes=dict(state.aux_hashes),
            )

        def push_from_local(self, _dataset: str, src: Path, **_kwargs) -> None:
            self.push_calls += 1

    remote = MirrorRemote()
    monkeypatch.setattr(sync_module, "SSHRemote", lambda _cfg: remote)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())

    event_scans: list[bool] = []
    original_event_delta = sync_module._event_delta_requires_push

    def checked_event_delta(events_path: Path, *, remote_lines: int) -> bool:
        event_scans.append(_event_lock_is_held(events_path))
        return original_event_delta(events_path, remote_lines=remote_lines)

    monkeypatch.setattr(sync_module, "_event_delta_requires_push", checked_event_delta)

    sync_module.execute_push(
        root,
        "demo",
        "remote",
        sync_module.SyncOptions(verify="size", primary_only=primary_only),
    )

    assert event_scans == expected_event_scans
    assert remote.push_calls == 0


def test_full_push_holds_local_event_snapshot_through_remote_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    dataset = Dataset(root, "demo")
    dataset.init(source="unit-test")
    dataset.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit-test",
            }
        ],
        source="unit-test",
    )

    remote_dir = tmp_path / "remote" / "demo"
    snapshot_captured = Event()
    release_transfer = Event()
    verification_started = Event()
    release_verification = Event()
    snapshot_unlocked = Event()
    push_event_recorded = Event()

    class SnapshotBlockingRemote(DummyRemote):
        def __init__(self) -> None:
            super().__init__()
            self.pushed = False
            self.snapshot_events = b""

        def stat_dataset(
            self,
            _dataset: str,
            *,
            verify: str = "auto",
            include_derived_hashes: bool = False,
        ) -> RemoteDatasetStat:
            if not self.pushed:
                return super().stat_dataset(_dataset, verify=verify, include_derived_hashes=include_derived_hashes)
            verification_started.set()
            assert release_verification.wait(timeout=5)
            state = local_sidecar_state(remote_dir, include_derived_hashes=include_derived_hashes)
            primary = self._stat_from_file(remote_dir / "records.parquet").primary
            return RemoteDatasetStat(
                primary=primary,
                meta_mtime=state.meta_mtime,
                events_lines=state.events_lines,
                snapshot_names=list(state.snapshot_names),
                derived_files=list(state.derived_files),
                derived_hashes=dict(state.derived_hashes),
                aux_files=list(state.aux_files),
                aux_hashes=dict(state.aux_hashes),
            )

        def push_from_local(self, _dataset: str, src: Path, **_kwargs) -> None:
            shutil.copytree(
                src,
                remote_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".events.lock", ".usr.lock"),
            )
            self.snapshot_events = (remote_dir / ".events.log").read_bytes()
            self.pushed = True
            snapshot_captured.set()
            assert release_transfer.wait(timeout=5)

    remote = SnapshotBlockingRemote()
    monkeypatch.setattr(sync_module, "SSHRemote", lambda _cfg: remote)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())

    original_event_log_lock = sync_module.event_log_lock

    @contextmanager
    def tracked_event_log_lock(event_path: Path):
        try:
            with original_event_log_lock(event_path):
                yield
        finally:
            snapshot_unlocked.set()

    def record_push_event(*_args, **_kwargs) -> None:
        assert snapshot_unlocked.is_set()
        push_event_recorded.set()

    monkeypatch.setattr(sync_module, "event_log_lock", tracked_event_log_lock)
    monkeypatch.setattr(sync_module, "record_event", record_push_event)

    append_started = Event()
    append_completed = Event()

    def append_concurrent_event() -> None:
        append_event_payload(
            dataset.events_path,
            encode_event_line('{"action":"local-concurrent"}'),
            on_start=append_started.set,
        )
        append_completed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        push = executor.submit(
            sync_module.execute_push,
            root,
            "demo",
            "remote",
            sync_module.SyncOptions(verify="size", verify_sidecars=True),
        )
        append = None
        try:
            assert snapshot_captured.wait(timeout=5)
            assert _event_lock_is_held(dataset.events_path)

            append = executor.submit(append_concurrent_event)
            assert append_started.wait(timeout=5)
            assert not append_completed.is_set()

            release_transfer.set()
            assert verification_started.wait(timeout=5)
            assert _event_lock_is_held(dataset.events_path)
            assert not append_completed.is_set()
            release_verification.set()

            push.result(timeout=5)
            append.result(timeout=5)
        finally:
            release_transfer.set()
            release_verification.set()
            if append is not None:
                append.result(timeout=5)
            push.result(timeout=5)

    assert b'"action":"local-concurrent"' not in remote.snapshot_events
    assert b'"action":"local-concurrent"' in dataset.events_path.read_bytes()
    assert push_event_recorded.is_set()
