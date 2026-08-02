"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/test_sync_event_transactions.py

Full-push event-log transaction contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.devtools.tests.support.usr import ensure_registry
from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.contracts import TransferError, VerificationError
from dnadesign.usr.src.sync.remote.remote import RemoteDatasetStat, RemotePrimaryStat
from dnadesign.usr.src.sync.remote.transfer import EventLogContentRevision


def _content_revision(payload: bytes) -> EventLogContentRevision:
    if not payload:
        return EventLogContentRevision(exists=False, size_bytes=0, sha256=None)
    return EventLogContentRevision(
        exists=True,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _dataset(root: Path) -> Dataset:
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
    return dataset


class _TransactionRemote:
    def __init__(self, *, event_payload: bytes = b"") -> None:
        self.event_payload = event_payload
        self.pushed_file: Path | None = None
        self.push_calls = 0
        self.dry_run_calls = 0
        self.event_lock = threading.Lock()
        self.active_lease: object | None = None

    def stat_dataset(
        self,
        _dataset: str,
        *,
        verify: str = "auto",
        include_derived_hashes: bool = False,
    ) -> RemoteDatasetStat:
        del verify, include_derived_hashes
        if self.pushed_file is None:
            primary = RemotePrimaryStat(False, None, None, None, None, None)
        else:
            parquet = pq.ParquetFile(str(self.pushed_file))
            primary = RemotePrimaryStat(
                True,
                self.pushed_file.stat().st_size,
                None,
                parquet.metadata.num_rows,
                parquet.metadata.num_columns,
                "0",
            )
        return RemoteDatasetStat(
            primary=primary,
            meta_mtime=None,
            events_lines=self.event_payload.count(b"\n"),
            snapshot_names=[],
        )

    def dataset_transfer_lock(self, _dataset: str):
        @contextmanager
        def _lock():
            yield

        return _lock()

    def event_log_transfer_lock(self, _dataset: str):
        @contextmanager
        def _lock():
            with self.event_lock:
                lease = object()
                self.active_lease = lease
                try:
                    yield lease
                finally:
                    self.active_lease = None

        return _lock()

    def event_log_revision(self, _dataset: str, *, event_lease: object) -> EventLogContentRevision:
        assert event_lease is self.active_lease
        return _content_revision(self.event_payload)

    def observe_event_log_revision(self, _dataset: str) -> EventLogContentRevision:
        return _content_revision(self.event_payload)

    def push_from_local(
        self,
        _dataset: str,
        src: Path,
        *,
        event_lease: object | None = None,
        dry_run: bool = False,
        **_kwargs,
    ) -> None:
        if dry_run:
            self.dry_run_calls += 1
            return
        assert event_lease is self.active_lease
        self.push_calls += 1
        self.pushed_file = Path(src) / "records.parquet"
        self.event_payload = (Path(src) / ".events.log").read_bytes()

    def pull_to_local(self, _dataset: str, dest: Path, *, dry_run: bool = False, **_kwargs) -> None:
        if dry_run:
            self.dry_run_calls += 1
            return
        assert self.pushed_file is not None
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.pushed_file, dest / "records.parquet")
        if self.event_payload:
            (dest / ".events.log").write_bytes(self.event_payload)

    def append_event(self, payload: bytes, *, attempted: threading.Event) -> None:
        attempted.set()
        with self.event_lock:
            self.event_payload += payload


def _install_remote(monkeypatch: pytest.MonkeyPatch, remote: _TransactionRemote) -> None:
    monkeypatch.setattr(sync_module, "SSHRemote", lambda _cfg: remote)
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())


@pytest.mark.parametrize(
    "remote_relation",
    ["same-size-divergent", "remote-ahead", "truncated-prefix"],
)
def test_full_push_rejects_remote_event_history_that_is_not_a_local_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_relation: str,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    local_events = dataset.events_path.read_bytes()
    if remote_relation == "same-size-divergent":
        remote_events = b"x" * len(local_events)
    elif remote_relation == "remote-ahead":
        remote_events = local_events + b'{"action":"remote-only"}\n'
    else:
        remote_events = local_events[:-1]

    remote = _TransactionRemote(event_payload=remote_events)
    _install_remote(monkeypatch, remote)

    with pytest.raises(VerificationError, match="not a prefix"):
        sync_module.execute_push(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert remote.push_calls == 0


def test_full_push_rejects_branches_created_after_a_pull_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    shared_history = dataset.events_path.read_bytes()
    dataset.events_path.write_bytes(shared_history + b'{"action":"local-only"}\n')
    remote = _TransactionRemote(event_payload=shared_history + b'{"action":"remote-write"}\n')
    _install_remote(monkeypatch, remote)

    with pytest.raises(VerificationError, match="not a prefix"):
        sync_module.execute_push(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert remote.push_calls == 0


def test_full_push_uses_one_ordered_event_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    _dataset(root)
    trace: list[str] = []

    class OrderedRemote(_TransactionRemote):
        def dataset_transfer_lock(self, _dataset: str):
            @contextmanager
            def _lock():
                trace.append("remote-dataset-enter")
                try:
                    yield
                finally:
                    trace.append("remote-dataset-exit")

            return _lock()

        def event_log_transfer_lock(self, _dataset: str):
            @contextmanager
            def _lock():
                trace.append("remote-event-enter")
                lease = object()
                self.active_lease = lease
                try:
                    yield lease
                finally:
                    self.active_lease = None
                    trace.append("remote-event-exit")

            return _lock()

        def push_from_local(self, _dataset: str, src: Path, *, event_lease: object, **kwargs) -> None:
            assert trace[-1] == "remote-event-enter"
            trace.append("transfer")
            super().push_from_local(_dataset, src, event_lease=event_lease, **kwargs)

    @contextmanager
    def local_dataset_lock(_path: Path):
        trace.append("local-dataset-enter")
        try:
            yield
        finally:
            trace.append("local-dataset-exit")

    original_event_log_lock = sync_module.event_log_lock

    @contextmanager
    def local_event_lock(event_path: Path):
        with original_event_log_lock(event_path):
            trace.append("local-event-enter")
            try:
                yield
            finally:
                trace.append("local-event-exit")

    remote = OrderedRemote()
    _install_remote(monkeypatch, remote)
    monkeypatch.setattr(sync_module, "dataset_write_lock", local_dataset_lock)
    monkeypatch.setattr(sync_module, "event_log_lock", local_event_lock)

    sync_module.execute_push(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert trace == [
        "local-dataset-enter",
        "remote-dataset-enter",
        "local-event-enter",
        "remote-event-enter",
        "transfer",
        "remote-event-exit",
        "local-event-exit",
        "remote-dataset-exit",
        "local-dataset-exit",
    ]


def test_remote_event_writer_waits_until_post_transfer_revision_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    _dataset(root)
    verification_started = threading.Event()
    release_verification = threading.Event()
    append_attempted = threading.Event()
    append_completed = threading.Event()

    class BlockingRevisionRemote(_TransactionRemote):
        def event_log_revision(self, dataset: str, *, event_lease: object) -> EventLogContentRevision:
            revision = super().event_log_revision(dataset, event_lease=event_lease)
            if self.push_calls:
                verification_started.set()
                assert release_verification.wait(timeout=5)
            return revision

    remote = BlockingRevisionRemote()
    _install_remote(monkeypatch, remote)

    def append_remote_event() -> None:
        remote.append_event(b'{"action":"remote-concurrent"}\n', attempted=append_attempted)
        append_completed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        push = executor.submit(
            sync_module.execute_push,
            root,
            "demo",
            "remote",
            sync_module.SyncOptions(verify="size"),
        )
        assert verification_started.wait(timeout=5)
        append = executor.submit(append_remote_event)
        try:
            assert append_attempted.wait(timeout=5)
            assert not append_completed.wait(timeout=0.1)
        finally:
            release_verification.set()
        push.result(timeout=5)
        append.result(timeout=5)

    assert append_completed.is_set()
    assert remote.event_payload.endswith(b'{"action":"remote-concurrent"}\n')


def test_full_push_does_not_append_a_transport_event_when_remote_dataset_lease_exit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    _dataset(root)
    events_before = (root / "demo" / ".events.log").read_bytes()

    class ReleaseFailingRemote(_TransactionRemote):
        def dataset_transfer_lock(self, _dataset: str):
            @contextmanager
            def _lock():
                yield
                raise TransferError("remote dataset lease release failed")

            return _lock()

    remote = ReleaseFailingRemote()
    _install_remote(monkeypatch, remote)
    with pytest.raises(TransferError, match="lease release failed"):
        sync_module.execute_push(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert remote.push_calls == 1
    assert (root / "demo" / ".events.log").read_bytes() == events_before


def test_full_pull_does_not_append_a_transport_event_when_remote_dataset_lease_exit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local"
    ensure_registry(local_root)
    remote_dataset = _dataset(tmp_path / "remote")

    class ReleaseFailingRemote(_TransactionRemote):
        def dataset_transfer_lock(self, _dataset: str):
            @contextmanager
            def _lock():
                yield
                raise TransferError("remote dataset lease release failed")

            return _lock()

        def pull_to_local(self, _dataset: str, dest_dir: Path, **_kwargs) -> None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_dataset.records_path, dest_dir / "records.parquet")
            shutil.copy2(remote_dataset.events_path, dest_dir / ".events.log")

    remote = ReleaseFailingRemote(event_payload=remote_dataset.events_path.read_bytes())
    remote.pushed_file = remote_dataset.records_path
    _install_remote(monkeypatch, remote)
    with pytest.raises(TransferError, match="lease release failed"):
        sync_module.execute_pull(local_root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert (local_root / "demo" / "records.parquet").is_file()
    assert (local_root / "demo" / ".events.log").read_bytes() == remote_dataset.events_path.read_bytes()


def test_full_push_dry_run_does_not_short_circuit_on_equal_line_count_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    remote_events = bytearray(dataset.events_path.read_bytes())
    remote_events[0] = ord("x") if remote_events[0] != ord("x") else ord("y")
    remote = _TransactionRemote(event_payload=bytes(remote_events))
    remote.pushed_file = dataset.records_path
    _install_remote(monkeypatch, remote)

    with pytest.raises(VerificationError, match="not a prefix"):
        sync_module.execute_push(
            root,
            "demo",
            "remote",
            sync_module.SyncOptions(verify="size", dry_run=True),
        )

    assert remote.dry_run_calls == 0
    assert remote.event_payload == bytes(remote_events)


def test_full_push_dry_run_observes_prefix_then_previews_without_remote_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    remote = _TransactionRemote(event_payload=dataset.events_path.read_bytes())
    remote.pushed_file = dataset.records_path
    _install_remote(monkeypatch, remote)

    summary = sync_module.execute_push(
        root,
        "demo",
        "remote",
        sync_module.SyncOptions(verify="size", dry_run=True),
    )

    assert remote.dry_run_calls == 1
    assert any("without mutation" in note for note in summary.verify_notes)


def test_plan_diff_detects_equal_line_event_content_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    dataset.events_path.write_bytes(b'{"action":"local"}\n')
    remote = _TransactionRemote(event_payload=b'{"action":"other"}\n')
    remote.pushed_file = dataset.records_path
    _install_remote(monkeypatch, remote)

    summary = sync_module.plan_diff(root, "demo", "remote", verify="size")

    assert summary.has_change is True
    assert summary.changes["events_content_diff"] is True
    assert summary.events_local_lines == summary.events_remote_lines == 1


def test_full_pull_rejects_equal_line_divergent_event_histories_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    dataset.events_path.write_bytes(b'{"action":"local"}\n')
    records_before = dataset.records_path.read_bytes()
    events_before = dataset.events_path.read_bytes()
    remote = _TransactionRemote(event_payload=b'{"action":"other"}\n')
    remote.pushed_file = dataset.records_path
    _install_remote(monkeypatch, remote)

    with pytest.raises(VerificationError, match="does not extend the local event history"):
        sync_module.execute_pull(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert dataset.records_path.read_bytes() == records_before
    assert dataset.events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"action":"partial"}', "partial JSONL"),
        (b"\xff\n", "not valid UTF-8"),
        (b"{broken}\n", "malformed JSON"),
        (b"[]\n", "not a JSON object"),
    ],
)
def test_full_push_rejects_invalid_local_event_history_before_transfer_when_remote_log_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    dataset.events_path.write_bytes(payload)
    (dataset.dir / ".events.lock").unlink(missing_ok=True)
    remote = _TransactionRemote()
    _install_remote(monkeypatch, remote)

    with pytest.raises(VerificationError, match=message):
        sync_module.execute_push(root, "demo", "remote", sync_module.SyncOptions(verify="size"))

    assert remote.push_calls == 0
    assert not (dataset.dir / ".events.lock").exists()


def test_full_pull_dry_run_does_not_create_the_absent_local_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    remote_dataset = _dataset(tmp_path / "remote")
    remote = _TransactionRemote(event_payload=remote_dataset.events_path.read_bytes())
    remote.pushed_file = remote_dataset.records_path
    _install_remote(monkeypatch, remote)

    sync_module.execute_pull(
        local_root,
        "demo",
        "remote",
        sync_module.SyncOptions(verify="size", dry_run=True),
    )

    assert remote.dry_run_calls == 1
    assert not (local_root / "demo").exists()


def test_full_push_dry_run_with_no_event_log_does_not_create_a_local_event_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    dataset = _dataset(root)
    dataset.events_path.unlink()
    (dataset.dir / ".events.lock").unlink(missing_ok=True)
    remote = _TransactionRemote()
    _install_remote(monkeypatch, remote)

    sync_module.execute_push(
        root,
        "demo",
        "remote",
        sync_module.SyncOptions(verify="size", dry_run=True),
    )

    assert remote.dry_run_calls == 1
    assert not (dataset.dir / ".events.lock").exists()


def test_single_file_transfers_do_not_append_to_an_adjacent_dataset_event_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    local_file = dataset_dir / "artifact.bin"
    local_file.write_bytes(b"local")
    event_path = dataset_dir / ".events.log"
    event_path.write_bytes(b'{"action":"scientific"}\n')
    events_before = event_path.read_bytes()

    class FileRemote:
        payload = b"remote"

        def stat_file(self, _remote_path: str, *, verify: str = "auto") -> RemotePrimaryStat:
            del verify
            return RemotePrimaryStat(True, len(self.payload), None, None, None, "0")

        def pull_file(self, _remote_path: str, destination: Path, *, dry_run: bool = False) -> None:
            assert dry_run is False
            Path(destination).write_bytes(self.payload)

        def push_file(self, source: Path, _remote_path: str, *, dry_run: bool = False) -> None:
            assert dry_run is False
            self.payload = Path(source).read_bytes()

    remote = FileRemote()
    _install_remote(monkeypatch, remote)  # type: ignore[arg-type]
    opts = sync_module.SyncOptions(verify="size")

    sync_module.execute_pull_file(local_file, "remote", "/remote/artifact.bin", opts)
    assert event_path.read_bytes() == events_before
    local_file.write_bytes(b"next")
    sync_module.execute_push_file(local_file, "remote", "/remote/artifact.bin", opts)

    assert event_path.read_bytes() == events_before
    assert remote.payload == b"next"
