"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_sync_locking.py

Ensure sync operations take the dataset write lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.remote import RemoteDatasetStat, RemotePrimaryStat
from dnadesign.usr.src.schema import REQUIRED_COLUMNS
from dnadesign.usr.tests.registry_helpers import ensure_registry


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

    def stat_dataset(self, _dataset: str, *, verify: str = "auto") -> RemoteDatasetStat:
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
