"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_schema_adversarial.py

Adversarial sync tests for schema and verification mismatch handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.errors import VerificationError
from dnadesign.usr.src.remote import RemoteDatasetStat, RemotePrimaryStat
from dnadesign.usr.src.schema import REQUIRED_COLUMNS
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _write_records(path: Path, rows: int) -> None:
    schema = pa.schema(REQUIRED_COLUMNS)
    payload = {}
    for field in schema:
        if field.name == "length":
            payload[field.name] = pa.array([4] * rows, type=field.type)
        elif field.name == "created_at":
            payload[field.name] = pa.array([None] * rows, type=field.type)
        else:
            payload[field.name] = pa.array([field.name] * rows, type=field.type)
    tbl = pa.table(payload, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _row(sequence: str, source: str) -> dict[str, str]:
    return {
        "sequence": sequence,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def test_execute_pull_file_fails_on_post_transfer_parquet_row_mismatch(tmp_path: Path, monkeypatch) -> None:
    local_file = tmp_path / "payload.parquet"
    remote_file = tmp_path / "remote" / "payload.parquet"
    _write_records(remote_file, rows=2)

    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

        def stat_file(self, _remote_path: str, *, verify: str = "auto"):
            pf = pq.ParquetFile(str(remote_file))
            return RemotePrimaryStat(
                exists=True,
                size=int(remote_file.stat().st_size),
                sha256=None,
                rows=pf.metadata.num_rows,
                cols=pf.metadata.num_columns,
                mtime="0",
            )

        def pull_file(self, _remote_path: str, local_path: Path, *, dry_run: bool = False):
            assert dry_run is False
            _write_records(Path(local_path), rows=1)

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="post-pull-file: row mismatch"):
        sync_module.execute_pull_file(
            local_file,
            "mock-remote",
            "remote/payload.parquet",
            sync_module.SyncOptions(verify="parquet"),
        )


def test_execute_push_file_fails_on_post_transfer_parquet_col_mismatch(tmp_path: Path, monkeypatch) -> None:
    local_file = tmp_path / "local" / "payload.parquet"
    _write_records(local_file, rows=1)
    stat_calls = {"count": 0}

    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

        def stat_file(self, _remote_path: str, *, verify: str = "auto"):
            stat_calls["count"] += 1
            if stat_calls["count"] == 1:
                return RemotePrimaryStat(
                    exists=False,
                    size=None,
                    sha256=None,
                    rows=None,
                    cols=None,
                    mtime=None,
                )
            return RemotePrimaryStat(
                exists=True,
                size=int(local_file.stat().st_size),
                sha256=None,
                rows=1,
                cols=6,
                mtime="0",
            )

        def push_file(self, _local_path: Path, _remote_path: str, *, dry_run: bool = False):
            assert dry_run is False

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="post-push-file: col mismatch"):
        sync_module.execute_push_file(
            local_file,
            "mock-remote",
            "remote/payload.parquet",
            sync_module.SyncOptions(verify="parquet"),
        )


def test_execute_pull_rejects_staged_symlink_payload_without_mutating_local_primary(
    tmp_path: Path, monkeypatch
) -> None:
    local_root = tmp_path / "local_usr"
    dataset_id = "densegen/demo_adversarial"
    local_dataset_dir = local_root / dataset_id
    local_records = local_dataset_dir / "records.parquet"
    _write_records(local_records, rows=1)

    remote_records = tmp_path / "remote_records.parquet"
    _write_records(remote_records, rows=2)

    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

        def stat_dataset(self, _dataset: str, *, verify: str = "auto"):
            assert verify in {"auto", "hash", "size", "parquet"}
            return RemoteDatasetStat(
                primary=RemotePrimaryStat(
                    exists=True,
                    size=int(remote_records.stat().st_size),
                    sha256=None,
                    rows=2,
                    cols=7,
                    mtime="0",
                ),
                meta_mtime=None,
                events_lines=0,
                snapshot_names=[],
            )

        def pull_to_local(
            self,
            _dataset: str,
            dest_dir: Path,
            *,
            primary_only: bool = False,
            skip_snapshots: bool = False,
            dry_run: bool = False,
        ) -> None:
            assert dry_run is False
            assert primary_only is False
            assert skip_snapshots is False
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            _write_records(dest_dir / "records.parquet", rows=2)
            derived = dest_dir / "_derived"
            derived.mkdir(parents=True, exist_ok=True)
            try:
                (derived / "danger.link").symlink_to(local_records)
            except OSError as exc:
                pytest.skip(f"symlink not supported in this environment: {exc}")

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="symlink"):
        sync_module.execute_pull(
            local_root,
            dataset_id,
            "mock-remote",
            sync_module.SyncOptions(verify="parquet"),
        )

    assert pq.read_table(local_records).num_rows == 1


def test_execute_pull_verify_sidecars_rejects_mismatched_staged_sidecars_before_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_sidecar_pull"

    local_dataset = Dataset(local_root, dataset_id)
    local_dataset.init(source="local-seed")
    local_dataset.import_rows([_row("AAAA", "local-seed")], source="local-seed")
    local_records = local_root / dataset_id / "records.parquet"
    local_sha_before = _sha256(local_records)

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("CCCC", "remote-seed")], source="remote-seed")
    remote_dataset.log_event("remote-only-sidecar", args={"source": "remote"})
    remote_records = remote_root / dataset_id / "records.parquet"
    remote_size = int(remote_records.stat().st_size)
    remote_mtime = str(int(remote_records.stat().st_mtime))

    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

        def dataset_transfer_lock(self, _dataset: str):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

        def stat_dataset(self, _dataset: str, *, verify: str = "auto"):
            snapshot_dir = remote_root / dataset_id / "_snapshots"
            snapshot_names = []
            snapshot_re = re.compile(r"^records-\d{8}T\d{6}\.parquet$")
            if snapshot_dir.exists():
                snapshot_names = sorted(
                    [item.name for item in snapshot_dir.iterdir() if item.is_file() and snapshot_re.match(item.name)]
                )
            events_path = remote_root / dataset_id / ".events.log"
            events_lines = sum(1 for _ in events_path.open("rb")) if events_path.exists() else 0
            meta_path = remote_root / dataset_id / "meta.md"
            meta_mtime = str(int(meta_path.stat().st_mtime)) if meta_path.exists() else None
            return RemoteDatasetStat(
                primary=RemotePrimaryStat(
                    exists=True,
                    size=remote_size,
                    sha256=None,
                    rows=1,
                    cols=7,
                    mtime=remote_mtime,
                ),
                meta_mtime=meta_mtime,
                events_lines=events_lines,
                snapshot_names=snapshot_names,
            )

        def pull_to_local(
            self,
            _dataset: str,
            dest_dir: Path,
            *,
            primary_only: bool = False,
            skip_snapshots: bool = False,
            dry_run: bool = False,
        ) -> None:
            assert dry_run is False
            assert primary_only is False
            assert skip_snapshots is False
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Intentionally write only primary to trigger sidecar fidelity failure.
            _write_records(dest_dir / "records.parquet", rows=1)

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="post-pull-sidecars"):
        sync_module.execute_pull(
            local_root,
            dataset_id,
            "mock-remote",
            sync_module.SyncOptions(verify="parquet", verify_sidecars=True),
        )

    assert _sha256(local_records) == local_sha_before


def test_execute_push_verify_sidecars_fails_when_remote_sidecars_do_not_match_local(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "densegen/demo_sidecar_push")
    ds.init(source="unit-test")
    ds.import_rows([_row("ACGT", "unit-test")], source="unit-test")
    local_records = root / "densegen" / "demo_sidecar_push" / "records.parquet"
    local_size = int(local_records.stat().st_size)

    stat_calls = {"count": 0}

    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

        def dataset_transfer_lock(self, _dataset: str):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

        def stat_dataset(self, _dataset: str, *, verify: str = "auto"):
            stat_calls["count"] += 1
            if stat_calls["count"] <= 2:
                return RemoteDatasetStat(
                    primary=RemotePrimaryStat(False, None, None, None, None, None),
                    meta_mtime=None,
                    events_lines=0,
                    snapshot_names=[],
                )
            return RemoteDatasetStat(
                primary=RemotePrimaryStat(True, local_size, None, 1, 7, "0"),
                meta_mtime=None,
                events_lines=0,
                snapshot_names=[],
            )

        def push_from_local(
            self,
            _dataset: str,
            _src_dir: Path,
            *,
            primary_only: bool = False,
            skip_snapshots: bool = False,
            dry_run: bool = False,
        ) -> None:
            assert dry_run is False
            assert primary_only is False
            assert skip_snapshots is False

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="post-push-sidecars"):
        sync_module.execute_push(
            root,
            "densegen/demo_sidecar_push",
            "mock-remote",
            sync_module.SyncOptions(verify="size", verify_sidecars=True),
        )


def test_verify_sidecars_requires_full_dataset_transfer_options(tmp_path: Path, monkeypatch) -> None:
    class _Remote:
        def __init__(self, _cfg) -> None:
            pass

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)

    with pytest.raises(VerificationError, match="requires full dataset transfer"):
        sync_module.execute_pull(
            tmp_path,
            "densegen/demo",
            "mock-remote",
            sync_module.SyncOptions(verify="auto", verify_sidecars=True, primary_only=True),
        )

    with pytest.raises(VerificationError, match="requires full dataset transfer"):
        sync_module.execute_push(
            tmp_path,
            "densegen/demo",
            "mock-remote",
            sync_module.SyncOptions(verify="auto", verify_sidecars=True, skip_snapshots=True),
        )
