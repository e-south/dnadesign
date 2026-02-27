"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_schema_adversarial.py

Adversarial sync tests for schema and verification mismatch handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.errors import VerificationError
from dnadesign.usr.src.remote import RemotePrimaryStat
from dnadesign.usr.src.schema import REQUIRED_COLUMNS


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
