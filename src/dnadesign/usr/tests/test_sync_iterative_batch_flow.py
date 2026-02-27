"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py

Pressure tests for iterative USR sync flows used by HPC batch workloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.config import SSHRemoteConfig
from dnadesign.usr.src.errors import TransferError
from dnadesign.usr.src.remote import RemoteDatasetStat, RemotePrimaryStat
from dnadesign.usr.tests.registry_helpers import ensure_registry, register_test_namespace


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


class _FilesystemRemote:
    fail_next_pull = False
    fail_next_push = False
    pull_transfer_calls = 0
    push_transfer_calls = 0
    remote_lock_calls = 0

    def __init__(self, cfg: SSHRemoteConfig) -> None:
        self.cfg = cfg

    def _dataset_dir(self, dataset: str) -> Path:
        return Path(self.cfg.base_dir) / dataset

    def dataset_transfer_lock(self, _dataset: str):
        @contextmanager
        def _ctx():
            _FilesystemRemote.remote_lock_calls += 1
            yield

        return _ctx()

    def _primary_stat(self, records_path: Path, *, verify: str) -> RemotePrimaryStat:
        if not records_path.exists():
            return RemotePrimaryStat(False, None, None, None, None, None)
        size = int(records_path.stat().st_size)
        mtime = str(int(records_path.stat().st_mtime))
        sha = _sha256(records_path) if verify in {"auto", "hash"} else None
        rows = cols = None
        if verify == "parquet":
            pf = pq.ParquetFile(str(records_path))
            rows, cols = int(pf.metadata.num_rows), int(pf.metadata.num_columns)
        return RemotePrimaryStat(True, size, sha, rows, cols, mtime)

    def stat_dataset(
        self, dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False
    ) -> RemoteDatasetStat:
        dataset_dir = self._dataset_dir(dataset)
        records_path = dataset_dir / "records.parquet"
        meta_path = dataset_dir / "meta.md"
        events_path = dataset_dir / ".events.log"
        snapshot_dir = dataset_dir / "_snapshots"
        derived_dir = dataset_dir / "_derived"
        snapshot_re = re.compile(r"^records-\d{8}T\d{6,}\.parquet$")
        snapshot_names = []
        if snapshot_dir.exists():
            snapshot_names = sorted([item.name for item in snapshot_dir.iterdir() if snapshot_re.match(item.name)])
        derived_files = []
        if derived_dir.exists():
            derived_files = sorted(
                [item.relative_to(derived_dir).as_posix() for item in derived_dir.rglob("*") if item.is_file()]
            )
        derived_hashes = {}
        if include_derived_hashes:
            for rel in derived_files:
                derived_hashes[rel] = _sha256(derived_dir / rel)
        aux_files = []
        if dataset_dir.exists():
            for item in sorted(dataset_dir.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(dataset_dir)
                rel_text = rel.as_posix()
                if rel_text in {"records.parquet", "meta.md", ".events.log", ".usr.lock"}:
                    continue
                if rel.parts and rel.parts[0] in {"_snapshots", "_derived"}:
                    continue
                aux_files.append(rel_text)
        events_lines = 0
        if events_path.exists():
            events_lines = sum(1 for _ in events_path.open("rb"))
        meta_mtime = str(int(meta_path.stat().st_mtime)) if meta_path.exists() else None
        return RemoteDatasetStat(
            primary=self._primary_stat(records_path, verify=verify),
            meta_mtime=meta_mtime,
            events_lines=events_lines,
            snapshot_names=snapshot_names,
            derived_files=derived_files,
            derived_hashes=derived_hashes,
            aux_files=sorted(aux_files),
        )

    def _copy_dataset(self, src_dir: Path, dst_dir: Path, *, primary_only: bool, skip_snapshots: bool) -> None:
        if not src_dir.exists():
            raise TransferError(f"missing source dataset directory: {src_dir}")
        if primary_only:
            src_primary = src_dir / "records.parquet"
            if not src_primary.exists():
                raise TransferError(f"missing source primary file: {src_primary}")
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_primary, dst_dir / "records.parquet")
            return

        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.rglob("*"):
            rel = item.relative_to(src_dir)
            if skip_snapshots and rel.parts and rel.parts[0] == "_snapshots":
                continue
            target = dst_dir / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

    def pull_to_local(
        self,
        dataset: str,
        dest_dir: Path,
        *,
        primary_only: bool = False,
        skip_snapshots: bool = False,
        dry_run: bool = False,
    ) -> None:
        if dry_run:
            return
        _FilesystemRemote.pull_transfer_calls += 1
        src_dir = self._dataset_dir(dataset)
        dest_dir = Path(dest_dir)
        if _FilesystemRemote.fail_next_pull:
            _FilesystemRemote.fail_next_pull = False
            src_primary = src_dir / "records.parquet"
            if src_primary.exists():
                payload = src_primary.read_bytes()
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / "records.parquet").write_bytes(payload[: max(1, len(payload) // 3)])
            raise TransferError("simulated rsync interruption")
        self._copy_dataset(src_dir, dest_dir, primary_only=primary_only, skip_snapshots=skip_snapshots)

    def push_from_local(
        self,
        dataset: str,
        src_dir: Path,
        *,
        primary_only: bool = False,
        skip_snapshots: bool = False,
        dry_run: bool = False,
    ) -> None:
        if dry_run:
            return
        _FilesystemRemote.push_transfer_calls += 1
        if _FilesystemRemote.fail_next_push:
            _FilesystemRemote.fail_next_push = False
            src_primary = Path(src_dir) / "records.parquet"
            payload = src_primary.read_bytes()
            dst_primary = self._dataset_dir(dataset) / "records.parquet"
            dst_primary.parent.mkdir(parents=True, exist_ok=True)
            dst_primary.write_bytes(payload[: max(1, len(payload) // 3)])
            raise TransferError("simulated rsync push interruption")
        self._copy_dataset(
            Path(src_dir),
            self._dataset_dir(dataset),
            primary_only=primary_only,
            skip_snapshots=skip_snapshots,
        )


def _dataset_file_fingerprints(dataset_dir: Path, *, include_events: bool = True) -> dict[str, str]:
    file_hashes: dict[str, str] = {}
    for path in sorted(Path(dataset_dir).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dataset_dir).as_posix()
        if rel == ".usr.lock":
            continue
        if not include_events and rel == ".events.log":
            continue
        file_hashes[rel] = _sha256(path)
    return file_hashes


def test_iterative_sync_flow_recovers_after_interrupted_pull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_hpc_sync"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    local_records = local_root / dataset_id / "records.parquet"
    remote_records = remote_root / dataset_id / "records.parquet"
    assert pq.read_table(local_records).num_rows == 1

    local_dataset = Dataset(local_root, dataset_id)
    local_dataset.import_rows([_row("CCCC", "local-batch")], source="local-batch")
    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)
    assert pq.read_table(remote_records).num_rows == 2

    remote_dataset.import_rows([_row("GGGG", "remote-batch")], source="remote-batch")
    local_sha_before_failed_pull = _sha256(local_records)
    _FilesystemRemote.fail_next_pull = True
    with pytest.raises(TransferError, match="simulated rsync interruption"):
        sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _sha256(local_records) == local_sha_before_failed_pull
    assert pq.read_table(local_records).num_rows == 2

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert pq.read_table(local_records).num_rows == 3
    assert _sha256(local_records) == _sha256(remote_records)
    assert _FilesystemRemote.remote_lock_calls >= 3


def test_iterative_sync_flow_skips_transfer_calls_when_already_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_noop_sync"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.push_transfer_calls == 0
    assert _FilesystemRemote.remote_lock_calls == 1


def test_iterative_sync_flow_recovers_after_interrupted_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_push_resume"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    local_dataset = Dataset(local_root, dataset_id)
    local_dataset.import_rows([_row("CCCC", "local-batch")], source="local-batch")

    local_records = local_root / dataset_id / "records.parquet"
    remote_records = remote_root / dataset_id / "records.parquet"
    local_sha = _sha256(local_records)

    _FilesystemRemote.fail_next_push = True
    with pytest.raises(TransferError, match="simulated rsync push interruption"):
        sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)

    assert _sha256(remote_records) != local_sha

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)
    assert _sha256(remote_records) == local_sha
    assert pq.read_table(remote_records).num_rows == 2
    assert _FilesystemRemote.remote_lock_calls >= 2


def test_iterative_cross_location_sidecar_fidelity_for_densegen_infer_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    register_test_namespace(remote_root, namespace="densegen", columns_spec="densegen__score:float64")
    register_test_namespace(remote_root, namespace="infer", columns_spec="infer__llr:float64")
    register_test_namespace(local_root, namespace="densegen", columns_spec="densegen__score:float64")
    register_test_namespace(local_root, namespace="infer", columns_spec="infer__llr:float64")
    dataset_id = "densegen/demo_hpc_fidelity"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="densegen-seed")
    remote_dataset.import_rows(
        [_row("AAAA", "densegen-batch"), _row("CCCC", "densegen-batch")],
        source="densegen-batch",
    )
    remote_dataset.write_overlay_part(
        "densegen",
        pa.table({"sequence": ["AAAA", "CCCC"], "densegen__score": [0.15, 0.72]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )
    remote_dataset.log_event("densegen_batch_complete", args={"phase": "hpc"})
    remote_dataset.snapshot()

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="parquet", verify_sidecars=True)
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)

    local_dataset_dir = local_root / dataset_id
    remote_dataset_dir = remote_root / dataset_id
    assert _dataset_file_fingerprints(local_dataset_dir, include_events=False) == _dataset_file_fingerprints(
        remote_dataset_dir, include_events=False
    )
    local_events_lines = sum(1 for _ in (local_dataset_dir / ".events.log").open("rb"))
    remote_events_lines = sum(1 for _ in (remote_dataset_dir / ".events.log").open("rb"))
    assert local_events_lines == remote_events_lines + 1

    local_dataset = Dataset(local_root, dataset_id)
    local_dataset.write_overlay_part(
        "infer",
        pa.table({"sequence": ["AAAA", "CCCC"], "infer__llr": [1.2, -0.7]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )
    local_dataset.log_event("infer_batch_complete", args={"phase": "local"})

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)
    assert _dataset_file_fingerprints(local_dataset_dir, include_events=False) == _dataset_file_fingerprints(
        remote_dataset_dir, include_events=False
    )
    local_events_lines = sum(1 for _ in (local_dataset_dir / ".events.log").open("rb"))
    remote_events_lines = sum(1 for _ in (remote_dataset_dir / ".events.log").open("rb"))
    assert local_events_lines == remote_events_lines + 1

    remote_dataset.import_rows([_row("GGGG", "densegen-batch-2")], source="densegen-batch-2")
    remote_dataset.write_overlay_part(
        "densegen",
        pa.table({"sequence": ["GGGG"], "densegen__score": [0.48]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )
    remote_dataset.log_event("densegen_batch_complete", args={"phase": "hpc-2"})
    remote_dataset.snapshot()

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _dataset_file_fingerprints(local_dataset_dir, include_events=False) == _dataset_file_fingerprints(
        remote_dataset_dir, include_events=False
    )
    local_events_lines = sum(1 for _ in (local_dataset_dir / ".events.log").open("rb"))
    remote_events_lines = sum(1 for _ in (remote_dataset_dir / ".events.log").open("rb"))
    assert local_events_lines == remote_events_lines + 1
    assert pq.read_table(local_dataset_dir / "records.parquet").num_rows == 3


def test_pull_detects_overlay_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    register_test_namespace(remote_root, namespace="densegen", columns_spec="densegen__score:float64")
    register_test_namespace(local_root, namespace="densegen", columns_spec="densegen__score:float64")
    dataset_id = "densegen/demo_overlay_pull_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1

    events_path = remote_root / dataset_id / ".events.log"
    baseline_lines = events_path.read_text(encoding="utf-8").splitlines()

    remote_dataset.write_overlay_part(
        "densegen",
        pa.table({"sequence": ["AAAA"], "densegen__score": [0.91]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )

    # Simulate interrupted logging where overlay payload changed but event-sidecar line count did not.
    events_path.write_text("\n".join(baseline_lines) + ("\n" if baseline_lines else ""), encoding="utf-8")

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.pull_transfer_calls == 2
    assert _dataset_file_fingerprints(local_root / dataset_id, include_events=False) == _dataset_file_fingerprints(
        remote_root / dataset_id, include_events=False
    )


def test_push_detects_overlay_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    register_test_namespace(remote_root, namespace="infer", columns_spec="infer__llr:float64")
    register_test_namespace(local_root, namespace="infer", columns_spec="infer__llr:float64")
    dataset_id = "densegen/demo_overlay_push_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1
    assert _FilesystemRemote.push_transfer_calls == 0

    local_dataset = Dataset(local_root, dataset_id)
    local_events_path = local_root / dataset_id / ".events.log"
    baseline_lines = local_events_path.read_text(encoding="utf-8").splitlines()

    local_dataset.write_overlay_part(
        "infer",
        pa.table({"sequence": ["AAAA"], "infer__llr": [1.7]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )

    # Simulate interrupted logging where overlay payload changed but event-sidecar line count did not.
    local_events_path.write_text("\n".join(baseline_lines) + ("\n" if baseline_lines else ""), encoding="utf-8")

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.push_transfer_calls == 1
    assert _dataset_file_fingerprints(local_root / dataset_id, include_events=False) == _dataset_file_fingerprints(
        remote_root / dataset_id, include_events=False
    )


def test_pull_detects_auxiliary_file_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_aux_pull_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1

    remote_aux = remote_root / dataset_id / "_artifacts" / "batch" / "checkpoint.json"
    remote_aux.parent.mkdir(parents=True, exist_ok=True)
    remote_aux.write_text('{"epoch": 7, "status": "running"}\n', encoding="utf-8")

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.pull_transfer_calls == 2
    local_aux = local_root / dataset_id / "_artifacts" / "batch" / "checkpoint.json"
    assert local_aux.read_text(encoding="utf-8") == remote_aux.read_text(encoding="utf-8")


def test_push_detects_auxiliary_file_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_aux_push_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1
    assert _FilesystemRemote.push_transfer_calls == 0

    local_aux = local_root / dataset_id / "_artifacts" / "batch" / "checkpoint.json"
    local_aux.parent.mkdir(parents=True, exist_ok=True)
    local_aux.write_text('{"epoch": 8, "status": "queued"}\n', encoding="utf-8")

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.push_transfer_calls == 1
    remote_aux = remote_root / dataset_id / "_artifacts" / "batch" / "checkpoint.json"
    assert remote_aux.read_text(encoding="utf-8") == local_aux.read_text(encoding="utf-8")


def test_pull_detects_registry_auxiliary_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_registry_pull_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1

    remote_registry_note = remote_root / dataset_id / "_registry" / "operator-note.yaml"
    remote_registry_note.parent.mkdir(parents=True, exist_ok=True)
    remote_registry_note.write_text("source: hpc\nphase: batch-2\n", encoding="utf-8")

    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.pull_transfer_calls == 2
    local_registry_note = local_root / dataset_id / "_registry" / "operator-note.yaml"
    assert local_registry_note.read_text(encoding="utf-8") == remote_registry_note.read_text(encoding="utf-8")


def test_push_detects_registry_auxiliary_drift_when_event_sidecar_delta_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "local_usr"
    remote_root = tmp_path / "remote_usr"
    ensure_registry(local_root)
    ensure_registry(remote_root)
    dataset_id = "densegen/demo_registry_push_gap"

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed")], source="remote-seed")

    remote_cfg = SSHRemoteConfig(
        name="bu-scc",
        host="mock",
        user="mock",
        base_dir=str(remote_root),
    )
    _FilesystemRemote.fail_next_pull = False
    _FilesystemRemote.fail_next_push = False
    _FilesystemRemote.pull_transfer_calls = 0
    _FilesystemRemote.push_transfer_calls = 0
    _FilesystemRemote.remote_lock_calls = 0
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: remote_cfg)
    monkeypatch.setattr(sync_module, "SSHRemote", _FilesystemRemote)

    opts = sync_module.SyncOptions(verify="auto")
    sync_module.execute_pull(local_root, dataset_id, "bu-scc", opts)
    assert _FilesystemRemote.pull_transfer_calls == 1
    assert _FilesystemRemote.push_transfer_calls == 0

    local_registry_note = local_root / dataset_id / "_registry" / "operator-note.yaml"
    local_registry_note.parent.mkdir(parents=True, exist_ok=True)
    local_registry_note.write_text("source: local\nphase: post-analysis\n", encoding="utf-8")

    sync_module.execute_push(local_root, dataset_id, "bu-scc", opts)

    assert _FilesystemRemote.push_transfer_calls == 1
    remote_registry_note = remote_root / dataset_id / "_registry" / "operator-note.yaml"
    assert remote_registry_note.read_text(encoding="utf-8") == local_registry_note.read_text(encoding="utf-8")
