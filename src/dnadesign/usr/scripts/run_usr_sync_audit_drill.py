"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py

Runs a deterministic local sync drill through diff/pull/push audit call paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.cli_commands import sync as sync_commands
from dnadesign.usr.src.config import SSHRemoteConfig
from dnadesign.usr.src.errors import TransferError
from dnadesign.usr.src.registry import parse_columns_spec, register_namespace
from dnadesign.usr.src.remote import RemoteDatasetStat, RemotePrimaryStat


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
    def __init__(self, cfg: SSHRemoteConfig) -> None:
        self.cfg = cfg

    def _dataset_dir(self, dataset: str) -> Path:
        return Path(self.cfg.base_dir) / dataset

    def dataset_transfer_lock(self, _dataset: str):
        @contextmanager
        def _ctx():
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
        aux_hashes = {}
        if include_derived_hashes:
            for rel in aux_files:
                aux_hashes[rel] = _sha256(dataset_dir / rel)
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
            aux_hashes=aux_hashes,
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
        self._copy_dataset(
            self._dataset_dir(dataset), Path(dest_dir), primary_only=primary_only, skip_snapshots=skip_snapshots
        )

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
        self._copy_dataset(
            Path(src_dir), self._dataset_dir(dataset), primary_only=primary_only, skip_snapshots=skip_snapshots
        )


def _register_default_namespaces(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    register_namespace(
        root,
        namespace="densegen",
        columns=parse_columns_spec("densegen__score:float64", namespace="densegen"),
        owner="sync-audit-drill",
        description="DenseGen drill namespace.",
        overwrite=True,
    )
    register_namespace(
        root,
        namespace="infer",
        columns=parse_columns_spec("infer__llr:float64", namespace="infer"),
        owner="sync-audit-drill",
        description="Infer drill namespace.",
        overwrite=True,
    )


def _sync_args(
    *,
    root: Path,
    dataset_id: str,
    remote_name: str,
    audit_json_out: Path,
    verify: str = "hash",
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset_id,
        remote=remote_name,
        verify=verify,
        root=root,
        rich=False,
        repo_root=None,
        remote_path=None,
        format="plain",
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=False,
        no_verify_sidecars=False,
        verify_derived_hashes=False,
        no_verify_derived_hashes=False,
        strict_bootstrap_id=False,
        audit_json_out=str(audit_json_out),
    )


def _read_audit(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_sync_audit_drill(*, work_dir: Path, dataset_id: str, report_json: Path) -> dict:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    local_root = work_dir / "local_usr"
    remote_root = work_dir / "remote_usr"
    audit_dir = work_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    remote_name = "local-drill-remote"

    _register_default_namespaces(local_root)
    _register_default_namespaces(remote_root)

    remote_dataset = Dataset(remote_root, dataset_id)
    remote_dataset.init(source="remote-seed")
    remote_dataset.import_rows([_row("AAAA", "remote-seed"), _row("CCCC", "remote-seed")], source="remote-seed")
    remote_dataset.write_overlay_part(
        "densegen",
        pa.table({"sequence": ["AAAA", "CCCC"], "densegen__score": [0.31, 0.73]}),
        key="sequence",
        key_col="sequence",
        allow_missing=False,
    )
    remote_dataset.log_event("densegen_batch_complete", args={"phase": "hpc"})
    remote_dataset.snapshot()

    remote_aux = remote_root / dataset_id / "_artifacts" / "batch" / "checkpoint.json"
    remote_aux.parent.mkdir(parents=True, exist_ok=True)
    remote_aux.write_text('{"epoch": 3, "status": "complete"}\n', encoding="utf-8")
    remote_registry_note = remote_root / dataset_id / "_registry" / "operator-note.yaml"
    remote_registry_note.parent.mkdir(parents=True, exist_ok=True)
    remote_registry_note.write_text("source: remote\nphase: seeded\n", encoding="utf-8")

    remote_cfg = SSHRemoteConfig(name=remote_name, host="mock", user="mock", base_dir=str(remote_root))
    old_get_remote = sync_module.get_remote
    old_ssh_remote = sync_module.SSHRemote
    sync_module.get_remote = lambda _name: remote_cfg
    sync_module.SSHRemote = _FilesystemRemote
    try:
        diff_before_pull_path = audit_dir / "diff-before-pull.json"
        pull_path = audit_dir / "pull.json"
        diff_before_push_path = audit_dir / "diff-before-push.json"
        push_path = audit_dir / "push.json"
        diff_after_push_path = audit_dir / "diff-after-push.json"

        diff_before_pull_args = _sync_args(
            root=local_root,
            dataset_id=dataset_id,
            remote_name=remote_name,
            audit_json_out=diff_before_pull_path,
        )
        sync_commands.cmd_diff(
            diff_before_pull_args,
            resolve_output_format=lambda _args: "plain",
            print_json=lambda _payload: None,
            output_version=sync_commands.USR_OUTPUT_VERSION,
        )

        pull_args = _sync_args(
            root=local_root, dataset_id=dataset_id, remote_name=remote_name, audit_json_out=pull_path
        )
        sync_commands.cmd_pull(pull_args)

        local_dataset = Dataset(local_root, dataset_id)
        local_dataset.write_overlay_part(
            "infer",
            pa.table({"sequence": ["AAAA", "CCCC"], "infer__llr": [1.22, -0.44]}),
            key="sequence",
            key_col="sequence",
            allow_missing=False,
        )
        local_aux = local_root / dataset_id / "_artifacts" / "local" / "analysis.json"
        local_aux.parent.mkdir(parents=True, exist_ok=True)
        local_aux.write_text('{"phase": "local-infer", "ok": true}\n', encoding="utf-8")
        local_registry_note = local_root / dataset_id / "_registry" / "local-note.yaml"
        local_registry_note.parent.mkdir(parents=True, exist_ok=True)
        local_registry_note.write_text("source: local\nphase: infer\n", encoding="utf-8")

        diff_before_push_args = _sync_args(
            root=local_root,
            dataset_id=dataset_id,
            remote_name=remote_name,
            audit_json_out=diff_before_push_path,
        )
        sync_commands.cmd_diff(
            diff_before_push_args,
            resolve_output_format=lambda _args: "plain",
            print_json=lambda _payload: None,
            output_version=sync_commands.USR_OUTPUT_VERSION,
        )

        push_args = _sync_args(
            root=local_root, dataset_id=dataset_id, remote_name=remote_name, audit_json_out=push_path
        )
        sync_commands.cmd_push(push_args)

        diff_after_push_args = _sync_args(
            root=local_root,
            dataset_id=dataset_id,
            remote_name=remote_name,
            audit_json_out=diff_after_push_path,
        )
        sync_commands.cmd_diff(
            diff_after_push_args,
            resolve_output_format=lambda _args: "plain",
            print_json=lambda _payload: None,
            output_version=sync_commands.USR_OUTPUT_VERSION,
        )
    finally:
        sync_module.get_remote = old_get_remote
        sync_module.SSHRemote = old_ssh_remote

    audits = {
        "diff_before_pull": _read_audit(diff_before_pull_path),
        "pull": _read_audit(pull_path),
        "diff_before_push": _read_audit(diff_before_push_path),
        "push": _read_audit(push_path),
        "diff_after_push": _read_audit(diff_after_push_path),
    }
    final_data = audits["diff_after_push"]["data"]
    final_up_to_date = (
        not bool(final_data["primary"]["changed"])
        and not bool(final_data["_derived"]["changed"])
        and not bool(final_data["_auxiliary"]["changed"])
    )
    payload = {
        "dataset_id": dataset_id,
        "local_root": str(local_root),
        "remote_root": str(remote_root),
        "audit_dir": str(audit_dir),
        "final_up_to_date": bool(final_up_to_date),
        "audits": audits,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic USR sync audit drill.")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Working directory for local and remote drill roots."
    )
    parser.add_argument("--dataset-id", type=str, default="densegen/demo_sync_audit_drill")
    parser.add_argument("--report-json", type=Path, required=True, help="Path for machine-readable drill report.")
    args = parser.parse_args()

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="usr-sync-audit-drill-") as tmp:
            payload = run_sync_audit_drill(
                work_dir=Path(tmp),
                dataset_id=args.dataset_id,
                report_json=Path(args.report_json),
            )
    else:
        payload = run_sync_audit_drill(
            work_dir=Path(args.work_dir),
            dataset_id=args.dataset_id,
            report_json=Path(args.report_json),
        )
    print(f"sync-audit-drill report: {args.report_json}")
    print(f"final_up_to_date={payload['final_up_to_date']}")
    return 0 if payload["final_up_to_date"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
