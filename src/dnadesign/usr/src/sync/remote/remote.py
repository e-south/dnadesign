"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sync/remote/remote.py

SSH remote stats and transfer helpers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Tuple

from ...contracts import RemoteUnavailableError, TransferError
from .config import SSHRemoteConfig


@dataclass
class RemotePrimaryStat:
    exists: bool
    size: Optional[int]
    sha256: Optional[str]
    rows: Optional[int]
    cols: Optional[int]
    mtime: Optional[str]


@dataclass
class RemoteDatasetStat:
    primary: RemotePrimaryStat
    meta_mtime: Optional[str]
    events_lines: int
    snapshot_names: List[str] = field(default_factory=list)
    derived_files: List[str] = field(default_factory=list)
    derived_hashes: dict[str, str] = field(default_factory=dict)
    aux_files: List[str] = field(default_factory=list)
    aux_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SSHControlSessionStatus:
    host: str
    user: str
    ssh_target: str
    batch_mode: bool
    control_master: Optional[str]
    control_path: Optional[str]
    control_persist: Optional[str]
    multiplex_enabled: bool
    socket_exists: bool
    socket_live: bool


class SSHRemote:
    """
    Thin wrapper around ssh/rsync CLI tools.
    No heavy dependencies; assertive with clear failures.
    """

    def __init__(self, cfg: SSHRemoteConfig):
        self.cfg = cfg
        self._effective_ssh_config_cache: Dict[str, str] | None = None

    # ---- subprocess helpers ----

    def _ssh_key_args(self) -> List[str]:
        if not self.cfg.ssh_key_env:
            return []
        key_env = self.cfg.ssh_key_env
        key_path = os.environ.get(key_env)
        if not key_path:
            raise RemoteUnavailableError(f"Environment variable '{key_env}' not set (SSH key path).")
        return ["-i", str(Path(key_path))]

    def _ssh_cmd(self) -> List[str]:
        cmd = ["ssh", "-o", "ConnectTimeout=10"]
        if self.cfg.batch_mode:
            cmd += ["-o", "BatchMode=yes"]
        cmd += self._ssh_key_args()
        return cmd + [f"{self.cfg.user}@{self.cfg.host}"]

    def _rsync_cmd(self) -> List[str]:
        cmd = [
            "rsync",
            # Preserve dataset contents and symlink topology, but avoid replaying
            # host-specific ownership/permission metadata onto the destination.
            "-rltz",
            "--partial",
            "--protect-args",
            "--info=progress2",
            "--delete-delay",
            "--delay-updates",
            "--no-perms",
            "--no-owner",
            "--no-group",
            "--omit-dir-times",
        ]
        ssh_parts = ["ssh", "-o", "ConnectTimeout=10"]
        if self.cfg.batch_mode:
            ssh_parts += ["-o", "BatchMode=yes"]
        key_args = self._ssh_key_args()
        if key_args:
            ssh_parts[1:1] = key_args
        ssh_opts = " ".join(shlex.quote(part) for part in ssh_parts)
        cmd += ["-e", ssh_opts]
        return cmd

    def _ssh_effective_config(self) -> Dict[str, str]:
        if self._effective_ssh_config_cache is not None:
            return dict(self._effective_ssh_config_cache)
        proc = subprocess.run(
            ["ssh", "-G", "-l", self.cfg.user, *self._ssh_key_args(), self.cfg.host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ssh config error"
            raise RemoteUnavailableError(f"ssh -G failed for {self.cfg.ssh_target}: {detail}")
        parsed: Dict[str, str] = {}
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line or " " not in line:
                continue
            key, value = line.split(None, 1)
            parsed[key.lower()] = value.strip()
        self._effective_ssh_config_cache = dict(parsed)
        return dict(parsed)

    def _keyboard_interactive_hint(self) -> str:
        return (
            f" Hint: remote '{self.cfg.name}' accepted publickey auth but still requires "
            "keyboard-interactive follow-up. Run "
            f"`usr remotes warm-auth --remote {self.cfg.name}` in a terminal, or establish "
            f"`ssh {self.cfg.host}` once before retrying sync."
        )

    def _ssh_run(self, remote_cmd: str, check: bool = True) -> Tuple[int, str, str]:
        full = self._ssh_cmd() + [remote_cmd]
        proc = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if check and proc.returncode != 0:
            raise RemoteUnavailableError(f"ssh failed ({proc.returncode}): {remote_cmd}\n{proc.stderr.strip()}")
        return proc.returncode, proc.stdout, proc.stderr

    def _ssh_probe(self, remote_cmd: str) -> Tuple[int, str, str]:
        rc, out, err = self._ssh_run(remote_cmd, check=False)
        if rc == 255:
            detail = err.strip() or "unknown ssh transport/auth error"
            if "keyboard-interactive" in detail.lower():
                detail = f"{detail}{self._keyboard_interactive_hint()}"
            raise RemoteUnavailableError(f"ssh failed ({rc}): {remote_cmd}\n{detail}")
        return rc, out, err

    def control_session_status(self) -> SSHControlSessionStatus:
        effective = self._ssh_effective_config()
        control_master = str(effective.get("controlmaster", "") or "").strip() or None
        control_path_raw = str(effective.get("controlpath", "") or "").strip() or None
        control_persist = str(effective.get("controlpersist", "") or "").strip() or None
        if control_path_raw and control_path_raw.lower() == "none":
            control_path_raw = None
        control_path = str(Path(control_path_raw).expanduser()) if control_path_raw else None
        multiplex_enabled = (
            bool(control_master) and control_master.lower() not in {"no", "false", "none"} and bool(control_path)
        )
        socket_exists = bool(control_path and Path(control_path).exists())
        socket_live = False
        if multiplex_enabled and socket_exists and control_path is not None:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    *self._ssh_key_args(),
                    "-S",
                    control_path,
                    "-O",
                    "check",
                    "-l",
                    self.cfg.user,
                    self.cfg.host,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            socket_live = proc.returncode == 0
        return SSHControlSessionStatus(
            host=self.cfg.host,
            user=self.cfg.user,
            ssh_target=self.cfg.ssh_target,
            batch_mode=bool(self.cfg.batch_mode),
            control_master=control_master,
            control_path=control_path,
            control_persist=control_persist,
            multiplex_enabled=multiplex_enabled,
            socket_exists=socket_exists,
            socket_live=socket_live,
        )

    def warm_auth_session(self) -> SSHControlSessionStatus:
        status = self.control_session_status()
        if status.socket_live:
            return status
        if not status.multiplex_enabled or not status.control_path:
            raise RemoteUnavailableError(
                f"SSH multiplexing is not configured for remote '{self.cfg.name}'. "
                "Configure `ControlMaster auto` plus `ControlPath ...` in SSH "
                "config before using `usr remotes warm-auth`."
            )
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise RemoteUnavailableError(
                f"Interactive SSH auth bootstrap requires a TTY for remote '{self.cfg.name}'. "
                f"Run `usr remotes warm-auth --remote {self.cfg.name}` in a "
                f"terminal or establish `ssh {self.cfg.host}` once before "
                "retrying sync."
            )
        Path(status.control_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "ssh",
                "-MNf",
                "-o",
                "BatchMode=no",
                *self._ssh_key_args(),
                "-S",
                status.control_path,
                "-l",
                self.cfg.user,
                self.cfg.host,
            ],
            text=True,
        )
        if proc.returncode != 0:
            raise RemoteUnavailableError(
                f"ssh warm-auth failed ({proc.returncode}) for {self.cfg.ssh_target}. "
                "Complete any site MFA prompt in a real terminal, or establish "
                f"`ssh {self.cfg.host}` manually before retrying `usr remotes status` or sync commands."
            )
        refreshed = self.control_session_status()
        if not refreshed.socket_live:
            raise RemoteUnavailableError(
                "ssh warm-auth exited but no live control socket was found for "
                f"{self.cfg.ssh_target} at {status.control_path}. "
                "Check whether MFA completed successfully, then rerun `usr remotes status`."
            )
        return refreshed

    def _dataset_lock_script(self, dataset: str, *, timeout_seconds: int) -> str:
        dataset_dir = shlex.quote(self.cfg.dataset_path(dataset))
        lock_path = shlex.quote(str(Path(self.cfg.dataset_path(dataset)) / ".usr.lock"))
        timeout = max(1, int(timeout_seconds))
        return (
            "set -eu; "
            f"mkdir -p {dataset_dir}; "
            f"exec 9>>{lock_path}; "
            f"if ! flock -x -w {timeout} 9; then echo USR_REMOTE_LOCK_TIMEOUT; exit 73; fi; "
            "echo USR_REMOTE_LOCK_ACQUIRED; "
            "IFS= read -r _usr_sync_unlock || true"
        )

    def _remote_event_locked_rsync_program(self, dataset: str, *, timeout_seconds: int = 300) -> str:
        lock_path = PurePosixPath(self.cfg.dataset_path(dataset)) / ".events.lock"
        timeout = max(1, int(timeout_seconds))
        return f"flock -x -w {timeout} {shlex.quote(str(lock_path))} rsync"

    @contextmanager
    def dataset_transfer_lock(self, dataset: str, *, timeout_seconds: int = 300) -> Iterator[None]:
        script = self._dataset_lock_script(dataset, timeout_seconds=timeout_seconds)
        proc = subprocess.Popen(
            self._ssh_cmd() + ["sh", "-lc", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        marker = ""
        noise_lines: list[str] = []
        if proc.stdout is not None:
            while True:
                line = proc.stdout.readline()
                if line == "":
                    break
                marker = line.strip()
                if marker in {"USR_REMOTE_LOCK_ACQUIRED", "USR_REMOTE_LOCK_TIMEOUT"}:
                    break
                if marker:
                    noise_lines.append(marker)
        if marker != "USR_REMOTE_LOCK_ACQUIRED":
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().strip()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
            if marker == "USR_REMOTE_LOCK_TIMEOUT":
                raise TransferError(
                    f"Remote dataset lock timeout for '{dataset}' on {self.cfg.ssh_target} "
                    f"after {max(1, int(timeout_seconds))} seconds."
                )
            detail = stderr_text or marker or "missing lock handshake marker"
            if noise_lines:
                detail = f"{detail}; stdout_before_marker={noise_lines[-1]}"
            raise TransferError(
                f"Failed to acquire remote dataset lock for '{dataset}' on {self.cfg.ssh_target}: {detail}"
            )

        release_error: str | None = None
        body_raised = False
        try:
            yield
        except Exception:
            body_raised = True
            raise
        finally:
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write("release\n")
                        proc.stdin.flush()
                        proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            if proc.returncode not in (0, None):
                stderr_text = ""
                if proc.stderr is not None:
                    stderr_text = proc.stderr.read().strip()
                release_error = stderr_text or f"remote lock session exited with code {proc.returncode}"
            if release_error is not None and not body_raised:
                raise TransferError(
                    f"Remote dataset lock release failed for '{dataset}' on {self.cfg.ssh_target}: {release_error}"
                )

    # ---- STAT helpers on remote ----

    def _remote_stat_file(self, path: str) -> Tuple[bool, Optional[int], Optional[str]]:
        # size (bytes) and mtime (epoch seconds) in a portable way
        # Try GNU coreutils:
        stat_gnu_cmd = f"stat -c '%s %Y' {shlex.quote(path)}"
        rc, out, _ = self._ssh_probe(stat_gnu_cmd)
        if rc == 0 and out.strip():
            size_s, mtime_s = out.strip().split()
            return True, int(size_s), mtime_s
        # BSD/macOS fallback:
        stat_bsd_cmd = f"stat -f '%z %m' {shlex.quote(path)}"
        rc, out, _ = self._ssh_probe(stat_bsd_cmd)
        if rc == 0 and out.strip():
            size_s, mtime_s = out.strip().split()
            return True, int(size_s), mtime_s
        # Not found or error
        # Check existence separately
        exists_cmd = f"test -f {shlex.quote(path)}"
        rc, _, _ = self._ssh_probe(exists_cmd)
        if rc == 0:
            return True, None, None
        return False, None, None

    def _remote_sha256(self, path: str) -> Optional[str]:
        # Prefer sha256sum
        sha256_cmd = f"sha256sum {shlex.quote(path)}"
        rc, out, _ = self._ssh_probe(sha256_cmd)
        if rc == 0 and out.strip():
            return out.split()[0]
        # macOS shasum
        shasum_cmd = f"shasum -a 256 {shlex.quote(path)}"
        rc, out, _ = self._ssh_probe(shasum_cmd)
        if rc == 0 and out.strip():
            return out.split()[0]
        return None

    def _remote_parquet_shape(self, path: str) -> Tuple[Optional[int], Optional[int]]:
        # Try python3 -> pyarrow; then python
        for py in ("python3", "python"):
            cmd = f"""{py} -c "import sys;import pyarrow.parquet as pq;f=pq.ParquetFile(sys.argv[1]);m=f.metadata;print(m.num_rows, m.num_columns)" {shlex.quote(path)}"""  # noqa
            rc, out, _ = self._ssh_probe(cmd)
            if rc == 0 and out.strip():
                try:
                    r, c = out.strip().split()
                    return int(r), int(c)
                except ValueError as e:
                    raise RemoteUnavailableError(f"Unexpected parquet stats output from {py} on remote: {out!r}") from e
        raise RemoteUnavailableError("Remote parquet stats unavailable. Install python + pyarrow on the remote host.")

    def _remote_wc_lines(self, path: str) -> int:
        wc_cmd = f"wc -l < {shlex.quote(path)}"
        rc, out, _ = self._ssh_probe(wc_cmd)
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
        return 0

    def _remote_list_snapshots(self, snap_dir: str) -> List[str]:
        # Names like records-YYYYMMDDThhmmss.parquet or records-YYYYMMDDThhmmssffffff.parquet
        snapshot_cmd = f"ls -1 {shlex.quote(snap_dir)} 2>/dev/null"
        rc, out, _ = self._ssh_probe(snapshot_cmd)
        if rc != 0 or not out.strip():
            return []
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        pat = re.compile(r"^records-\d{8}T\d{6,}\.parquet$")
        return [n for n in names if pat.match(n)]

    def _remote_list_derived_files(self, derived_dir: str) -> List[str]:
        # Returns file inventory relative to _derived for overlay-fidelity diffing.
        derived_cmd = f"cd {shlex.quote(derived_dir)} 2>/dev/null && find . -type f -print"
        rc, out, _ = self._ssh_probe(derived_cmd)
        if rc != 0 or not out.strip():
            return []
        files = [line.strip() for line in out.splitlines() if line.strip()]
        return self._normalize_inventory_paths(files, context="remote derived inventory")

    def _remote_list_aux_files(self, dataset_dir: str) -> List[str]:
        # Returns non-core file inventory relative to dataset root for full-fidelity sync planning.
        aux_cmd = (
            "cd "
            + shlex.quote(dataset_dir)
            + " 2>/dev/null && find . -type f "
            + "! -path './records.parquet' "
            + "! -path './meta.md' "
            + "! -path './.events.log' "
            + "! -path './.events.lock' "
            + "! -path './.usr.lock' "
            + "! -path './_snapshots/*' "
            + "! -path './_derived/*' "
            + "-print"
        )
        rc, out, _ = self._ssh_probe(aux_cmd)
        if rc != 0 or not out.strip():
            return []
        files = [line.strip() for line in out.splitlines() if line.strip()]
        return self._normalize_inventory_paths(files, context="remote auxiliary inventory")

    def _normalize_inventory_paths(self, entries: List[str], *, context: str) -> List[str]:
        normalized: List[str] = []
        for raw in entries:
            entry = raw[2:] if raw.startswith("./") else raw
            entry = entry.strip()
            path = PurePosixPath(entry)
            if not entry or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise RemoteUnavailableError(f"{context} contains unsafe relative path '{raw}'.")
            normalized.append(path.as_posix())
        return sorted(normalized)

    def _remote_hash_derived_files(self, derived_dir: str, derived_files: List[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for rel in derived_files:
            full_path = str(PurePosixPath(derived_dir).joinpath(rel))
            sha = self._remote_sha256(full_path)
            if not sha:
                raise RemoteUnavailableError(
                    "verify-derived-hashes requires remote sha256 support (sha256sum or shasum)."
                )
            hashes[rel] = sha
        return hashes

    def _remote_hash_aux_files(self, dataset_dir: str, aux_files: List[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for rel in aux_files:
            full_path = str(PurePosixPath(dataset_dir).joinpath(rel))
            sha = self._remote_sha256(full_path)
            if not sha:
                raise RemoteUnavailableError(
                    "verify-derived-hashes requires remote sha256 support (sha256sum or shasum)."
                )
            hashes[rel] = sha
        return hashes

    # ---- Public: stat/pull/push ----

    def stat_dataset(
        self, dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False
    ) -> RemoteDatasetStat:
        base = self.cfg.dataset_path(dataset)
        primary = f"{base}/records.parquet"
        meta = f"{base}/meta.md"
        events = f"{base}/.events.log"
        snaps_d = f"{base}/_snapshots"
        derived_d = f"{base}/_derived"

        exists, size_b, mtime = self._remote_stat_file(primary)
        sha = rows = cols = None
        if exists:
            if verify in {"hash", "auto"}:
                sha = self._remote_sha256(primary)
            if verify == "parquet" or (verify == "auto" and not sha and size_b is None):
                rows, cols = self._remote_parquet_shape(primary)

        meta_mtime = None
        m_exists, _, meta_mtime = self._remote_stat_file(meta)
        if not m_exists:
            meta_mtime = None

        evt_lines = self._remote_wc_lines(events)

        snapshot_names = self._remote_list_snapshots(snaps_d)
        derived_files = self._remote_list_derived_files(derived_d)
        derived_hashes = self._remote_hash_derived_files(derived_d, derived_files) if include_derived_hashes else {}
        aux_files = self._remote_list_aux_files(base)
        aux_hashes = self._remote_hash_aux_files(base, aux_files) if include_derived_hashes else {}

        return RemoteDatasetStat(
            primary=RemotePrimaryStat(
                exists=bool(exists),
                size=size_b,
                sha256=sha,
                rows=rows,
                cols=cols,
                mtime=mtime,
            ),
            meta_mtime=meta_mtime,
            events_lines=evt_lines,
            snapshot_names=snapshot_names,
            derived_files=derived_files,
            derived_hashes=derived_hashes,
            aux_files=aux_files,
            aux_hashes=aux_hashes,
        )

    def stat_file(self, remote_path: str, *, verify: str = "auto") -> RemotePrimaryStat:
        exists, size_b, mtime = self._remote_stat_file(remote_path)
        if not exists:
            return RemotePrimaryStat(False, None, None, None, None, None)
        sha = self._remote_sha256(remote_path) if verify in {"hash", "auto"} else None
        rows = cols = None
        wants_parquet = verify == "parquet" or (verify == "auto" and not sha and size_b is None)
        if remote_path.endswith(".parquet") and wants_parquet:
            rows, cols = self._remote_parquet_shape(remote_path)
        return RemotePrimaryStat(True, size_b, sha, rows, cols, mtime)

    def pull_file(self, remote_src: str, local_dst: Path, *, dry_run: bool = False) -> None:
        local_dst = Path(local_dst)
        local_dst.parent.mkdir(parents=True, exist_ok=True)
        rsync = self._rsync_cmd()
        cmd = rsync + (["--dry-run"] if dry_run else []) + [f"{self.cfg.ssh_target}:{remote_src}", str(local_dst)]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise TransferError(f"rsync file pull failed with code {proc.returncode}")

    def push_file(self, local_src: Path, remote_dst: str, *, dry_run: bool = False) -> None:
        local_src = Path(local_src)
        # ensure remote parent exists
        import shlex

        parent = Path(remote_dst).parent.as_posix()
        self._ssh_run(f"mkdir -p {shlex.quote(parent)}", check=True)
        rsync = self._rsync_cmd()
        cmd = rsync + (["--dry-run"] if dry_run else []) + [str(local_src), f"{self.cfg.ssh_target}:{remote_dst}"]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise TransferError(f"rsync file push failed with code {proc.returncode}")

    def pull_to_local(
        self,
        dataset: str,
        dest_dir: Path,
        *,
        primary_only: bool = False,
        skip_snapshots: bool = False,
        dry_run: bool = False,
    ) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = self.cfg.rsync_url(dataset)
        rsync = self._rsync_cmd()

        include_args: List[str] = ["--exclude", ".events.lock", "--exclude", ".usr.lock"]
        if primary_only:
            include_args += ["--include", "records.parquet", "--exclude", "*"]
        else:
            include_args += ["--rsync-path", self._remote_event_locked_rsync_program(dataset)]
            if skip_snapshots:
                include_args += ["--exclude", "_snapshots/**"]

        cmd = rsync + include_args + (["--dry-run"] if dry_run else []) + [src, str(dest_dir)]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise TransferError(f"rsync pull failed with code {proc.returncode}")

    def push_from_local(
        self,
        dataset: str,
        src_dir: Path,
        *,
        primary_only: bool = False,
        skip_snapshots: bool = False,
        dry_run: bool = False,
    ) -> None:
        src = str(src_dir)
        dst = self.cfg.rsync_url(dataset)
        rsync = self._rsync_cmd()

        include_args: List[str] = ["--exclude", ".events.lock", "--exclude", ".usr.lock"]
        if primary_only:
            include_args += ["--include", "records.parquet", "--exclude", "*"]
        else:
            include_args += ["--rsync-path", self._remote_event_locked_rsync_program(dataset)]
            if skip_snapshots:
                include_args += ["--exclude", "_snapshots/**"]

        # Ensure remote dataset directory exists
        self._ssh_run(f"mkdir -p {shlex.quote(self.cfg.dataset_path(dataset))}", check=True)

        cmd = rsync + include_args + (["--dry-run"] if dry_run else []) + [src + "/", dst]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise TransferError(f"rsync push failed with code {proc.returncode}")
