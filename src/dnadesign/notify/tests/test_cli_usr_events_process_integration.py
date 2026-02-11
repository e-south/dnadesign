"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_usr_events_process_integration.py

Process-level notify watcher integration tests using local HTTP capture only.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _python_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src_root = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing else f"{src_root}:{existing}"
    return env


def _event(action: str = "materialize") -> dict:
    return {
        "event_version": 1,
        "timestamp_utc": "2026-02-06T00:00:00+00:00",
        "action": action,
        "dataset": {"name": "demo", "root": "/tmp/datasets"},
        "args": {"namespace": "densegen"},
        "metrics": {"rows_written": 3},
        "artifacts": {"overlay": {"namespace": "densegen"}},
        "fingerprint": {"rows": 1, "cols": 2, "size_bytes": 128, "sha256": None},
        "registry_hash": "abc123",
        "actor": {"tool": "densegen", "run_id": "run-1", "host": "host", "pid": 123},
        "version": "0.1.0",
    }


def _append_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        self.server.calls.append(payload)  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, _format: str, *_args) -> None:
        return


def _start_capture_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.calls = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/webhook"
    return server, thread, url


def _wait_for(predicate, timeout_seconds: float = 10.0, interval_seconds: float = 0.05) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_seconds)
    return False


def _spawn_watch_process(
    *,
    repo_root: Path,
    events: Path,
    cursor: Path,
    url: str,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    args = [
        sys.executable,
        "-m",
        "dnadesign.notify.cli",
        "usr-events",
        "watch",
        "--provider",
        "generic",
        "--events",
        str(events),
        "--cursor",
        str(cursor),
        "--url",
        url,
        "--poll-interval-seconds",
        "0.05",
        "--wait-for-events",
        "--follow",
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.Popen(
        args,
        cwd=str(repo_root),
        env=_python_env(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _unused_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.mark.skipif(shutil.which("bash") is None, reason="shell runtime required")
def test_watch_process_resumes_after_interrupt_without_duplicate_posts(tmp_path: Path) -> None:
    repo_root = _repo_root()
    events = tmp_path / "usr" / "demo" / ".events.log"
    cursor = tmp_path / "outputs" / "notify" / "densegen" / "cursor"
    _append_events(events, [_event(action="materialize")])

    server, thread, url = _start_capture_server()
    proc1: subprocess.Popen[str] | None = None
    proc2: subprocess.Popen[str] | None = None
    try:
        proc1 = _spawn_watch_process(repo_root=repo_root, events=events, cursor=cursor, url=url)
        ok = _wait_for(lambda: len(server.calls) >= 1 and cursor.exists() and cursor.read_text().strip() != "")
        assert ok, "first watcher did not deliver initial event and advance cursor in time"

        proc1.terminate()
        try:
            proc1.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait(timeout=10)

        terminal_event = _event(action="densegen_health")
        terminal_event["args"] = {"status": "completed"}
        _append_events(events, [terminal_event])

        proc2 = _spawn_watch_process(
            repo_root=repo_root,
            events=events,
            cursor=cursor,
            url=url,
            extra_args=["--stop-on-terminal-status", "--idle-timeout", "15"],
        )
        return_code = proc2.wait(timeout=20)
        assert return_code == 0
        assert _wait_for(lambda: len(server.calls) >= 2)
        assert len(server.calls) == 2
        assert [payload["status"] for payload in server.calls] == ["running", "success"]
    finally:
        if proc1 is not None and proc1.poll() is None:
            proc1.kill()
            proc1.wait(timeout=10)
        if proc2 is not None and proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(shutil.which("bash") is None, reason="shell runtime required")
def test_watch_process_spools_when_endpoint_unavailable_and_spool_drain_replays(tmp_path: Path) -> None:
    repo_root = _repo_root()
    events = tmp_path / "usr" / "demo" / ".events.log"
    cursor = tmp_path / "outputs" / "notify" / "densegen" / "cursor"
    spool_dir = tmp_path / "outputs" / "notify" / "densegen" / "spool"
    _append_events(events, [_event(action="materialize")])

    dead_url = f"http://127.0.0.1:{_unused_local_port()}/webhook"
    watch_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.notify.cli",
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--cursor",
            str(cursor),
            "--spool-dir",
            str(spool_dir),
            "--url",
            dead_url,
            "--retry-max",
            "1",
            "--connect-timeout",
            "0.2",
            "--read-timeout",
            "0.2",
        ],
        cwd=str(repo_root),
        env=_python_env(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert watch_result.returncode == 0, watch_result.stderr
    assert list(spool_dir.glob("*.json"))

    server, thread, url = _start_capture_server()
    try:
        drain_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnadesign.notify.cli",
                "spool",
                "drain",
                "--spool-dir",
                str(spool_dir),
                "--provider",
                "generic",
                "--url",
                url,
                "--retry-max",
                "1",
                "--connect-timeout",
                "1",
                "--read-timeout",
                "1",
            ],
            cwd=str(repo_root),
            env=_python_env(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        assert drain_result.returncode == 0, drain_result.stderr
        assert _wait_for(lambda: len(server.calls) == 1)
        assert not list(spool_dir.glob("*.json"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
