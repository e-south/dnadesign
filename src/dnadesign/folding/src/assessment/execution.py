"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/execution.py

Isolated prediction execution for one structure-assessment request.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import psutil

from dnadesign.contracts.folding import StructureAssessmentRequestV1

from ..errors import FoldingExecutionError
from .projection import project_prediction_request, project_target_sequence
from .publication import write_model_json

_PREDICTION_REQUEST = "prediction-request.json"
_TARGET_SEQUENCE = "assessment-target-sequence.json"
_TERMINATION_DRAIN_SECONDS = 0.5
_DESCENDANT_TERMINATION_SECONDS = 0.5
_DESCENDANT_POLL_SECONDS = 0.005


class _DescendantTracker:
    """Retain process identities even when a backend later reparents them."""

    def __init__(self, root_pid: int) -> None:
        self._root_pid = root_pid
        self._processes: set[psutil.Process] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._watch, name="folding-descendant-tracker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def capture(self) -> None:
        descendants = _descendants(self._root_pid)
        with self._lock:
            self._processes.update(descendants)

    def stop(self) -> tuple[psutil.Process, ...]:
        self._stop.set()
        self._thread.join(timeout=_DESCENDANT_TERMINATION_SECONDS)
        if self._thread.is_alive():
            raise FoldingExecutionError("Structure assessment descendant tracking did not stop.")
        if self._error is not None:
            raise FoldingExecutionError("Structure assessment could not inspect backend descendants.") from self._error
        with self._lock:
            return tuple(self._processes)

    def _watch(self) -> None:
        try:
            while not self._stop.is_set():
                self.capture()
                self._stop.wait(_DESCENDANT_POLL_SECONDS)
        except BaseException as exc:
            self._error = exc
            self._stop.set()


def write_target_sequence(path: Path, request: StructureAssessmentRequestV1) -> bytes:
    """Write the exact target bytes consumed by the isolated worker."""
    return write_model_json(path, project_target_sequence(request))


def run_worker(request_path: Path, output_path: Path, *, timeout_seconds: float) -> None:
    """Run and, on timeout, terminate the assessment worker process group."""
    command = [
        sys.executable,
        "-m",
        "dnadesign.folding.src.assessment.worker",
        request_path.as_posix(),
        output_path.as_posix(),
    ]
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    tracker = _DescendantTracker(process.pid)
    tracker.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_worker_tree(process, tracker)
        _bounded_post_kill_wait(process)
        raise FoldingExecutionError(f"Structure assessment timed out after {timeout_seconds:g} seconds.") from exc
    except BaseException:
        _terminate_worker_tree(process, tracker)
        _bounded_post_kill_wait(process)
        raise
    _terminate_worker_tree(process, tracker)
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"worker exited with status {process.returncode}"
        raise FoldingExecutionError(f"Structure assessment worker failed: {detail}")


def _terminate_worker_group(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


def _terminate_worker_tree(process: subprocess.Popen[str], tracker: _DescendantTracker) -> None:
    tracker.capture()
    _terminate_worker_group(process)
    descendants = tracker.stop()
    _terminate_processes(descendants)


def terminate_current_process_descendants() -> None:
    """Terminate backend descendants before an isolated worker exits."""
    descendants = _descendants(os.getpid())
    _terminate_processes(descendants)


def _descendants(pid: int) -> tuple[psutil.Process, ...]:
    try:
        return tuple(psutil.Process(pid).children(recursive=True))
    except psutil.NoSuchProcess:
        return ()
    except psutil.AccessDenied as exc:
        raise FoldingExecutionError(
            f"Structure assessment cannot inspect backend process tree rooted at {pid}."
        ) from exc


def _terminate_processes(processes: tuple[psutil.Process, ...]) -> None:
    for process in reversed(processes):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=_DESCENDANT_TERMINATION_SECONDS)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=_DESCENDANT_TERMINATION_SECONDS)
    if alive:
        pids = ", ".join(str(process.pid) for process in alive)
        raise FoldingExecutionError(f"Structure assessment could not terminate backend descendants: {pids}.")


def _bounded_post_kill_wait(process: subprocess.Popen[str]) -> None:
    try:
        process.communicate(timeout=_TERMINATION_DRAIN_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()
    try:
        process.wait(timeout=_TERMINATION_DRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_TERMINATION_DRAIN_SECONDS)


def prepare_prediction_request(stage: Path, request: StructureAssessmentRequestV1) -> tuple[Path, bytes]:
    """Write the target and backend request used by one worker invocation."""
    target_content = write_target_sequence(stage / _TARGET_SEQUENCE, request)
    prediction_dir = stage / "prediction"
    prediction_dir.mkdir()
    low_level_path = prediction_dir / _PREDICTION_REQUEST
    write_model_json(low_level_path, project_prediction_request(request))
    return low_level_path, target_content
