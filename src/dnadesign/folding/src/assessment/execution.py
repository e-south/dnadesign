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
import tempfile
from pathlib import Path
from typing import BinaryIO

from dnadesign.contracts.folding import StructureAssessmentRequestV1

from ..errors import FoldingExecutionError
from .projection import project_prediction_request, project_target_sequence
from .publication import write_model_json

_PREDICTION_REQUEST = "prediction-request.json"
_TARGET_SEQUENCE = "assessment-target-sequence.json"
_TERMINATION_DRAIN_SECONDS = 0.5
_WORKER_STREAM_LIMIT_BYTES = 1_048_576

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on non-POSIX hosts
    resource = None  # type: ignore[assignment]


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
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=os.name == "posix",
            preexec_fn=_limit_worker_file_output if resource is not None else None,
        )
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_worker_group(process)
            _bounded_post_kill_wait(process)
            raise FoldingExecutionError(f"Structure assessment timed out after {timeout_seconds:g} seconds.") from exc
        except BaseException:
            _terminate_worker_group(process)
            _bounded_post_kill_wait(process)
            raise
        _terminate_worker_group(process)
        stdout = _read_worker_stream(stdout_file, label="stdout")
        stderr = _read_worker_stream(stderr_file, label="stderr")
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"worker exited with status {process.returncode}"
            raise FoldingExecutionError(f"Structure assessment worker failed: {detail}")


def _limit_worker_file_output() -> None:
    if resource is None or not hasattr(resource, "RLIMIT_FSIZE"):
        return
    resource.setrlimit(resource.RLIMIT_FSIZE, (_WORKER_STREAM_LIMIT_BYTES, _WORKER_STREAM_LIMIT_BYTES))


def _read_worker_stream(stream: BinaryIO, *, label: str) -> str:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size >= _WORKER_STREAM_LIMIT_BYTES:
        raise FoldingExecutionError(
            f"Structure assessment worker {label} exceeded the {_WORKER_STREAM_LIMIT_BYTES}-byte limit."
        )
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace")


def _terminate_worker_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


def _bounded_post_kill_wait(process: subprocess.Popen[bytes]) -> None:
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
