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
from pathlib import Path

from dnadesign.contracts.folding import StructureAssessmentRequestV1

from ..errors import FoldingExecutionError
from .projection import project_prediction_request, project_target_sequence
from .publication import write_model_json

_PREDICTION_REQUEST = "prediction-request.json"
_TARGET_SEQUENCE = "assessment-target-sequence.json"
_TERMINATION_DRAIN_SECONDS = 0.5


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
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_worker_group(process)
        _bounded_post_kill_wait(process)
        raise FoldingExecutionError(f"Structure assessment timed out after {timeout_seconds:g} seconds.") from exc
    except BaseException:
        _terminate_worker_group(process)
        _bounded_post_kill_wait(process)
        raise
    _terminate_worker_group(process)
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
