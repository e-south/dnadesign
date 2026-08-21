"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/execution.py

Isolated prediction execution for one structure-assessment request.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from dnadesign.contracts.folding import StructureAssessmentRequestV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionRequestV1,
)

from ..errors import FoldingExecutionError
from .publication import write_model_json

_PREDICTION_REQUEST = "prediction-request.json"
_TARGET_SEQUENCE = "assessment-target-sequence.json"


def prediction_request(request: StructureAssessmentRequestV1) -> SecondaryStructurePredictionRequestV1:
    """Project an exact assessment target into the backend execution contract."""
    target = request.target
    return SecondaryStructurePredictionRequestV1.model_validate(
        {
            "request_id": request.assessment_id,
            "input": {
                "sequence_id": target.sequence_id,
                "sequence_sha256": target.sequence_sha256.removeprefix("sha256:"),
                "alphabet": target.alphabet,
                "topology": "linear_ssdna",
                "length": len(target.sequence),
                "sequence_artifact": f"../{_TARGET_SEQUENCE}",
            },
            "backend": request.backend.model_dump(mode="json"),
            "policy": {
                "required": request.policy.required,
                "fail_on_malformed_output": request.policy.fail_on_malformed_output,
                "fail_on_length_mismatch": request.policy.fail_on_length_mismatch,
            },
        }
    )


def write_target_sequence(path: Path, request: StructureAssessmentRequestV1) -> None:
    """Write the exact target bytes consumed by the isolated worker."""
    target = request.target
    payload = {
        "contract": "assessment_target_sequence_v1",
        "sequence": {
            "id": target.sequence_id,
            "sha256": target.sequence_sha256.removeprefix("sha256:"),
            "sequence": target.sequence,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise FoldingExecutionError(f"Structure assessment timed out after {timeout_seconds:g} seconds.") from exc
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"worker exited with status {process.returncode}"
        raise FoldingExecutionError(f"Structure assessment worker failed: {detail}")


def prepare_prediction_request(stage: Path, request: StructureAssessmentRequestV1) -> Path:
    """Write the target and backend request used by one worker invocation."""
    write_target_sequence(stage / _TARGET_SEQUENCE, request)
    prediction_dir = stage / "prediction"
    prediction_dir.mkdir()
    low_level_path = prediction_dir / _PREDICTION_REQUEST
    write_model_json(low_level_path, prediction_request(request))
    return low_level_path
