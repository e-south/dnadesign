"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/api.py

Public secondary-structure folding API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, BinaryIO

import yaml
from pydantic import ValidationError as PydanticValidationError

from dnadesign.contracts.folding import (
    SecondaryStructureFailureKindV2,
    SecondaryStructureFailureV2,
    SecondaryStructurePredictionRequestV1,
    SecondaryStructurePredictionV2,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v2 import (
    SecondaryStructurePredictionBackendV1,
    SecondaryStructurePredictionDnaPolicyV1,
    SecondaryStructurePredictionInputV1,
    SecondaryStructureQaV1,
)

from .errors import (
    FoldingConfigError,
    FoldingError,
    FoldingExecutionError,
    FoldingLengthMismatchError,
    FoldingMalformedOutputError,
)
from .execution_metadata import (
    cli_failure_evidence_text,
    exception_evidence_text,
    prediction_command,
    prediction_log_paths,
    python_api_success_stdout,
)
from .rnafold import parse_rnafold_stdout

try:
    import resource
except ImportError:  # pragma: no cover - assessment execution fails closed off POSIX
    resource = None  # type: ignore[assignment]

_PREDICTION_FILENAME = "secondary_structure_prediction_v2.json"
_PREFLIGHT_FILENAME = "folding_preflight.json"
_BACKEND_STREAM_LIMIT_BYTES = 1_048_576


@dataclass(frozen=True)
class FoldingPreflightResult:
    status: str
    backend_name: str
    interface: str
    version: str | None
    output_dir: Path
    executable: str | None = None
    python_module: str | None = None
    resolved_executable: Path | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    failure: SecondaryStructureFailureV2 | None = None

    @property
    def backend_available(self) -> bool:
        if self.interface == "python_api":
            return self.python_module is not None and self.version is not None
        return self.resolved_executable is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "secondary_structure_folding_preflight_v2",
            "status": self.status,
            "backend": {
                "name": self.backend_name,
                "interface": self.interface,
                "executable": self.executable,
                "python_module": self.python_module,
                "resolved_executable": self.resolved_executable.as_posix()
                if self.resolved_executable is not None
                else None,
                "available": self.backend_available,
                "version": self.version,
            },
            "output_dir": self.output_dir.as_posix(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "failure": self.failure.model_dump(mode="json") if self.failure is not None else None,
        }


@dataclass(frozen=True)
class _AssembledSequence:
    sequence_id: str
    sequence_sha256: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def load_prediction_request(path: str | Path) -> tuple[SecondaryStructurePredictionRequestV1, Path]:
    request_path = Path(path).expanduser().resolve()
    if not request_path.exists():
        raise FoldingConfigError(f"Folding request not found: {request_path}")
    try:
        payload = yaml.safe_load(request_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise FoldingConfigError(f"Invalid YAML in folding request: {request_path}") from exc
    try:
        return SecondaryStructurePredictionRequestV1.model_validate(payload), request_path
    except PydanticValidationError as exc:
        raise FoldingConfigError(f"Invalid folding request {request_path}: {exc}") from exc


def preflight_request(
    request: SecondaryStructurePredictionRequestV1,
    *,
    output_dir: str | Path,
    deny_backend_child_processes: bool = False,
) -> FoldingPreflightResult:
    output_path = Path(output_dir).expanduser().resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        probe_path = output_path / ".write_probe"
        probe_path.write_text("ok\n", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        return FoldingPreflightResult(
            status="blocker_output_unwritable",
            backend_name=request.backend.name,
            interface=request.backend.interface,
            version=None,
            output_dir=output_path,
            executable=request.backend.executable,
            python_module=request.backend.python_module,
            errors=(f"Folding output path is not writable: {exc}",),
        )

    if request.backend.interface == "python_api":
        module_name = request.backend.python_module
        if module_name is None:
            status = "blocker_required_missing" if request.policy.required else "warning_optional_missing"
            message = "Folding backend python module is not configured."
            return FoldingPreflightResult(
                status=status,
                backend_name=request.backend.name,
                interface=request.backend.interface,
                version=None,
                output_dir=output_path,
                executable=request.backend.executable,
                python_module=module_name,
                warnings=() if request.policy.required else (message,),
                errors=(message,) if request.policy.required else (),
            )
        try:
            module_spec = find_spec(module_name)
        except Exception as exc:  # Discovery failures are evidence; process-control exceptions remain fatal.
            return _python_import_failure_preflight(
                request,
                output_path=output_path,
                module_name=module_name,
                error=exc,
            )
        if module_spec is None:
            status = "blocker_required_missing" if request.policy.required else "warning_optional_missing"
            message = f"Folding backend Python module '{module_name}' is not available."
            return FoldingPreflightResult(
                status=status,
                backend_name=request.backend.name,
                interface=request.backend.interface,
                version=None,
                output_dir=output_path,
                executable=request.backend.executable,
                python_module=module_name,
                warnings=() if request.policy.required else (message,),
                errors=(message,) if request.policy.required else (),
            )
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # Import failures are evidence; process-control exceptions remain fatal.
            return _python_import_failure_preflight(
                request,
                output_path=output_path,
                module_name=module_name,
                error=exc,
            )
        return FoldingPreflightResult(
            status="ok",
            backend_name=request.backend.name,
            interface=request.backend.interface,
            version=str(getattr(module, "__version__", "unknown")),
            output_dir=output_path,
            executable=request.backend.executable,
            python_module=module_name,
        )

    if request.backend.executable is None:
        status = "blocker_required_missing" if request.policy.required else "warning_optional_missing"
        message = "Folding backend executable is not configured."
        return FoldingPreflightResult(
            status=status,
            backend_name=request.backend.name,
            interface=request.backend.interface,
            version=None,
            output_dir=output_path,
            executable=request.backend.executable,
            python_module=request.backend.python_module,
            warnings=() if request.policy.required else (message,),
            errors=(message,) if request.policy.required else (),
        )
    resolved = _resolve_executable(request.backend.executable)
    if resolved is None:
        status = "blocker_required_missing" if request.policy.required else "warning_optional_missing"
        message = f"Folding backend '{request.backend.executable}' is not available."
        return FoldingPreflightResult(
            status=status,
            backend_name=request.backend.name,
            interface=request.backend.interface,
            version=None,
            output_dir=output_path,
            executable=request.backend.executable,
            resolved_executable=None,
            python_module=request.backend.python_module,
            warnings=() if request.policy.required else (message,),
            errors=(message,) if request.policy.required else (),
        )

    return FoldingPreflightResult(
        status="ok",
        backend_name=request.backend.name,
        interface=request.backend.interface,
        version=_capture_version(
            resolved,
            deny_backend_child_processes=deny_backend_child_processes,
        ),
        output_dir=output_path,
        executable=request.backend.executable,
        resolved_executable=resolved,
        python_module=request.backend.python_module,
    )


def _python_import_failure_preflight(
    request: SecondaryStructurePredictionRequestV1,
    *,
    output_path: Path,
    module_name: str,
    error: Exception,
) -> FoldingPreflightResult:
    message = f"ViennaRNA Python API import failed: {error}"
    failure = SecondaryStructureFailureV2(
        kind="backend_import_exception",
        message=message,
        exception_type=type(error).__name__,
    )
    return FoldingPreflightResult(
        status="error",
        backend_name=request.backend.name,
        interface=request.backend.interface,
        version=None,
        output_dir=output_path,
        executable=request.backend.executable,
        python_module=module_name,
        errors=(message,),
        failure=failure,
    )


def run_prediction_request(
    request: SecondaryStructurePredictionRequestV1,
    *,
    output_dir: str | Path,
    request_path: str | Path | None = None,
    raise_on_required_failure: bool = True,
    backend_timeout_seconds: float | None = 60.0,
    deny_backend_child_processes: bool = False,
) -> SecondaryStructurePredictionV2:
    if backend_timeout_seconds is not None and backend_timeout_seconds <= 0:
        raise FoldingConfigError("backend_timeout_seconds must be positive or None.")
    output_path = Path(output_dir).expanduser().resolve()
    if deny_backend_child_processes and request.backend.interface == "python_api":
        _deny_process_creation()
    preflight = preflight_request(
        request,
        output_dir=output_path,
        deny_backend_child_processes=deny_backend_child_processes,
    )
    _write_json(output_path / _PREFLIGHT_FILENAME, preflight.to_dict())
    if preflight.failure is not None:
        if (
            preflight.interface != "python_api"
            or preflight.failure.kind != "backend_import_exception"
            or preflight.failure.exception_type is None
        ):
            raise FoldingExecutionError("Python import preflight returned an invalid typed failure.")
        assembled = _load_assembled_sequence(request, request_path=request_path)
        _submitted_value, submitted_alphabet = _submitted_sequence(
            assembled.sequence,
            dna_policy=request.backend.dna_policy.mode,
        )
        command = prediction_command(
            interface=request.backend.interface,
            python_module=request.backend.python_module,
            resolved_executable=preflight.resolved_executable,
            parameters=request.backend.parameters,
        )
        _write_backend_logs(
            output_path,
            interface=request.backend.interface,
            stdout="",
            stderr=exception_evidence_text(
                exception_type=preflight.failure.exception_type,
                message=preflight.failure.message,
            ),
        )
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=preflight.failure.message,
            failure_kind=preflight.failure.kind,
            exception_type=preflight.failure.exception_type,
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0])
        return prediction
    if preflight.status != "ok":
        prediction = _prediction_for_preflight_blocker(request, preflight)
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError("; ".join(preflight.errors or preflight.warnings))
        return prediction

    assembled = _load_assembled_sequence(request, request_path=request_path)
    submitted_sequence, submitted_alphabet = _submitted_sequence(
        assembled.sequence,
        dna_policy=request.backend.dna_policy.mode,
    )
    if request.backend.interface == "python_api":
        return _run_python_api_prediction_request(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_sequence=submitted_sequence,
            submitted_alphabet=submitted_alphabet,
            output_path=output_path,
            raise_on_required_failure=raise_on_required_failure,
        )

    if preflight.resolved_executable is None:
        raise FoldingExecutionError("Successful CLI preflight lacks a resolved executable.")
    command = prediction_command(
        interface=request.backend.interface,
        python_module=request.backend.python_module,
        resolved_executable=preflight.resolved_executable,
        parameters=request.backend.parameters,
    )
    try:
        completed = _run_bounded_cli_command(
            command,
            input_text=f">{request.input.sequence_id}\n{submitted_sequence}\n",
            timeout=backend_timeout_seconds,
            deny_backend_child_processes=deny_backend_child_processes,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"ViennaRNA RNAfold CLI execution failed: {exc}"
        exception_type = "OSError" if isinstance(exc, OSError) else "SubprocessError"
        _write_backend_logs(
            output_path,
            interface=request.backend.interface,
            stdout="",
            stderr=exception_evidence_text(exception_type=exception_type, message=error),
        )
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=error,
            failure_kind="backend_invocation_exception",
            exception_type=exception_type,
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0]) from exc
        return prediction

    if completed.returncode != 0:
        _write_backend_logs(
            output_path,
            interface=request.backend.interface,
            stdout=completed.stdout,
            stderr=cli_failure_evidence_text(
                returncode=completed.returncode,
                backend_stderr=completed.stderr,
            ),
        )
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=f"ViennaRNA RNAfold CLI exited with status {completed.returncode}.",
            failure_kind="backend_nonzero_exit",
            returncode=completed.returncode,
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0])
        return prediction

    _write_backend_logs(
        output_path,
        interface=request.backend.interface,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    try:
        result = parse_rnafold_stdout(
            stdout=completed.stdout,
            submitted_sequence=submitted_sequence,
            input_length=assembled.length,
        )
        prediction = SecondaryStructurePredictionV2(
            prediction_id=request.request_id,
            status="ok",
            input=_prediction_input(assembled, request),
            backend=SecondaryStructurePredictionBackendV1(
                name=request.backend.name,
                version=preflight.version or "unknown",
                command=command,
                parameters=request.backend.parameters,
            ),
            dna_policy=SecondaryStructurePredictionDnaPolicyV1(
                mode=request.backend.dna_policy.mode,
                submitted_alphabet=submitted_alphabet,
                coordinates_mapped_to=request.backend.dna_policy.output_coordinates,
            ),
            result=result,
            qa=SecondaryStructureQaV1(length_matches_input=True),
            artifacts=_artifact_refs(interface=request.backend.interface),
        )
    except FoldingError as exc:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=str(exc),
            failure_kind="output_parse_exception",
            exception_type=type(exc).__name__,
        )
        if _parse_failure_requires_raise(
            request,
            exc,
            raise_on_required_failure=raise_on_required_failure,
        ):
            _write_prediction(output_path, prediction)
            raise
    _write_prediction(output_path, prediction)
    return prediction


def _run_python_api_prediction_request(
    request: SecondaryStructurePredictionRequestV1,
    *,
    assembled: _AssembledSequence,
    preflight: FoldingPreflightResult,
    submitted_sequence: str,
    submitted_alphabet: str,
    output_path: Path,
    raise_on_required_failure: bool,
) -> SecondaryStructurePredictionV2:
    module_name = request.backend.python_module
    if module_name is None:
        raise FoldingExecutionError("Successful Python preflight lacks a configured module.")

    command = prediction_command(
        interface=request.backend.interface,
        python_module=module_name,
        resolved_executable=None,
        parameters=request.backend.parameters,
    )
    try:
        stdout = _run_python_api_mfe(
            module_name=module_name,
            submitted_sequence=submitted_sequence,
            sequence_id=request.input.sequence_id,
            parameters=request.backend.parameters,
        )
    except Exception as exc:  # Backend failures are evidence; process-control exceptions remain fatal.
        error = f"ViennaRNA Python API execution failed: {exc}"
        exception_type = type(exc).__name__
        _write_backend_logs(
            output_path,
            interface=request.backend.interface,
            stdout="",
            stderr=exception_evidence_text(exception_type=exception_type, message=error),
        )
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=error,
            failure_kind="backend_exception",
            exception_type=exception_type,
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0]) from exc
        return prediction

    _write_backend_logs(
        output_path,
        interface=request.backend.interface,
        stdout=stdout,
        stderr="",
    )
    try:
        result = parse_rnafold_stdout(
            stdout=stdout,
            submitted_sequence=submitted_sequence,
            input_length=assembled.length,
        )
        prediction = SecondaryStructurePredictionV2(
            prediction_id=request.request_id,
            status="ok",
            input=_prediction_input(assembled, request),
            backend=SecondaryStructurePredictionBackendV1(
                name=request.backend.name,
                version=preflight.version or "unknown",
                command=command,
                parameters=request.backend.parameters,
            ),
            dna_policy=SecondaryStructurePredictionDnaPolicyV1(
                mode=request.backend.dna_policy.mode,
                submitted_alphabet=submitted_alphabet,
                coordinates_mapped_to=request.backend.dna_policy.output_coordinates,
            ),
            result=result,
            qa=SecondaryStructureQaV1(length_matches_input=True),
            artifacts=_artifact_refs(interface=request.backend.interface),
        )
    except FoldingError as exc:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=str(exc),
            failure_kind="output_parse_exception",
            exception_type=type(exc).__name__,
        )
        if _parse_failure_requires_raise(
            request,
            exc,
            raise_on_required_failure=raise_on_required_failure,
        ):
            _write_prediction(output_path, prediction)
            raise
    _write_prediction(output_path, prediction)
    return prediction


def _run_python_api_mfe(
    *,
    module_name: str,
    submitted_sequence: str,
    sequence_id: str,
    parameters: dict[str, Any],
) -> str:
    module = importlib.import_module(module_name)
    fold_compound = getattr(module, "fold_compound")
    model_details = _python_api_model_details(module, parameters=parameters)
    if model_details is None:
        compound = fold_compound(submitted_sequence)
    else:
        compound = fold_compound(submitted_sequence, model_details)
    raw_result = compound.mfe()
    if not isinstance(raw_result, (tuple, list)) or len(raw_result) != 2:
        raise FoldingExecutionError("ViennaRNA fold_compound.mfe() returned an unsupported result.")
    dot_bracket, mfe_kcal_mol = raw_result
    if not isinstance(dot_bracket, str) or "\n" in dot_bracket or "\r" in dot_bracket:
        raise FoldingExecutionError("ViennaRNA fold_compound.mfe() returned an unsupported dot-bracket value.")
    energy = float(mfe_kcal_mol)
    if not math.isfinite(energy):
        raise FoldingExecutionError("ViennaRNA fold_compound.mfe() returned a non-finite energy.")
    return python_api_success_stdout(
        sequence_id=sequence_id,
        submitted_sequence=submitted_sequence,
        dot_bracket=dot_bracket,
        mfe_kcal_mol=energy,
    )


def _parse_failure_requires_raise(
    request: SecondaryStructurePredictionRequestV1,
    error: FoldingError,
    *,
    raise_on_required_failure: bool,
) -> bool:
    if request.policy.required and raise_on_required_failure:
        return True
    if isinstance(error, FoldingLengthMismatchError):
        return request.policy.fail_on_length_mismatch
    if isinstance(error, FoldingMalformedOutputError):
        return request.policy.fail_on_malformed_output
    return False


def _python_api_model_details(module: Any, *, parameters: dict[str, Any]) -> object | None:
    temperature_c = parameters.get("temperature_c")
    if temperature_c is None:
        return None
    md_factory = getattr(module, "md", None)
    if not callable(md_factory):
        return None
    model_details = md_factory()
    model_details.temperature = float(temperature_c)
    return model_details


def _prediction_for_preflight_blocker(
    request: SecondaryStructurePredictionRequestV1,
    preflight: FoldingPreflightResult,
) -> SecondaryStructurePredictionV2:
    return SecondaryStructurePredictionV2(
        prediction_id=request.request_id,
        status=preflight.status,  # type: ignore[arg-type]
        input=SecondaryStructurePredictionInputV1(
            sequence_id=request.input.sequence_id,
            sequence_sha256=request.input.sequence_sha256,
            alphabet=request.input.alphabet,
            topology=request.input.topology,
            length=request.input.length,
        ),
        qa=SecondaryStructureQaV1(
            length_matches_input=None,
            warnings=list(preflight.warnings),
            errors=list(preflight.errors),
        ),
    )


def _error_prediction(
    request: SecondaryStructurePredictionRequestV1,
    *,
    assembled: _AssembledSequence,
    preflight: FoldingPreflightResult,
    submitted_alphabet: str,
    command: list[str],
    error: str,
    failure_kind: SecondaryStructureFailureKindV2,
    returncode: int | None = None,
    exception_type: str | None = None,
) -> SecondaryStructurePredictionV2:
    return SecondaryStructurePredictionV2(
        prediction_id=request.request_id,
        status="error",
        input=_prediction_input(assembled, request),
        backend=SecondaryStructurePredictionBackendV1(
            name=request.backend.name,
            version=preflight.version or "unknown",
            command=command,
            parameters=request.backend.parameters,
        ),
        dna_policy=SecondaryStructurePredictionDnaPolicyV1(
            mode=request.backend.dna_policy.mode,
            submitted_alphabet=submitted_alphabet,
            coordinates_mapped_to=request.backend.dna_policy.output_coordinates,
        ),
        failure=SecondaryStructureFailureV2(
            kind=failure_kind,
            message=error,
            returncode=returncode,
            exception_type=exception_type,
        ),
        qa=SecondaryStructureQaV1(length_matches_input=None, errors=[error]),
        artifacts=_artifact_refs(interface=preflight.interface),
    )


def _load_assembled_sequence(
    request: SecondaryStructurePredictionRequestV1,
    *,
    request_path: str | Path | None,
) -> _AssembledSequence:
    artifact_path = Path(request.input.sequence_artifact).expanduser()
    if not artifact_path.is_absolute() and request_path is not None:
        artifact_path = Path(request_path).expanduser().resolve().parent / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.exists():
        raise FoldingConfigError(f"Assembled sequence artifact not found: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoldingConfigError(f"Invalid assembled sequence JSON: {artifact_path}") from exc
    sequence_payload = payload.get("sequence")
    if not isinstance(sequence_payload, dict):
        raise FoldingConfigError("Assembled sequence artifact missing sequence object.")
    sequence = str(sequence_payload.get("sequence") or "")
    if not sequence:
        raise FoldingConfigError("Assembled sequence artifact contains an empty sequence.")
    sequence_sha256 = str(sequence_payload.get("sha256") or "")
    sequence_id = str(sequence_payload.get("id") or "")
    observed_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    if sequence_id != request.input.sequence_id:
        raise FoldingConfigError(
            f"Request sequence_id '{request.input.sequence_id}' does not match artifact id '{sequence_id}'."
        )
    if sequence_sha256 != request.input.sequence_sha256 or observed_sha256 != request.input.sequence_sha256:
        raise FoldingConfigError("Request sequence_sha256 does not match assembled sequence artifact.")
    if len(sequence) != request.input.length:
        raise FoldingConfigError("Request input length does not match assembled sequence artifact.")
    return _AssembledSequence(
        sequence_id=sequence_id,
        sequence_sha256=sequence_sha256,
        sequence=sequence,
    )


def _submitted_sequence(sequence: str, *, dna_policy: str) -> tuple[str, str]:
    upper = sequence.upper()
    if dna_policy == "convert_t_to_u_for_rna_backend":
        return upper.replace("T", "U"), "rna_surrogate"
    if dna_policy == "backend_accepts_dna_directly":
        return upper, "dna"
    raise FoldingConfigError(f"Unsupported DNA/RNA backend policy: {dna_policy}")


def _prediction_input(
    assembled: _AssembledSequence,
    request: SecondaryStructurePredictionRequestV1,
) -> SecondaryStructurePredictionInputV1:
    return SecondaryStructurePredictionInputV1(
        sequence_id=assembled.sequence_id,
        sequence_sha256=assembled.sequence_sha256,
        alphabet=request.input.alphabet,
        topology=request.input.topology,
        length=assembled.length,
    )


def _artifact_refs(*, interface: str):
    from dnadesign.contracts.folding.secondary_structure_prediction_v2 import SecondaryStructureArtifactsV1

    stdout, stderr = prediction_log_paths(interface=interface)
    return SecondaryStructureArtifactsV1(stdout=stdout, stderr=stderr)


def _write_backend_logs(
    output_path: Path,
    *,
    interface: str,
    stdout: str,
    stderr: str,
) -> None:
    artifacts = _artifact_refs(interface=interface)
    if artifacts.stdout is None or artifacts.stderr is None:
        raise FoldingExecutionError("Folding backend log references are incomplete.")
    (output_path / artifacts.stdout).write_text(stdout, encoding="utf-8")
    (output_path / artifacts.stderr).write_text(stderr, encoding="utf-8")


def _resolve_executable(executable: str) -> Path | None:
    if "/" in executable or "\\" in executable:
        path = Path(executable).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
        return None
    resolved = shutil.which(executable)
    return Path(resolved).resolve() if resolved else None


def _deny_process_creation() -> None:
    if (
        os.name != "posix"
        or resource is None
        or not hasattr(resource, "RLIMIT_NPROC")
        or not hasattr(resource, "RLIMIT_FSIZE")
    ):
        raise FoldingConfigError("Kernel no-fork containment is unavailable for this assessment backend.")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise FoldingConfigError("Kernel no-fork containment is not enforceable for a root assessment process.")
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_BACKEND_STREAM_LIMIT_BYTES, _BACKEND_STREAM_LIMIT_BYTES))


def _run_bounded_cli_command(
    command: list[str],
    *,
    input_text: str | None,
    timeout: float | None,
    deny_backend_child_processes: bool,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            preexec_fn=_deny_process_creation if deny_backend_child_processes else None,
        )
        try:
            process.communicate(input=input_text, timeout=timeout)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.communicate()
            raise
        stdout = _read_bounded_cli_stream(stdout_file, label="stdout")
        stderr = _read_bounded_cli_stream(stderr_file, label="stderr")
        if process.returncode is None:
            raise FoldingExecutionError("ViennaRNA backend completed without a return code.")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _read_bounded_cli_stream(stream: BinaryIO, *, label: str) -> str:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size >= _BACKEND_STREAM_LIMIT_BYTES:
        raise FoldingExecutionError(f"ViennaRNA backend {label} exceeded the {_BACKEND_STREAM_LIMIT_BYTES}-byte limit.")
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace")


def _capture_version(
    executable: Path,
    *,
    deny_backend_child_processes: bool = False,
) -> str:
    try:
        completed = _run_bounded_cli_command(
            [executable.as_posix(), "--version"],
            input_text=None,
            timeout=10,
            deny_backend_child_processes=deny_backend_child_processes,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0].strip() if text else "unknown"


def _write_prediction(output_path: Path, prediction: SecondaryStructurePredictionV2) -> None:
    _write_json(output_path / _PREDICTION_FILENAME, prediction.model_dump(mode="json"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "FoldingPreflightResult",
    "load_prediction_request",
    "parse_rnafold_stdout",
    "preflight_request",
    "run_prediction_request",
]
