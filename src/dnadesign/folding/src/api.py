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
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1, SecondaryStructurePredictionV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
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
from .rnafold import parse_rnafold_stdout

_PREDICTION_FILENAME = "secondary_structure_prediction_v1.json"
_PREFLIGHT_FILENAME = "folding_preflight.json"
_STDOUT_FILENAME = "RNAfold.stdout.txt"
_STDERR_FILENAME = "RNAfold.stderr.txt"
_PYTHON_API_STDOUT_FILENAME = "ViennaRNA.python_api.stdout.txt"
_PYTHON_API_STDERR_FILENAME = "ViennaRNA.python_api.stderr.txt"


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

    @property
    def backend_available(self) -> bool:
        if self.interface == "python_api":
            return self.python_module is not None and self.version is not None
        return self.resolved_executable is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "secondary_structure_folding_preflight_v1",
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
            module = importlib.import_module(module_name)
        except ImportError as exc:
            status = "blocker_required_missing" if request.policy.required else "warning_optional_missing"
            message = f"Folding backend Python module '{module_name}' is not available: {exc}"
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
        version=_capture_version(resolved),
        output_dir=output_path,
        executable=request.backend.executable,
        resolved_executable=resolved,
        python_module=request.backend.python_module,
    )


def run_prediction_request(
    request: SecondaryStructurePredictionRequestV1,
    *,
    output_dir: str | Path,
    request_path: str | Path | None = None,
    raise_on_required_failure: bool = True,
    backend_timeout_seconds: float | None = 60.0,
) -> SecondaryStructurePredictionV1:
    if backend_timeout_seconds is not None and backend_timeout_seconds <= 0:
        raise FoldingConfigError("backend_timeout_seconds must be positive or None.")
    output_path = Path(output_dir).expanduser().resolve()
    preflight = preflight_request(request, output_dir=output_path)
    _write_json(output_path / _PREFLIGHT_FILENAME, preflight.to_dict())
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
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=[request.backend.executable or request.backend.name],
            error="Folding executable preflight succeeded without a resolved executable.",
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0])
        return prediction
    command = _rnafold_cli_command(preflight.resolved_executable, parameters=request.backend.parameters)
    stdout_path = output_path / _STDOUT_FILENAME
    stderr_path = output_path / _STDERR_FILENAME
    try:
        completed = subprocess.run(
            command,
            input=f">{request.input.sequence_id}\n{submitted_sequence}\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=backend_timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=f"ViennaRNA RNAfold CLI execution failed: {exc}",
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0]) from exc
        return prediction

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=f"ViennaRNA RNAfold CLI exited with status {completed.returncode}.",
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0])
        return prediction

    try:
        result = parse_rnafold_stdout(
            stdout=completed.stdout,
            submitted_sequence=submitted_sequence,
            input_length=assembled.length,
        )
        prediction = SecondaryStructurePredictionV1(
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


def _rnafold_cli_command(executable: Path, *, parameters: dict[str, Any]) -> list[str]:
    command = [executable.as_posix(), "--noPS"]
    temperature_c = parameters.get("temperature_c")
    if temperature_c is not None:
        command.extend(["--temp", f"{float(temperature_c):g}"])
    return command


def _run_python_api_prediction_request(
    request: SecondaryStructurePredictionRequestV1,
    *,
    assembled: _AssembledSequence,
    preflight: FoldingPreflightResult,
    submitted_sequence: str,
    submitted_alphabet: str,
    output_path: Path,
    raise_on_required_failure: bool,
) -> SecondaryStructurePredictionV1:
    module_name = request.backend.python_module
    if module_name is None:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=[request.backend.name],
            error="Folding backend python module is not configured.",
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0])
        return prediction

    command = [f"{module_name}.fold_compound", "mfe"]
    stdout_path = output_path / _PYTHON_API_STDOUT_FILENAME
    stderr_path = output_path / _PYTHON_API_STDERR_FILENAME
    try:
        stdout = _run_python_api_mfe(
            module_name=module_name,
            submitted_sequence=submitted_sequence,
            sequence_id=request.input.sequence_id,
            parameters=request.backend.parameters,
        )
    except (FoldingError, ImportError, AttributeError, TypeError, ValueError) as exc:
        prediction = _error_prediction(
            request,
            assembled=assembled,
            preflight=preflight,
            submitted_alphabet=submitted_alphabet,
            command=command,
            error=f"ViennaRNA Python API execution failed: {exc}",
        )
        _write_prediction(output_path, prediction)
        if request.policy.required and raise_on_required_failure:
            raise FoldingExecutionError(prediction.qa.errors[0]) from exc
        return prediction

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    try:
        result = parse_rnafold_stdout(
            stdout=stdout,
            submitted_sequence=submitted_sequence,
            input_length=assembled.length,
        )
        prediction = SecondaryStructurePredictionV1(
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
    return f">{sequence_id}\n{submitted_sequence}\n{dot_bracket} ({float(mfe_kcal_mol):.2f})\n"


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
) -> SecondaryStructurePredictionV1:
    return SecondaryStructurePredictionV1(
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
) -> SecondaryStructurePredictionV1:
    return SecondaryStructurePredictionV1(
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
    from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructureArtifactsV1

    if interface == "python_api":
        return SecondaryStructureArtifactsV1(
            stdout=_PYTHON_API_STDOUT_FILENAME,
            stderr=_PYTHON_API_STDERR_FILENAME,
        )
    return SecondaryStructureArtifactsV1(
        stdout=_STDOUT_FILENAME,
        stderr=_STDERR_FILENAME,
    )


def _resolve_executable(executable: str) -> Path | None:
    if "/" in executable or "\\" in executable:
        path = Path(executable).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
        return None
    resolved = shutil.which(executable)
    return Path(resolved).resolve() if resolved else None


def _capture_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [executable.as_posix(), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0].strip() if text else "unknown"


def _write_prediction(output_path: Path, prediction: SecondaryStructurePredictionV1) -> None:
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
