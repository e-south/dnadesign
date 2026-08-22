"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/publication.py

Canonical serialization and replay verification for structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from dnadesign.artifacts.errors import PublicationError
from dnadesign.artifacts.portable_paths import validate_publication_metadata_paths
from dnadesign.contracts.folding import (
    AssessmentTargetSequenceV1,
    StructureAssessmentPublicationV1,
    StructureAssessmentRecordV1,
    StructureAssessmentRequestV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v2 import (
    SecondaryStructurePredictionRequestV1,
    SecondaryStructurePredictionV2,
    SecondaryStructureQaV1,
)

from ..errors import FoldingError, FoldingLengthMismatchError, FoldingMalformedOutputError
from ..execution_metadata import (
    exception_evidence_text,
    parse_cli_failure_evidence,
    prediction_command,
    prediction_log_paths,
    python_api_success_stdout,
)
from ..rnafold import parse_rnafold_stdout
from ._limits import (
    ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES,
    ARTIFACT_ENTRY_COUNT_LIMIT,
    ARTIFACT_FILE_SIZE_LIMIT_BYTES,
)
from .projection import project_prediction_request

_MANIFEST = "manifest.json"
_STAGING_OWNER = ".dnadesign-publication-owner.json"
_PREFLIGHT = "folding_preflight.json"
_HASH_CHUNK_BYTES = 65_536
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


def _filesystem_root_parts(path: Path) -> tuple[str, ...]:
    parts = path.parts[1:]
    # Darwin exposes these fixed root aliases as symlinks. Expand the known
    # mapping lexically so later user-writable components still open no-follow.
    if sys.platform == "darwin" and parts and parts[0] in {"etc", "tmp", "var"}:
        return ("private", *parts)
    return parts


class _PreflightContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _PreflightBackend(_PreflightContractModel):
    name: str
    interface: str
    executable: str | None
    python_module: str | None
    resolved_executable: str | None
    available: bool
    version: str | None


class _PreflightArtifact(_PreflightContractModel):
    contract: str
    status: str
    backend: _PreflightBackend
    output_dir: str
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True, slots=True)
class PublishedStructureAssessment:
    """One verified create-only assessment publication."""

    manifest: StructureAssessmentPublicationV1
    request: StructureAssessmentRequestV1
    record: StructureAssessmentRecordV1


@dataclass(frozen=True, slots=True)
class _AnchoredInventory:
    files: dict[str, str]
    contents: dict[str, bytes]
    directories: set[str]
    symbolic_links: set[str]
    special_entries: set[str]
    total_bytes: int


class _AnchoredPublicationReader:
    """Read publication files through one no-follow root descriptor."""

    def __init__(self, root: Path) -> None:
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW") or not _OPEN_SUPPORTS_DIR_FD:
            raise ValueError("Descriptor-anchored assessment reads are unavailable on this platform.")
        if ".." in root.parts:
            raise ValueError("Assessment publication root cannot use parent traversal.")
        absolute_root = root if root.is_absolute() else Path.cwd() / root
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(absolute_root.anchor, flags)
            for part in _filesystem_root_parts(absolute_root):
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise ValueError("Assessment publication root is missing or unsafe.") from exc
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("Assessment publication root must be a directory.")
        self._root_descriptor = descriptor

    @classmethod
    def from_descriptor(cls, descriptor: int) -> _AnchoredPublicationReader:
        """Duplicate an already anchored publication-root descriptor."""
        instance = cls.__new__(cls)
        instance._root_descriptor = os.dup(descriptor)
        if not stat.S_ISDIR(os.fstat(instance._root_descriptor).st_mode):
            os.close(instance._root_descriptor)
            raise ValueError("Assessment publication root descriptor must name a directory.")
        return instance

    def close(self) -> None:
        if self._root_descriptor >= 0:
            os.close(self._root_descriptor)
            self._root_descriptor = -1

    def __enter__(self) -> _AnchoredPublicationReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _open_file(self, relative: str, *, label: str) -> int:
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
            raise ValueError(f"{label} must stay inside the assessment publication.")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        opened_directories: list[int] = []
        current_descriptor = self._root_descriptor
        try:
            for part in relative_path.parts[:-1]:
                directory_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
                opened_directories.append(directory_descriptor)
                current_descriptor = directory_descriptor
            descriptor = os.open(
                relative_path.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=current_descriptor,
            )
        except OSError as exc:
            raise ValueError(f"{label} cannot use a symbolic link or unsafe or missing path.") from exc
        finally:
            for directory_descriptor in reversed(opened_directories):
                os.close(directory_descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"{label} must be a regular file.")
        if metadata.st_size >= ARTIFACT_FILE_SIZE_LIMIT_BYTES:
            os.close(descriptor)
            raise ValueError(f"{label} exceeds the {ARTIFACT_FILE_SIZE_LIMIT_BYTES}-byte limit.")
        return descriptor

    def write_new_bytes(self, relative: str, content: bytes, *, label: str) -> None:
        """Create one new regular file below the anchored root."""
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
            raise ValueError(f"{label} must stay inside the assessment publication.")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current_descriptor = os.dup(self._root_descriptor)
        try:
            for part in relative_path.parts[:-1]:
                next_descriptor = os.open(part, directory_flags, dir_fd=current_descriptor)
                os.close(current_descriptor)
                current_descriptor = next_descriptor
            descriptor = os.open(
                relative_path.parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=current_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
        except OSError as exc:
            raise ValueError(f"{label} cannot be created safely in the assessment publication.") from exc
        finally:
            os.close(current_descriptor)

    def read_bytes(self, relative: str, *, label: str) -> bytes:
        """Read bounded bytes from the same descriptor that was validated."""
        descriptor = self._open_file(relative, label=label)
        return self._read_descriptor(descriptor, label=label)

    @staticmethod
    def _read_descriptor(descriptor: int, *, label: str) -> bytes:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"{label} must be a regular file.")
        if metadata.st_size >= ARTIFACT_FILE_SIZE_LIMIT_BYTES:
            os.close(descriptor)
            raise ValueError(f"{label} exceeds the {ARTIFACT_FILE_SIZE_LIMIT_BYTES}-byte limit.")
        chunks: list[bytes] = []
        total = 0
        with os.fdopen(descriptor, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                total += len(chunk)
                if total >= ARTIFACT_FILE_SIZE_LIMIT_BYTES:
                    raise ValueError(f"{label} exceeds the {ARTIFACT_FILE_SIZE_LIMIT_BYTES}-byte limit.")
                chunks.append(chunk)
        return b"".join(chunks)

    def inventory(self) -> _AnchoredInventory:
        """Enumerate and hash the tree below the anchored root descriptor."""
        files: dict[str, str] = {}
        contents: dict[str, bytes] = {}
        directories: set[str] = set()
        symbolic_links: set[str] = set()
        special_entries: set[str] = set()
        totals = [0, 0]
        self._inventory_directory(
            self._root_descriptor,
            PurePosixPath(),
            files=files,
            contents=contents,
            directories=directories,
            symbolic_links=symbolic_links,
            special_entries=special_entries,
            totals=totals,
        )
        return _AnchoredInventory(
            files=files,
            contents=contents,
            directories=directories,
            symbolic_links=symbolic_links,
            special_entries=special_entries,
            total_bytes=totals[1],
        )

    def _inventory_directory(
        self,
        directory_descriptor: int,
        prefix: PurePosixPath,
        *,
        files: dict[str, str],
        contents: dict[str, bytes],
        directories: set[str],
        symbolic_links: set[str],
        special_entries: set[str],
        totals: list[int],
    ) -> None:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise ValueError("Assessment artifact inventory cannot enumerate the anchored publication.") from exc
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for name in names:
            relative = (prefix / name).as_posix()
            try:
                metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"Assessment artifact inventory cannot inspect: {relative}") from exc
            totals[0] += 1
            if totals[0] > ARTIFACT_ENTRY_COUNT_LIMIT:
                raise ValueError(f"Assessment artifact inventory exceeds the {ARTIFACT_ENTRY_COUNT_LIMIT}-entry limit.")
            if stat.S_ISLNK(metadata.st_mode):
                symbolic_links.add(relative)
                continue
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_descriptor = os.open(name, directory_flags, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise ValueError(f"Assessment artifact inventory cannot open directory: {relative}") from exc
                directories.add(relative)
                try:
                    self._inventory_directory(
                        child_descriptor,
                        prefix / name,
                        files=files,
                        contents=contents,
                        directories=directories,
                        symbolic_links=symbolic_links,
                        special_entries=special_entries,
                        totals=totals,
                    )
                finally:
                    os.close(child_descriptor)
                continue
            if stat.S_ISREG(metadata.st_mode):
                try:
                    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise ValueError(f"Assessment artifact inventory cannot open file: {relative}") from exc
                content = self._read_descriptor(
                    descriptor,
                    label=f"assessment artifact {relative}",
                )
                totals[1] += len(content)
                if totals[1] >= ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES:
                    raise ValueError(
                        "Assessment artifact inventory exceeds the "
                        f"{ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES}-byte aggregate limit."
                    )
                files[relative] = content_digest(content)
                contents[relative] = content
                continue
            special_entries.add(relative)


def model_json_bytes(model: BaseModel) -> bytes:
    """Return canonical indented JSON bytes for one contract model."""
    payload = model.model_dump(mode="json", by_alias=True)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def content_digest(content: bytes) -> str:
    """Return the contract-form SHA-256 digest for exact bytes."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def write_model_json(path: Path, model: BaseModel) -> bytes:
    """Write and return one canonical contract representation."""
    content = model_json_bytes(model)
    path.write_bytes(content)
    return content


def artifact_digests(
    root: Path,
    *,
    reader: _AnchoredPublicationReader | None = None,
) -> dict[str, str]:
    """Inventory every non-manifest file in one staged publication."""
    owned_reader = reader is None
    active_reader = reader or _AnchoredPublicationReader(root)
    try:
        inventory = active_reader.inventory()
    finally:
        if owned_reader:
            active_reader.close()
    if inventory.symbolic_links:
        first = sorted(inventory.symbolic_links)[0]
        raise ValueError(f"Assessment artifact inventory cannot use a symbolic link: {first}")
    if inventory.special_entries:
        first = sorted(inventory.special_entries)[0]
        raise ValueError(f"Assessment artifact inventory contains an unsupported filesystem entry: {first}")
    return {
        relative: digest for relative, digest in inventory.files.items() if relative not in {_MANIFEST, _STAGING_OWNER}
    }


def verify_publication(
    root: Path,
    *,
    allow_staging_owner: bool = False,
    reader: _AnchoredPublicationReader | None = None,
) -> PublishedStructureAssessment:
    """Replay all byte and cross-object invariants in one publication."""
    if ".." in root.parts:
        raise ValueError("Assessment publication root cannot use parent traversal.")
    root = root if root.is_absolute() else Path.cwd() / root
    reader_context = _AnchoredPublicationReader(root) if reader is None else nullcontext(reader)
    with reader_context as active_reader:
        inventory = active_reader.inventory()
        manifest_content = _inventory_bytes(inventory, _MANIFEST, label="assessment manifest")
        manifest = StructureAssessmentPublicationV1.model_validate_json(manifest_content)
        _verify_artifact_inventory_structure(
            root,
            inventory,
            allow_staging_owner=allow_staging_owner,
        )
        request_content = _inventory_bytes(inventory, manifest.request_path, label="assessment request")
        target_sequence_content = _inventory_bytes(
            inventory,
            manifest.target_sequence_path,
            label="assessment target sequence",
        )
        worker_request_content = _inventory_bytes(
            inventory,
            manifest.worker_request_path,
            label="assessment worker request",
        )
        prediction_content = _inventory_bytes(
            inventory,
            manifest.prediction_path,
            label="assessment prediction",
        )
        prediction_root = PurePosixPath(manifest.prediction_path).parent
        preflight_content = _inventory_bytes(
            inventory,
            (prediction_root / _PREFLIGHT).as_posix(),
            label="assessment preflight",
        )
        record_content = _inventory_bytes(inventory, manifest.record_path, label="assessment record")
        if content_digest(request_content) != manifest.request_digest:
            raise ValueError("Assessment request digest does not match the publication manifest.")
        if content_digest(target_sequence_content) != manifest.target_sequence_artifact_digest:
            raise ValueError("Assessment target-sequence artifact digest does not match the publication manifest.")
        if content_digest(worker_request_content) != manifest.worker_request_digest:
            raise ValueError("Assessment worker-request digest does not match the publication manifest.")
        if content_digest(prediction_content) != manifest.prediction_digest:
            raise ValueError("Assessment prediction digest does not match the publication manifest.")
        if content_digest(record_content) != manifest.record_digest:
            raise ValueError("Assessment record digest does not match the publication manifest.")
        _verify_artifact_inventory_contents(inventory, manifest.artifact_digests)
        request = StructureAssessmentRequestV1.model_validate_json(request_content)
        target_sequence = AssessmentTargetSequenceV1.model_validate_json(target_sequence_content)
        worker_request = SecondaryStructurePredictionRequestV1.model_validate_json(worker_request_content)
        prediction = SecondaryStructurePredictionV2.model_validate_json(prediction_content)
        preflight = _PreflightArtifact.model_validate_json(preflight_content)
        record = StructureAssessmentRecordV1.model_validate_json(record_content)
        if request.assessment_id != manifest.assessment_id or record.assessment_id != manifest.assessment_id:
            raise ValueError("Assessment identifiers do not agree across the publication.")
        if record.request_digest != manifest.request_digest or record.prediction_digest != manifest.prediction_digest:
            raise ValueError("Assessment record digests do not agree with the publication manifest.")
        if record.target != request.target or record.prediction != prediction:
            raise ValueError("Assessment record does not replay its request target and prediction.")
        if worker_request != project_prediction_request(request):
            raise ValueError("Assessment worker request does not match its deterministic high-level projection.")
        _verify_preflight(
            preflight,
            worker_request=worker_request,
            prediction=prediction,
            prediction_root=prediction_root,
        )
        _verify_prediction_execution_metadata(preflight, worker_request=worker_request, prediction=prediction)
        _verify_prediction_artifacts(inventory, prediction_root, prediction)
        _verify_prediction_backend_output(inventory, prediction_root, request=request, prediction=prediction)
        if (
            target_sequence.sequence.id != request.target.sequence_id
            or f"sha256:{target_sequence.sequence.sha256}" != request.target.sequence_sha256
            or target_sequence.sequence.sequence != request.target.sequence
        ):
            raise ValueError("Assessment target artifact does not match the assessment request.")
        if (
            request.target.state_digest != manifest.target_state_digest
            or request.target.sequence_sha256 != manifest.target_sequence_sha256
        ):
            raise ValueError("Assessment target digests do not agree with the publication manifest.")
    return PublishedStructureAssessment(manifest=manifest, request=request, record=record)


def _inventory_bytes(inventory: _AnchoredInventory, relative: str, *, label: str) -> bytes:
    try:
        return inventory.contents[relative]
    except KeyError as exc:
        raise ValueError(f"{label} is missing from the assessment publication.") from exc


def _verify_prediction_artifacts(
    inventory: _AnchoredInventory,
    prediction_root: PurePosixPath,
    prediction: SecondaryStructurePredictionV2,
) -> None:
    for label, reference in (
        ("prediction stdout", prediction.artifacts.stdout),
        ("prediction stderr", prediction.artifacts.stderr),
    ):
        if reference is not None:
            _inventory_bytes(inventory, (prediction_root / reference).as_posix(), label=label)


def _verify_prediction_execution_metadata(
    preflight: _PreflightArtifact,
    *,
    worker_request: SecondaryStructurePredictionRequestV1,
    prediction: SecondaryStructurePredictionV2,
) -> None:
    request_input = worker_request.input
    observed_input = prediction.input
    if prediction.prediction_id != worker_request.request_id or (
        observed_input.sequence_id,
        observed_input.sequence_sha256,
        observed_input.alphabet,
        observed_input.topology,
        observed_input.length,
    ) != (
        request_input.sequence_id,
        request_input.sequence_sha256,
        request_input.alphabet,
        request_input.topology,
        request_input.length,
    ):
        raise ValueError("Assessment prediction input does not match the worker request.")
    if preflight.status != "ok":
        if (
            prediction.dna_policy is not None
            or prediction.artifacts.stdout is not None
            or prediction.artifacts.stderr is not None
        ):
            raise ValueError("Assessment blocked prediction contains execution metadata.")
        expected_qa = SecondaryStructureQaV1(
            length_matches_input=None,
            warnings=[] if worker_request.policy.required else [_missing_backend_diagnostic(worker_request)],
            errors=[_missing_backend_diagnostic(worker_request)] if worker_request.policy.required else [],
        )
        if prediction.qa != expected_qa:
            raise ValueError("Assessment blocked prediction contains derived QA without backend execution.")
        return
    backend = prediction.backend
    dna_policy = prediction.dna_policy
    if backend is None or dna_policy is None:
        raise ValueError("Assessment prediction lacks deterministic execution metadata.")
    expected_command = prediction_command(
        interface=worker_request.backend.interface,
        python_module=worker_request.backend.python_module,
        resolved_executable=preflight.backend.resolved_executable,
        parameters=worker_request.backend.parameters,
    )
    expected_submitted_alphabet = (
        "rna_surrogate" if worker_request.backend.dna_policy.mode == "convert_t_to_u_for_rna_backend" else "dna"
    )
    if (
        backend.name != worker_request.backend.name
        or backend.version != preflight.backend.version
        or backend.command != expected_command
        or backend.parameters != worker_request.backend.parameters
        or dna_policy.mode != worker_request.backend.dna_policy.mode
        or dna_policy.submitted_alphabet != expected_submitted_alphabet
        or dna_policy.coordinates_mapped_to != worker_request.backend.dna_policy.output_coordinates
    ):
        raise ValueError("Assessment prediction execution metadata does not match the worker request.")
    expected_stdout, expected_stderr = prediction_log_paths(interface=worker_request.backend.interface)
    if (
        prediction.artifacts.stdout != expected_stdout
        or prediction.artifacts.stderr != expected_stderr
        or expected_stdout == expected_stderr
    ):
        raise ValueError("Assessment prediction log references do not match the backend interface.")


def _missing_backend_diagnostic(worker_request: SecondaryStructurePredictionRequestV1) -> str:
    backend = worker_request.backend
    if backend.interface == "python_api":
        if backend.python_module is None:
            return "Folding backend python module is not configured."
        return f"Folding backend Python module '{backend.python_module}' is not available."
    if backend.executable is None:
        return "Folding backend executable is not configured."
    return f"Folding backend '{backend.executable}' is not available."


def _verify_prediction_backend_output(
    inventory: _AnchoredInventory,
    prediction_root: PurePosixPath,
    *,
    request: StructureAssessmentRequestV1,
    prediction: SecondaryStructurePredictionV2,
) -> None:
    if prediction.status == "error":
        _verify_prediction_failure(
            inventory,
            prediction_root,
            request=request,
            prediction=prediction,
        )
        return
    if prediction.status != "ok":
        return
    stdout_ref = prediction.artifacts.stdout
    stderr_ref = prediction.artifacts.stderr
    if stdout_ref is None or stderr_ref is None:
        raise ValueError("Successful assessment prediction lacks backend output evidence.")
    stdout = _inventory_bytes(
        inventory,
        (prediction_root / stdout_ref).as_posix(),
        label="prediction stdout",
    ).decode("utf-8")
    stderr = _inventory_bytes(
        inventory,
        (prediction_root / stderr_ref).as_posix(),
        label="prediction stderr",
    ).decode("utf-8")
    submitted_sequence = request.target.sequence.upper()
    if request.backend.dna_policy.mode == "convert_t_to_u_for_rna_backend":
        submitted_sequence = submitted_sequence.replace("T", "U")
    if request.backend.interface == "python_api":
        result = prediction.result
        if result is None or result.mfe_kcal_mol is None:
            raise ValueError("Successful Python assessment lacks a complete structure result.")
        expected_stdout = python_api_success_stdout(
            sequence_id=request.target.sequence_id,
            submitted_sequence=submitted_sequence,
            dot_bracket=result.dot_bracket,
            mfe_kcal_mol=result.mfe_kcal_mol,
        )
        if stdout != expected_stdout or stderr:
            raise ValueError("Successful Python assessment logs are not canonical producer evidence.")
    try:
        replayed_result = parse_rnafold_stdout(
            stdout=stdout,
            submitted_sequence=submitted_sequence,
            input_length=len(request.target.sequence),
        )
    except FoldingError as exc:
        raise ValueError(f"Assessment backend output evidence cannot be replayed: {exc}") from exc
    if prediction.result != replayed_result or prediction.qa != SecondaryStructureQaV1(length_matches_input=True):
        raise ValueError("Assessment prediction does not match its backend output evidence.")


def _verify_prediction_failure(
    inventory: _AnchoredInventory,
    prediction_root: PurePosixPath,
    *,
    request: StructureAssessmentRequestV1,
    prediction: SecondaryStructurePredictionV2,
) -> None:
    failure = prediction.failure
    stdout_ref = prediction.artifacts.stdout
    stderr_ref = prediction.artifacts.stderr
    if failure is None or stdout_ref is None or stderr_ref is None:
        raise ValueError("Failed assessment prediction lacks typed backend evidence.")
    stdout = _inventory_bytes(
        inventory,
        (prediction_root / stdout_ref).as_posix(),
        label="prediction stdout",
    ).decode("utf-8")
    stderr = _inventory_bytes(
        inventory,
        (prediction_root / stderr_ref).as_posix(),
        label="prediction stderr",
    ).decode("utf-8")
    if failure.kind == "output_parse_exception":
        if request.backend.interface == "python_api" and stderr:
            raise ValueError("Assessment Python parse-failure stderr is not canonical producer evidence.")
        submitted_sequence = request.target.sequence.upper()
        if request.backend.dna_policy.mode == "convert_t_to_u_for_rna_backend":
            submitted_sequence = submitted_sequence.replace("T", "U")
        try:
            parse_rnafold_stdout(
                stdout=stdout,
                submitted_sequence=submitted_sequence,
                input_length=len(request.target.sequence),
            )
        except FoldingError as exc:
            if failure.exception_type != type(exc).__name__ or failure.message != str(exc):
                raise ValueError("Assessment parse-failure evidence does not match backend output.") from exc
            if (
                isinstance(exc, FoldingLengthMismatchError)
                and request.policy.fail_on_length_mismatch
                or isinstance(exc, FoldingMalformedOutputError)
                and request.policy.fail_on_malformed_output
            ):
                raise ValueError("Assessment parse-failure status contradicts the persisted failure policy.") from exc
        else:
            raise ValueError("Assessment parse-failure claim contradicts successful backend output replay.")
        return
    expected_interface = {
        "backend_invocation_exception": "cli",
        "backend_nonzero_exit": "cli",
        "backend_exception": "python_api",
    }.get(failure.kind)
    if expected_interface is not None and request.backend.interface != expected_interface:
        raise ValueError("Assessment failure kind does not match the backend interface.")
    if failure.kind == "backend_nonzero_exit":
        expected_message = f"ViennaRNA RNAfold CLI exited with status {failure.returncode}."
        if failure.message != expected_message:
            raise ValueError("Assessment nonzero-exit evidence is internally inconsistent.")
        try:
            evidence_returncode, _backend_stderr = parse_cli_failure_evidence(stderr)
        except ValueError as exc:
            raise ValueError("Assessment nonzero-exit evidence does not match backend logs.") from exc
        if evidence_returncode != failure.returncode:
            raise ValueError("Assessment nonzero-exit evidence does not match backend logs.")
        return
    exception_type = failure.exception_type
    if exception_type is None:
        raise ValueError("Assessment exception evidence lacks its exception type.")
    expected_stderr = exception_evidence_text(
        exception_type=exception_type,
        message=failure.message,
    )
    if stdout or stderr != expected_stderr:
        raise ValueError("Assessment exception evidence does not match backend logs.")


def _verify_preflight(
    preflight: _PreflightArtifact,
    *,
    worker_request: SecondaryStructurePredictionRequestV1,
    prediction: SecondaryStructurePredictionV2,
    prediction_root: PurePosixPath,
) -> None:
    if preflight.contract != "secondary_structure_folding_preflight_v1":
        raise ValueError("Assessment preflight contract is unsupported.")
    backend = preflight.backend
    request_backend = worker_request.backend
    if (
        backend.name != request_backend.name
        or backend.interface != request_backend.interface
        or backend.executable != request_backend.executable
        or backend.python_module != request_backend.python_module
    ):
        raise ValueError("Assessment preflight backend does not match the worker request.")
    if backend.interface == "cli" and backend.resolved_executable is not None:
        resolved_executable = PurePosixPath(backend.resolved_executable)
        if (
            not backend.resolved_executable.strip()
            or not resolved_executable.is_absolute()
            or "." in resolved_executable.parts
            or ".." in resolved_executable.parts
            or resolved_executable.as_posix() != backend.resolved_executable
        ):
            raise ValueError("Assessment preflight resolved executable must be an absolute normalized path.")
    expected_available = (
        backend.python_module is not None and backend.version is not None
        if backend.interface == "python_api"
        else backend.resolved_executable is not None
    )
    if backend.available is not expected_available:
        raise ValueError("Assessment preflight backend availability is internally inconsistent.")
    if preflight.output_dir != "." or prediction_root.name != "prediction":
        raise ValueError("Assessment preflight output directory is not the portable worker root.")
    if worker_request.policy.required and prediction.status != "ok":
        raise ValueError("A required assessment cannot replay non-ok status.")
    if preflight.status == "ok":
        if not backend.available or backend.version is None or prediction.backend is None:
            raise ValueError("Assessment preflight success lacks a usable prediction backend.")
        if prediction.status not in {"ok", "error"}:
            raise ValueError("A successful assessment preflight has an impossible prediction status.")
        if prediction.status == "error":
            failure = prediction.failure
            if failure is None or prediction.qa != SecondaryStructureQaV1(
                length_matches_input=None,
                errors=[failure.message],
            ):
                raise ValueError("Assessment execution error lacks canonical diagnostic evidence.")
        if preflight.warnings or preflight.errors:
            raise ValueError("Assessment preflight success cannot contain warnings or errors.")
        if prediction.backend.name != backend.name:
            raise ValueError("Assessment preflight backend name does not match the prediction.")
        if prediction.backend.version != backend.version:
            raise ValueError("Assessment preflight backend version does not match the prediction.")
        expected_command = prediction_command(
            interface=backend.interface,
            python_module=backend.python_module,
            resolved_executable=backend.resolved_executable,
            parameters=request_backend.parameters,
        )
        if (
            backend.interface == "python_api" and backend.resolved_executable is not None
        ) or prediction.backend.command != expected_command:
            raise ValueError("Assessment preflight backend does not match the prediction command.")
        return
    expected_missing_status = (
        "blocker_required_missing" if worker_request.policy.required else "warning_optional_missing"
    )
    if (
        backend.available
        or backend.version is not None
        or backend.resolved_executable is not None
        or prediction.backend is not None
        or preflight.status != expected_missing_status
        or prediction.status != expected_missing_status
    ):
        raise ValueError("Assessment preflight blocker does not match the prediction status.")
    if preflight.warnings != prediction.qa.warnings or preflight.errors != prediction.qa.errors:
        raise ValueError("Assessment preflight diagnostics do not match the prediction.")


def _verify_artifact_inventory_structure(
    root: Path,
    inventory: _AnchoredInventory,
    *,
    allow_staging_owner: bool,
) -> None:
    relative_entry_files = {
        **{relative: True for relative in inventory.files},
        **{relative: False for relative in inventory.directories},
        **{relative: False for relative in inventory.symbolic_links},
        **{relative: False for relative in inventory.special_entries},
    }
    try:
        validate_publication_metadata_paths(
            root,
            required_manifest=Path(_MANIFEST),
            owner_file=_STAGING_OWNER,
            require_owner=allow_staging_owner,
            relative_entry_files=relative_entry_files,
        )
    except PublicationError as exc:
        raise ValueError(str(exc)) from exc
    if inventory.symbolic_links:
        first = sorted(inventory.symbolic_links)[0]
        raise ValueError(f"Assessment artifact inventory cannot use a symbolic link: {first}")
    if inventory.special_entries:
        first = sorted(inventory.special_entries)[0]
        raise ValueError(f"Assessment artifact inventory contains an unsupported filesystem entry: {first}")


def _verify_artifact_inventory_contents(
    inventory: _AnchoredInventory,
    expected: dict[str, str],
) -> None:
    actual_files = {
        relative: digest for relative, digest in inventory.files.items() if relative not in {_MANIFEST, _STAGING_OWNER}
    }
    actual_directories = inventory.directories
    expected_directories = {
        parent.as_posix()
        for artifact_path in expected
        for parent in PurePosixPath(artifact_path).parents
        if parent.as_posix() != "."
    }
    if set(actual_files) != set(expected) or actual_directories != expected_directories:
        raise ValueError("Assessment artifact inventory does not match the publication contents.")
    mismatches = [path for path, digest in expected.items() if actual_files[path] != digest]
    if mismatches:
        raise ValueError(f"Assessment artifact inventory digest mismatch: {', '.join(sorted(mismatches))}")


def load_published_assessment(output_dir: str | Path) -> PublishedStructureAssessment:
    """Load and verify one create-only structure assessment publication."""
    path = Path(output_dir).expanduser()
    if ".." in path.parts:
        raise ValueError("Assessment publication directory cannot use parent traversal.")
    if not path.is_absolute():
        path = Path.cwd() / path
    return verify_publication(path)


__all__ = ["PublishedStructureAssessment", "artifact_digests", "load_published_assessment"]
