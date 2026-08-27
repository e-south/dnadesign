"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/design_results.py

Admission boundary for completed LigandMPNN design output trees.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.alphabets import LigandMpnnResidueAlphabetSidecar
from dnadesign.thread.adapters.ligandmpnn.commands import build_ligandmpnn_commands
from dnadesign.thread.adapters.ligandmpnn.context_inventory import _read_descriptor_relative_regular_bytes
from dnadesign.thread.adapters.ligandmpnn.design_fasta import parse_official_design_fasta
from dnadesign.thread.adapters.ligandmpnn.design_manifest import build_design_output_manifest
from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnCommand, LigandMpnnRequest
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import (
    parse_pinned_runtime_prefix,
    pinned_runtime_completion_contract,
)

_COMPLETION_RECORD_NAME = ".dnadesign-ligandmpnn-execution.json"


@dataclass(frozen=True)
class LigandMpnnDesignOutput:
    """One admitted per-seed design tree and its immutable manifest claim."""

    seed: int
    output_dir: Path
    manifest: dict[str, object]
    sequence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "output_dir": self.output_dir.as_posix(),
            "manifest": self.manifest,
            "sequence_count": self.sequence_count,
        }


@dataclass(frozen=True)
class LigandMpnnDesignResult:
    """Validated design outputs admitted against atomic completion records."""

    request_id: str
    expected_sequence_count: int
    outputs: tuple[LigandMpnnDesignOutput, ...]

    @property
    def sequence_count(self) -> int:
        """Return the admitted designed-record count across every seed."""

        return sum(output.sequence_count for output in self.outputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.design_result",
            "schema_version": 2,
            "status": "completed_validated",
            "request_id": self.request_id,
            "expected_sequence_count": self.expected_sequence_count,
            "sequence_count": self.sequence_count,
            "outputs": [output.to_dict() for output in self.outputs],
        }


def parse_ligandmpnn_design_outputs(
    request: LigandMpnnRequest,
    commands: tuple[LigandMpnnCommand, ...],
    *,
    execution_root: Path,
    residue_alphabet_sidecar: LigandMpnnResidueAlphabetSidecar | None = None,
) -> LigandMpnnDesignResult:
    """Admit only artifact trees exactly bound by their completion records."""

    if not isinstance(commands, tuple) or not commands:
        raise ValueError("commands must be a nonempty tuple")
    root = execution_root
    if not root.is_absolute():
        raise ValueError("execution_root must be an absolute directory")
    first = commands[0]
    checkout_root, python_executable, _completion_path, _execution_sha256 = parse_pinned_runtime_prefix(
        first.argv,
        upstream_commit=request.upstream.commit,
        checkpoint_sha256=request.upstream.checkpoint_sha256,
        pdb_sha256=request.pdb_sha256,
        context_inventory_path=request.context_inventory.path,
        context_inventory_sha256=request.context_inventory.sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=(request.upstream.packing_checkpoint_sha256 if request.packing.enabled else None),
        residue_alphabet_sha256=(
            residue_alphabet_sidecar.sha256.removeprefix("sha256:") if residue_alphabet_sidecar is not None else None
        ),
        entrypoint="run.py",
    )
    expected_commands = build_ligandmpnn_commands(
        request,
        checkout_root=checkout_root,
        execution_root=execution_root,
        python_executable=python_executable,
        residue_alphabet_sidecar=residue_alphabet_sidecar,
    )
    if commands != expected_commands:
        raise ValueError("commands do not exactly match the design request")

    outputs: list[LigandMpnnDesignOutput] = []
    for command in commands:
        completion_relative_path = command.output_dir / _COMPLETION_RECORD_NAME
        completion_bytes = _read_descriptor_relative_regular_bytes(
            root,
            completion_relative_path,
            label="design completion record",
        )
        try:
            completion = json.loads(completion_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"design completion record is not valid JSON: {completion_relative_path}") from error
        if not isinstance(completion, dict):
            raise ValueError("design completion record root must be an object")
        output_root = root / command.output_dir
        observed_manifest = build_design_output_manifest(output_root)
        if completion.get("design_output_manifest") != observed_manifest:
            raise ValueError(f"design output manifest mismatch: {command.output_dir}")
        expected_path, expected_completion = pinned_runtime_completion_contract(
            command.argv,
            upstream_commit=request.upstream.commit,
            checkpoint_sha256=request.upstream.checkpoint_sha256,
            pdb_sha256=request.pdb_sha256,
            context_inventory_path=request.context_inventory.path,
            context_inventory_sha256=request.context_inventory.sha256,
            execution_root=execution_root,
            packing_checkpoint_sha256=(request.upstream.packing_checkpoint_sha256 if request.packing.enabled else None),
            residue_alphabet_sha256=(
                residue_alphabet_sidecar.sha256.removeprefix("sha256:")
                if residue_alphabet_sidecar is not None
                else None
            ),
            entrypoint="run.py",
            design_output_manifest=observed_manifest,
        )
        if expected_path.is_absolute():
            try:
                expected_relative_path = expected_path.relative_to(root)
            except ValueError as error:
                raise ValueError("design completion record escapes execution_root") from error
        else:
            expected_relative_path = expected_path
        if expected_relative_path != completion_relative_path:
            raise ValueError("design completion record path does not match planned output")
        if completion != expected_completion:
            raise ValueError(f"design completion record does not match planned execution: {command.output_dir}")
        input_name = _official_input_name(request.pdb_path)
        fasta_relative_path = command.output_dir / "seqs" / f"{input_name}.fa"
        try:
            fasta_bytes = _read_descriptor_relative_regular_bytes(
                root,
                fasta_relative_path,
                label="official LigandMPNN FASTA",
            )
        except ValueError as error:
            raise ValueError(f"official LigandMPNN FASTA is missing or unreadable: {fasta_relative_path}") from error
        _validate_fasta_manifest_binding(
            observed_manifest,
            relative_path=Path("seqs") / f"{input_name}.fa",
            payload=fasta_bytes,
        )
        sequence_count = parse_official_design_fasta(
            fasta_bytes,
            input_stem=input_name,
            expected_design_count=request.batch_size * request.number_of_batches,
        )
        outputs.append(
            LigandMpnnDesignOutput(
                seed=command.seed,
                output_dir=command.output_dir,
                manifest=observed_manifest,
                sequence_count=sequence_count,
            )
        )
    result = LigandMpnnDesignResult(
        request_id=request.request_id,
        expected_sequence_count=request.expected_sequence_count,
        outputs=tuple(outputs),
    )
    if result.sequence_count != request.expected_sequence_count:
        raise ValueError(
            f"design result expected {request.expected_sequence_count} designed records; "
            f"observed {result.sequence_count}"
        )
    return result


def _official_input_name(pdb_path: Path) -> str:
    name = pdb_path.name
    return name[:-4] if name.endswith(".pdb") else name


def _validate_fasta_manifest_binding(
    manifest: dict[str, object],
    *,
    relative_path: Path,
    payload: bytes,
) -> None:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("design output manifest entries are invalid")
    expected_path = relative_path.as_posix()
    matching_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == expected_path]
    observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if len(matching_entries) != 1 or matching_entries[0] != {
        "path": expected_path,
        "type": "file",
        "size_bytes": len(payload),
        "sha256": observed_digest,
    }:
        raise ValueError(f"official LigandMPNN FASTA does not match admitted manifest: {expected_path}")


__all__ = [
    "LigandMpnnDesignOutput",
    "LigandMpnnDesignResult",
    "parse_ligandmpnn_design_outputs",
]
