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
from dnadesign.thread.adapters.ligandmpnn.commands import (
    build_ligandmpnn_commands,
    resolve_execution_root_for_execution,
)
from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    LigandMpnnProteinStructureEvidence,
    _read_descriptor_relative_regular_bytes,
    load_ligandmpnn_context_inventory,
    validate_context_inventory_for_input,
)
from dnadesign.thread.adapters.ligandmpnn.design_fasta import (
    OfficialLigandMpnnDesignFasta,
    parse_official_design_fasta_records,
)
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
    root = resolve_execution_root_for_execution(execution_root)
    first = commands[0]
    try:
        checkout_root, python_executable, _completion_path, _execution_sha256 = parse_pinned_runtime_prefix(
            first.argv,
            upstream_commit=request.upstream.commit,
            checkpoint_sha256=request.upstream.checkpoint_sha256,
            pdb_sha256=request.pdb_sha256,
            request_id=request.request_id,
            context_inventory_path=request.context_inventory.path,
            context_inventory_sha256=request.context_inventory.sha256,
            execution_root=root,
            packing_checkpoint_sha256=(request.upstream.packing_checkpoint_sha256 if request.packing.enabled else None),
            residue_alphabet_sha256=(
                residue_alphabet_sidecar.sha256.removeprefix("sha256:")
                if residue_alphabet_sidecar is not None
                else None
            ),
            entrypoint="run.py",
        )
    except ValueError as error:
        raise ValueError("commands do not exactly match the design request") from error
    expected_commands = build_ligandmpnn_commands(
        request,
        checkout_root=checkout_root,
        execution_root=root,
        python_executable=python_executable,
        residue_alphabet_sidecar=residue_alphabet_sidecar,
    )
    if commands != expected_commands:
        raise ValueError("commands do not exactly match the design request")
    context_inventory = load_ligandmpnn_context_inventory(
        request.context_inventory,
        execution_root=root,
    )
    protein_evidence = validate_context_inventory_for_input(
        context_inventory,
        pdb_path=request.pdb_path,
        pdb_sha256=request.pdb_sha256,
        upstream=request.upstream,
        use_side_chain_context=request.use_side_chain_context,
        checkout_root=checkout_root,
        execution_root=root,
    )

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
            request_id=request.request_id,
            context_inventory_path=request.context_inventory.path,
            context_inventory_sha256=request.context_inventory.sha256,
            execution_root=root,
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
        _validate_manifest_file_binding(
            observed_manifest,
            relative_path=Path("seqs") / f"{input_name}.fa",
            payload=fasta_bytes,
            label="official LigandMPNN FASTA",
        )
        parsed_fasta = parse_official_design_fasta_records(
            fasta_bytes,
            input_stem=input_name,
            expected_design_count=request.batch_size * request.number_of_batches,
        )
        _validate_design_sequence_contract(request, protein_evidence, parsed_fasta)
        sequence_count = parsed_fasta.design_count
        if request.packing.enabled:
            packed_paths = _validate_packed_artifact_manifest(
                observed_manifest,
                input_name=input_name,
                design_count=request.batch_size * request.number_of_batches,
                pack_count=request.packing.number_of_packs_per_design,
            )
            _validate_packed_artifact_contents(
                root=root,
                output_dir=command.output_dir,
                manifest=observed_manifest,
                packed_paths=packed_paths,
                checkout_root=checkout_root,
                upstream_commit=request.upstream.commit,
                input_evidence=protein_evidence,
                fasta=parsed_fasta,
                pack_count=request.packing.number_of_packs_per_design,
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


def _validate_design_sequence_contract(
    request: LigandMpnnRequest,
    evidence: LigandMpnnProteinStructureEvidence,
    fasta: OfficialLigandMpnnDesignFasta,
) -> None:
    """Require every admitted sequence to obey the exact pinned-parser design mask."""

    if fasta.native_segments != evidence.fasta_native_segments:
        raise ValueError(
            "official LigandMPNN FASTA native sequence does not match pinned parser protein sequence: "
            f"expected {evidence.fasta_native_segments}; observed {fasta.native_segments}"
        )
    residue_ids = evidence.fasta_residue_ids
    native_sequence = tuple("".join(fasta.native_segments))
    if len(residue_ids) != len(native_sequence):
        raise ValueError("official LigandMPNN FASTA residue count does not match pinned parser protein residues")
    fixed_ids = {item.upstream_id for item in request.fixed_residues}
    redesigned_ids = {item.upstream_id for item in request.redesigned_residues}
    alphabets = {item.residue.upstream_id: frozenset(item.allowed_amino_acids) for item in request.residue_alphabets}
    for design_index, segments in enumerate(fasta.designed_segments, start=1):
        designed_sequence = tuple("".join(segments))
        for residue_id, native_residue, designed_residue in zip(
            residue_ids,
            native_sequence,
            designed_sequence,
            strict=True,
        ):
            if residue_id in fixed_ids and designed_residue != native_residue:
                raise ValueError(
                    f"official LigandMPNN FASTA design {design_index} fixed residue {residue_id} was mutated"
                )
            if redesigned_ids and residue_id not in redesigned_ids and designed_residue != native_residue:
                raise ValueError(
                    f"official LigandMPNN FASTA design {design_index} mutated residue {residue_id} "
                    "outside redesigned_residues"
                )
            allowed = alphabets.get(residue_id)
            if allowed is not None and designed_residue not in allowed:
                raise ValueError(
                    f"official LigandMPNN FASTA design {design_index} violates residue alphabet constraint "
                    f"{residue_id}: observed {designed_residue}"
                )


def _official_input_name(pdb_path: Path) -> str:
    name = pdb_path.name
    return name[:-4] if name.endswith(".pdb") else name


def _validate_manifest_file_binding(
    manifest: dict[str, object],
    *,
    relative_path: Path,
    payload: bytes,
    label: str,
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
        raise ValueError(f"{label} does not match admitted manifest: {expected_path}")


def _validate_packed_artifact_manifest(
    manifest: dict[str, object],
    *,
    input_name: str,
    design_count: int,
    pack_count: int,
) -> tuple[Path, ...]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("design output manifest entries are invalid")
    observed = {
        (entry.get("path"), entry.get("type"))
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and (entry["path"] == "packed" or entry["path"].startswith("packed/"))
    }
    expected_paths = tuple(
        Path("packed") / f"{input_name}_packed_{design_id}_{pack_id}.pdb"
        for design_id in range(1, design_count + 1)
        for pack_id in range(1, pack_count + 1)
    )
    expected = {("packed", "directory")}
    expected.update((path.as_posix(), "file") for path in expected_paths)
    if observed != expected:
        raise ValueError("official LigandMPNN packed artifacts do not exactly match the packing request")
    return expected_paths


def _validate_packed_artifact_contents(
    *,
    root: Path,
    output_dir: Path,
    manifest: dict[str, object],
    packed_paths: tuple[Path, ...],
    checkout_root: Path,
    upstream_commit: str,
    input_evidence: LigandMpnnProteinStructureEvidence,
    fasta: OfficialLigandMpnnDesignFasta,
    pack_count: int,
) -> None:
    payloads: list[tuple[str, bytes]] = []
    for packed_path in packed_paths:
        execution_relative_path = output_dir / packed_path
        try:
            payload = _read_descriptor_relative_regular_bytes(
                root,
                execution_relative_path,
                label="official LigandMPNN packed PDB",
            )
        except ValueError as error:
            raise ValueError(f"official LigandMPNN packed PDB is invalid: {packed_path}") from error
        _validate_manifest_file_binding(
            manifest,
            relative_path=packed_path,
            payload=payload,
            label="official LigandMPNN packed PDB",
        )
        payloads.append((packed_path.name, payload))

    from dnadesign.thread.adapters.ligandmpnn.context_probe import (
        _derive_pinned_protein_evidence_for_payloads,
    )

    try:
        packed_evidence = _derive_pinned_protein_evidence_for_payloads(
            checkout_root,
            expected_commit=upstream_commit,
            inputs=tuple(payloads),
        )
    except Exception as error:
        raise ValueError("official LigandMPNN packed PDB is invalid") from error

    expected_sequences = tuple(
        input_evidence.parser_sequence_from_fasta_segments(design_segments)
        for design_segments in fasta.designed_segments
        for _pack_index in range(pack_count)
    )
    for packed_path, observed, expected_sequence in zip(
        packed_paths,
        packed_evidence,
        expected_sequences,
        strict=True,
    ):
        if not observed.residue_ids or any(value != 1 for value in observed.residue_validity_mask):
            raise ValueError(f"official LigandMPNN packed PDB is invalid: {packed_path}")
        if observed.residue_ids != input_evidence.residue_ids:
            raise ValueError(
                f"official LigandMPNN packed PDB structural identity does not match pinned input: {packed_path}"
            )
        if observed.native_sequence != expected_sequence:
            raise ValueError(f"official LigandMPNN packed PDB does not match designed sequence identity: {packed_path}")


__all__ = [
    "LigandMpnnDesignOutput",
    "LigandMpnnDesignResult",
    "parse_ligandmpnn_design_outputs",
]
