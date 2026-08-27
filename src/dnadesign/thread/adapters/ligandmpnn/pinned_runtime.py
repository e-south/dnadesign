"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/pinned_runtime.py

Execute attested LigandMPNN entrypoints with attested parser source.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    load_ligandmpnn_context_inventory,
    validate_context_inventory_for_input,
)
from dnadesign.thread.adapters.ligandmpnn.design_manifest import build_design_output_manifest
from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnContextInventoryReference, LigandMpnnUpstreamPin
from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import materialize_pinned_tree

_ENTRYPOINTS = frozenset({"run.py", "score.py"})
_MODULE = "dnadesign.thread.adapters.ligandmpnn.pinned_runtime"
_CHECKPOINT_FLAG = "--checkpoint_ligand_mpnn"
_PACKING_CHECKPOINT_FLAG = "--checkpoint_path_sc"
_PDB_FLAG = "--pdb_path"
_RESIDUE_ALPHABET_FLAG = "--omit_AA_per_residue"
_FIXED_RESIDUES_FLAG = "--fixed_residues"
_REDESIGNED_RESIDUES_FLAG = "--redesigned_residues"
_MODEL_TYPE_FLAG = "--model_type"
_OUTPUT_FOLDER_FLAG = "--out_folder"
_COMPLETION_RECORD_NAME = ".dnadesign-ligandmpnn-execution.json"
_ALTERNATE_SOURCE_FLAGS = frozenset(
    {
        "--checkpoint_protein_mpnn",
        "--checkpoint_per_residue_label_membrane_mpnn",
        "--checkpoint_global_label_membrane_mpnn",
        "--checkpoint_soluble_mpnn",
        "--pdb_path_multi",
        "--fixed_residues_multi",
        "--redesigned_residues_multi",
        "--bias_AA_per_residue",
        "--bias_AA_per_residue_multi",
        "--omit_AA_per_residue_multi",
    }
)
_ATTESTATION_SENSITIVE_FLAGS = frozenset(
    {
        _MODEL_TYPE_FLAG,
        _CHECKPOINT_FLAG,
        _PACKING_CHECKPOINT_FLAG,
        _PDB_FLAG,
        _RESIDUE_ALPHABET_FLAG,
        _FIXED_RESIDUES_FLAG,
        _REDESIGNED_RESIDUES_FLAG,
        _OUTPUT_FOLDER_FLAG,
        *_ALTERNATE_SOURCE_FLAGS,
    }
)
_SEPARATELY_VALIDATED_SINGLETON_FLAGS = frozenset(
    {
        _MODEL_TYPE_FLAG,
        _CHECKPOINT_FLAG,
        _PACKING_CHECKPOINT_FLAG,
        _PDB_FLAG,
        _RESIDUE_ALPHABET_FLAG,
    }
)
_CANONICAL_RUNTIME_FLAGS = frozenset(
    {
        *_ATTESTATION_SENSITIVE_FLAGS,
        "--autoregressive_score",
        "--batch_size",
        "--ligand_mpnn_use_atom_context",
        "--ligand_mpnn_use_side_chain_context",
        "--number_of_batches",
        "--number_of_packs_per_design",
        "--pack_side_chains",
        "--pack_with_ligand_context",
        "--repack_everything",
        "--seed",
        "--single_aa_score",
        "--temperature",
        "--use_sequence",
    }
)


class LigandMpnnCompletionPublicationUncertainError(RuntimeError):
    """Completion rollback could not establish a durable absent lifecycle."""


class LigandMpnnScorePublicationUncertainError(RuntimeError):
    """Score rollback could not establish a durable absent artifact."""


class LigandMpnnDesignPublicationUncertainError(RuntimeError):
    """Design-directory rollback could not establish a durable absence."""


def pinned_runtime_prefix(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None,
    context_inventory_sha256: str | None,
    execution_root: Path | None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    python_executable: str,
    planned_execution_sha256: str,
    completion_record_path: Path,
) -> tuple[str, ...]:
    """Return the deterministic wrapper prefix for one official entrypoint."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    _validate_context_binding_fields(
        entrypoint=entrypoint,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
    )
    prefix = [
        python_executable,
        "-m",
        _MODULE,
    ]
    _append_cli_option(prefix, "--checkout-root", str(checkout_root))
    prefix.extend(
        [
            "--upstream-commit",
            upstream_commit,
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--pdb-sha256",
            pdb_sha256,
        ]
    )
    if context_inventory_path is not None:
        if context_inventory_sha256 is None or execution_root is None:
            raise ValueError("context inventory runtime binding must be complete")
        _append_cli_option(prefix, "--execution-root", str(execution_root))
        _append_cli_option(prefix, "--context-inventory-path", str(context_inventory_path))
        prefix.extend(["--context-inventory-sha256", context_inventory_sha256])
    elif context_inventory_sha256 is not None or execution_root is not None:
        raise ValueError("context inventory runtime binding must be complete")
    if packing_checkpoint_sha256 is not None:
        prefix.extend(["--packing-checkpoint-sha256", packing_checkpoint_sha256])
    if residue_alphabet_sha256 is not None:
        prefix.extend(["--residue-alphabet-sha256", residue_alphabet_sha256])
    prefix.extend(
        [
            "--planned-execution-sha256",
            planned_execution_sha256,
            "--completion-record",
            str(completion_record_path),
            "--entrypoint",
            entrypoint,
            "--",
        ]
    )
    return tuple(prefix)


def build_pinned_runtime_command(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None = None,
    context_inventory_sha256: str | None = None,
    execution_root: Path | None = None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    python_executable: str,
    output_dir: Path,
    arguments: tuple[str, ...],
) -> tuple[str, ...]:
    """Bind a generated command to its complete semantic execution payload."""

    _validate_context_binding_fields(
        entrypoint=entrypoint,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
    )
    completion_record_path = output_dir / _COMPLETION_RECORD_NAME
    planned_execution_sha256 = pinned_execution_sha256(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=arguments,
    )
    return (
        *pinned_runtime_prefix(
            checkout_root=checkout_root,
            upstream_commit=upstream_commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            context_inventory_path=context_inventory_path,
            context_inventory_sha256=context_inventory_sha256,
            execution_root=execution_root,
            packing_checkpoint_sha256=packing_checkpoint_sha256,
            residue_alphabet_sha256=residue_alphabet_sha256,
            entrypoint=entrypoint,
            python_executable=python_executable,
            planned_execution_sha256=planned_execution_sha256,
            completion_record_path=completion_record_path,
        ),
        *arguments,
    )


def pinned_execution_sha256(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None = None,
    context_inventory_sha256: str | None = None,
    execution_root: Path | None = None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    completion_record_path: Path,
    arguments: tuple[str, ...],
) -> str:
    """Return the canonical digest of every result-affecting execution field."""

    payload = _pinned_execution_payload(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=arguments,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_pinned_runtime_prefix(
    argv: tuple[str, ...],
    *,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None = None,
    context_inventory_sha256: str | None = None,
    execution_root: Path | None = None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
) -> tuple[Path, str, Path, str]:
    """Recover only the two caller-owned fields from an exact wrapper prefix."""

    if "--" not in argv:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    delimiter = argv.index("--")
    arguments = argv[delimiter + 1 :]
    prefix = argv[: delimiter + 1]
    checkout_root = Path(_split_option_value(prefix, "--checkout-root"))
    completion_record_path = Path(_split_option_value(prefix, "--completion-record"))
    planned_execution_sha256 = pinned_execution_sha256(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=arguments,
    )
    expected = pinned_runtime_prefix(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        python_executable=argv[0],
        planned_execution_sha256=planned_execution_sha256,
        completion_record_path=completion_record_path,
    )
    if prefix != expected:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    return checkout_root, argv[0], completion_record_path, planned_execution_sha256


def pinned_runtime_completion_contract(
    argv: tuple[str, ...],
    *,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None = None,
    context_inventory_sha256: str | None = None,
    execution_root: Path | None = None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    score_output_sha256: str | None = None,
    design_output_manifest: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Return the exact completion record required for one planned command."""

    checkout_root, _python, completion_record_path, execution_sha256 = parse_pinned_runtime_prefix(
        argv,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
    )
    arguments = argv[argv.index("--") + 1 :]
    execution = _pinned_execution_payload(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=arguments,
    )
    return completion_record_path, _completion_record(
        execution,
        execution_sha256,
        score_output_sha256=score_output_sha256,
        design_output_manifest=design_output_manifest,
    )


def execute_pinned_entrypoint(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None,
    context_inventory_sha256: str | None,
    execution_root: Path | None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    planned_execution_sha256: str,
    completion_record_path: Path,
    arguments: tuple[str, ...],
) -> None:
    """Execute one pinned source snapshot with digest-verified weight bytes."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    _validate_context_binding_fields(
        entrypoint=entrypoint,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
    )
    execution = _pinned_execution_payload(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        context_inventory_path=context_inventory_path,
        context_inventory_sha256=context_inventory_sha256,
        execution_root=execution_root,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=arguments,
    )
    observed_execution_sha256 = hashlib.sha256(
        json.dumps(execution, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_execution_sha256 != planned_execution_sha256:
        raise ValueError("actual LigandMPNN execution does not match the complete planned arguments")
    if os.path.lexists(completion_record_path):
        raise ValueError(f"LigandMPNN completion record already exists: {completion_record_path}")
    _validate_runtime_option_contract(arguments)
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    observed_commit = _git_head(checkout)
    if observed_commit != upstream_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {upstream_commit}, observed {observed_commit}")
    with tempfile.TemporaryDirectory(prefix="dnadesign-ligandmpnn-") as temporary:
        snapshot = Path(temporary) / "source"
        snapshot.mkdir()
        materialize_pinned_tree(checkout, upstream_commit, snapshot)
        entrypoint_path = snapshot / entrypoint
        if not entrypoint_path.is_file():
            raise ValueError(f"pinned LigandMPNN commit does not contain {entrypoint}")
        runtime_arguments = list(arguments)
        requested_pdb_path = Path(_runtime_option_value(runtime_arguments, _PDB_FLAG)).expanduser()
        weights_root = snapshot / ".dnadesign-weights"
        weights_root.mkdir()
        _replace_verified_file(
            runtime_arguments,
            flag=_CHECKPOINT_FLAG,
            expected_sha256=checkpoint_sha256,
            destination=weights_root / "ligandmpnn.pt",
        )
        if packing_checkpoint_sha256 is None:
            if _has_flag(runtime_arguments, _PACKING_CHECKPOINT_FLAG):
                raise ValueError("packing checkpoint was supplied without a pinned digest")
        else:
            _replace_verified_file(
                runtime_arguments,
                flag=_PACKING_CHECKPOINT_FLAG,
                expected_sha256=packing_checkpoint_sha256,
                destination=weights_root / "packing.pt",
            )
        inputs_root = snapshot / ".dnadesign-inputs"
        inputs_root.mkdir()
        staged_pdb = _replace_verified_file(
            runtime_arguments,
            flag=_PDB_FLAG,
            expected_sha256=pdb_sha256,
            destination=inputs_root / "pdb",
            preserve_source_name=True,
        )
        if entrypoint == "run.py":
            assert context_inventory_path is not None
            assert context_inventory_sha256 is not None
            assert execution_root is not None
            _validate_runtime_context_inventory(
                reference=LigandMpnnContextInventoryReference(
                    path=context_inventory_path,
                    sha256=context_inventory_sha256,
                ),
                execution_root=execution_root,
                checkout_root=checkout,
                upstream_commit=upstream_commit,
                checkpoint_sha256=checkpoint_sha256,
                packing_checkpoint_sha256=packing_checkpoint_sha256,
                requested_pdb_path=requested_pdb_path,
                requested_pdb_sha256=pdb_sha256,
                use_side_chain_context=_runtime_boolean_option(
                    runtime_arguments,
                    "--ligand_mpnn_use_side_chain_context",
                ),
            )
        if residue_alphabet_sha256 is None:
            if _has_flag(runtime_arguments, _RESIDUE_ALPHABET_FLAG):
                raise ValueError("residue alphabet sidecar was supplied without a pinned digest")
        else:
            _replace_verified_file(
                runtime_arguments,
                flag=_RESIDUE_ALPHABET_FLAG,
                expected_sha256=residue_alphabet_sha256,
                destination=inputs_root / "residue-alphabet.json",
            )
        score_temporary: tempfile.TemporaryDirectory[str] | None = None
        score_publication: tuple[Path, Path] | None = None
        published_score_path: Path | None = None
        published_score_identity: tuple[int, int] | None = None
        published_score_sha256: str | None = None
        design_temporary: tempfile.TemporaryDirectory[str] | None = None
        design_publication: tuple[Path, Path] | None = None
        if entrypoint == "run.py" and _has_flag(runtime_arguments, _OUTPUT_FOLDER_FLAG):
            output_value_index, output_root = _runtime_output_root(runtime_arguments)
            if _lexical_absolute_path(completion_record_path) != output_root / _COMPLETION_RECORD_NAME:
                raise ValueError("design completion record must be inside its per-seed output directory")
            if os.path.lexists(output_root):
                raise ValueError(f"design output directory already exists: {output_root}")
            try:
                output_parent_fd = _open_directory_path(output_root.parent, create=True)
            except OSError as exc:
                raise ValueError(f"design output parent could not be opened safely: {output_root.parent}") from exc
            else:
                os.close(output_parent_fd)
            design_temporary = tempfile.TemporaryDirectory(
                prefix=f".{output_root.name}.attempt-",
                dir=output_root.parent,
            )
            temporary_output_root = Path(design_temporary.name)
            runtime_arguments[output_value_index] = str(temporary_output_root)
            design_publication = (temporary_output_root, output_root)
        if entrypoint == "score.py":
            output_value_index, output_root, expected_output = _reject_existing_score_output(
                runtime_arguments,
                pdb_path=staged_pdb,
            )
            try:
                output_directory_fd = _open_directory_path(output_root, create=True)
            except OSError as exc:
                raise ValueError(f"score output directory could not be opened safely: {output_root}") from exc
            else:
                os.close(output_directory_fd)
            score_attempt_root = output_root.parent
            try:
                score_attempt_root_fd = _open_directory_path(score_attempt_root, create=False)
            except OSError as exc:
                raise ValueError(f"score attempt directory could not be opened safely: {score_attempt_root}") from exc
            else:
                os.close(score_attempt_root_fd)
            score_temporary = tempfile.TemporaryDirectory(
                prefix=f".dnadesign-score-{output_root.parent.name}-{output_root.name}-",
                dir=score_attempt_root,
            )
            temporary_output_root = Path(score_temporary.name)
            runtime_arguments[output_value_index] = str(temporary_output_root)
            score_publication = (temporary_output_root / staged_pdb.with_suffix(".pt").name, expected_output)
        execution_failure: BaseException | None = None
        try:
            subprocess.run(
                [sys.executable, "-B", "-E", "-s", str(entrypoint_path), *runtime_arguments],
                check=True,
            )
            if score_publication is not None:
                published_score_sha256, published_score_identity = _publish_score_output(*score_publication)
                published_score_path = score_publication[1]
        except BaseException as error:
            execution_failure = error
            raise
        finally:
            if score_temporary is not None:
                try:
                    score_temporary.cleanup()
                except BaseException as cleanup_error:
                    if execution_failure is not None:
                        execution_failure.add_note(f"private score attempt cleanup also failed: {cleanup_error}")
                    elif published_score_path is not None and published_score_identity is not None:
                        _rollback_score_after_cleanup_failure(
                            published_score_path,
                            published_identity=published_score_identity,
                        )
                        if not isinstance(cleanup_error, Exception):
                            raise
                        raise ValueError("score attempt cleanup failed after publication") from cleanup_error
                    else:
                        raise
        if design_publication is not None:
            temporary_output_root, output_root = design_publication
            try:
                _sync_regular_directory_tree(temporary_output_root)
                design_output_manifest = build_design_output_manifest(temporary_output_root)
                _write_completion_record(
                    temporary_output_root / _COMPLETION_RECORD_NAME,
                    _completion_record(
                        execution,
                        observed_execution_sha256,
                        score_output_sha256=None,
                        design_output_manifest=design_output_manifest,
                    ),
                    rollback_output_path=None,
                    rollback_output_identity=None,
                )
                _sync_regular_directory_tree(temporary_output_root)
                if build_design_output_manifest(temporary_output_root) != design_output_manifest:
                    raise ValueError("design output tree changed before atomic publication")
                _publish_design_output_directory(temporary_output_root, output_root)
            finally:
                if design_temporary is not None:
                    design_temporary.cleanup()
            return
    _write_completion_record(
        completion_record_path,
        _completion_record(
            execution,
            observed_execution_sha256,
            score_output_sha256=published_score_sha256,
            design_output_manifest=None,
        ),
        rollback_output_path=published_score_path,
        rollback_output_identity=published_score_identity,
    )


def _pinned_execution_payload(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    context_inventory_path: Path | None,
    context_inventory_sha256: str | None,
    execution_root: Path | None,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    completion_record_path: Path,
    arguments: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(arguments, tuple) or any(not isinstance(value, str) for value in arguments):
        raise ValueError("LigandMPNN runtime arguments must be a tuple of strings")
    payload: dict[str, object] = {
        "checkout_root": str(checkout_root),
        "upstream_commit": upstream_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "pdb_sha256": pdb_sha256,
        "packing_checkpoint_sha256": packing_checkpoint_sha256,
        "residue_alphabet_sha256": residue_alphabet_sha256,
        "entrypoint": entrypoint,
        "completion_record_path": str(completion_record_path),
        "arguments": list(arguments),
    }
    if context_inventory_path is not None:
        assert context_inventory_sha256 is not None
        assert execution_root is not None
        payload["context_inventory_path"] = str(context_inventory_path)
        payload["context_inventory_sha256"] = context_inventory_sha256
        payload["execution_root"] = str(execution_root)
    return payload


def _validate_context_binding_fields(
    *,
    entrypoint: str,
    context_inventory_path: Path | None,
    context_inventory_sha256: str | None,
    execution_root: Path | None,
) -> None:
    bound = (
        context_inventory_path is not None,
        context_inventory_sha256 is not None,
        execution_root is not None,
    )
    if entrypoint == "run.py" and not all(bound):
        raise ValueError("design runtime requires a complete context inventory binding")
    if entrypoint == "score.py" and any(bound):
        raise ValueError("score runtime does not accept a design context inventory binding")


def _validate_runtime_context_inventory(
    *,
    reference: LigandMpnnContextInventoryReference,
    execution_root: Path,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    packing_checkpoint_sha256: str | None,
    requested_pdb_path: Path,
    requested_pdb_sha256: str,
    use_side_chain_context: bool,
) -> None:
    """Revalidate bound design evidence immediately before execution."""

    root = execution_root.expanduser().resolve()
    if requested_pdb_path.is_absolute():
        try:
            relative_pdb_path = requested_pdb_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("runtime PDB path is outside the bound execution root") from exc
    else:
        relative_pdb_path = requested_pdb_path
    inventory = load_ligandmpnn_context_inventory(reference, execution_root=root)
    if inventory.input_path != relative_pdb_path:
        raise ValueError("context inventory input path does not match runtime PDB path")
    validate_context_inventory_for_input(
        inventory,
        pdb_path=relative_pdb_path,
        pdb_sha256=requested_pdb_sha256,
        upstream=LigandMpnnUpstreamPin(
            commit=upstream_commit,
            checkpoint_sha256=checkpoint_sha256,
            packing_checkpoint_sha256=packing_checkpoint_sha256,
        ),
        use_side_chain_context=use_side_chain_context,
        checkout_root=checkout_root,
        execution_root=root,
        require_clean_parser_checkout=False,
    )


def _completion_record(
    execution: dict[str, object],
    execution_sha256: str,
    *,
    score_output_sha256: str | None,
    design_output_manifest: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "thread.ligandmpnn.execution_completion",
        "schema_version": 3 if design_output_manifest is not None else 2,
        "status": "completed",
        "execution_sha256": f"sha256:{execution_sha256}",
        "score_output_sha256": score_output_sha256,
        "execution": execution,
    }
    if design_output_manifest is not None:
        payload["design_output_manifest"] = design_output_manifest
    return payload


def _write_completion_record(
    path: Path,
    payload: dict[str, object],
    *,
    rollback_output_path: Path | None,
    rollback_output_identity: tuple[int, int] | None,
) -> None:
    if not isinstance(path, Path):
        raise ValueError("completion record path must be a Path")
    if not path.name or ".." in path.parts:
        raise ValueError("completion record path must not contain traversal")
    if (rollback_output_path is None) != (rollback_output_identity is None):
        raise ValueError("completion rollback output path and identity must be supplied together")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        directory_fd = _open_directory_path(path.parent, create=True)
    except OSError as exc:
        _rollback_output_after_completion_failure(
            rollback_output_path,
            rollback_output_identity=rollback_output_identity,
        )
        raise ValueError(f"LigandMPNN completion record directory could not be opened safely: {path}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    completion_created = False
    completion_identity: tuple[int, int] | None = None
    durability_failure = False
    try:
        try:
            file_descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
            completion_created = True
            completion_status = os.fstat(file_descriptor)
            completion_identity = (completion_status.st_dev, completion_status.st_ino)
            try:
                handle = os.fdopen(file_descriptor, "wb")
            except BaseException:
                os.close(file_descriptor)
                raise
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.fsync(directory_fd)
            except OSError:
                durability_failure = True
                raise
        except BaseException as publication_error:
            try:
                _rollback_completion_and_output(
                    directory_fd,
                    completion_name=path.name,
                    completion_created=completion_created,
                    completion_identity=completion_identity,
                    rollback_output_path=rollback_output_path,
                    rollback_output_identity=rollback_output_identity,
                )
            except OSError as rollback_error:
                raise LigandMpnnCompletionPublicationUncertainError(
                    "LigandMPNN completion publication rollback durability is uncertain"
                ) from rollback_error
            if isinstance(publication_error, OSError):
                message = (
                    "LigandMPNN completion record publication could not be made durable"
                    if durability_failure
                    else "LigandMPNN completion record could not be created durably"
                )
                raise ValueError(f"{message}: {path}") from publication_error
            raise
    finally:
        os.close(directory_fd)


def _open_directory_path(path: Path, *, create: bool) -> int:
    """Open one directory through an entirely no-follow descriptor chain."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    if path.is_absolute():
        current_fd = os.open(path.anchor, directory_flags)
        components = path.parts[1:]
    else:
        current_fd = os.open(".", directory_flags)
        components = path.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            if component == "..":
                raise OSError("directory traversal is not allowed")
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    os.fsync(current_fd)
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _rollback_completion_and_output(
    completion_directory_fd: int,
    *,
    completion_name: str,
    completion_created: bool,
    completion_identity: tuple[int, int] | None,
    rollback_output_path: Path | None,
    rollback_output_identity: tuple[int, int] | None,
) -> None:
    """Remove incomplete lifecycle entries and make their absence durable."""

    output_directory_fd: int | None = None
    if rollback_output_path is not None:
        output_directory_fd = _open_directory_path(rollback_output_path.parent, create=False)
    try:
        completion_owned = False
        if completion_created:
            if completion_identity is None:
                raise LigandMpnnCompletionPublicationUncertainError(
                    "LigandMPNN completion publication ownership was not recorded"
                )
            completion_owned = _owned_leaf_exists(
                completion_directory_fd,
                completion_name,
                completion_identity,
                error_type=LigandMpnnCompletionPublicationUncertainError,
                changed_message="LigandMPNN completion publication rollback target changed",
                inspect_message="LigandMPNN completion publication rollback target could not be inspected",
            )
        output_owned = False
        if output_directory_fd is not None and rollback_output_path is not None:
            if rollback_output_identity is None:
                raise LigandMpnnCompletionPublicationUncertainError(
                    "LigandMPNN completion publication output ownership was not recorded"
                )
            try:
                output_owned = _owned_leaf_exists(
                    output_directory_fd,
                    rollback_output_path.name,
                    rollback_output_identity,
                    error_type=LigandMpnnCompletionPublicationUncertainError,
                    changed_message="LigandMPNN completion publication output rollback target changed",
                    inspect_message="LigandMPNN completion publication output rollback target could not be inspected",
                )
            except LigandMpnnCompletionPublicationUncertainError:
                if completion_owned:
                    os.unlink(completion_name, dir_fd=completion_directory_fd)
                    os.fsync(completion_directory_fd)
                raise
        if completion_owned:
            os.unlink(completion_name, dir_fd=completion_directory_fd)
        if output_owned and output_directory_fd is not None and rollback_output_path is not None:
            os.unlink(rollback_output_path.name, dir_fd=output_directory_fd)
        if completion_created:
            os.fsync(completion_directory_fd)
        if output_directory_fd is not None:
            os.fsync(output_directory_fd)
    finally:
        if output_directory_fd is not None:
            os.close(output_directory_fd)


def _rollback_output_after_completion_failure(
    rollback_output_path: Path | None,
    *,
    rollback_output_identity: tuple[int, int] | None,
) -> None:
    """Roll back a score if its completion directory cannot be opened safely."""

    if rollback_output_path is None:
        return
    if rollback_output_identity is None:
        raise LigandMpnnCompletionPublicationUncertainError(
            "LigandMPNN completion publication output ownership was not recorded"
        )
    try:
        output_directory_fd = _open_directory_path(rollback_output_path.parent, create=False)
        try:
            if _owned_leaf_exists(
                output_directory_fd,
                rollback_output_path.name,
                rollback_output_identity,
                error_type=LigandMpnnCompletionPublicationUncertainError,
                changed_message="LigandMPNN completion publication output rollback target changed",
                inspect_message="LigandMPNN completion publication output rollback target could not be inspected",
            ):
                os.unlink(rollback_output_path.name, dir_fd=output_directory_fd)
            os.fsync(output_directory_fd)
        finally:
            os.close(output_directory_fd)
    except OSError as rollback_error:
        raise LigandMpnnCompletionPublicationUncertainError(
            "LigandMPNN completion publication rollback durability is uncertain"
        ) from rollback_error


def _append_cli_option(argv: list[str], option: str, value: str) -> None:
    if value.startswith("-"):
        argv.append(f"{option}={value}")
    else:
        argv.extend([option, value])


def _split_option_value(argv: tuple[str, ...], option: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == option]
    attached_prefix = f"{option}="
    attached = [value.removeprefix(attached_prefix) for value in argv if value.startswith(attached_prefix)]
    if len(positions) + len(attached) != 1:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    if attached:
        return attached[0]
    if positions[0] + 1 >= len(argv):
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    return argv[positions[0] + 1]


def _replace_verified_file(
    arguments: list[str],
    *,
    flag: str,
    expected_sha256: str,
    destination: Path,
    preserve_source_name: bool = False,
) -> Path:
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"{flag} expected digest must be a lowercase SHA256")
    attached_prefix = f"{flag}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {flag} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {flag}")
    value_index = positions[0] + 1
    source_path = Path(arguments[value_index]).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"{flag} must reference a regular file")
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{flag} file could not be read") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{flag} SHA256 mismatch: expected sha256:{expected_sha256}, observed sha256:{observed_sha256}"
        )
    staged_path = destination / source_path.name if preserve_source_name else destination
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(payload)
    staged_path.chmod(0o400)
    arguments[value_index] = str(staged_path)
    return staged_path


def _validate_runtime_option_contract(arguments: tuple[str, ...]) -> None:
    option_names = tuple(value.partition("=")[0] for value in arguments if value.startswith("--"))
    for option_name in option_names:
        if option_name in _ALTERNATE_SOURCE_FLAGS or (
            option_name not in _CANONICAL_RUNTIME_FLAGS
            and any(canonical.startswith(option_name) for canonical in _CANONICAL_RUNTIME_FLAGS)
        ):
            raise ValueError(f"unattested or ambiguous LigandMPNN runtime option: {option_name}")
    model_positions = [index for index, value in enumerate(arguments) if value == _MODEL_TYPE_FLAG]
    if (
        option_names.count(_MODEL_TYPE_FLAG) != 1
        or len(model_positions) != 1
        or model_positions[0] + 1 >= len(arguments)
        or arguments[model_positions[0] + 1] != "ligand_mpnn"
    ):
        raise ValueError(f"unattested or ambiguous LigandMPNN runtime option: {_MODEL_TYPE_FLAG}")
    if _FIXED_RESIDUES_FLAG in option_names and _REDESIGNED_RESIDUES_FLAG in option_names:
        raise ValueError("fixed_residues and redesigned_residues runtime options are mutually exclusive")
    for index, option_name in enumerate(option_names):
        for earlier_name in option_names[:index]:
            if option_name == earlier_name and option_name in _SEPARATELY_VALIDATED_SINGLETON_FLAGS:
                continue
            if option_name.startswith(earlier_name) or earlier_name.startswith(option_name):
                raise ValueError(f"duplicate LigandMPNN runtime option or abbreviation: {option_name}")


def _reject_existing_score_output(arguments: list[str], *, pdb_path: Path) -> tuple[int, Path, Path]:
    output_value_index, output_root = _runtime_output_root(arguments)
    expected_output = output_root / f"{pdb_path.stem}.pt"
    if expected_output.exists() or expected_output.is_symlink():
        raise ValueError(f"score output already exists; refuse stale or ambiguous result: {expected_output}")
    return output_value_index, output_root, expected_output


def _runtime_output_root(arguments: list[str]) -> tuple[int, Path]:
    attached_prefix = f"{_OUTPUT_FOLDER_FLAG}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {_OUTPUT_FOLDER_FLAG} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == _OUTPUT_FOLDER_FLAG]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {_OUTPUT_FOLDER_FLAG}")
    output_root = _lexical_absolute_path(Path(arguments[positions[0] + 1]).expanduser())
    return positions[0] + 1, output_root


def _runtime_option_value(arguments: list[str], flag: str) -> str:
    attached_prefix = f"{flag}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {flag} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {flag}")
    return arguments[positions[0] + 1]


def _runtime_boolean_option(arguments: list[str], flag: str) -> bool:
    if not _has_flag(arguments, flag):
        return False
    value = _runtime_option_value(arguments, flag)
    if value not in {"0", "1"}:
        raise ValueError(f"runtime argument {flag} must be 0 or 1")
    return value == "1"


def _lexical_absolute_path(path: Path) -> Path:
    """Anchor a relative path without resolving away symlink evidence."""

    return path if path.is_absolute() else Path.cwd() / path


def _sync_regular_directory_tree(root: Path) -> None:
    """Make one private output tree durable before its directory publication."""

    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    directory_flags = file_flags | os.O_DIRECTORY
    for directory, child_directories, filenames in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in child_directories):
            raise ValueError("design output tree must not contain symlinked directories")
        for filename in filenames:
            path = directory_path / filename
            try:
                file_descriptor = os.open(path, file_flags)
                try:
                    if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                        raise OSError("design output is not regular")
                    os.fsync(file_descriptor)
                finally:
                    os.close(file_descriptor)
            except OSError as exc:
                raise ValueError(f"design output could not be synced safely: {path}") from exc
        try:
            directory_fd = os.open(directory_path, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise ValueError(f"design output directory could not be synced safely: {directory_path}") from exc


def _owned_leaf_exists(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    error_type: type[RuntimeError],
    changed_message: str,
    inspect_message: str,
) -> bool:
    """Return whether a no-follow leaf still belongs to this publication."""

    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise error_type(inspect_message) from exc
    if (observed.st_dev, observed.st_ino) != expected_identity:
        raise error_type(changed_message)
    return True


def _publish_design_output_directory(source_path: Path, destination_path: Path) -> None:
    """Publish one complete private output tree through an exclusive reservation."""

    if source_path.parent != destination_path.parent:
        raise ValueError("design output publication requires one parent directory")
    try:
        parent_fd = _open_directory_path(destination_path.parent, create=False)
    except OSError as exc:
        raise ValueError(f"design output parent could not be opened safely: {destination_path.parent}") from exc
    placeholder_identity: tuple[int, int] | None = None
    published = False
    source_identity: tuple[int, int] | None = None
    try:
        try:
            source_fd = os.open(
                source_path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ValueError(f"design output attempt directory is not safe: {source_path}") from exc
        else:
            try:
                source_status = os.fstat(source_fd)
                source_identity = (source_status.st_dev, source_status.st_ino)
            except OSError as exc:
                raise ValueError(f"design output attempt directory is not safe: {source_path}") from exc
            finally:
                os.close(source_fd)
        try:
            os.mkdir(destination_path.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise ValueError(f"design output directory already exists: {destination_path}") from exc
        try:
            placeholder_status = os.stat(destination_path.name, dir_fd=parent_fd, follow_symlinks=False)
            placeholder_identity = (placeholder_status.st_dev, placeholder_status.st_ino)
        except OSError as exc:
            raise LigandMpnnDesignPublicationUncertainError(
                "LigandMPNN design placeholder ownership could not be recorded"
            ) from exc
        try:
            if not _owned_leaf_exists(
                parent_fd,
                destination_path.name,
                placeholder_identity,
                error_type=LigandMpnnDesignPublicationUncertainError,
                changed_message="LigandMPNN design placeholder changed before publication",
                inspect_message="LigandMPNN design placeholder could not be inspected before publication",
            ):
                raise LigandMpnnDesignPublicationUncertainError(
                    "LigandMPNN design placeholder disappeared before publication"
                )
            os.rename(
                source_path.name,
                destination_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            published = True
            os.fsync(parent_fd)
        except OSError as publication_error:
            try:
                if published:
                    assert source_identity is not None
                    if _owned_leaf_exists(
                        parent_fd,
                        destination_path.name,
                        source_identity,
                        error_type=LigandMpnnDesignPublicationUncertainError,
                        changed_message="LigandMPNN design publication rollback target changed",
                        inspect_message="LigandMPNN design publication rollback target could not be inspected",
                    ):
                        os.rename(
                            destination_path.name,
                            source_path.name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                        )
                elif placeholder_identity is not None and _owned_leaf_exists(
                    parent_fd,
                    destination_path.name,
                    placeholder_identity,
                    error_type=LigandMpnnDesignPublicationUncertainError,
                    changed_message="LigandMPNN design placeholder rollback target changed",
                    inspect_message="LigandMPNN design placeholder rollback target could not be inspected",
                ):
                    os.rmdir(destination_path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as rollback_error:
                raise LigandMpnnDesignPublicationUncertainError(
                    "LigandMPNN design publication rollback durability is uncertain"
                ) from rollback_error
            raise ValueError(f"design output publication could not be made durable: {destination_path}") from (
                publication_error
            )
    finally:
        os.close(parent_fd)


def _publish_score_output(source_path: Path, destination_path: Path) -> tuple[str, tuple[int, int]]:
    """Publish and durably commit one score without concurrent replacement."""

    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"pinned score execution did not produce a regular output: {source_path}")
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        source_fd = os.open(source_path, source_flags)
        try:
            source_status = os.fstat(source_fd)
            if not stat.S_ISREG(source_status.st_mode):
                raise OSError("score output is not regular")
            source_identity = (source_status.st_dev, source_status.st_ino)
            digest = hashlib.sha256()
            while payload := os.read(source_fd, 1024 * 1024):
                digest.update(payload)
            os.fsync(source_fd)
        finally:
            os.close(source_fd)
    except OSError as exc:
        raise ValueError(f"score output could not be read durably: {source_path}") from exc
    try:
        directory_fd = _open_directory_path(destination_path.parent, create=False)
    except OSError as exc:
        raise ValueError(f"score output could not be published atomically: {destination_path}") from exc
    try:
        try:
            os.link(
                source_path,
                destination_path.name,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError(f"score output already exists; refuse concurrent result: {destination_path}") from exc
        except OSError as exc:
            raise ValueError(f"score output could not be published atomically: {destination_path}") from exc
        try:
            os.fsync(directory_fd)
        except OSError as publication_error:
            try:
                if _owned_leaf_exists(
                    directory_fd,
                    destination_path.name,
                    source_identity,
                    error_type=LigandMpnnScorePublicationUncertainError,
                    changed_message="LigandMPNN score publication rollback target changed",
                    inspect_message="LigandMPNN score publication rollback target could not be inspected",
                ):
                    os.unlink(destination_path.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError as rollback_error:
                raise LigandMpnnScorePublicationUncertainError(
                    "LigandMPNN score publication rollback durability is uncertain"
                ) from rollback_error
            raise ValueError(f"score output publication could not be made durable: {destination_path}") from (
                publication_error
            )
        return f"sha256:{digest.hexdigest()}", source_identity
    finally:
        os.close(directory_fd)


def _rollback_score_after_cleanup_failure(
    published_path: Path,
    *,
    published_identity: tuple[int, int],
) -> None:
    """Remove only this attempt's score after private cleanup fails."""

    try:
        directory_fd = _open_directory_path(published_path.parent, create=False)
    except OSError as exc:
        raise LigandMpnnScorePublicationUncertainError(
            "LigandMPNN score cleanup rollback durability is uncertain"
        ) from exc
    try:
        try:
            published_status = os.stat(published_path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            published_status = None
        except OSError as exc:
            raise LigandMpnnScorePublicationUncertainError(
                "LigandMPNN score cleanup rollback durability is uncertain"
            ) from exc
        if published_status is not None and (published_status.st_dev, published_status.st_ino) != published_identity:
            raise LigandMpnnScorePublicationUncertainError("LigandMPNN score cleanup rollback target changed")
        if published_status is not None:
            try:
                os.unlink(published_path.name, dir_fd=directory_fd)
            except OSError as exc:
                raise LigandMpnnScorePublicationUncertainError(
                    "LigandMPNN score cleanup rollback durability is uncertain"
                ) from exc
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise LigandMpnnScorePublicationUncertainError(
                "LigandMPNN score cleanup rollback durability is uncertain"
            ) from exc
    finally:
        os.close(directory_fd)


def _has_flag(arguments: list[str], flag: str) -> bool:
    attached_prefix = f"{flag}="
    return any(value == flag or value.startswith(attached_prefix) for value in arguments)


def _git_head(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(checkout), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("LigandMPNN checkout Git commit could not be read") from exc


def main() -> None:
    """Run one attested official entrypoint from a generated command."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--pdb-sha256", required=True)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--context-inventory-path", type=Path)
    parser.add_argument("--context-inventory-sha256")
    parser.add_argument("--packing-checkpoint-sha256")
    parser.add_argument("--residue-alphabet-sha256")
    parser.add_argument("--planned-execution-sha256", required=True)
    parser.add_argument("--completion-record", type=Path, required=True)
    parser.add_argument("--entrypoint", choices=sorted(_ENTRYPOINTS), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    execute_pinned_entrypoint(
        checkout_root=args.checkout_root,
        upstream_commit=args.upstream_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        pdb_sha256=args.pdb_sha256,
        context_inventory_path=args.context_inventory_path,
        context_inventory_sha256=args.context_inventory_sha256,
        execution_root=args.execution_root,
        packing_checkpoint_sha256=args.packing_checkpoint_sha256,
        residue_alphabet_sha256=args.residue_alphabet_sha256,
        entrypoint=args.entrypoint,
        planned_execution_sha256=args.planned_execution_sha256,
        completion_record_path=args.completion_record,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()
