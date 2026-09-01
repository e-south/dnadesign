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
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    load_ligandmpnn_context_inventory,
    validate_context_inventory_for_input,
    validate_ligandmpnn_residue_selection,
)
from dnadesign.thread.adapters.ligandmpnn.design_manifest import build_design_output_manifest
from dnadesign.thread.adapters.ligandmpnn.models import (
    MAX_LIGANDMPNN_SEED,
    MIN_LIGANDMPNN_SEED,
    LigandMpnnContextInventoryReference,
    LigandMpnnUpstreamPin,
)
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
_RUNTIME_PATH_FLAGS = (
    _CHECKPOINT_FLAG,
    _PACKING_CHECKPOINT_FLAG,
    _PDB_FLAG,
    _RESIDUE_ALPHABET_FLAG,
    _OUTPUT_FOLDER_FLAG,
)
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

_LINUX_RENAME_NOREPLACE = 1
_MACOS_RENAME_EXCL = 0x00000004
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
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
    request_id: str,
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
    _validate_request_id(request_id)
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
    _append_cli_option(prefix, "--request-id", request_id)
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
    request_id: str,
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
        request_id=request_id,
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
            request_id=request_id,
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
    request_id: str,
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
        request_id=request_id,
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
    request_id: str,
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
        request_id=request_id,
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
        request_id=request_id,
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
    request_id: str,
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
        request_id=request_id,
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
        request_id=request_id,
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
    request_id: str,
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
    assert execution_root is not None
    runtime_execution_root = _canonical_runtime_execution_root(execution_root)
    publication_completion_record_path = _lexical_absolute_path(
        completion_record_path,
        execution_root=runtime_execution_root,
        label="completion record path",
    )
    execution = _pinned_execution_payload(
        checkout_root=checkout_root,
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        request_id=request_id,
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
    if os.path.lexists(publication_completion_record_path):
        raise ValueError(f"LigandMPNN completion record already exists: {publication_completion_record_path}")
    _validate_runtime_option_contract(arguments)
    try:
        _resolve_rename_no_replace()
    except OSError as exc:
        raise ValueError("atomic no-replace publication is unavailable on this platform") from exc
    checkout = _lexical_absolute_path(
        checkout_root,
        execution_root=runtime_execution_root,
        label="checkout_root",
    ).resolve()
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
        requested_pdb_path = Path(_runtime_option_value(runtime_arguments, _PDB_FLAG))
        _anchor_runtime_path_options(runtime_arguments, execution_root=runtime_execution_root)
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
        if entrypoint in _ENTRYPOINTS:
            assert context_inventory_path is not None
            assert context_inventory_sha256 is not None
            assert execution_root is not None
            _validate_runtime_context_inventory(
                reference=LigandMpnnContextInventoryReference(
                    path=context_inventory_path,
                    sha256=context_inventory_sha256,
                ),
                execution_root=runtime_execution_root,
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
                fixed_residue_ids=_runtime_residue_selector_ids(runtime_arguments, _FIXED_RESIDUES_FLAG),
                redesigned_residue_ids=_runtime_residue_selector_ids(
                    runtime_arguments,
                    _REDESIGNED_RESIDUES_FLAG,
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
        score_attempt_path: Path | None = None
        score_attempt_identity: tuple[int, int] | None = None
        score_publication: tuple[Path, Path] | None = None
        published_score_path: Path | None = None
        published_score_identity: tuple[int, int] | None = None
        published_score_sha256: str | None = None
        design_attempt_path: Path | None = None
        design_attempt_identity: tuple[int, int] | None = None
        design_publication: tuple[Path, Path] | None = None
        if entrypoint == "run.py" and _has_flag(runtime_arguments, _OUTPUT_FOLDER_FLAG):
            output_value_index, output_root = _runtime_output_root(
                runtime_arguments,
                execution_root=runtime_execution_root,
            )
            if publication_completion_record_path != output_root / _COMPLETION_RECORD_NAME:
                raise ValueError("design completion record must be inside its per-seed output directory")
            if os.path.lexists(output_root):
                raise ValueError(f"design output directory already exists: {output_root}")
            try:
                output_parent_fd = _open_directory_path(output_root.parent, create=True)
            except OSError as exc:
                raise ValueError(f"design output parent could not be opened safely: {output_root.parent}") from exc
            else:
                os.close(output_parent_fd)
            temporary_output_root, design_attempt_identity = _create_private_attempt_directory(
                prefix=f".{output_root.name}.attempt-",
                parent=output_root.parent,
            )
            design_attempt_path = temporary_output_root
            runtime_arguments[output_value_index] = str(temporary_output_root)
            design_publication = (temporary_output_root, output_root)
        if entrypoint == "score.py":
            output_value_index, output_root, expected_output = _reject_existing_score_output(
                runtime_arguments,
                pdb_path=staged_pdb,
                execution_root=runtime_execution_root,
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
            temporary_output_root, score_attempt_identity = _create_private_attempt_directory(
                prefix=f".dnadesign-score-{output_root.parent.name}-{output_root.name}-",
                parent=score_attempt_root,
            )
            score_attempt_path = temporary_output_root
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
            if score_attempt_path is not None and score_attempt_identity is not None:
                try:
                    _cleanup_private_attempt_directory(
                        score_attempt_path,
                        score_attempt_identity,
                        error_type=LigandMpnnScorePublicationUncertainError,
                        changed_message="LigandMPNN score attempt cleanup target changed",
                        durability_message="LigandMPNN score attempt cleanup durability is uncertain",
                    )
                except BaseException as cleanup_error:
                    if execution_failure is not None:
                        execution_failure.add_note(f"private score attempt cleanup also failed: {cleanup_error}")
                    elif (
                        published_score_path is not None
                        and published_score_identity is not None
                        and published_score_sha256 is not None
                    ):
                        _rollback_score_after_cleanup_failure(
                            published_score_path,
                            published_identity=published_score_identity,
                            published_sha256=published_score_sha256,
                        )
                        if not isinstance(cleanup_error, Exception):
                            raise
                        raise ValueError("score attempt cleanup failed after publication") from cleanup_error
                    else:
                        raise
            if (
                execution_failure is not None
                and design_attempt_path is not None
                and design_attempt_identity is not None
            ):
                try:
                    _cleanup_private_attempt_directory(
                        design_attempt_path,
                        design_attempt_identity,
                        error_type=LigandMpnnDesignPublicationUncertainError,
                        changed_message="LigandMPNN design attempt cleanup target changed",
                        durability_message="LigandMPNN design attempt cleanup durability is uncertain",
                    )
                except BaseException as cleanup_error:
                    execution_failure.add_note(f"private design attempt cleanup also failed: {cleanup_error}")
        if design_publication is not None:
            temporary_output_root, output_root = design_publication
            assert design_attempt_identity is not None
            design_failure: BaseException | None = None
            design_published = False
            try:
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
                _sync_regular_directory_tree(temporary_output_root)
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
                design_output_manifest = build_design_output_manifest(
                    temporary_output_root,
                    expected_root_identity=design_attempt_identity,
                )
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
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
                    rollback_output_sha256=None,
                )
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
                _sync_regular_directory_tree(temporary_output_root)
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
                if (
                    build_design_output_manifest(
                        temporary_output_root,
                        expected_root_identity=design_attempt_identity,
                    )
                    != design_output_manifest
                ):
                    raise ValueError("design output tree changed before atomic publication")
                _require_design_attempt_identity(temporary_output_root, design_attempt_identity)
                _publish_design_output_directory(
                    temporary_output_root,
                    output_root,
                    expected_identity=design_attempt_identity,
                )
                design_published = True
            except BaseException as error:
                design_failure = error
                raise
            finally:
                if not design_published and design_attempt_path is not None and design_attempt_identity is not None:
                    try:
                        _cleanup_private_attempt_directory(
                            design_attempt_path,
                            design_attempt_identity,
                            error_type=LigandMpnnDesignPublicationUncertainError,
                            changed_message="LigandMPNN design attempt cleanup target changed",
                            durability_message="LigandMPNN design attempt cleanup durability is uncertain",
                        )
                    except BaseException as cleanup_error:
                        if design_failure is not None:
                            design_failure.add_note(f"private design attempt cleanup also failed: {cleanup_error}")
                        else:
                            raise
            return
    _write_completion_record(
        publication_completion_record_path,
        _completion_record(
            execution,
            observed_execution_sha256,
            score_output_sha256=published_score_sha256,
            design_output_manifest=None,
        ),
        rollback_output_path=published_score_path,
        rollback_output_identity=published_score_identity,
        rollback_output_sha256=published_score_sha256,
    )


def _pinned_execution_payload(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    request_id: str,
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
    _validate_request_id(request_id)
    payload: dict[str, object] = {
        "checkout_root": str(checkout_root),
        "upstream_commit": upstream_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "pdb_sha256": pdb_sha256,
        "request_id": request_id,
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
    if entrypoint == "score.py" and not all(bound):
        raise ValueError("score runtime requires a complete context inventory binding")


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("runtime request_id must contain only letters, numbers, dots, underscores, or hyphens")


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
    fixed_residue_ids: tuple[str, ...],
    redesigned_residue_ids: tuple[str, ...],
) -> None:
    """Revalidate bound context evidence immediately before execution."""

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
    protein_evidence = validate_context_inventory_for_input(
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
    validate_ligandmpnn_residue_selection(
        fixed_residue_ids=fixed_residue_ids,
        redesigned_residue_ids=redesigned_residue_ids,
        protein_residue_ids=protein_evidence.residue_id_set,
    )


def _runtime_residue_selector_ids(arguments: list[str], flag: str) -> tuple[str, ...]:
    if not _has_flag(arguments, flag):
        return ()
    return tuple(_runtime_option_value(arguments, flag).split())


def _completion_record(
    execution: dict[str, object],
    execution_sha256: str,
    *,
    score_output_sha256: str | None,
    design_output_manifest: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "thread.ligandmpnn.execution_completion",
        "schema_version": 4 if design_output_manifest is not None else 3,
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
    rollback_output_sha256: str | None,
) -> None:
    if not isinstance(path, Path):
        raise ValueError("completion record path must be a Path")
    if not path.name or ".." in path.parts:
        raise ValueError("completion record path must not contain traversal")
    rollback_output_fields = (rollback_output_path, rollback_output_identity, rollback_output_sha256)
    if sum(value is not None for value in rollback_output_fields) not in {0, len(rollback_output_fields)}:
        raise ValueError("completion rollback output path, identity, and digest must be supplied together")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        directory_fd = _open_directory_path(path.parent, create=True)
    except OSError as exc:
        _rollback_output_after_completion_failure(
            rollback_output_path,
            rollback_output_identity=rollback_output_identity,
            rollback_output_sha256=rollback_output_sha256,
        )
        raise ValueError(f"LigandMPNN completion record directory could not be opened safely: {path}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    completion_created = False
    completion_identity: tuple[int, int] | None = None
    completion_descriptor: int | None = None
    completion_fully_written = False
    durability_failure = False
    try:
        try:
            completion_descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
            completion_created = True
            completion_status = os.fstat(completion_descriptor)
            completion_identity = (completion_status.st_dev, completion_status.st_ino)
            handle = os.fdopen(completion_descriptor, "wb", closefd=False)
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            completion_fully_written = True
            try:
                os.fsync(directory_fd)
            except OSError:
                durability_failure = True
                raise
            if rollback_output_path is not None:
                assert rollback_output_identity is not None
                assert rollback_output_sha256 is not None
                output_directory_fd = _open_directory_path(rollback_output_path.parent, create=False)
                try:
                    output_is_owned = _owned_regular_leaf_matches_sha256(
                        output_directory_fd,
                        rollback_output_path.name,
                        rollback_output_identity,
                        rollback_output_sha256,
                        error_type=LigandMpnnCompletionPublicationUncertainError,
                        changed_message="LigandMPNN completion publication output rollback target changed",
                        inspect_message=(
                            "LigandMPNN completion publication output rollback target could not be inspected"
                        ),
                    )
                    if not output_is_owned:
                        raise LigandMpnnCompletionPublicationUncertainError(
                            "LigandMPNN completion publication output rollback target changed"
                        )
                finally:
                    os.close(output_directory_fd)
        except BaseException as publication_error:
            try:
                _rollback_completion_and_output(
                    directory_fd,
                    completion_name=path.name,
                    completion_created=completion_created,
                    completion_identity=completion_identity,
                    completion_expected_bytes=encoded if completion_fully_written else None,
                    rollback_output_path=rollback_output_path,
                    rollback_output_identity=rollback_output_identity,
                    rollback_output_sha256=rollback_output_sha256,
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
        if completion_descriptor is not None:
            os.close(completion_descriptor)
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


def _create_private_attempt_directory(*, parent: Path, prefix: str) -> tuple[Path, tuple[int, int]]:
    """Create one private attempt without registering path-based cleanup authority."""

    attempt_path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    parent_fd = _open_directory_path(parent, create=False)
    try:
        observed = os.stat(attempt_path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode):
            raise OSError("private attempt is not a directory")
        return attempt_path, (observed.st_dev, observed.st_ino)
    finally:
        os.close(parent_fd)


def _require_design_attempt_identity(path: Path, expected_identity: tuple[int, int]) -> None:
    """Require the path to remain the private directory created for this execution."""

    try:
        parent_fd = _open_directory_path(path.parent, create=False)
        try:
            directory_fd = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                observed = os.fstat(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise LigandMpnnDesignPublicationUncertainError("LigandMPNN design attempt identity changed") from exc
    if (observed.st_dev, observed.st_ino) != expected_identity:
        raise LigandMpnnDesignPublicationUncertainError("LigandMPNN design attempt identity changed")


def _remove_directory_tree_contents(directory_fd: int) -> None:
    """Remove one already-quarantined directory tree without following links."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for name in os.listdir(directory_fd):
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
                    raise OSError("private attempt child changed during cleanup")
                _remove_directory_tree_contents(child_fd)
            finally:
                os.close(child_fd)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
                raise OSError("private attempt child changed during cleanup")
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _cleanup_private_attempt_directory(
    attempt_path: Path,
    expected_identity: tuple[int, int],
    *,
    error_type: type[RuntimeError],
    changed_message: str,
    durability_message: str,
) -> None:
    """Quarantine and remove only the attempt directory created by this execution."""

    try:
        parent_fd = _open_directory_path(attempt_path.parent, create=False)
    except OSError as exc:
        raise error_type(durability_message) from exc
    quarantine_name = f".dnadesign-attempt-cleanup-{secrets.token_hex(12)}"
    quarantine_fd: int | None = None
    quarantined = False
    quarantine_leaf = "attempt"
    try:
        os.mkdir(quarantine_name, mode=0o700, dir_fd=parent_fd)
        quarantine_fd = os.open(
            quarantine_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            os.rename(
                attempt_path.name,
                quarantine_leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=quarantine_fd,
            )
        except FileNotFoundError:
            os.rmdir(quarantine_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            return
        quarantined = True
        os.fsync(parent_fd)
        try:
            observed = os.stat(quarantine_leaf, dir_fd=quarantine_fd, follow_symlinks=False)
        except OSError as exc:
            raise error_type(durability_message) from exc
        if not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != expected_identity:
            try:
                _rename_no_replace(
                    quarantine_leaf,
                    attempt_path.name,
                    src_dir_fd=quarantine_fd,
                    dst_dir_fd=parent_fd,
                )
                quarantined = False
                os.fsync(parent_fd)
                os.fsync(quarantine_fd)
                os.rmdir(quarantine_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as restore_error:
                raise error_type(
                    f"{changed_message}; displaced attempt retained in {quarantine_name}/{quarantine_leaf}"
                ) from restore_error
            raise error_type(changed_message)
        attempt_fd = os.open(
            quarantine_leaf,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=quarantine_fd,
        )
        try:
            opened = os.fstat(attempt_fd)
            if (opened.st_dev, opened.st_ino) != expected_identity:
                raise error_type(changed_message)
            _remove_directory_tree_contents(attempt_fd)
        finally:
            os.close(attempt_fd)
        current = os.stat(quarantine_leaf, dir_fd=quarantine_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise error_type(changed_message)
        os.rmdir(quarantine_leaf, dir_fd=quarantine_fd)
        quarantined = False
        os.fsync(quarantine_fd)
        os.rmdir(quarantine_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(attempt_path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise error_type(durability_message) from exc
        raise error_type(changed_message)
    except error_type:
        raise
    except OSError as exc:
        recovery = f"; displaced attempt retained in {quarantine_name}/{quarantine_leaf}" if quarantined else ""
        raise error_type(f"{durability_message}{recovery}") from exc
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        os.close(parent_fd)


def _resolve_rename_no_replace() -> tuple[ctypes._CFuncPtr, int]:
    """Resolve the platform's native atomic no-replace rename before execution."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        if sys.platform.startswith("linux"):
            rename_function = libc.renameat2
            flags = _LINUX_RENAME_NOREPLACE
        elif sys.platform == "darwin":
            rename_function = libc.renameatx_np
            flags = _MACOS_RENAME_EXCL
        else:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is not supported on this platform")
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from error
    rename_function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_function.restype = ctypes.c_int
    return rename_function, flags


def _rename_no_replace(
    source_name: str,
    destination_name: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
) -> None:
    """Atomically rename one leaf without replacing any destination type."""

    rename_function, flags = _resolve_rename_no_replace()
    result = rename_function(
        src_dir_fd,
        os.fsencode(source_name),
        dst_dir_fd,
        os.fsencode(destination_name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _rollback_completion_and_output(
    completion_directory_fd: int,
    *,
    completion_name: str,
    completion_created: bool,
    completion_identity: tuple[int, int] | None,
    completion_expected_bytes: bytes | None,
    rollback_output_path: Path | None,
    rollback_output_identity: tuple[int, int] | None,
    rollback_output_sha256: str | None,
) -> None:
    """Remove incomplete lifecycle entries and make their absence durable."""

    output_directory_fd: int | None = None
    if rollback_output_path is not None:
        output_directory_fd = _open_directory_path(rollback_output_path.parent, create=False)
    try:
        if completion_created:
            if completion_identity is None:
                raise LigandMpnnCompletionPublicationUncertainError(
                    "LigandMPNN completion publication ownership was not recorded"
                )
            _quarantine_and_remove_owned_leaf(
                completion_directory_fd,
                completion_name,
                completion_identity,
                expected_bytes=completion_expected_bytes,
                expected_sha256=None,
                error_type=LigandMpnnCompletionPublicationUncertainError,
                changed_message="LigandMPNN completion publication rollback target changed",
                inspect_message="LigandMPNN completion publication rollback target could not be inspected",
                durability_message="LigandMPNN completion publication rollback durability is uncertain",
            )
        if output_directory_fd is not None and rollback_output_path is not None:
            if rollback_output_identity is None or rollback_output_sha256 is None:
                raise LigandMpnnCompletionPublicationUncertainError(
                    "LigandMPNN completion publication output ownership was not recorded"
                )
            _quarantine_and_remove_owned_leaf(
                output_directory_fd,
                rollback_output_path.name,
                rollback_output_identity,
                expected_bytes=None,
                expected_sha256=rollback_output_sha256,
                error_type=LigandMpnnCompletionPublicationUncertainError,
                changed_message="LigandMPNN completion publication output rollback target changed",
                inspect_message="LigandMPNN completion publication output rollback target could not be inspected",
                durability_message="LigandMPNN completion publication rollback durability is uncertain",
            )
    finally:
        if output_directory_fd is not None:
            os.close(output_directory_fd)


def _rollback_output_after_completion_failure(
    rollback_output_path: Path | None,
    *,
    rollback_output_identity: tuple[int, int] | None,
    rollback_output_sha256: str | None,
) -> None:
    """Roll back a score if its completion directory cannot be opened safely."""

    if rollback_output_path is None:
        return
    if rollback_output_identity is None or rollback_output_sha256 is None:
        raise LigandMpnnCompletionPublicationUncertainError(
            "LigandMPNN completion publication output ownership was not recorded"
        )
    try:
        output_directory_fd = _open_directory_path(rollback_output_path.parent, create=False)
        try:
            _quarantine_and_remove_owned_leaf(
                output_directory_fd,
                rollback_output_path.name,
                rollback_output_identity,
                expected_bytes=None,
                expected_sha256=rollback_output_sha256,
                error_type=LigandMpnnCompletionPublicationUncertainError,
                changed_message="LigandMPNN completion publication output rollback target changed",
                inspect_message="LigandMPNN completion publication output rollback target could not be inspected",
                durability_message="LigandMPNN completion publication rollback durability is uncertain",
            )
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
    _validate_runtime_seed(arguments)


def _validate_runtime_seed(arguments: tuple[str, ...]) -> None:
    """Reject seed zero because pinned upstream replaces falsey seeds randomly."""

    seed_flag = "--seed"
    if not _has_flag(list(arguments), seed_flag):
        raise ValueError(
            f"runtime argument {seed_flag} must be an integer from {MIN_LIGANDMPNN_SEED} through {MAX_LIGANDMPNN_SEED}"
        )
    value = _runtime_option_value(list(arguments), seed_flag)
    if not value.isascii() or not value.isdecimal():
        raise ValueError(
            f"runtime argument {seed_flag} must be an integer from {MIN_LIGANDMPNN_SEED} through {MAX_LIGANDMPNN_SEED}"
        )
    seed = int(value)
    if not MIN_LIGANDMPNN_SEED <= seed <= MAX_LIGANDMPNN_SEED:
        raise ValueError(
            f"runtime argument {seed_flag} must be an integer from {MIN_LIGANDMPNN_SEED} through {MAX_LIGANDMPNN_SEED}"
        )


def _reject_existing_score_output(
    arguments: list[str],
    *,
    pdb_path: Path,
    execution_root: Path,
) -> tuple[int, Path, Path]:
    output_value_index, output_root = _runtime_output_root(arguments, execution_root=execution_root)
    expected_output = output_root / f"{pdb_path.stem}.pt"
    if expected_output.exists() or expected_output.is_symlink():
        raise ValueError(f"score output already exists; refuse stale or ambiguous result: {expected_output}")
    return output_value_index, output_root, expected_output


def _runtime_output_root(arguments: list[str], *, execution_root: Path) -> tuple[int, Path]:
    attached_prefix = f"{_OUTPUT_FOLDER_FLAG}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {_OUTPUT_FOLDER_FLAG} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == _OUTPUT_FOLDER_FLAG]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {_OUTPUT_FOLDER_FLAG}")
    output_root = _lexical_absolute_path(
        Path(arguments[positions[0] + 1]),
        execution_root=execution_root,
        label=_OUTPUT_FOLDER_FLAG,
    )
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


def _canonical_runtime_execution_root(execution_root: Path) -> Path:
    if not isinstance(execution_root, Path) or not execution_root.is_absolute():
        raise ValueError("runtime execution_root must be an absolute directory")
    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("runtime execution_root must be an existing directory")
    return root


def _anchor_runtime_path_options(arguments: list[str], *, execution_root: Path) -> None:
    for flag in _RUNTIME_PATH_FLAGS:
        if not _has_flag(arguments, flag):
            continue
        positions = [index for index, value in enumerate(arguments) if value == flag]
        if not positions:
            continue
        if len(positions) != 1 or positions[0] + 1 >= len(arguments):
            raise ValueError(f"runtime arguments must contain exactly one {flag}")
        value_index = positions[0] + 1
        arguments[value_index] = str(
            _lexical_absolute_path(
                Path(arguments[value_index]),
                execution_root=execution_root,
                label=flag,
            )
        )


def _lexical_absolute_path(path: Path, *, execution_root: Path, label: str) -> Path:
    """Anchor a safe relative path without resolving descendant symlink evidence."""

    if str(path).startswith("~"):
        raise ValueError(f"runtime {label} must not begin with '~'")
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise ValueError(f"runtime {label} must not contain traversal")
    return execution_root / path


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


def _quarantine_and_remove_owned_leaf(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    expected_bytes: bytes | None,
    expected_sha256: str | None,
    error_type: type[RuntimeError],
    changed_message: str,
    inspect_message: str,
    durability_message: str,
    restore_owned_name: str | None = None,
) -> bool:
    """Atomically displace, verify, and durably remove only an owned leaf."""

    if expected_bytes is not None and expected_sha256 is not None:
        raise ValueError("rollback leaf cannot use both byte and digest evidence")
    quarantine_name = f".dnadesign-rollback-{secrets.token_hex(12)}"
    try:
        os.mkdir(quarantine_name, mode=0o700, dir_fd=directory_fd)
        quarantine_fd = os.open(
            quarantine_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise error_type(durability_message) from exc
    quarantined = False
    quarantine_leaf = "publication"
    try:
        try:
            os.rename(
                name,
                quarantine_leaf,
                src_dir_fd=directory_fd,
                dst_dir_fd=quarantine_fd,
            )
        except FileNotFoundError:
            os.rmdir(quarantine_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return False
        quarantined = True
        os.fsync(directory_fd)
        try:
            if expected_bytes is not None:
                owned = _owned_regular_leaf_matches_bytes(
                    quarantine_fd,
                    quarantine_leaf,
                    expected_identity,
                    expected_bytes,
                    error_type=error_type,
                    changed_message=changed_message,
                    inspect_message=inspect_message,
                )
            elif expected_sha256 is not None:
                owned = _owned_regular_leaf_matches_sha256(
                    quarantine_fd,
                    quarantine_leaf,
                    expected_identity,
                    expected_sha256,
                    error_type=error_type,
                    changed_message=changed_message,
                    inspect_message=inspect_message,
                )
            else:
                owned = _owned_leaf_exists(
                    quarantine_fd,
                    quarantine_leaf,
                    expected_identity,
                    error_type=error_type,
                    changed_message=changed_message,
                    inspect_message=inspect_message,
                )
        except error_type as ownership_error:
            try:
                _rename_no_replace(
                    quarantine_leaf,
                    name,
                    src_dir_fd=quarantine_fd,
                    dst_dir_fd=directory_fd,
                )
                os.fsync(directory_fd)
                os.fsync(quarantine_fd)
                os.rmdir(quarantine_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError as restore_error:
                raise error_type(
                    f"{changed_message}; displaced leaf retained in {quarantine_name}/{quarantine_leaf}"
                ) from restore_error
            raise ownership_error
        if not owned:
            os.rmdir(quarantine_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return False
        if restore_owned_name is None:
            os.unlink(quarantine_leaf, dir_fd=quarantine_fd)
        else:
            _rename_no_replace(
                quarantine_leaf,
                restore_owned_name,
                src_dir_fd=quarantine_fd,
                dst_dir_fd=directory_fd,
            )
        quarantined = False
        os.fsync(quarantine_fd)
        os.rmdir(quarantine_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise error_type(inspect_message) from exc
        raise error_type(changed_message)
    except error_type:
        raise
    except OSError as exc:
        recovery = f"; displaced leaf retained in {quarantine_name}/{quarantine_leaf}" if quarantined else ""
        raise error_type(f"{durability_message}{recovery}") from exc
    finally:
        os.close(quarantine_fd)


def _owned_regular_leaf_matches_bytes(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    expected_bytes: bytes,
    *,
    error_type: type[RuntimeError],
    changed_message: str,
    inspect_message: str,
) -> bool:
    """Return whether one no-follow regular leaf is the exact publication."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        file_descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise error_type(inspect_message) from exc
    try:
        observed = os.fstat(file_descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise error_type(changed_message)
        if (observed.st_dev, observed.st_ino) != expected_identity:
            raise error_type(changed_message)
        if observed.st_size != len(expected_bytes):
            raise error_type(changed_message)
        observed_bytes = bytearray()
        while chunk := os.read(file_descriptor, 65536):
            observed_bytes.extend(chunk)
        if bytes(observed_bytes) != expected_bytes:
            raise error_type(changed_message)
    except error_type:
        raise
    except OSError as exc:
        raise error_type(inspect_message) from exc
    finally:
        os.close(file_descriptor)
    return True


def _owned_regular_leaf_matches_sha256(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    expected_sha256: str,
    *,
    error_type: type[RuntimeError],
    changed_message: str,
    inspect_message: str,
) -> bool:
    """Return whether one no-follow regular leaf is the exact digest-bound publication."""

    expected_digest = expected_sha256.removeprefix("sha256:")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        file_descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise error_type(inspect_message) from exc
    try:
        observed = os.fstat(file_descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise error_type(changed_message)
        if (observed.st_dev, observed.st_ino) != expected_identity:
            raise error_type(changed_message)
        digest = hashlib.sha256()
        while chunk := os.read(file_descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise error_type(changed_message)
    except error_type:
        raise
    except OSError as exc:
        raise error_type(inspect_message) from exc
    finally:
        os.close(file_descriptor)
    return True


def _publish_design_output_directory(
    source_path: Path,
    destination_path: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Publish one complete private output tree through an exclusive reservation."""

    if source_path.parent != destination_path.parent:
        raise ValueError("design output publication requires one parent directory")
    try:
        parent_fd = _open_directory_path(destination_path.parent, create=False)
    except OSError as exc:
        raise ValueError(f"design output parent could not be opened safely: {destination_path.parent}") from exc
    source_fd: int | None = None
    source_identity: tuple[int, int] | None = None
    try:
        try:
            source_fd = os.open(
                source_path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            source_status = os.fstat(source_fd)
            source_identity = (source_status.st_dev, source_status.st_ino)
            if source_identity != expected_identity:
                raise LigandMpnnDesignPublicationUncertainError("LigandMPNN design attempt identity changed")
        except OSError as exc:
            raise ValueError(f"design output attempt directory is not safe: {source_path}") from exc
        try:
            _rename_no_replace(
                source_path.name,
                destination_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise ValueError(f"design output directory already exists: {destination_path}") from exc
        except OSError as exc:
            raise ValueError(f"design output could not be published atomically: {destination_path}") from exc
        try:
            published_status = os.stat(destination_path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise LigandMpnnDesignPublicationUncertainError(
                "LigandMPNN design publication could not be verified"
            ) from exc
        published_identity = (published_status.st_dev, published_status.st_ino)
        if not stat.S_ISDIR(published_status.st_mode) or published_identity != expected_identity:
            try:
                _rename_no_replace(
                    destination_path.name,
                    source_path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
            except OSError as recovery_error:
                raise LigandMpnnDesignPublicationUncertainError(
                    "LigandMPNN design attempt identity changed; foreign publication was preserved"
                ) from recovery_error
            raise LigandMpnnDesignPublicationUncertainError("LigandMPNN design attempt identity changed")
        try:
            os.fsync(parent_fd)
        except OSError as publication_error:
            try:
                assert source_identity is not None
                _quarantine_and_remove_owned_leaf(
                    parent_fd,
                    destination_path.name,
                    source_identity,
                    expected_bytes=None,
                    expected_sha256=None,
                    error_type=LigandMpnnDesignPublicationUncertainError,
                    changed_message="LigandMPNN design publication rollback target changed",
                    inspect_message="LigandMPNN design publication rollback target could not be inspected",
                    durability_message="LigandMPNN design publication rollback durability is uncertain",
                    restore_owned_name=source_path.name,
                )
            except OSError as rollback_error:
                raise LigandMpnnDesignPublicationUncertainError(
                    "LigandMPNN design publication rollback durability is uncertain"
                ) from rollback_error
            raise ValueError(f"design output publication could not be made durable: {destination_path}") from (
                publication_error
            )
        try:
            durable_status = os.stat(destination_path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise LigandMpnnDesignPublicationUncertainError(
                "LigandMPNN durable design publication could not be verified"
            ) from exc
        if (
            not stat.S_ISDIR(durable_status.st_mode)
            or (
                durable_status.st_dev,
                durable_status.st_ino,
            )
            != expected_identity
        ):
            raise LigandMpnnDesignPublicationUncertainError("LigandMPNN durable design publication identity changed")
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def _publish_score_output(source_path: Path, destination_path: Path) -> tuple[str, tuple[int, int]]:
    """Publish and durably commit one score without concurrent replacement."""

    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"pinned score execution did not produce a regular output: {source_path}")
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    source_fd: int | None = None
    try:
        source_fd = os.open(source_path, source_flags)
        source_status = os.fstat(source_fd)
        if not stat.S_ISREG(source_status.st_mode):
            raise OSError("score output is not regular")
        source_identity = (source_status.st_dev, source_status.st_ino)
        digest = hashlib.sha256()
        while payload := os.read(source_fd, 1024 * 1024):
            digest.update(payload)
        os.fsync(source_fd)
        source_sha256 = f"sha256:{digest.hexdigest()}"
    except OSError as exc:
        if source_fd is not None:
            os.close(source_fd)
        raise ValueError(f"score output could not be read durably: {source_path}") from exc
    try:
        directory_fd = _open_directory_path(destination_path.parent, create=False)
    except OSError as exc:
        if source_fd is not None:
            os.close(source_fd)
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
        destination_fd: int | None = None
        try:
            destination_fd = os.open(
                destination_path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            destination_status = os.fstat(destination_fd)
            destination_identity = (destination_status.st_dev, destination_status.st_ino)
            destination_digest = hashlib.sha256()
            while payload := os.read(destination_fd, 1024 * 1024):
                destination_digest.update(payload)
            destination_sha256 = f"sha256:{destination_digest.hexdigest()}"
        except OSError as exc:
            raise LigandMpnnScorePublicationUncertainError(
                "LigandMPNN score publication could not be verified"
            ) from exc
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
        if destination_identity != source_identity or destination_sha256 != source_sha256:
            _quarantine_and_remove_owned_leaf(
                directory_fd,
                destination_path.name,
                destination_identity,
                expected_bytes=None,
                expected_sha256=destination_sha256,
                error_type=LigandMpnnScorePublicationUncertainError,
                changed_message="LigandMPNN changed score publication could not be removed safely",
                inspect_message="LigandMPNN changed score publication could not be inspected",
                durability_message="LigandMPNN changed score publication removal is uncertain",
            )
            raise ValueError("score output changed before atomic publication")
        try:
            os.fsync(directory_fd)
        except OSError as publication_error:
            try:
                _quarantine_and_remove_owned_leaf(
                    directory_fd,
                    destination_path.name,
                    source_identity,
                    expected_bytes=None,
                    expected_sha256=source_sha256,
                    error_type=LigandMpnnScorePublicationUncertainError,
                    changed_message="LigandMPNN score publication rollback target changed",
                    inspect_message="LigandMPNN score publication rollback target could not be inspected",
                    durability_message="LigandMPNN score publication rollback durability is uncertain",
                )
            except OSError as rollback_error:
                raise LigandMpnnScorePublicationUncertainError(
                    "LigandMPNN score publication rollback durability is uncertain"
                ) from rollback_error
            raise ValueError(f"score output publication could not be made durable: {destination_path}") from (
                publication_error
            )
        score_is_owned = _owned_regular_leaf_matches_sha256(
            directory_fd,
            destination_path.name,
            source_identity,
            source_sha256,
            error_type=LigandMpnnScorePublicationUncertainError,
            changed_message="LigandMPNN durable score publication identity changed",
            inspect_message="LigandMPNN durable score publication could not be inspected",
        )
        if not score_is_owned:
            raise LigandMpnnScorePublicationUncertainError("LigandMPNN durable score publication identity changed")
        return source_sha256, source_identity
    finally:
        os.close(directory_fd)
        if source_fd is not None:
            os.close(source_fd)


def _rollback_score_after_cleanup_failure(
    published_path: Path,
    *,
    published_identity: tuple[int, int],
    published_sha256: str,
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
            _quarantine_and_remove_owned_leaf(
                directory_fd,
                published_path.name,
                published_identity,
                expected_bytes=None,
                expected_sha256=published_sha256,
                error_type=LigandMpnnScorePublicationUncertainError,
                changed_message="LigandMPNN score cleanup rollback target changed",
                inspect_message="LigandMPNN score cleanup rollback target could not be inspected",
                durability_message="LigandMPNN score cleanup rollback durability is uncertain",
            )
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
    parser.add_argument("--request-id", required=True)
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
        request_id=args.request_id,
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
