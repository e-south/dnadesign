"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/commands.py

Deterministic official LigandMPNN command construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.alphabets import LigandMpnnResidueAlphabetSidecar
from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    load_ligandmpnn_context_inventory,
    validate_context_inventory_for_input,
    validate_ligandmpnn_residue_selection,
)
from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnCommand, LigandMpnnRequest
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import build_pinned_runtime_command


def build_ligandmpnn_commands(
    request: LigandMpnnRequest,
    *,
    checkout_root: Path,
    execution_root: Path,
    python_executable: str = "python",
    residue_alphabet_sidecar: LigandMpnnResidueAlphabetSidecar | None = None,
) -> tuple[LigandMpnnCommand, ...]:
    """Build one official ``run.py`` invocation per seed; relative interpreter paths use the current directory."""

    execution_root = resolve_execution_root_for_execution(execution_root)
    checkout_root = resolve_checkout_root_for_execution(checkout_root, execution_root=execution_root)
    _validate_command_input_output_separation(
        request,
        checkout_root=checkout_root,
        execution_root=execution_root,
        python_executable=python_executable,
        residue_alphabet_sidecar=residue_alphabet_sidecar,
    )
    context_inventory = load_ligandmpnn_context_inventory(
        request.context_inventory,
        execution_root=execution_root,
    )
    protein_evidence = validate_context_inventory_for_input(
        context_inventory,
        pdb_path=request.pdb_path,
        pdb_sha256=request.pdb_sha256,
        upstream=request.upstream,
        use_side_chain_context=request.use_side_chain_context,
        checkout_root=checkout_root,
        execution_root=execution_root,
    )
    validate_ligandmpnn_residue_selection(
        fixed_residue_ids=tuple(item.upstream_id for item in request.fixed_residues),
        redesigned_residue_ids=tuple(item.upstream_id for item in request.redesigned_residues),
        protein_residue_ids=protein_evidence.residue_id_set,
    )
    _validate_alphabet_sidecar(
        request,
        residue_alphabet_sidecar,
        execution_root=execution_root,
    )
    commands: list[LigandMpnnCommand] = []
    for seed in request.seeds:
        output_dir = request.output_dir / f"seed_{seed}"
        runtime_arguments = [
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkout_root / request.upstream.checkpoint_path),
            "--pdb_path",
            str(request.pdb_path),
            "--out_folder",
            str(output_dir),
            "--seed",
            str(seed),
            "--temperature",
            str(request.temperature),
            "--batch_size",
            str(request.batch_size),
            "--number_of_batches",
            str(request.number_of_batches),
            "--ligand_mpnn_use_atom_context",
            _flag(request.use_atom_context),
            "--ligand_mpnn_use_side_chain_context",
            _flag(request.use_side_chain_context),
        ]
        _append_residue_selection(runtime_arguments, request)
        if residue_alphabet_sidecar is not None:
            runtime_arguments.extend(["--omit_AA_per_residue", str(residue_alphabet_sidecar.path)])
        runtime_arguments.extend(
            [
                "--pack_side_chains",
                _flag(request.packing.enabled),
                "--number_of_packs_per_design",
                str(request.packing.number_of_packs_per_design),
                "--repack_everything",
                _flag(request.packing.repack_everything),
                "--pack_with_ligand_context",
                _flag(request.packing.use_ligand_context),
            ]
        )
        if request.packing.enabled:
            runtime_arguments.extend(
                ["--checkpoint_path_sc", str(checkout_root / request.upstream.packing_checkpoint_path)]
            )
        argv = build_pinned_runtime_command(
            checkout_root=checkout_root,
            upstream_commit=request.upstream.commit,
            checkpoint_sha256=request.upstream.checkpoint_sha256,
            pdb_sha256=request.pdb_sha256,
            request_id=request.request_id,
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
            python_executable=python_executable,
            output_dir=output_dir,
            arguments=tuple(runtime_arguments),
        )
        commands.append(LigandMpnnCommand(seed=seed, output_dir=output_dir, argv=argv))
    return tuple(commands)


def _validate_command_input_output_separation(
    request: LigandMpnnRequest,
    *,
    checkout_root: Path,
    execution_root: Path,
    python_executable: str,
    residue_alphabet_sidecar: LigandMpnnResidueAlphabetSidecar | None,
) -> None:
    """Reject inputs that make a per-seed output exist before execution."""

    inputs = _command_input_paths(
        request,
        checkout_root=checkout_root,
        execution_root=execution_root,
        python_executable=python_executable,
        residue_alphabet_sidecar=residue_alphabet_sidecar,
    )
    for seed in request.seeds:
        output_path = execution_root / request.output_dir / f"seed_{seed}"
        for field_name, input_path in inputs.items():
            if input_path == output_path or input_path.is_relative_to(output_path):
                raise ValueError(f"{field_name} must not be nested inside a per-seed output directory")


def _command_input_paths(
    request: LigandMpnnRequest,
    *,
    checkout_root: Path,
    execution_root: Path,
    python_executable: str,
    residue_alphabet_sidecar: LigandMpnnResidueAlphabetSidecar | None,
) -> dict[str, Path]:
    """Inventory paths read during construction or launch using their actual resolution bases."""

    inputs = {
        "checkout_root": checkout_root,
        "pdb_path": execution_root / request.pdb_path,
        "context inventory path": execution_root / request.context_inventory.path,
    }
    executable_path = Path(python_executable)
    if executable_path.is_absolute():
        inputs["python_executable"] = executable_path
    elif python_executable != executable_path.name:
        inputs["python_executable"] = (Path.cwd() / executable_path).resolve()
    if residue_alphabet_sidecar is not None:
        sidecar_path = residue_alphabet_sidecar.path
        inputs["residue alphabet sidecar path"] = (
            sidecar_path if sidecar_path.is_absolute() else execution_root / sidecar_path
        )
        materialized_path = residue_alphabet_sidecar.materialized_path
        if materialized_path is not None:
            inputs["residue alphabet sidecar materialized_path"] = (
                materialized_path if materialized_path.is_absolute() else Path.cwd() / materialized_path
            )
    return inputs


def resolve_checkout_root_for_execution(checkout_root: Path, *, execution_root: Path) -> Path:
    """Anchor a safe relative checkout at the command's execution root."""

    if not execution_root.is_absolute():
        raise ValueError("execution_root must be an absolute directory")
    if checkout_root.is_absolute():
        return checkout_root
    if ".." in checkout_root.parts:
        raise ValueError("relative checkout_root must not contain traversal")
    if str(checkout_root).startswith("~"):
        raise ValueError("relative checkout_root must not begin with '~'")
    return execution_root / checkout_root


def resolve_execution_root_for_execution(execution_root: Path) -> Path:
    """Canonicalize one absolute workspace root before binding command evidence."""

    if not isinstance(execution_root, Path) or not execution_root.is_absolute():
        raise ValueError("execution_root must be an absolute directory")
    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("execution_root must be an existing directory")
    return root


def _validate_alphabet_sidecar(
    request: LigandMpnnRequest,
    sidecar: LigandMpnnResidueAlphabetSidecar | None,
    *,
    execution_root: Path,
) -> None:
    if request.residue_alphabets and sidecar is None:
        raise ValueError("residue alphabets require a typed residue alphabet sidecar")
    if not request.residue_alphabets and sidecar is not None:
        raise ValueError("typed residue alphabet sidecar requires residue alphabets")
    if sidecar is not None:
        sidecar.validate_for(request, execution_root=execution_root)


def _append_residue_selection(argv: list[str], request: LigandMpnnRequest) -> None:
    if request.fixed_residues:
        argv.extend(["--fixed_residues", " ".join(residue.upstream_id for residue in request.fixed_residues)])
    elif request.redesigned_residues:
        argv.extend(["--redesigned_residues", " ".join(residue.upstream_id for residue in request.redesigned_residues)])


def _flag(value: bool) -> str:
    return "1" if value else "0"
