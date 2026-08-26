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
from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnCommand, LigandMpnnRequest
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import pinned_runtime_prefix


def build_ligandmpnn_commands(
    request: LigandMpnnRequest,
    *,
    checkout_root: Path,
    python_executable: str = "python",
    residue_alphabet_sidecar: LigandMpnnResidueAlphabetSidecar | None = None,
) -> tuple[LigandMpnnCommand, ...]:
    """Build one explicit official ``run.py`` invocation per requested seed."""

    _validate_alphabet_sidecar(request, residue_alphabet_sidecar)
    commands: list[LigandMpnnCommand] = []
    for seed in request.seeds:
        output_dir = request.output_dir / f"seed_{seed}"
        argv = [
            *pinned_runtime_prefix(
                checkout_root=checkout_root,
                upstream_commit=request.upstream.commit,
                checkpoint_sha256=request.upstream.checkpoint_sha256,
                pdb_sha256=request.pdb_sha256,
                packing_checkpoint_sha256=(
                    request.upstream.packing_checkpoint_sha256 if request.packing.enabled else None
                ),
                residue_alphabet_sha256=(
                    residue_alphabet_sidecar.sha256.removeprefix("sha256:")
                    if residue_alphabet_sidecar is not None
                    else None
                ),
                entrypoint="run.py",
                python_executable=python_executable,
            ),
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
            f"{request.temperature:g}",
            "--batch_size",
            str(request.batch_size),
            "--number_of_batches",
            str(request.number_of_batches),
            "--ligand_mpnn_use_atom_context",
            _flag(request.use_atom_context),
            "--ligand_mpnn_use_side_chain_context",
            _flag(request.use_side_chain_context),
        ]
        _append_residue_selection(argv, request)
        if residue_alphabet_sidecar is not None:
            argv.extend(["--omit_AA_per_residue", str(residue_alphabet_sidecar.path)])
        argv.extend(
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
            argv.extend(["--checkpoint_path_sc", str(checkout_root / request.upstream.packing_checkpoint_path)])
        commands.append(LigandMpnnCommand(seed=seed, output_dir=output_dir, argv=tuple(argv)))
    return tuple(commands)


def _validate_alphabet_sidecar(
    request: LigandMpnnRequest,
    sidecar: LigandMpnnResidueAlphabetSidecar | None,
) -> None:
    if request.residue_alphabets and sidecar is None:
        raise ValueError("residue alphabets require a typed residue alphabet sidecar")
    if not request.residue_alphabets and sidecar is not None:
        raise ValueError("typed residue alphabet sidecar requires residue alphabets")
    if sidecar is not None:
        sidecar.validate_for(request)


def _append_residue_selection(argv: list[str], request: LigandMpnnRequest) -> None:
    if request.fixed_residues:
        argv.extend(["--fixed_residues", " ".join(residue.upstream_id for residue in request.fixed_residues)])
    elif request.redesigned_residues:
        argv.extend(["--redesigned_residues", " ".join(residue.upstream_id for residue in request.redesigned_residues)])


def _flag(value: bool) -> str:
    return "1" if value else "0"
