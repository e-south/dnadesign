"""Deterministic official LigandMPNN command construction."""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnCommand, LigandMpnnRequest


def build_ligandmpnn_commands(
    request: LigandMpnnRequest,
    *,
    checkout_root: Path,
    python_executable: str = "python",
) -> tuple[LigandMpnnCommand, ...]:
    """Build one explicit official ``run.py`` invocation per requested seed."""

    commands: list[LigandMpnnCommand] = []
    for seed in request.seeds:
        output_dir = request.output_dir / f"seed_{seed}"
        argv = [
            python_executable,
            str(checkout_root / "run.py"),
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


def _append_residue_selection(argv: list[str], request: LigandMpnnRequest) -> None:
    if request.fixed_residues:
        argv.extend(["--fixed_residues", " ".join(residue.upstream_id for residue in request.fixed_residues)])
    elif request.redesigned_residues:
        argv.extend(["--redesigned_residues", " ".join(residue.upstream_id for residue in request.redesigned_residues)])


def _flag(value: bool) -> str:
    return "1" if value else "0"
