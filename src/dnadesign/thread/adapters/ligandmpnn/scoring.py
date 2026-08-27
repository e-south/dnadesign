"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/scoring.py

Typed LigandMPNN per-position probability-scoring commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.commands import (
    resolve_checkout_root_for_execution,
    resolve_execution_root_for_execution,
)
from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    load_ligandmpnn_context_inventory,
    validate_context_inventory_for_input,
)
from dnadesign.thread.adapters.ligandmpnn.models import (
    LigandMpnnCommand,
    LigandMpnnContextInventoryReference,
    LigandMpnnResidue,
    LigandMpnnUpstreamPin,
    validate_ligandmpnn_seeds,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import build_pinned_runtime_command

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_HEX_64 = re.compile(r"[0-9a-fA-F]{64}")


class LigandMpnnScoreMode(str, Enum):
    """The two mutually exclusive probability modes exposed by ``score.py``."""

    SINGLE_AA = "single_aa"
    AUTOREGRESSIVE = "autoregressive"


@dataclass(frozen=True)
class LigandMpnnScoreRequest:
    """Validated request for official per-position probability output."""

    request_id: str
    pdb_path: Path
    pdb_sha256: str
    output_dir: Path
    upstream: LigandMpnnUpstreamPin
    context_inventory: LigandMpnnContextInventoryReference
    fixed_residues: tuple[LigandMpnnResidue, ...] = field(default_factory=tuple)
    redesigned_residues: tuple[LigandMpnnResidue, ...] = field(default_factory=tuple)
    seeds: tuple[int, ...] = (1,)
    batch_size: int = 1
    number_of_batches: int = 10
    mode: LigandMpnnScoreMode = LigandMpnnScoreMode.SINGLE_AA
    use_sequence: bool = False
    use_atom_context: bool = True
    use_side_chain_context: bool = False

    def __post_init__(self) -> None:
        if _REQUEST_ID.fullmatch(self.request_id) is None:
            raise ValueError("request_id must contain only letters, numbers, dots, underscores, or hyphens")
        if (
            not isinstance(self.pdb_path, Path)
            or self.pdb_path.is_absolute()
            or ".." in self.pdb_path.parts
            or str(self.pdb_path).startswith("~")
            or str(self.pdb_path).startswith("-")
            or self.pdb_path.suffix.lower() != ".pdb"
        ):
            raise ValueError("pdb_path must be a safe non-option relative Path ending in .pdb")
        if _HEX_64.fullmatch(self.pdb_sha256) is None:
            raise ValueError("pdb_sha256 must be a 64-character SHA256 digest")
        object.__setattr__(self, "pdb_sha256", self.pdb_sha256.lower())
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path")
        if self.output_dir.is_absolute() or str(self.output_dir).startswith("~"):
            raise ValueError("output_dir must be a safe non-option relative Path")
        if ".." in self.output_dir.parts:
            raise ValueError("output_dir must not contain traversal")
        if str(self.output_dir).startswith("-"):
            raise ValueError("output_dir must not begin with a hyphen")
        if not isinstance(self.upstream, LigandMpnnUpstreamPin):
            raise ValueError("upstream must be a LigandMpnnUpstreamPin")
        if not isinstance(self.context_inventory, LigandMpnnContextInventoryReference):
            raise ValueError("context_inventory must be a LigandMpnnContextInventoryReference")
        if self.fixed_residues and self.redesigned_residues:
            raise ValueError("fixed_residues and redesigned_residues are mutually exclusive")
        _validate_residues(self.fixed_residues, field_name="fixed_residues")
        _validate_residues(self.redesigned_residues, field_name="redesigned_residues")
        validate_ligandmpnn_seeds(self.seeds)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            isinstance(self.number_of_batches, bool)
            or not isinstance(self.number_of_batches, int)
            or self.number_of_batches < 10
        ):
            raise ValueError("number_of_batches must be at least 10 for probability scoring")
        if not isinstance(self.mode, LigandMpnnScoreMode):
            raise ValueError("mode must be a LigandMpnnScoreMode")
        for field_name in ("use_sequence", "use_atom_context", "use_side_chain_context"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")


def build_ligandmpnn_score_commands(
    request: LigandMpnnScoreRequest,
    *,
    checkout_root: Path,
    execution_root: Path,
    python_executable: str = "python",
) -> tuple[LigandMpnnCommand, ...]:
    """Build one explicit official ``score.py`` invocation per seed."""

    execution_root = resolve_execution_root_for_execution(execution_root)
    checkout_root = resolve_checkout_root_for_execution(checkout_root, execution_root=execution_root)
    context_inventory = load_ligandmpnn_context_inventory(
        request.context_inventory,
        execution_root=execution_root,
    )
    validate_context_inventory_for_input(
        context_inventory,
        pdb_path=request.pdb_path,
        pdb_sha256=request.pdb_sha256,
        upstream=request.upstream,
        use_side_chain_context=request.use_side_chain_context,
        checkout_root=checkout_root,
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
            "--batch_size",
            str(request.batch_size),
            "--number_of_batches",
            str(request.number_of_batches),
            "--ligand_mpnn_use_atom_context",
            _flag(request.use_atom_context),
            "--ligand_mpnn_use_side_chain_context",
            _flag(request.use_side_chain_context),
            "--use_sequence",
            _flag(request.use_sequence),
            "--autoregressive_score",
            _flag(request.mode is LigandMpnnScoreMode.AUTOREGRESSIVE),
            "--single_aa_score",
            _flag(request.mode is LigandMpnnScoreMode.SINGLE_AA),
        ]
        _append_residue_selection(runtime_arguments, request)
        argv = build_pinned_runtime_command(
            checkout_root=checkout_root,
            upstream_commit=request.upstream.commit,
            checkpoint_sha256=request.upstream.checkpoint_sha256,
            pdb_sha256=request.pdb_sha256,
            context_inventory_path=request.context_inventory.path,
            context_inventory_sha256=request.context_inventory.sha256,
            execution_root=execution_root,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            python_executable=python_executable,
            output_dir=output_dir,
            arguments=tuple(runtime_arguments),
        )
        commands.append(LigandMpnnCommand(seed=seed, output_dir=output_dir, argv=argv))
    return tuple(commands)


def _append_residue_selection(argv: list[str], request: LigandMpnnScoreRequest) -> None:
    if request.fixed_residues:
        argv.extend(["--fixed_residues", " ".join(item.upstream_id for item in request.fixed_residues)])
    elif request.redesigned_residues:
        argv.extend(["--redesigned_residues", " ".join(item.upstream_id for item in request.redesigned_residues)])


def _validate_residues(residues: tuple[LigandMpnnResidue, ...], *, field_name: str) -> None:
    if not isinstance(residues, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    identifiers: list[str] = []
    for residue in residues:
        if not isinstance(residue, LigandMpnnResidue):
            raise ValueError(f"{field_name} must contain LigandMpnnResidue values")
        identifiers.append(residue.upstream_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field_name} must contain unique residues")


def _flag(value: bool) -> str:
    return "1" if value else "0"
