"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/models.py

Typed study-neutral LigandMPNN request models.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/dauparas/LigandMPNN"
DEFAULT_CHECKPOINT_PATH = Path("model_params/ligandmpnn_v_32_010_25.pt")
DEFAULT_PACKING_CHECKPOINT_PATH = Path("model_params/ligandmpnn_sc_v_32_002_16.pt")
_HEX_40 = re.compile(r"[0-9a-fA-F]{40}")
_HEX_64 = re.compile(r"[0-9a-fA-F]{64}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CANONICAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True, order=True)
class LigandMpnnResidue:
    """One upstream PDB residue selector: chain, residue number, insertion code."""

    chain_id: str
    residue_number: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.chain_id, str) or re.fullmatch(r"[A-Za-z0-9]", self.chain_id) is None:
            raise ValueError("chain_id must be one ASCII alphanumeric character")
        if self.insertion_code and re.fullmatch(r"[A-Za-z]", self.insertion_code) is None:
            raise ValueError("insertion_code must be one ASCII letter")
        if isinstance(self.residue_number, bool) or not isinstance(self.residue_number, int):
            raise ValueError("residue_number must be an integer")

    @property
    def upstream_id(self) -> str:
        """Render the exact chain-number-insertion token parsed by upstream."""

        return f"{self.chain_id}{self.residue_number}{self.insertion_code}"


@dataclass(frozen=True)
class LigandMpnnResidueAlphabet:
    """Allowed canonical amino acids for one redesigned residue."""

    residue: LigandMpnnResidue
    allowed_amino_acids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.residue, LigandMpnnResidue):
            raise ValueError("residue must be a LigandMpnnResidue")
        if not isinstance(self.allowed_amino_acids, tuple):
            raise ValueError("allowed_amino_acids must be a tuple")
        if not self.allowed_amino_acids:
            raise ValueError("allowed_amino_acids must not be empty")
        seen: set[str] = set()
        for amino_acid in self.allowed_amino_acids:
            if not isinstance(amino_acid, str) or len(amino_acid) != 1 or amino_acid not in CANONICAL_AA_ALPHABET:
                raise ValueError("allowed_amino_acids must use the canonical 20 amino acids")
            if amino_acid in seen:
                raise ValueError(f"allowed_amino_acids contains duplicate amino acid {amino_acid}")
            seen.add(amino_acid)

    @property
    def omitted_amino_acids(self) -> str:
        """Render the official upstream omission alphabet in canonical order."""

        allowed = set(self.allowed_amino_acids)
        return "".join(amino_acid for amino_acid in f"{CANONICAL_AA_ALPHABET}X" if amino_acid not in allowed)


@dataclass(frozen=True)
class LigandMpnnUpstreamPin:
    """Immutable upstream source and checkpoint identities required by preflight."""

    commit: str
    checkpoint_sha256: str
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    packing_checkpoint_sha256: str | None = None
    packing_checkpoint_path: Path = DEFAULT_PACKING_CHECKPOINT_PATH

    def __post_init__(self) -> None:
        if _HEX_40.fullmatch(self.commit) is None:
            raise ValueError("commit must be a 40-character Git commit hash")
        if _HEX_64.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("checkpoint_sha256 must be a 64-character SHA256 digest")
        if self.packing_checkpoint_sha256 is not None and _HEX_64.fullmatch(self.packing_checkpoint_sha256) is None:
            raise ValueError("packing_checkpoint_sha256 must be a 64-character SHA256 digest")
        _require_relative_file(self.checkpoint_path, field_name="checkpoint_path")
        _require_relative_file(self.packing_checkpoint_path, field_name="packing_checkpoint_path")
        object.__setattr__(self, "commit", self.commit.lower())
        object.__setattr__(self, "checkpoint_sha256", self.checkpoint_sha256.lower())
        if self.packing_checkpoint_sha256 is not None:
            object.__setattr__(self, "packing_checkpoint_sha256", self.packing_checkpoint_sha256.lower())


@dataclass(frozen=True)
class LigandMpnnPackingConfig:
    """Official LigandMPNN side-chain packing controls."""

    enabled: bool = False
    number_of_packs_per_design: int = 4
    repack_everything: bool = False
    use_ligand_context: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.number_of_packs_per_design, bool) or not isinstance(self.number_of_packs_per_design, int):
            raise ValueError("number_of_packs_per_design must be a positive integer")
        if self.number_of_packs_per_design <= 0:
            raise ValueError("number_of_packs_per_design must be positive")
        _require_bools(
            enabled=self.enabled,
            repack_everything=self.repack_everything,
            use_ligand_context=self.use_ligand_context,
        )


@dataclass(frozen=True)
class LigandMpnnContextInventoryReference:
    """Portable identity of one observed pinned-parser context inventory."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_file(self.path, field_name="context inventory path")
        _require_sha256(self.sha256, field_name="context inventory SHA256")
        object.__setattr__(self, "sha256", self.sha256.lower())

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path.as_posix(), "sha256": f"sha256:{self.sha256}"}


@dataclass(frozen=True)
class LigandMpnnRequest:
    """Validated request for deterministic official LigandMPNN CLI adaptation."""

    request_id: str
    pdb_path: Path
    pdb_sha256: str
    output_dir: Path
    upstream: LigandMpnnUpstreamPin
    context_inventory: LigandMpnnContextInventoryReference
    fixed_residues: tuple[LigandMpnnResidue, ...] = field(default_factory=tuple)
    redesigned_residues: tuple[LigandMpnnResidue, ...] = field(default_factory=tuple)
    residue_alphabets: tuple[LigandMpnnResidueAlphabet, ...] = field(default_factory=tuple)
    seeds: tuple[int, ...] = (1,)
    temperature: float = 0.1
    batch_size: int = 1
    number_of_batches: int = 1
    use_atom_context: bool = True
    use_side_chain_context: bool = False
    packing: LigandMpnnPackingConfig = field(default_factory=LigandMpnnPackingConfig)

    def __post_init__(self) -> None:
        if _REQUEST_ID.fullmatch(self.request_id) is None:
            raise ValueError("request_id must contain only letters, numbers, dots, underscores, or hyphens")
        if not isinstance(self.upstream, LigandMpnnUpstreamPin):
            raise ValueError("upstream must be a LigandMpnnUpstreamPin")
        if self.fixed_residues and self.redesigned_residues:
            raise ValueError("fixed_residues and redesigned_residues are mutually exclusive")
        if not isinstance(self.fixed_residues, tuple) or not isinstance(self.redesigned_residues, tuple):
            raise ValueError("fixed_residues and redesigned_residues must be tuples")
        _require_unique_residues(self.fixed_residues, field_name="fixed_residues")
        _require_unique_residues(self.redesigned_residues, field_name="redesigned_residues")
        _require_residue_alphabets(self.residue_alphabets, redesigned_residues=self.redesigned_residues)
        if not isinstance(self.seeds, tuple):
            raise ValueError("seeds must be a tuple of positive integers")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in self.seeds
        ):
            raise ValueError("seeds must contain positive integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (float, int)):
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            isinstance(self.number_of_batches, bool)
            or not isinstance(self.number_of_batches, int)
            or self.number_of_batches <= 0
        ):
            raise ValueError("number_of_batches must be positive")
        if (
            not isinstance(self.pdb_path, Path)
            or self.pdb_path.is_absolute()
            or ".." in self.pdb_path.parts
            or str(self.pdb_path).startswith("~")
            or str(self.pdb_path).startswith("-")
            or self.pdb_path.suffix.lower() != ".pdb"
        ):
            raise ValueError("pdb_path must be a safe non-option relative Path ending in .pdb")
        _require_sha256(self.pdb_sha256, field_name="pdb_sha256")
        object.__setattr__(self, "pdb_sha256", self.pdb_sha256.lower())
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path")
        if self.output_dir.is_absolute() or str(self.output_dir).startswith("~"):
            raise ValueError("output_dir must be a safe non-option relative Path")
        if ".." in self.output_dir.parts:
            raise ValueError("output_dir must not contain traversal")
        if str(self.output_dir).startswith("-"):
            raise ValueError("output_dir must not begin with a hyphen")
        if not isinstance(self.context_inventory, LigandMpnnContextInventoryReference):
            raise ValueError("context_inventory must be a LigandMpnnContextInventoryReference")
        _require_bools(
            use_atom_context=self.use_atom_context,
            use_side_chain_context=self.use_side_chain_context,
        )
        if self.packing.enabled and self.upstream.packing_checkpoint_sha256 is None:
            raise ValueError("packing requires a pinned packing_checkpoint_sha256")

    @property
    def expected_sequence_count(self) -> int:
        """Return upstream batch size times batch count across explicit seeds."""

        return len(self.seeds) * self.batch_size * self.number_of_batches


@dataclass(frozen=True)
class LigandMpnnCommand:
    """One deterministic argv vector and its seed/output identity."""

    seed: int
    output_dir: Path
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "output_dir": str(self.output_dir), "argv": list(self.argv)}


def _require_unique_residues(residues: tuple[LigandMpnnResidue, ...], *, field_name: str) -> None:
    seen: set[str] = set()
    for residue in residues:
        if not isinstance(residue, LigandMpnnResidue):
            raise ValueError(f"{field_name} must contain LigandMpnnResidue values")
        if residue.upstream_id in seen:
            raise ValueError(f"{field_name} contains duplicate residue {residue.upstream_id}")
        seen.add(residue.upstream_id)


def _require_residue_alphabets(
    alphabets: tuple[LigandMpnnResidueAlphabet, ...],
    *,
    redesigned_residues: tuple[LigandMpnnResidue, ...],
) -> None:
    if not isinstance(alphabets, tuple):
        raise ValueError("residue_alphabets must be a tuple")
    redesigned_ids = {residue.upstream_id for residue in redesigned_residues}
    seen: set[str] = set()
    for alphabet in alphabets:
        if not isinstance(alphabet, LigandMpnnResidueAlphabet):
            raise ValueError("residue_alphabets must contain LigandMpnnResidueAlphabet values")
        residue_id = alphabet.residue.upstream_id
        if residue_id in seen:
            raise ValueError(f"residue_alphabets contains duplicate residue {residue_id}")
        if residue_id not in redesigned_ids:
            raise ValueError(f"residue alphabet constraint {residue_id} must be redesigned")
        seen.add(residue_id)


def _require_relative_file(path: Path, *, field_name: str) -> None:
    if not isinstance(path, Path):
        raise ValueError(f"{field_name} must be a Path")
    if path.is_absolute() or not path.name or ".." in path.parts or str(path).startswith("~"):
        raise ValueError(f"{field_name} must be a checkout-relative file path")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA256 digest")


def _require_bools(**values: bool) -> None:
    for field_name, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} must be a boolean")
