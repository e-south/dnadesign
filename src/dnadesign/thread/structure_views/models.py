"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/models.py

Neutral data models for browser structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

StructureFormat = Literal["pdb", "mmcif"]
_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
DNA_RESIDUE_NAMES = frozenset({"DA", "DC", "DG", "DT"})
RNA_RESIDUE_NAMES = frozenset({"A", "C", "G", "I", "U"})
STANDARD_AMINO_ACID_RESIDUE_NAMES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
ViewProjection = Literal["", "orthographic", "perspective"]
ViewStyle = Literal["", "outline"]
MoleculeClass = Literal["protein", "dna", "rna"]
MoleculeRenderStyle = Literal["", "cartoon", "line", "stick"]


@dataclass(frozen=True)
class StructureAtomContent:
    """Heavy-atom content summary for a rendered structure text."""

    atom_count: int
    residue_count: int
    sidechain_atom_count: int
    sidechain_residue_count: int

    @property
    def has_sidechain_atoms(self) -> bool:
        return self.sidechain_atom_count > 0

    @property
    def scope_label(self) -> str:
        if self.atom_count == 0:
            return "no_atoms_detected"
        if self.has_sidechain_atoms:
            return "sidechain_atoms_present"
        return "backbone_only_or_no_sidechain_atoms"


def summarize_pdb_atom_content(structure_text: str) -> StructureAtomContent:
    """Summarize PDB ATOM/HETATM records for side-chain rendering decisions."""

    return summarize_structure_atom_content(structure_text, structure_format="pdb")


def summarize_structure_atom_content(
    structure_text: str,
    *,
    structure_format: StructureFormat = "pdb",
) -> StructureAtomContent:
    """Summarize protein atom content for side-chain rendering decisions."""

    if structure_format == "pdb":
        return _summarize_pdb_atom_content(structure_text)
    if structure_format == "mmcif":
        return _summarize_mmcif_atom_content(structure_text)
    raise ValueError(f"Unsupported structure format for atom summary: {structure_format}")


def _summarize_pdb_atom_content(structure_text: str) -> StructureAtomContent:
    atom_count = 0
    residue_keys: set[tuple[str, str, str, str]] = set()
    sidechain_atom_count = 0
    sidechain_residue_keys: set[tuple[str, str, str, str]] = set()
    for line in structure_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip().upper()
        if residue_name not in STANDARD_AMINO_ACID_RESIDUE_NAMES:
            continue
        residue_key = (line[21:22], line[22:26].strip(), line[26:27].strip(), residue_name)
        atom_count += 1
        residue_keys.add(residue_key)
        if atom_name.upper() not in _BACKBONE_ATOM_NAMES:
            sidechain_atom_count += 1
            sidechain_residue_keys.add(residue_key)
    return StructureAtomContent(
        atom_count=atom_count,
        residue_count=len(residue_keys),
        sidechain_atom_count=sidechain_atom_count,
        sidechain_residue_count=len(sidechain_residue_keys),
    )


def _summarize_mmcif_atom_content(structure_text: str) -> StructureAtomContent:
    atom_count = 0
    residue_keys: set[tuple[str, str, str, str]] = set()
    sidechain_atom_count = 0
    sidechain_residue_keys: set[tuple[str, str, str, str]] = set()
    for line in structure_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("ATOM ", "HETATM ")):
            continue
        try:
            parts = shlex.split(stripped)
        except ValueError:
            continue
        if len(parts) < 10:
            continue
        atom_name = parts[3].strip().upper()
        residue_name = parts[5].strip().upper()
        if residue_name not in STANDARD_AMINO_ACID_RESIDUE_NAMES:
            continue
        chain_id = parts[12] if len(parts) > 12 else parts[6]
        residue_number = parts[13] if len(parts) > 13 else parts[8]
        insertion_code = parts[14] if len(parts) > 14 else "?"
        residue_key = (chain_id, residue_number, insertion_code, residue_name)
        atom_count += 1
        residue_keys.add(residue_key)
        if atom_name not in _BACKBONE_ATOM_NAMES:
            sidechain_atom_count += 1
            sidechain_residue_keys.add(residue_key)
    return StructureAtomContent(
        atom_count=atom_count,
        residue_count=len(residue_keys),
        sidechain_atom_count=sidechain_atom_count,
        sidechain_residue_count=len(sidechain_residue_keys),
    )


@dataclass(frozen=True)
class StructureViewModel:
    """One structure model to load into a browser viewer."""

    model_id: str
    structure_text: str
    structure_format: StructureFormat = "pdb"
    label: str = ""
    color: str = "#0072B2"
    opacity: float = 1.0
    show_sidechains: bool = False
    sidechain_color: str = ""
    sidechain_radius: float = 0.16

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("StructureViewModel.model_id is required")
        if not self.structure_text.strip():
            raise ValueError(f"StructureViewModel.structure_text is required for {self.model_id}")
        if self.structure_format not in {"pdb", "mmcif"}:
            raise ValueError(f"Unsupported structure format for {self.model_id}: {self.structure_format}")
        if not (0.0 < float(self.opacity) <= 1.0):
            raise ValueError(f"StructureViewModel.opacity must be in (0, 1] for {self.model_id}")
        if self.show_sidechains and float(self.sidechain_radius) <= 0.0:
            raise ValueError(f"StructureViewModel.sidechain_radius must be positive for {self.model_id}")


@dataclass(frozen=True)
class StructureViewMoleculeStyle:
    """Optional molecule-class styling for a loaded structure model."""

    molecule_class: MoleculeClass
    model_id: str
    label: str
    color: str
    opacity: float = 1.0
    style: MoleculeRenderStyle = ""
    radius: float = 0.18

    def validate(self, *, model_ids: set[str]) -> None:
        if self.molecule_class not in {"protein", "dna", "rna"}:
            raise ValueError(f"Unsupported molecule class: {self.molecule_class}")
        if self.model_id not in model_ids:
            raise ValueError(f"Molecule style references unknown model_id: {self.model_id}")
        if not self.label.strip():
            raise ValueError(f"StructureViewMoleculeStyle.label is required for {self.molecule_class}")
        if not (0.0 < float(self.opacity) <= 1.0):
            raise ValueError(f"StructureViewMoleculeStyle.opacity must be in (0, 1] for {self.molecule_class}")
        if self.style not in {"", "cartoon", "line", "stick"}:
            raise ValueError(f"Unsupported molecule render style: {self.style}")
        if float(self.radius) <= 0.0:
            raise ValueError(f"StructureViewMoleculeStyle.radius must be positive for {self.molecule_class}")


@dataclass(frozen=True)
class StructureViewSelectionStyle:
    """Residue-level style overlay for one loaded structure model."""

    selection_id: str
    model_id: str
    label: str
    residue_numbers: tuple[int, ...]
    color: str = "#D55E00"
    opacity: float = 1.0
    residue_scope: MoleculeClass = "protein"

    def validate(self, *, model_ids: set[str]) -> None:
        if not self.selection_id.strip():
            raise ValueError("StructureViewSelectionStyle.selection_id is required")
        if self.model_id not in model_ids:
            raise ValueError(f"Selection style references unknown model_id: {self.model_id}")
        if not self.label.strip():
            raise ValueError(f"StructureViewSelectionStyle.label is required for {self.selection_id}")
        if not self.residue_numbers:
            raise ValueError(f"StructureViewSelectionStyle.residue_numbers is required for {self.selection_id}")
        if any(int(residue) < 1 for residue in self.residue_numbers):
            raise ValueError(f"Residue numbers must be one-based positive integers for {self.selection_id}")
        if not (0.0 < float(self.opacity) <= 1.0):
            raise ValueError(f"StructureViewSelectionStyle.opacity must be in (0, 1] for {self.selection_id}")
        if self.residue_scope not in {"protein", "dna", "rna"}:
            raise ValueError(f"Unsupported selection residue scope for {self.selection_id}: {self.residue_scope}")


@dataclass(frozen=True)
class StructureViewSpec:
    """Backend-independent browser structure-view specification."""

    title: str
    models: tuple[StructureViewModel, ...]
    subtitle: str = ""
    description: str = ""
    interpretation_limit: str = ""
    molecule_styles: tuple[StructureViewMoleculeStyle, ...] = ()
    selection_styles: tuple[StructureViewSelectionStyle, ...] = ()
    width: int = 700
    height: int = 500
    background_color: str = "#ffffff"
    style: str = "cartoon"
    projection: ViewProjection = "orthographic"
    view_style: ViewStyle = "outline"
    camera_memory_key: str = ""

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("StructureViewSpec.title is required")
        if self.description and not self.description.strip():
            raise ValueError("StructureViewSpec.description must be non-empty when provided")
        if self.interpretation_limit and not self.interpretation_limit.strip():
            raise ValueError("StructureViewSpec.interpretation_limit must be non-empty when provided")
        if not self.models:
            raise ValueError("StructureViewSpec.models must contain at least one model")
        if int(self.width) < 240 or int(self.height) < 180:
            raise ValueError("StructureViewSpec width/height are too small for review use")
        if self.style not in {"cartoon", "line", "stick"}:
            raise ValueError(f"Unsupported structure-view style: {self.style}")
        if self.projection not in {"", "orthographic", "perspective"}:
            raise ValueError(f"Unsupported structure-view projection: {self.projection}")
        if self.view_style not in {"", "outline"}:
            raise ValueError(f"Unsupported structure-view style enhancement: {self.view_style}")
        if self.camera_memory_key and not self.camera_memory_key.strip():
            raise ValueError("StructureViewSpec.camera_memory_key must be non-empty when provided")
        for model in self.models:
            model.validate()
        model_ids = {model.model_id for model in self.models}
        for molecule_style in self.molecule_styles:
            molecule_style.validate(model_ids=model_ids)
        for selection_style in self.selection_styles:
            selection_style.validate(model_ids=model_ids)
