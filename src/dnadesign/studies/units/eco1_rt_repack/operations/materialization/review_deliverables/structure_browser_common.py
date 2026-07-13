"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_browser_common.py

Shared helpers for Eco1 review-deliverable structure-browser manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dnadesign.thread.structure_views.models import STANDARD_AMINO_ACID_RESIDUE_NAMES
from dnadesign.thread.structure_views.styles import DNA_COLOR, MOLECULE_CLASS_COLORS, RNA_COLOR

REFERENCE_STRUCTURE_RELATIVE_PATH = "structures/ec86kit_chain_a_backbone_reference.pdb"
REFERENCE_ALL_ATOM_RELATIVE_PATH = "structures/ec86kit_protomer1_all_atom_reference.cif"
REFERENCE_BROWSER_PDB_RELATIVE_PATH = "structures/ec86kit_protomer1_all_atom_reference.pdb"
REFERENCE_COLOR = "#F7F3EA"
PROTEIN_CLASS_COLOR = MOLECULE_CLASS_COLORS["protein"]
DNA_CLASS_COLOR = DNA_COLOR
RNA_CLASS_COLOR = RNA_COLOR
RESIDUE_CATEGORY_HIGHLIGHT_COLOR = "#C00000"
CANDIDATE_PASS_COLOR = "#0072B2"
CANDIDATE_LOW_CONFIDENCE_COLOR = "#6A3D9A"
_STRUCTURE_PREPROCESSING_PROVENANCE_PATH = Path(
    "docs/studies/eco1_rt_repack/workbench/provenance/structure-preprocessing.yaml"
)


@dataclass(frozen=True)
class BrowserReferenceStructure:
    """Visual reference structure staged for browser rendering."""

    local_path: Path
    structure_format: str
    display_label: str
    source_status: str
    source_path: Path


def display_label(candidate_id: str, row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "WT ColabFold baseline"
    label = str(row.get("display_label") or "")
    if label:
        return label
    return f"ProteinMPNN variant {candidate_id.removeprefix('thread_candidate_')[:12]}"


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def repo_relative_hint(path: Path) -> str:
    if path.parent.name == "foldcheck_review":
        return str(Path("foldcheck_review") / path.name)
    return path.name


def stage_browser_reference_structure(
    *,
    repo_root: Path,
    reference_backbone_path: Path,
) -> BrowserReferenceStructure:
    """Regenerate the browser PDB from the complete all-atom Ec86 mmCIF."""

    staged_all_atom_path = reference_backbone_path.parent / Path(REFERENCE_ALL_ATOM_RELATIVE_PATH).name
    staged_browser_pdb_path = reference_backbone_path.parent / Path(REFERENCE_BROWSER_PDB_RELATIVE_PATH).name
    source_status = "regenerated_browser_pdb_from_all_atom_mmcif"
    if not staged_all_atom_path.exists():
        source_path = _resolve_ec86kit_model_source(repo_root)
        if not source_path.exists():
            raise FileNotFoundError(
                "Ec86 all-atom reference mmCIF is required for browser rendering. "
                f"Expected source from study provenance: {source_path}"
            )
        staged_all_atom_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, staged_all_atom_path)
        source_status = "staged_all_atom_mmcif_and_regenerated_browser_pdb"
    _write_browser_pdb_from_mmcif(source_path=staged_all_atom_path, target_path=staged_browser_pdb_path)
    return BrowserReferenceStructure(
        local_path=staged_browser_pdb_path,
        structure_format="pdb",
        display_label="Ec86/7V9U all-atom reference",
        source_status=source_status,
        source_path=staged_all_atom_path,
    )


def reference_residue_number_by_canonical(
    mask_residues: list[dict[str, Any]],
    *,
    reference_structure_format: str,
) -> dict[int, int]:
    """Return the residue-number lookup for the selected visual reference."""

    if reference_structure_format == "mmcif":
        return {
            int(row["canonical_position"]): int(row.get("structure_residue_id") or row["canonical_position"])
            for row in mask_residues
            if str(row.get("mapping_status") or "") == "mapped" or bool(row.get("has_backbone_coordinates")) is True
        }
    return proteinmpnn_export_number_by_canonical(mask_residues)


def reference_selection_coordinate_basis(*, reference_structure_format: str) -> str:
    """Name the coordinate basis used by reference-browser residue selections."""

    if reference_structure_format == "mmcif":
        return "ec86kit_auth_seq_id"
    return "proteinmpnn_export_residue_number"


def proteinmpnn_export_number_by_canonical(mask_residues: list[dict[str, Any]]) -> dict[int, int]:
    """Map canonical positions to the renumbered ProteinMPNN backbone export."""

    mapped_rows = sorted(
        [
            row
            for row in mask_residues
            if str(row.get("mapping_status") or "") == "mapped" or bool(row.get("has_backbone_coordinates")) is True
        ],
        key=lambda row: int(row["canonical_position"]),
    )
    return {int(row["canonical_position"]): index for index, row in enumerate(mapped_rows, start=1)}


def _resolve_ec86kit_model_source(repo_root: Path) -> Path:
    provenance_path = repo_root / _STRUCTURE_PREPROCESSING_PROVENANCE_PATH
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    payload = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {provenance_path}")
    upstream_refs = payload.get("upstream_refs")
    if not isinstance(upstream_refs, dict):
        raise ValueError(f"Missing upstream_refs in {provenance_path}")
    source_ref = str(upstream_refs.get("ec86kit_model_ref") or "")
    if not source_ref:
        raise ValueError(f"Missing upstream_refs.ec86kit_model_ref in {provenance_path}")
    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    return (repo_root / source_ref).resolve()


def _write_browser_pdb_from_mmcif(*, source_path: Path, target_path: Path) -> None:
    """Write a PDB text that keeps protein residues on the existing browser-selection numbering."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    atom_site = MMCIF2Dict(str(source_path))
    required_columns = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    )
    missing_columns = [column for column in required_columns if column not in atom_site]
    if missing_columns:
        raise ValueError(f"Missing mmCIF atom-site columns in {source_path}: {', '.join(missing_columns)}")
    column_lengths = {column: len(atom_site[column]) for column in required_columns}
    if len(set(column_lengths.values())) != 1:
        raise ValueError(f"Inconsistent mmCIF atom-site column lengths in {source_path}: {column_lengths}")

    lines: list[str] = []
    protein_residue_number_by_key: dict[tuple[str, str, str], int] = {}
    source_atom_row_count = next(iter(column_lengths.values()))
    for row_index in range(source_atom_row_count):
        lines.append(
            _mmcif_atom_row_to_pdb_line(
                atom_site,
                row_index=row_index,
                protein_residue_number_by_key=protein_residue_number_by_key,
            )
        )
    if not lines:
        raise ValueError(f"No ATOM/HETATM rows could be converted from {source_path}")
    target_path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _mmcif_atom_row_to_pdb_line(
    atom_site: dict[str, list[str]],
    *,
    row_index: int,
    protein_residue_number_by_key: dict[tuple[str, str, str], int],
) -> str:
    def value(column: str) -> str:
        return str(atom_site[column][row_index])

    group = "HETATM" if value("_atom_site.group_PDB") == "HETATM" else "ATOM"
    try:
        serial = int(value("_atom_site.id"))
        x_coord = float(value("_atom_site.Cartn_x"))
        y_coord = float(value("_atom_site.Cartn_y"))
        z_coord = float(value("_atom_site.Cartn_z"))
        occupancy = float(value("_atom_site.occupancy"))
        b_factor = float(value("_atom_site.B_iso_or_equiv"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric atom-site value at row {row_index + 1}") from exc
    atom_name = _pdb_atom_name(value("_atom_site.label_atom_id"), value("_atom_site.type_symbol"))
    alt_loc = _pdb_optional_char(value("_atom_site.label_alt_id"))
    residue_name = value("_atom_site.label_comp_id")[:3].upper()
    chain_id = _pdb_optional_char(value("_atom_site.auth_asym_id") or value("_atom_site.label_asym_id"))
    auth_seq_id = value("_atom_site.auth_seq_id")
    insertion_code_value = value("_atom_site.pdbx_PDB_ins_code")
    residue_number = _browser_pdb_residue_number(
        residue_name=residue_name,
        chain_id=chain_id,
        auth_seq_id=auth_seq_id,
        insertion_code=insertion_code_value,
        protein_residue_number_by_key=protein_residue_number_by_key,
    )
    insertion_code = _pdb_optional_char(insertion_code_value)
    element = value("_atom_site.type_symbol").strip().upper()[:2]
    return (
        f"{group:<6}{serial:5d} {atom_name}{alt_loc}{residue_name:>3} {chain_id}"
        f"{residue_number:4d}{insertion_code}   {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2}"
    )


def _pdb_atom_name(atom_name: str, element: str) -> str:
    atom = str(atom_name).strip()
    if len(atom) >= 4:
        return atom[:4]
    if len(str(element).strip()) == 1:
        return f" {atom:<3}"
    return f"{atom:<4}"


def _pdb_optional_char(value: str) -> str:
    text = str(value).strip()
    if not text or text in {".", "?"}:
        return " "
    return text[:1]


def _browser_pdb_residue_number(
    *,
    residue_name: str,
    chain_id: str,
    auth_seq_id: str,
    insertion_code: str,
    protein_residue_number_by_key: dict[tuple[str, str, str], int],
) -> int:
    if residue_name not in STANDARD_AMINO_ACID_RESIDUE_NAMES:
        return _pdb_residue_number(auth_seq_id)
    key = (chain_id, str(auth_seq_id).strip(), str(insertion_code).strip())
    if key not in protein_residue_number_by_key:
        protein_residue_number_by_key[key] = len(protein_residue_number_by_key) + 1
    return protein_residue_number_by_key[key]


def _pdb_residue_number(value: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
