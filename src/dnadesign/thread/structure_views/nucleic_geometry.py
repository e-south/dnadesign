"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/nucleic_geometry.py

Coordinate-derived geometry for browser nucleic-acid ribbons.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from math import dist, sqrt

from Bio.PDB import MMCIFParser, PDBParser

from dnadesign.thread.structure_views.models import (
    DNA_RESIDUE_NAMES,
    RNA_RESIDUE_NAMES,
    MoleculeClass,
    StructureFormat,
)

_BACKBONE_ANCHOR_NAMES = ("C4'", "C4*")
_PURINE_RESIDUE_NAMES = frozenset({"A", "G", "I", "DA", "DG"})
_PURINE_RING_ATOMS = frozenset({"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"})
_PYRIMIDINE_RING_ATOMS = frozenset({"N1", "C2", "N3", "C4", "C5", "C6"})
_MAX_ADJACENT_ANCHOR_DISTANCE_ANGSTROM = 10.0
_MIN_VECTOR_LENGTH = 1e-6

Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class NucleotideGeometry:
    """One nucleotide anchor and its base-facing endpoint."""

    molecule_class: MoleculeClass
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    backbone_anchor: Point3D
    base_centroid: Point3D


@dataclass(frozen=True)
class NucleicRibbonMesh:
    """One extruded ribbon mesh for a contiguous nucleic-acid chain span."""

    vertices: tuple[Point3D, ...]
    faces: tuple[int, ...]
    residue_count: int


@dataclass(frozen=True)
class NucleicRibbonGeometry:
    """Ordered nucleotide geometry for one DNA or RNA role."""

    molecule_class: MoleculeClass
    residues: tuple[NucleotideGeometry, ...]
    chain_ribbons: tuple[tuple[NucleotideGeometry, ...], ...]

    @property
    def base_spoke_count(self) -> int:
        return len(self.residues)

    @property
    def backbone_segment_count(self) -> int:
        return sum(max(0, len(ribbon) - 1) for ribbon in self.chain_ribbons)

    def ribbon_meshes(self, *, width: float, thickness: float) -> tuple[NucleicRibbonMesh, ...]:
        """Build one thin rectangular mesh per contiguous chain span."""

        if width <= 0.0 or thickness <= 0.0:
            raise ValueError("Nucleic ribbon width and thickness must be positive")
        return tuple(_ribbon_mesh(ribbon, width=width, thickness=thickness) for ribbon in self.chain_ribbons)

    def filtered(self, residue_numbers: tuple[int, ...]) -> NucleicRibbonGeometry:
        """Return geometry restricted to the declared residue numbers."""

        allowed = frozenset(int(value) for value in residue_numbers)
        return _geometry_from_residues(
            self.molecule_class,
            tuple(residue for residue in self.residues if residue.residue_number in allowed),
        )

    def audit_row(self, *, model_id: str, ribbon_width: float, ribbon_thickness: float) -> dict[str, object]:
        """Return a serializable primitive-count record."""

        meshes = self.ribbon_meshes(width=ribbon_width, thickness=ribbon_thickness)
        return {
            "model_id": model_id,
            "molecule_class": self.molecule_class,
            "nucleotide_count": len(self.residues),
            "ribbon_mesh_count": len(meshes),
            "ribbon_vertex_count": sum(len(mesh.vertices) for mesh in meshes),
            "ribbon_triangle_count": sum(len(mesh.faces) // 3 for mesh in meshes),
            "backbone_segment_count": self.backbone_segment_count,
            "base_spoke_count": self.base_spoke_count,
            "ribbon_width_angstrom": float(ribbon_width),
            "ribbon_thickness_angstrom": float(ribbon_thickness),
        }


def extract_nucleic_ribbon_geometries(
    structure_text: str,
    *,
    structure_format: StructureFormat,
    molecule_classes: tuple[MoleculeClass, ...],
    source_label: str,
) -> dict[MoleculeClass, NucleicRibbonGeometry]:
    """Extract DNA and RNA ribbon geometry with one coordinate parse."""

    return dict(
        _extract_nucleic_ribbon_geometry_items(
            structure_text,
            structure_format,
            tuple(sorted(molecule_classes)),
            source_label,
        )
    )


@lru_cache(maxsize=64)
def _extract_nucleic_ribbon_geometry_items(
    structure_text: str,
    structure_format: StructureFormat,
    molecule_classes: tuple[MoleculeClass, ...],
    source_label: str,
) -> tuple[tuple[MoleculeClass, NucleicRibbonGeometry], ...]:
    """Parse immutable nucleic geometry once per coordinate and role contract."""

    requested_classes = frozenset(molecule_classes)
    if requested_classes - {"dna", "rna"}:
        raise ValueError(f"Nucleic ribbon geometry received unsupported classes: {sorted(requested_classes)}")
    parser = PDBParser(QUIET=True) if structure_format == "pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure(source_label, StringIO(structure_text))
    residues_by_class: dict[MoleculeClass, list[NucleotideGeometry]] = {
        molecule_class: [] for molecule_class in requested_classes
    }
    first_model = next(structure.get_models())
    for chain in first_model:
        for residue in chain:
            residue_name = str(residue.get_resname()).strip().upper()
            molecule_class: MoleculeClass | None = None
            if "dna" in requested_classes and residue_name in DNA_RESIDUE_NAMES:
                molecule_class = "dna"
            elif "rna" in requested_classes and residue_name in RNA_RESIDUE_NAMES:
                molecule_class = "rna"
            if molecule_class is None:
                continue
            atom_coordinates = {
                str(atom.get_name()).strip().upper(): tuple(float(value) for value in atom.coord)
                for atom in residue.get_atoms()
            }
            backbone_anchor = _first_coordinate(atom_coordinates, _BACKBONE_ANCHOR_NAMES)
            base_centroid = _base_centroid(atom_coordinates, residue_name=residue_name)
            residue_number = int(residue.id[1])
            missing: list[str] = []
            if backbone_anchor is None:
                missing.append("C4-prime backbone anchor")
            if base_centroid is None:
                missing.append("base-ring atoms")
            if missing:
                joined = " and ".join(missing)
                raise ValueError(
                    f"{source_label} {molecule_class} chain {chain.id} residue {residue_number} "
                    f"({residue_name}) lacks {joined}; ribbon-with-spokes rendering cannot proceed"
                )
            residues_by_class[molecule_class].append(
                NucleotideGeometry(
                    molecule_class=molecule_class,
                    chain_id=str(chain.id),
                    residue_number=residue_number,
                    insertion_code=str(residue.id[2]).strip(),
                    residue_name=residue_name,
                    backbone_anchor=backbone_anchor,
                    base_centroid=base_centroid,
                )
            )
    return tuple(
        (molecule_class, _geometry_from_residues(molecule_class, tuple(residues)))
        for molecule_class, residues in sorted(residues_by_class.items())
    )


def _first_coordinate(atom_coordinates: dict[str, Point3D], names: tuple[str, ...]) -> Point3D | None:
    for name in names:
        coordinate = atom_coordinates.get(name.upper())
        if coordinate is not None:
            return coordinate
    return None


def _base_centroid(atom_coordinates: dict[str, Point3D], *, residue_name: str) -> Point3D | None:
    atom_names = _PURINE_RING_ATOMS if residue_name in _PURINE_RESIDUE_NAMES else _PYRIMIDINE_RING_ATOMS
    coordinates = [coordinate for atom_name, coordinate in atom_coordinates.items() if atom_name in atom_names]
    if not coordinates:
        return None
    count = float(len(coordinates))
    return tuple(sum(point[axis] for point in coordinates) / count for axis in range(3))  # type: ignore[return-value]


def _geometry_from_residues(
    molecule_class: MoleculeClass,
    residues: tuple[NucleotideGeometry, ...],
) -> NucleicRibbonGeometry:
    ribbons: list[tuple[NucleotideGeometry, ...]] = []
    current: list[NucleotideGeometry] = []
    previous: NucleotideGeometry | None = None
    for residue in residues:
        starts_new_ribbon = (
            previous is None
            or residue.chain_id != previous.chain_id
            or dist(previous.backbone_anchor, residue.backbone_anchor) > _MAX_ADJACENT_ANCHOR_DISTANCE_ANGSTROM
        )
        if starts_new_ribbon:
            if len(current) >= 2:
                ribbons.append(tuple(current))
            current = [residue]
        else:
            current.append(residue)
        previous = residue
    if len(current) >= 2:
        ribbons.append(tuple(current))
    return NucleicRibbonGeometry(
        molecule_class=molecule_class,
        residues=residues,
        chain_ribbons=tuple(ribbons),
    )


def _ribbon_mesh(
    residues: tuple[NucleotideGeometry, ...],
    *,
    width: float,
    thickness: float,
) -> NucleicRibbonMesh:
    half_width = width / 2.0
    half_thickness = thickness / 2.0
    vertices: list[Point3D] = []
    previous_width_axis: Point3D | None = None
    for index, residue in enumerate(residues):
        tangent = _tangent(residues, index)
        base_direction = _normalize(_subtract(residue.base_centroid, residue.backbone_anchor))
        width_axis = _normalize(_cross(tangent, base_direction), fallback=_fallback_perpendicular(tangent))
        if previous_width_axis is not None and _dot(width_axis, previous_width_axis) < 0.0:
            width_axis = _scale(width_axis, -1.0)
        normal_axis = _normalize(_cross(width_axis, tangent), fallback=base_direction)
        anchor = residue.backbone_anchor
        vertices.extend(
            (
                _offset(anchor, width_axis, half_width, normal_axis, half_thickness),
                _offset(anchor, width_axis, -half_width, normal_axis, half_thickness),
                _offset(anchor, width_axis, -half_width, normal_axis, -half_thickness),
                _offset(anchor, width_axis, half_width, normal_axis, -half_thickness),
            )
        )
        previous_width_axis = width_axis

    faces: list[int] = []
    for index in range(len(residues) - 1):
        current = index * 4
        following = (index + 1) * 4
        for edge in range(4):
            next_edge = (edge + 1) % 4
            _append_quad(
                faces,
                current + edge,
                following + edge,
                following + next_edge,
                current + next_edge,
            )
    _append_quad(faces, 0, 1, 2, 3)
    end = (len(residues) - 1) * 4
    _append_quad(faces, end + 3, end + 2, end + 1, end)
    return NucleicRibbonMesh(vertices=tuple(vertices), faces=tuple(faces), residue_count=len(residues))


def _tangent(residues: tuple[NucleotideGeometry, ...], index: int) -> Point3D:
    if index == 0:
        vector = _subtract(residues[1].backbone_anchor, residues[0].backbone_anchor)
    elif index == len(residues) - 1:
        vector = _subtract(residues[-1].backbone_anchor, residues[-2].backbone_anchor)
    else:
        vector = _subtract(residues[index + 1].backbone_anchor, residues[index - 1].backbone_anchor)
    return _normalize(vector)


def _append_quad(faces: list[int], first: int, second: int, third: int, fourth: int) -> None:
    faces.extend((first, second, third, first, third, fourth))


def _subtract(left: Point3D, right: Point3D) -> Point3D:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(vector: Point3D, factor: float) -> Point3D:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(left: Point3D, right: Point3D) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Point3D, right: Point3D) -> Point3D:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Point3D, *, fallback: Point3D | None = None) -> Point3D:
    length = sqrt(_dot(vector, vector))
    if length <= _MIN_VECTOR_LENGTH:
        if fallback is None:
            raise ValueError("Cannot normalize a zero-length ribbon geometry vector")
        return _normalize(fallback)
    return _scale(vector, 1.0 / length)


def _fallback_perpendicular(tangent: Point3D) -> Point3D:
    axis = (1.0, 0.0, 0.0) if abs(tangent[0]) < 0.8 else (0.0, 1.0, 0.0)
    return _cross(tangent, axis)


def _offset(
    anchor: Point3D,
    width_axis: Point3D,
    width_offset: float,
    normal_axis: Point3D,
    normal_offset: float,
) -> Point3D:
    return tuple(
        anchor[index] + width_axis[index] * width_offset + normal_axis[index] * normal_offset for index in range(3)
    )  # type: ignore[return-value]
