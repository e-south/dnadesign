"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/context_inventory.py

Typed inventory of nonprotein atoms observed by the pinned LigandMPNN parser.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from dnadesign.thread.adapters.ligandmpnn.models import (
    UPSTREAM_REPOSITORY,
    LigandMpnnContextInventoryReference,
    LigandMpnnUpstreamPin,
)

_HEX_64 = re.compile(r"[0-9a-fA-F]{64}")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "status",
        "request_id",
        "request_sha256",
        "input",
        "upstream",
        "parser",
        "requirements",
        "observed",
    }
)


class LigandMpnnContextPolymer(str, Enum):
    """Neutral polymer classes used only to state observed context identity."""

    DNA = "dna"
    RNA = "rna"
    OTHER = "other"


@dataclass(frozen=True)
class LigandMpnnContextAtom:
    """One effective nonprotein atom returned in upstream ``Y/Y_t/Y_m``."""

    serial: int
    atom_name: str
    element: str
    upstream_element_type: int
    chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str
    polymer_type: LigandMpnnContextPolymer

    def __post_init__(self) -> None:
        if isinstance(self.serial, bool) or not isinstance(self.serial, int):
            raise ValueError("context atom serial must be an integer")
        if not self.atom_name or not isinstance(self.atom_name, str):
            raise ValueError("context atom name must be nonempty")
        if not self.element or not isinstance(self.element, str) or self.element != self.element.upper():
            raise ValueError("context atom element must be nonempty uppercase text")
        if (
            isinstance(self.upstream_element_type, bool)
            or not isinstance(self.upstream_element_type, int)
            or self.upstream_element_type <= 0
        ):
            raise ValueError("upstream element type must be a positive integer")
        if not isinstance(self.chain_id, str) or not isinstance(self.residue_name, str) or not self.residue_name:
            raise ValueError("context atom chain and residue identity must be strings")
        if isinstance(self.residue_number, bool) or not isinstance(self.residue_number, int):
            raise ValueError("context atom residue number must be an integer")
        if not isinstance(self.insertion_code, str):
            raise ValueError("context atom insertion code must be text")
        if not isinstance(self.polymer_type, LigandMpnnContextPolymer):
            raise ValueError("context atom polymer type must be a LigandMpnnContextPolymer")

    def to_dict(self) -> dict[str, object]:
        return {
            "serial": self.serial,
            "atom_name": self.atom_name,
            "element": self.element,
            "upstream_element_type": self.upstream_element_type,
            "chain_id": self.chain_id,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
            "polymer_type": self.polymer_type.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LigandMpnnContextAtom:
        expected = frozenset(
            {
                "serial",
                "atom_name",
                "element",
                "upstream_element_type",
                "chain_id",
                "residue_name",
                "residue_number",
                "insertion_code",
                "polymer_type",
            }
        )
        _require_exact_keys(payload, expected, label="context atom")
        return cls(
            serial=payload["serial"],
            atom_name=payload["atom_name"],
            element=payload["element"],
            upstream_element_type=payload["upstream_element_type"],
            chain_id=payload["chain_id"],
            residue_name=payload["residue_name"],
            residue_number=payload["residue_number"],
            insertion_code=payload["insertion_code"],
            polymer_type=LigandMpnnContextPolymer(payload["polymer_type"]),
        )


@dataclass(frozen=True)
class LigandMpnnContextInventory:
    """Validated receipt proving which atoms the pinned parser exposed."""

    request_id: str
    request_sha256: str
    input_path: Path
    input_sha256: str
    upstream_commit: str
    parser_path: Path
    parser_sha256: str
    parser_callable: str
    chains: tuple[str, ...]
    parse_all_atoms: bool
    parse_atoms_with_zero_occupancy: bool
    minimum_nucleotide_atoms: int
    required_polymer_types: tuple[LigandMpnnContextPolymer, ...]
    atoms: tuple[LigandMpnnContextAtom, ...]

    def __post_init__(self) -> None:
        for field_name in ("request_sha256", "input_sha256", "parser_sha256"):
            _require_sha256(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, getattr(self, field_name).lower())
        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("context inventory request_id must be nonempty")
        _require_relative_path(self.input_path, field_name="context inventory input path")
        if not self.upstream_commit or not re.fullmatch(r"[0-9a-fA-F]{40}", self.upstream_commit):
            raise ValueError("context inventory upstream commit must be a 40-character Git hash")
        object.__setattr__(self, "upstream_commit", self.upstream_commit.lower())
        if self.parser_path != Path("data_utils.py") or self.parser_callable != "parse_PDB":
            raise ValueError("context inventory must bind upstream data_utils.py parse_PDB")
        if not isinstance(self.chains, tuple) or any(not isinstance(chain, str) for chain in self.chains):
            raise ValueError("context inventory chains must be a tuple of strings")
        if not isinstance(self.parse_all_atoms, bool) or not isinstance(self.parse_atoms_with_zero_occupancy, bool):
            raise ValueError("context inventory parser options must be booleans")
        if (
            isinstance(self.minimum_nucleotide_atoms, bool)
            or not isinstance(self.minimum_nucleotide_atoms, int)
            or self.minimum_nucleotide_atoms <= 0
        ):
            raise ValueError("minimum_nucleotide_atoms must be a positive integer")
        if not isinstance(self.required_polymer_types, tuple) or any(
            item not in {LigandMpnnContextPolymer.DNA, LigandMpnnContextPolymer.RNA}
            for item in self.required_polymer_types
        ):
            raise ValueError("required_polymer_types may contain only DNA and RNA")
        if len(set(self.required_polymer_types)) != len(self.required_polymer_types):
            raise ValueError("required_polymer_types must be unique")
        if not isinstance(self.atoms, tuple) or any(not isinstance(atom, LigandMpnnContextAtom) for atom in self.atoms):
            raise ValueError("context inventory atoms must be a tuple of LigandMpnnContextAtom values")
        self._validate_requirements()

    @property
    def effective_nonprotein_atom_count(self) -> int:
        return len(self.atoms)

    @property
    def effective_nucleotide_atom_count(self) -> int:
        return sum(atom.polymer_type is not LigandMpnnContextPolymer.OTHER for atom in self.atoms)

    @property
    def polymer_atom_counts(self) -> dict[str, int]:
        counts = Counter(atom.polymer_type.value for atom in self.atoms)
        return {polymer.value: counts.get(polymer.value, 0) for polymer in LigandMpnnContextPolymer}

    @property
    def element_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(atom.element for atom in self.atoms).items()))

    @property
    def residues(self) -> tuple[dict[str, object], ...]:
        grouped: dict[tuple[str, str, int, str, str], list[LigandMpnnContextAtom]] = defaultdict(list)
        for atom in self.atoms:
            key = (
                atom.chain_id,
                atom.residue_name,
                atom.residue_number,
                atom.insertion_code,
                atom.polymer_type.value,
            )
            grouped[key].append(atom)
        rows: list[dict[str, object]] = []
        for key, atoms in sorted(grouped.items()):
            chain_id, residue_name, residue_number, insertion_code, polymer_type = key
            rows.append(
                {
                    "chain_id": chain_id,
                    "residue_name": residue_name,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "polymer_type": polymer_type,
                    "effective_atom_count": len(atoms),
                    "elements": dict(sorted(Counter(atom.element for atom in atoms).items())),
                }
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.context_inventory",
            "schema_version": 1,
            "status": "completed_validated",
            "request_id": self.request_id,
            "request_sha256": f"sha256:{self.request_sha256}",
            "input": {"path": self.input_path.as_posix(), "sha256": f"sha256:{self.input_sha256}"},
            "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": self.upstream_commit},
            "parser": {
                "path": self.parser_path.as_posix(),
                "sha256": f"sha256:{self.parser_sha256}",
                "callable": self.parser_callable,
                "chains": list(self.chains),
                "parse_all_atoms": self.parse_all_atoms,
                "parse_atoms_with_zero_occupancy": self.parse_atoms_with_zero_occupancy,
            },
            "requirements": {
                "minimum_nucleotide_atoms": self.minimum_nucleotide_atoms,
                "required_polymer_types": [item.value for item in self.required_polymer_types],
            },
            "observed": {
                "effective_nonprotein_atom_count": self.effective_nonprotein_atom_count,
                "effective_nucleotide_atom_count": self.effective_nucleotide_atom_count,
                "polymer_atom_counts": self.polymer_atom_counts,
                "element_counts": self.element_counts,
                "chain_ids": sorted({atom.chain_id for atom in self.atoms}),
                "residues": list(self.residues),
                "atoms": [atom.to_dict() for atom in self.atoms],
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LigandMpnnContextInventory:
        _require_exact_keys(payload, _TOP_LEVEL_KEYS, label="context inventory")
        if payload["schema_id"] != "thread.ligandmpnn.context_inventory" or payload["schema_version"] != 1:
            raise ValueError("unsupported LigandMPNN context inventory schema")
        if payload["status"] != "completed_validated":
            raise ValueError("context inventory status must be completed_validated")
        input_payload = _require_mapping(payload["input"], label="context inventory input")
        upstream = _require_mapping(payload["upstream"], label="context inventory upstream")
        parser = _require_mapping(payload["parser"], label="context inventory parser")
        requirements = _require_mapping(payload["requirements"], label="context inventory requirements")
        observed = _require_mapping(payload["observed"], label="context inventory observations")
        _require_exact_keys(input_payload, frozenset({"path", "sha256"}), label="context inventory input")
        _require_exact_keys(upstream, frozenset({"repository", "commit"}), label="context inventory upstream")
        _require_exact_keys(
            parser,
            frozenset(
                {
                    "path",
                    "sha256",
                    "callable",
                    "chains",
                    "parse_all_atoms",
                    "parse_atoms_with_zero_occupancy",
                }
            ),
            label="context inventory parser",
        )
        _require_exact_keys(
            requirements,
            frozenset({"minimum_nucleotide_atoms", "required_polymer_types"}),
            label="context inventory requirements",
        )
        _require_exact_keys(
            observed,
            frozenset(
                {
                    "effective_nonprotein_atom_count",
                    "effective_nucleotide_atom_count",
                    "polymer_atom_counts",
                    "element_counts",
                    "chain_ids",
                    "residues",
                    "atoms",
                }
            ),
            label="context inventory observations",
        )
        atoms_payload = observed.get("atoms")
        if not isinstance(atoms_payload, list):
            raise ValueError("context inventory atoms must be a list")
        if upstream.get("repository") != UPSTREAM_REPOSITORY:
            raise ValueError("context inventory upstream repository mismatch")
        inventory = cls(
            request_id=payload["request_id"],
            request_sha256=_strip_sha256(payload["request_sha256"], field_name="request_sha256"),
            input_path=Path(input_payload["path"]),
            input_sha256=_strip_sha256(input_payload["sha256"], field_name="input sha256"),
            upstream_commit=upstream["commit"],
            parser_path=Path(parser["path"]),
            parser_sha256=_strip_sha256(parser["sha256"], field_name="parser sha256"),
            parser_callable=parser["callable"],
            chains=tuple(parser["chains"]),
            parse_all_atoms=parser["parse_all_atoms"],
            parse_atoms_with_zero_occupancy=parser["parse_atoms_with_zero_occupancy"],
            minimum_nucleotide_atoms=requirements["minimum_nucleotide_atoms"],
            required_polymer_types=tuple(
                LigandMpnnContextPolymer(item) for item in requirements["required_polymer_types"]
            ),
            atoms=tuple(LigandMpnnContextAtom.from_dict(item) for item in atoms_payload),
        )
        expected_observed = inventory.to_dict()["observed"]
        if observed != expected_observed:
            raise ValueError("context inventory observed summaries do not match atom records")
        return inventory

    def _validate_requirements(self) -> None:
        if self.effective_nucleotide_atom_count < self.minimum_nucleotide_atoms:
            raise ValueError(
                f"expected at least {self.minimum_nucleotide_atoms} effective DNA/RNA context atoms; "
                f"observed {self.effective_nucleotide_atom_count}"
            )
        counts = self.polymer_atom_counts
        missing = [item.value for item in self.required_polymer_types if counts[item.value] == 0]
        if missing:
            raise ValueError("required nucleotide polymer context was not observed: " + ", ".join(missing))


def load_ligandmpnn_context_inventory(
    reference: LigandMpnnContextInventoryReference,
    *,
    execution_root: Path,
) -> LigandMpnnContextInventory:
    """Load one digest-bound inventory without following a symlink or path escape."""

    if not isinstance(reference, LigandMpnnContextInventoryReference):
        raise ValueError("reference must be a LigandMpnnContextInventoryReference")
    root = execution_root.expanduser().resolve()
    payload_bytes = _read_descriptor_relative_regular_bytes(
        root,
        reference.path,
        label="context inventory",
    )
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != reference.sha256:
        raise ValueError(
            f"context inventory SHA256 mismatch: expected sha256:{reference.sha256}, observed sha256:{observed_sha256}"
        )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"context inventory is not valid UTF-8 JSON: {reference.path}") from error
    if not isinstance(payload, dict):
        raise ValueError("context inventory root must be an object")
    return LigandMpnnContextInventory.from_dict(payload)


def _read_descriptor_relative_regular_bytes(
    execution_root: Path,
    relative_path: Path,
    *,
    label: str,
) -> bytes:
    """Read one regular leaf through a no-follow descriptor chain exactly once."""

    _require_relative_path(relative_path, field_name=f"{label} path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    leaf_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    root_parts = execution_root.parts
    if not execution_root.is_absolute() or not root_parts:
        raise ValueError("execution_root must resolve to an absolute directory")
    try:
        directory_fd = os.open(execution_root.anchor, directory_flags)
        try:
            for component in (*root_parts[1:], *relative_path.parent.parts):
                if component in {"", "."}:
                    continue
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            leaf_fd = os.open(relative_path.name, leaf_flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {relative_path}") from error
    except OSError as error:
        raise ValueError(f"{label} could not be opened safely: {relative_path}") from error
    try:
        leaf_status = os.fstat(leaf_fd)
    except OSError as error:
        os.close(leaf_fd)
        raise ValueError(f"{label} could not be inspected safely: {relative_path}") from error
    if not stat.S_ISREG(leaf_status.st_mode):
        os.close(leaf_fd)
        raise ValueError(f"{label} must be a regular file")
    try:
        handle = os.fdopen(leaf_fd, "rb")
    except BaseException:
        os.close(leaf_fd)
        raise
    try:
        with handle:
            return handle.read()
    except OSError as error:
        raise ValueError(f"{label} could not be read: {relative_path}") from error


def validate_context_inventory_for_input(
    inventory: LigandMpnnContextInventory,
    *,
    pdb_path: Path,
    pdb_sha256: str,
    upstream: LigandMpnnUpstreamPin,
    use_side_chain_context: bool,
    checkout_root: Path,
    execution_root: Path,
    require_clean_parser_checkout: bool = True,
) -> frozenset[str]:
    """Require an inventory produced for the exact input and context settings."""

    if inventory.input_path != pdb_path or inventory.input_sha256 != pdb_sha256:
        raise ValueError("context inventory input identity does not match request")
    if inventory.upstream_commit != upstream.commit:
        raise ValueError("context inventory upstream commit does not match request")
    expected_parser_sha256 = _pinned_parser_sha256(
        checkout_root,
        upstream_commit=upstream.commit,
        parser_path=inventory.parser_path,
    )
    if inventory.parser_sha256 != expected_parser_sha256:
        raise ValueError("context inventory parser digest does not match pinned upstream commit")
    if inventory.chains:
        raise ValueError("score requests require a context inventory parsed with all input chains")
    if inventory.parse_all_atoms is not use_side_chain_context:
        raise ValueError("context inventory parse_all_atoms does not match side-chain-context mode")
    if inventory.parse_atoms_with_zero_occupancy:
        raise ValueError("score requests require the upstream positive-occupancy parser default")
    if inventory.effective_nucleotide_atom_count <= 0:
        raise ValueError("context inventory proves zero effective DNA/RNA context atoms")
    from dnadesign.thread.adapters.ligandmpnn.context_probe import (
        LigandMpnnContextProbeRequest,
        _derive_ligandmpnn_context_evidence,
    )

    derived, protein_residue_ids = _derive_ligandmpnn_context_evidence(
        LigandMpnnContextProbeRequest(
            request_id=inventory.request_id,
            pdb_path=inventory.input_path,
            pdb_sha256=inventory.input_sha256,
            output_path=Path("context-inventory.json"),
            upstream=upstream,
            minimum_nucleotide_atoms=inventory.minimum_nucleotide_atoms,
            required_polymer_types=inventory.required_polymer_types,
            chains=inventory.chains,
            parse_all_atoms=inventory.parse_all_atoms,
            parse_atoms_with_zero_occupancy=inventory.parse_atoms_with_zero_occupancy,
        ),
        execution_root=execution_root,
        checkout_root=checkout_root,
        require_clean_parser_checkout=require_clean_parser_checkout,
    )
    if inventory != derived:
        raise ValueError("context inventory does not match pinned parser derivation")
    return protein_residue_ids


def validate_ligandmpnn_residue_selection(
    *,
    fixed_residue_ids: tuple[str, ...],
    redesigned_residue_ids: tuple[str, ...],
    protein_residue_ids: frozenset[str],
) -> None:
    """Reject selectors that pinned upstream would silently ignore."""

    for field_name, selected in (
        ("fixed_residues", fixed_residue_ids),
        ("redesigned_residues", redesigned_residue_ids),
    ):
        missing = tuple(item for item in selected if item not in protein_residue_ids)
        if missing:
            raise ValueError(
                f"{field_name} selector(s) {', '.join(missing)} are not present in pinned parser protein residues"
            )


def _pinned_parser_sha256(checkout_root: Path, *, upstream_commit: str, parser_path: Path) -> str:
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    try:
        payload = subprocess.check_output(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(checkout),
                "show",
                f"{upstream_commit}:{parser_path.as_posix()}",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("pinned LigandMPNN parser blob could not be read") from error
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(payload: dict[str, Any], expected: frozenset[str], *, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA256 digest")


def _strip_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{field_name} must use a sha256: prefix")
    digest = value.removeprefix("sha256:")
    _require_sha256(digest, field_name=field_name)
    return digest.lower()


def _require_relative_path(path: Path, *, field_name: str) -> None:
    if not isinstance(path, Path) or path.is_absolute() or not path.name or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe relative file path")


def _within_root(root: Path, relative_path: Path) -> Path:
    _require_relative_path(relative_path, field_name="context inventory path")
    candidate = root / relative_path
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError("context inventory path escapes execution_root")
    return resolved_parent / candidate.name
