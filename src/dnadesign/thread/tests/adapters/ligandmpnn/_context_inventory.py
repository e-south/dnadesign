"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/_context_inventory.py

Builds real digest-bound LigandMPNN context inventory fixtures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextAtom,
    LigandMpnnContextInventory,
    LigandMpnnContextInventoryReference,
    LigandMpnnContextPolymer,
    LigandMpnnContextProbeRequest,
    LigandMpnnUpstreamPin,
)
from dnadesign.thread.adapters.ligandmpnn.context_probe import _probe_request_sha256

PINNED_CONTEXT_PARSER_PAYLOAD = b"""from pathlib import Path

import numpy as np

VALUE = "attested"

element_dict_rev = {7: "N", 15: "P"}
restype_int_to_str = dict(enumerate("ACDEFGHIKLMNPQRSTVWYX"))
restype_str_to_int = {value: key for key, value in restype_int_to_str.items()}
restype_1to3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "Y": "TYR"}
restype_3to1 = {value: key for key, value in restype_1to3.items()}


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self._values


class _Atom:
    def __init__(self, serial, name, element, chain, residue, number, coordinates):
        self._values = serial, name, element, chain, residue, number, coordinates

    def getSerial(self): return self._values[0]
    def getName(self): return self._values[1]
    def getElement(self): return self._values[2]
    def getChid(self): return self._values[3]
    def getResname(self): return self._values[4]
    def getResnum(self): return self._values[5]
    def getIcode(self): return ""
    def getCoords(self): return np.asarray(self._values[6], dtype=np.float32)


class _Selection:
    def iterAtoms(self):
        return iter((
            _Atom(1, "P", "P", "D", "DC", 12, (1.0, 0.0, 0.0)),
            _Atom(2, "N9", "N", "E", "G", 66, (2.0, 0.0, 0.0)),
        ))


def packed_pdb_payload(sequence):
    identities = (("A", 12, ""), ("A", 13, "B"), ("B", -2, "A"), ("B", 2, ""))
    if len(sequence) != len(identities):
        raise ValueError("packed fixture sequence length mismatch")
    lines = ["REMARK DNADESIGN PACKED FIXTURE"]
    serial = 1
    for amino_acid, (chain, residue_number, insertion_code) in zip(sequence, identities):
        residue_name = restype_1to3[amino_acid]
        for atom_name, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
            coordinate = float(serial)
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain}{residue_number:4d}{insertion_code:1s}"
                f"   {coordinate:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    return "\\n".join(lines) + "\\n"


def _parse_packed_pdb(input_path):
    lines = Path(input_path).read_text(encoding="utf-8").splitlines()
    atom_lines = [line for line in lines if line.startswith("ATOM  ")]
    if not atom_lines or any(len(line) < 78 for line in atom_lines):
        raise ValueError("invalid packed PDB fixture")
    residue_order = []
    residue_names = {}
    residue_atoms = {}
    for line in atom_lines:
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain = line[21:22]
        residue_number = int(line[22:26])
        insertion_code = line[26:27].strip()
        float(line[30:38]); float(line[38:46]); float(line[46:54])
        key = chain, residue_number, insertion_code
        if key not in residue_names:
            residue_order.append(key)
            residue_names[key] = residue_name
            residue_atoms[key] = set()
        if residue_names[key] != residue_name:
            raise ValueError("inconsistent packed PDB residue name")
        residue_atoms[key].add(atom_name)
    try:
        native_sequence = [restype_str_to_int[restype_3to1[residue_names[key]]] for key in residue_order]
    except KeyError as error:
        raise ValueError("unknown packed PDB residue") from error
    parsed = {
        "Y": _Tensor(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        "Y_t": _Tensor((15, 7)),
        "Y_m": _Tensor((1, 1)),
        "R_idx": _Tensor(tuple(key[1] for key in residue_order)),
        "chain_letters": np.asarray(tuple(key[0] for key in residue_order)),
        "S": _Tensor(tuple(native_sequence)),
        "mask": _Tensor(tuple(int({"N", "CA", "C", "O"}.issubset(residue_atoms[key])) for key in residue_order)),
    }
    return parsed, None, _Selection(), tuple(key[2] for key in residue_order), None


def parse_PDB(input_path, *args, **kwargs):
    del args, kwargs
    if "_packed_" in Path(input_path).name:
        return _parse_packed_pdb(input_path)
    parsed = {
        "Y": _Tensor(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        "Y_t": _Tensor((15, 7)),
        "Y_m": _Tensor((1, 1)),
        "R_idx": _Tensor((12, 13, -2, 2)),
        "chain_letters": np.asarray(("A", "A", "B", "B")),
        "S": _Tensor((0, 1, 2, 3)),
        "mask": _Tensor((1, 0, 1, 1)),
    }
    return parsed, None, _Selection(), ("", "B", "A", ""), None
"""


def write_context_inventory(
    root: Path,
    *,
    input_path: Path,
    input_sha256: str,
    upstream_commit: str,
    parse_all_atoms: bool,
    parser_sha256: str = "c" * 64,
    relative_path: Path = Path("evidence/context-inventory.json"),
) -> LigandMpnnContextInventoryReference:
    """Write one valid inventory and return its exact relative reference."""

    probe_request = LigandMpnnContextProbeRequest(
        request_id="generic_context_inventory",
        pdb_path=input_path,
        pdb_sha256=input_sha256,
        output_path=relative_path,
        upstream=LigandMpnnUpstreamPin(commit=upstream_commit, checkpoint_sha256="0" * 64),
        minimum_nucleotide_atoms=1,
        required_polymer_types=(LigandMpnnContextPolymer.DNA, LigandMpnnContextPolymer.RNA),
        parse_all_atoms=parse_all_atoms,
    )
    inventory = LigandMpnnContextInventory(
        request_id="generic_context_inventory",
        request_sha256=_probe_request_sha256(probe_request),
        input_path=input_path,
        input_sha256=input_sha256,
        upstream_commit=upstream_commit,
        parser_path=Path("data_utils.py"),
        parser_sha256=parser_sha256,
        parser_callable="parse_PDB",
        chains=(),
        parse_all_atoms=parse_all_atoms,
        parse_atoms_with_zero_occupancy=False,
        minimum_nucleotide_atoms=1,
        required_polymer_types=(LigandMpnnContextPolymer.DNA, LigandMpnnContextPolymer.RNA),
        atoms=(
            LigandMpnnContextAtom(1, "P", "P", 15, "D", "DC", 12, "", LigandMpnnContextPolymer.DNA),
            LigandMpnnContextAtom(2, "N9", "N", 7, "E", "G", 66, "", LigandMpnnContextPolymer.RNA),
        ),
    )
    payload = (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return LigandMpnnContextInventoryReference(
        path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def create_pinned_context_checkout(root: Path) -> tuple[Path, str, str]:
    """Create a minimal Git fixture and return its parser identity."""

    checkout = root / "LigandMPNN"
    checkout.mkdir()
    parser_payload = PINNED_CONTEXT_PARSER_PAYLOAD
    (checkout / "data_utils.py").write_bytes(parser_payload)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "data_utils.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return checkout, commit, hashlib.sha256(parser_payload).hexdigest()


__all__ = ["PINNED_CONTEXT_PARSER_PAYLOAD", "create_pinned_context_checkout", "write_context_inventory"]
