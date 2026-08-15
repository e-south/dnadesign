"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/context_probe.py

Headless proof of the atom context returned by pinned upstream ``parse_PDB``.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    LigandMpnnContextAtom,
    LigandMpnnContextInventory,
    LigandMpnnContextPolymer,
)
from dnadesign.thread.adapters.ligandmpnn.models import (
    LigandMpnnContextInventoryReference,
    LigandMpnnUpstreamPin,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import attested_working_tree_path_bytes

_DNA_RESIDUE_NAMES = frozenset({"DA", "DC", "DG", "DI", "DT", "DU"})
_RNA_RESIDUE_NAMES = frozenset({"A", "C", "G", "I", "U", "RA", "RC", "RG", "RI", "RU"})


@dataclass(frozen=True)
class LigandMpnnContextProbeRequest:
    """Study-neutral request for one pinned-parser atom inventory."""

    request_id: str
    pdb_path: Path
    pdb_sha256: str
    output_path: Path
    upstream: LigandMpnnUpstreamPin
    minimum_nucleotide_atoms: int = 1
    required_polymer_types: tuple[LigandMpnnContextPolymer, ...] = ()
    chains: tuple[str, ...] = ()
    parse_all_atoms: bool = False
    parse_atoms_with_zero_occupancy: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("context probe request_id must be nonempty")
        _require_relative_file(self.pdb_path, field_name="context probe pdb_path", suffix=".pdb")
        _require_relative_file(self.output_path, field_name="context probe output_path", suffix=".json")
        if not isinstance(self.pdb_sha256, str) or len(self.pdb_sha256) != 64:
            raise ValueError("context probe pdb_sha256 must be a 64-character SHA256 digest")
        try:
            int(self.pdb_sha256, 16)
        except ValueError as error:
            raise ValueError("context probe pdb_sha256 must be a 64-character SHA256 digest") from error
        object.__setattr__(self, "pdb_sha256", self.pdb_sha256.lower())
        if not isinstance(self.upstream, LigandMpnnUpstreamPin):
            raise ValueError("context probe upstream must be a LigandMpnnUpstreamPin")
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
        if not isinstance(self.chains, tuple) or any(not isinstance(chain, str) for chain in self.chains):
            raise ValueError("context probe chains must be a tuple of strings")
        if not isinstance(self.parse_all_atoms, bool) or not isinstance(self.parse_atoms_with_zero_occupancy, bool):
            raise ValueError("context probe parser options must be booleans")


@dataclass(frozen=True)
class LigandMpnnContextProbeCommand:
    """One portable CLI invocation for the pinned-parser probe."""

    output_path: Path
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"output_path": self.output_path.as_posix(), "argv": list(self.argv)}


def build_ligandmpnn_context_probe_command(
    request: LigandMpnnContextProbeRequest,
    *,
    checkout_root: Path,
    python_executable: str = "python",
) -> LigandMpnnContextProbeCommand:
    """Build a deterministic module command; it runs from the execution root."""

    argv = [
        python_executable,
        "-m",
        "dnadesign.thread.adapters.ligandmpnn.context_probe_cli",
        "materialize",
        "--request-id",
        request.request_id,
        "--checkout-root",
        str(checkout_root),
        "--upstream-commit",
        request.upstream.commit,
        "--pdb-path",
        request.pdb_path.as_posix(),
        "--pdb-sha256",
        request.pdb_sha256,
        "--output-path",
        request.output_path.as_posix(),
        "--minimum-nucleotide-atoms",
        str(request.minimum_nucleotide_atoms),
        "--required-polymer-types",
        ",".join(item.value for item in request.required_polymer_types),
        "--chains",
        ",".join(request.chains),
        "--parse-all-atoms",
        _flag(request.parse_all_atoms),
        "--parse-atoms-with-zero-occupancy",
        _flag(request.parse_atoms_with_zero_occupancy),
    ]
    return LigandMpnnContextProbeCommand(output_path=request.output_path, argv=tuple(argv))


def materialize_ligandmpnn_context_inventory(
    request: LigandMpnnContextProbeRequest,
    *,
    execution_root: Path,
    checkout_root: Path,
) -> LigandMpnnContextInventoryReference:
    """Run upstream ``parse_PDB`` and persist its effective context inventory."""

    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("execution_root must be an existing directory")
    input_path = _within_root(root, request.pdb_path, field_name="context probe pdb_path")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("context probe input must be an existing regular file, not a symlink")
    observed_input_sha256 = _sha256_file(input_path)
    if observed_input_sha256 != request.pdb_sha256:
        raise ValueError(
            f"context probe input SHA256 mismatch: expected {request.pdb_sha256}, observed {observed_input_sha256}"
        )
    parser, element_dict_rev, parser_sha256 = _load_pinned_upstream_parser(
        checkout_root, expected_commit=request.upstream.commit
    )
    parsed, _, other_atoms, _, _ = parser(
        str(input_path),
        device="cpu",
        chains=list(request.chains),
        parse_all_atoms=request.parse_all_atoms,
        parse_atoms_with_zero_occupancy=request.parse_atoms_with_zero_occupancy,
    )
    atoms = _effective_context_atoms(parsed, other_atoms, element_dict_rev=element_dict_rev)
    inventory = LigandMpnnContextInventory(
        request_id=request.request_id,
        request_sha256=_probe_request_sha256(request),
        input_path=request.pdb_path,
        input_sha256=observed_input_sha256,
        upstream_commit=request.upstream.commit,
        parser_path=Path("data_utils.py"),
        parser_sha256=parser_sha256,
        parser_callable="parse_PDB",
        chains=request.chains,
        parse_all_atoms=request.parse_all_atoms,
        parse_atoms_with_zero_occupancy=request.parse_atoms_with_zero_occupancy,
        minimum_nucleotide_atoms=request.minimum_nucleotide_atoms,
        required_polymer_types=request.required_polymer_types,
        atoms=atoms,
    )
    output_path = _within_root(root, request.output_path, field_name="context probe output_path")
    if output_path.is_symlink():
        raise ValueError("context probe output must not be a symlink")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.write_bytes(payload)
    return LigandMpnnContextInventoryReference(path=request.output_path, sha256=hashlib.sha256(payload).hexdigest())


def _load_pinned_upstream_parser(
    checkout_root: Path,
    *,
    expected_commit: str,
) -> tuple[Callable[..., tuple[Any, ...]], dict[int, str], str]:
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    head = _git(checkout, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {expected_commit}, observed {head}")
    tracked = _git(checkout, "ls-files", "--error-unmatch", "data_utils.py")
    if tracked != "data_utils.py":
        raise ValueError("pinned LigandMPNN checkout does not track data_utils.py")
    source_bytes = attested_working_tree_path_bytes(checkout, expected_commit, "data_utils.py")
    if source_bytes is None:
        raise ValueError("data_utils.py must be clean at the pinned commit")
    source_path = checkout / "data_utils.py"
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("pinned LigandMPNN data_utils.py must be a regular file")
    module = _import_upstream_module(source_bytes, source_path=source_path, checkout=checkout)
    parser = getattr(module, "parse_PDB", None)
    if not callable(parser):
        raise ValueError("pinned LigandMPNN data_utils.py does not expose parse_PDB")
    element_dict_rev = getattr(module, "element_dict_rev", None)
    if not isinstance(element_dict_rev, dict) or any(
        not isinstance(key, int) or not isinstance(value, str) for key, value in element_dict_rev.items()
    ):
        raise ValueError("pinned LigandMPNN data_utils.py does not expose element_dict_rev")
    return parser, element_dict_rev, hashlib.sha256(source_bytes).hexdigest()


def _import_upstream_module(source_bytes: bytes, *, source_path: Path, checkout: Path) -> ModuleType:
    module_name = f"_dnadesign_ligandmpnn_data_utils_{hashlib.sha256(source_bytes).hexdigest()[:12]}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.path.insert(0, str(checkout))
    try:
        exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)
    except Exception as error:
        raise ValueError(f"could not import pinned LigandMPNN data_utils.py: {error}") from error
    finally:
        sys.path.pop(0)
    return module


def _effective_context_atoms(
    parsed: Any,
    other_atoms: Any,
    *,
    element_dict_rev: dict[int, str],
) -> tuple[LigandMpnnContextAtom, ...]:
    if not isinstance(parsed, dict) or not {"Y", "Y_t", "Y_m"}.issubset(parsed):
        raise ValueError("upstream parse_PDB did not return Y, Y_t, and Y_m")
    coordinates = _to_numpy(parsed["Y"])
    element_types = _to_numpy(parsed["Y_t"]).reshape(-1)
    masks = _to_numpy(parsed["Y_m"]).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("upstream Y context coordinates must have shape [atoms, 3]")
    if coordinates.shape[0] != element_types.shape[0] or element_types.shape[0] != masks.shape[0]:
        raise ValueError("upstream Y, Y_t, and Y_m context lengths differ")
    effective_rows = [index for index, mask in enumerate(masks) if int(mask) != 0]
    raw_atoms = list(other_atoms.iterAtoms()) if other_atoms is not None else []
    observed: list[LigandMpnnContextAtom] = []
    cursor = 0
    for row_index in effective_rows:
        element_type = int(element_types[row_index])
        element = element_dict_rev.get(element_type)
        if not isinstance(element, str) or not element:
            raise ValueError(f"upstream emitted unknown effective element type {element_type}")
        element = element.upper()
        atom, cursor = _match_upstream_atom(
            raw_atoms,
            start=cursor,
            coordinates=coordinates[row_index],
            element=element,
        )
        residue_name = str(atom.getResname()).strip().upper()
        observed.append(
            LigandMpnnContextAtom(
                serial=int(atom.getSerial()),
                atom_name=str(atom.getName()).strip(),
                element=element,
                upstream_element_type=element_type,
                chain_id=str(atom.getChid()).strip(),
                residue_name=residue_name,
                residue_number=int(atom.getResnum()),
                insertion_code=str(atom.getIcode() or "").strip(),
                polymer_type=_polymer_type(residue_name),
            )
        )
    if len(observed) != int(np.asarray(masks).astype(bool).sum()):
        raise ValueError("effective atom inventory does not equal upstream Y_m count")
    return tuple(observed)


def _match_upstream_atom(
    atoms: list[Any],
    *,
    start: int,
    coordinates: np.ndarray,
    element: str,
) -> tuple[Any, int]:
    for index in range(start, len(atoms)):
        atom = atoms[index]
        atom_element = str(atom.getElement() or "").strip().upper()
        atom_coordinates = np.asarray(atom.getCoords(), dtype=np.float32)
        if atom_element == element and np.allclose(atom_coordinates, coordinates, rtol=0.0, atol=1e-6):
            return atom, index + 1
    raise ValueError("could not map an effective upstream Y/Y_t/Y_m atom back to its parsed residue identity")


def _polymer_type(residue_name: str) -> LigandMpnnContextPolymer:
    if residue_name in _DNA_RESIDUE_NAMES:
        return LigandMpnnContextPolymer.DNA
    if residue_name in _RNA_RESIDUE_NAMES:
        return LigandMpnnContextPolymer.RNA
    return LigandMpnnContextPolymer.OTHER


def _probe_request_sha256(request: LigandMpnnContextProbeRequest) -> str:
    payload = {
        "schema_id": "thread.ligandmpnn.context_probe_request",
        "schema_version": 1,
        "request_id": request.request_id,
        "pdb_path": request.pdb_path.as_posix(),
        "pdb_sha256": f"sha256:{request.pdb_sha256}",
        "upstream_commit": request.upstream.commit,
        "parser": {
            "path": "data_utils.py",
            "callable": "parse_PDB",
            "chains": list(request.chains),
            "parse_all_atoms": request.parse_all_atoms,
            "parse_atoms_with_zero_occupancy": request.parse_atoms_with_zero_occupancy,
        },
        "minimum_nucleotide_atoms": request.minimum_nucleotide_atoms,
        "required_polymer_types": [item.value for item in request.required_polymer_types],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ValueError(f"could not verify pinned LigandMPNN checkout: {message}")
    return result.stdout.strip()


def _within_root(root: Path, path: Path, *, field_name: str) -> Path:
    _require_relative_file(path, field_name=field_name)
    candidate = root / path
    parent = candidate.parent.resolve()
    if parent != root and root not in parent.parents:
        raise ValueError(f"{field_name} escapes execution_root")
    return parent / candidate.name


def _require_relative_file(path: Path, *, field_name: str, suffix: str | None = None) -> None:
    if not isinstance(path, Path) or path.is_absolute() or not path.name or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe relative file path")
    if suffix is not None and path.suffix.lower() != suffix:
        raise ValueError(f"{field_name} must end in {suffix}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _parse_bool_flag(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return value == "1"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize pinned LigandMPNN parser context evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--request-id", required=True)
    materialize.add_argument("--checkout-root", type=Path, required=True)
    materialize.add_argument("--upstream-commit", required=True)
    materialize.add_argument("--pdb-path", type=Path, required=True)
    materialize.add_argument("--pdb-sha256", required=True)
    materialize.add_argument("--output-path", type=Path, required=True)
    materialize.add_argument("--minimum-nucleotide-atoms", type=int, required=True)
    materialize.add_argument("--required-polymer-types", default="")
    materialize.add_argument("--chains", default="")
    materialize.add_argument("--parse-all-atoms", type=_parse_bool_flag, required=True)
    materialize.add_argument("--parse-atoms-with-zero-occupancy", type=_parse_bool_flag, required=True)
    args = parser.parse_args(argv)
    required = tuple(LigandMpnnContextPolymer(item) for item in args.required_polymer_types.split(",") if item)
    request = LigandMpnnContextProbeRequest(
        request_id=args.request_id,
        pdb_path=args.pdb_path,
        pdb_sha256=args.pdb_sha256,
        output_path=args.output_path,
        upstream=LigandMpnnUpstreamPin(commit=args.upstream_commit, checkpoint_sha256="0" * 64),
        minimum_nucleotide_atoms=args.minimum_nucleotide_atoms,
        required_polymer_types=required,
        chains=tuple(item for item in args.chains.split(",") if item),
        parse_all_atoms=args.parse_all_atoms,
        parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
    )
    reference = materialize_ligandmpnn_context_inventory(
        request,
        execution_root=Path.cwd(),
        checkout_root=args.checkout_root,
    )
    print(json.dumps(reference.to_dict(), sort_keys=True))
    return 0
