"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_context_probe.py

Pinned-upstream LigandMPNN context-inventory contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

import dnadesign.thread.adapters.ligandmpnn.context_probe as context_probe_module
from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnContextPolymer,
    LigandMpnnContextProbeRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_context_probe_command,
    load_ligandmpnn_context_inventory,
    materialize_ligandmpnn_context_inventory,
)

_DIGEST = "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fake_upstream_checkout(root: Path) -> tuple[Path, str]:
    checkout = root / "LigandMPNN"
    checkout.mkdir()
    source = """
import numpy as np

element_dict_rev = {6: "C", 7: "N", 8: "O", 15: "P", 30: "ZN"}


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Atom:
    def __init__(self, serial, name, element, chain, resname, resnum, icode, xyz):
        self.values = serial, name, element, chain, resname, resnum, icode, xyz

    def getSerial(self): return self.values[0]
    def getName(self): return self.values[1]
    def getElement(self): return self.values[2]
    def getChid(self): return self.values[3]
    def getResname(self): return self.values[4]
    def getResnum(self): return self.values[5]
    def getIcode(self): return self.values[6]
    def getCoords(self): return np.asarray(self.values[7], dtype=np.float32)


class _Selection:
    def __init__(self, atoms): self._atoms = atoms
    def iterAtoms(self): return iter(self._atoms)


def parse_PDB(input_path, device="cpu", chains=[], parse_all_atoms=False,
              parse_atoms_with_zero_occupancy=False):
    del device, chains, parse_all_atoms, parse_atoms_with_zero_occupancy
    if "protein_only" in input_path:
        atoms = [_Atom(1, "ZN", "ZN", "Z", "ZN", 1, "", (1, 1, 1))]
        types = [30]
    else:
        atoms = [
            _Atom(1, "P", "P", "D", "DC", 12, "", (1, 0, 0)),
            _Atom(2, "C1'", "C", "D", "DC", 12, "", (2, 0, 0)),
            _Atom(3, "P", "P", "E", "G", 66, "", (3, 0, 0)),
            _Atom(4, "N9", "N", "E", "G", 66, "", (4, 0, 0)),
            _Atom(5, "H1", "H", "E", "G", 66, "", (5, 0, 0)),
            _Atom(6, "ZN", "ZN", "Z", "ZN", 1, "", (6, 0, 0)),
        ]
        types = [15, 6, 15, 7, 30]
    retained = [atom for atom in atoms if atom.getElement() != "H"]
    xyz = [atom.getCoords() for atom in retained]
    parsed = {"Y": _Tensor(xyz), "Y_t": _Tensor(types), "Y_m": _Tensor([1] * len(types))}
    return parsed, None, _Selection(atoms), [], None
""".lstrip()
    (checkout / "data_utils.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "data_utils.py"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Thread Test",
            "-c",
            "user.email=thread-test@example.invalid",
            "commit",
            "-qm",
            "fake pinned parser",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    return checkout, commit


def _request(
    root: Path,
    checkout: Path,
    commit: str,
    *,
    pdb_name: str = "target.pdb",
) -> LigandMpnnContextProbeRequest:
    del checkout
    pdb_payload = b"ATOM pinned probe fixture\n"
    pdb_path = root / "inputs" / pdb_name
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_bytes(pdb_payload)
    return LigandMpnnContextProbeRequest(
        request_id="generic_nucleotide_context",
        pdb_path=Path("inputs") / pdb_name,
        pdb_sha256=_sha256(pdb_payload),
        output_path=Path("evidence/context-inventory.json"),
        upstream=LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_DIGEST),
        minimum_nucleotide_atoms=1,
        required_polymer_types=(LigandMpnnContextPolymer.DNA, LigandMpnnContextPolymer.RNA),
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=False,
    )


def test_probe_records_the_effective_upstream_y_context_and_nucleotide_identities(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)

    reference = materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout,
    )
    inventory = load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)

    assert reference.path == Path("evidence/context-inventory.json")
    assert inventory.upstream_commit == commit
    assert inventory.parser_path == Path("data_utils.py")
    assert inventory.effective_nonprotein_atom_count == 5
    assert inventory.effective_nucleotide_atom_count == 4
    assert inventory.polymer_atom_counts == {"dna": 2, "rna": 2, "other": 1}
    assert inventory.element_counts == {"C": 1, "N": 1, "P": 2, "ZN": 1}
    assert {(atom.chain_id, atom.residue_name, atom.element) for atom in inventory.atoms} == {
        ("D", "DC", "P"),
        ("D", "DC", "C"),
        ("E", "G", "P"),
        ("E", "G", "N"),
        ("Z", "ZN", "ZN"),
    }
    payload = json.loads((tmp_path / reference.path).read_text(encoding="utf-8"))
    assert payload["parser"]["callable"] == "parse_PDB"
    assert payload["observed"]["effective_nonprotein_atom_count"] == 5
    assert payload["observed"]["effective_nucleotide_atom_count"] == 4


def test_probe_fails_before_writing_when_expected_nucleotide_context_is_absent(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit, pdb_name="protein_only.pdb")

    with pytest.raises(ValueError, match="expected at least 1 effective DNA/RNA context atoms"):
        materialize_ligandmpnn_context_inventory(request, execution_root=tmp_path, checkout_root=checkout)

    assert not (tmp_path / request.output_path).exists()


def test_probe_rejects_modified_parser_and_inventory_digest_drift(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    (checkout / "data_utils.py").write_text("# modified\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data_utils.py must be clean at the pinned commit"):
        materialize_ligandmpnn_context_inventory(request, execution_root=tmp_path, checkout_root=checkout)

    subprocess.run(["git", "restore", "data_utils.py"], cwd=checkout, check=True)
    reference = materialize_ligandmpnn_context_inventory(request, execution_root=tmp_path, checkout_root=checkout)
    (tmp_path / reference.path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="context inventory SHA256 mismatch"):
        load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)


def test_probe_rejects_assume_unchanged_modified_parser(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "data_utils.py"],
        cwd=checkout,
        check=True,
    )
    parser_path = checkout / "data_utils.py"
    parser_path.write_text(parser_path.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "data_utils.py"],
        cwd=checkout,
        text=True,
    )
    assert status == ""
    with pytest.raises(ValueError, match="data_utils.py must be clean at the pinned commit"):
        materialize_ligandmpnn_context_inventory(request, execution_root=tmp_path, checkout_root=checkout)

    assert not (tmp_path / request.output_path).exists()


def test_probe_executes_attested_source_instead_of_valid_cached_bytecode(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    parser_path = checkout / "data_utils.py"
    clean_source = parser_path.read_text(encoding="utf-8")
    cached_source = clean_source.replace(
        "types = [15, 6, 15, 7, 30]",
        "types = [30, 6, 15, 7, 30]",
    )
    assert len(cached_source) == len(clean_source)
    timestamp = 1_700_000_000
    parser_path.write_text(cached_source, encoding="utf-8")
    os.utime(parser_path, (timestamp, timestamp))
    py_compile.compile(str(parser_path), doraise=True)
    parser_path.write_text(clean_source, encoding="utf-8")
    os.utime(parser_path, (timestamp, timestamp))

    reference = materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout,
    )
    inventory = load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)

    assert inventory.effective_nucleotide_atom_count == 4


def test_probe_does_not_import_helpers_from_mutable_checkout(tmp_path: Path) -> None:
    checkout, _ = _fake_upstream_checkout(tmp_path)
    parser_path = checkout / "data_utils.py"
    parser_path.write_text(
        "from _ligandmpnn_untracked_probe_helper import VALUE\n" + parser_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(checkout), "add", "data_utils.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Thread Test",
            "-c",
            "user.email=thread-test@example.invalid",
            "commit",
            "-qm",
            "parser with unavailable dependency",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    (checkout / "_ligandmpnn_untracked_probe_helper.py").write_text(
        "VALUE = 'mutable-checkout-import'\n", encoding="utf-8"
    )
    request = _request(tmp_path, checkout, commit)

    with pytest.raises(ValueError, match="could not import pinned LigandMPNN data_utils.py"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert not (tmp_path / request.output_path).exists()


def test_probe_publishes_receipt_without_following_a_raced_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path.resolve() / request.output_path
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("sentinel", encoding="utf-8")
    path_type = type(output_path)
    original_write_bytes = path_type.write_bytes

    def _race_output_write(path: Path, payload: bytes) -> int:
        if path == output_path and not path.exists() and not path.is_symlink():
            path.symlink_to(outside_path)
        return original_write_bytes(path, payload)

    monkeypatch.setattr(path_type, "write_bytes", _race_output_write)

    materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout,
    )

    assert not output_path.is_symlink()
    assert outside_path.read_text(encoding="utf-8") == "sentinel"


def test_probe_command_is_explicit_and_portable(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)

    command = build_ligandmpnn_context_probe_command(
        request,
        checkout_root=Path("/opt/LigandMPNN"),
        python_executable="python3",
    )

    assert command.output_path == request.output_path
    assert command.argv[:4] == (
        "python3",
        "-m",
        "dnadesign.thread.adapters.ligandmpnn.context_probe_cli",
        "materialize",
    )
    assert command.argv[command.argv.index("--upstream-commit") + 1] == commit
    assert command.argv[command.argv.index("--minimum-nucleotide-atoms") + 1] == "1"
    assert command.argv[command.argv.index("--required-polymer-types") + 1] == "dna,rna"


def test_probe_command_preserves_blank_chain_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    request = LigandMpnnContextProbeRequest(
        request_id=request.request_id,
        pdb_path=request.pdb_path,
        pdb_sha256=request.pdb_sha256,
        output_path=request.output_path,
        upstream=request.upstream,
        minimum_nucleotide_atoms=request.minimum_nucleotide_atoms,
        required_polymer_types=request.required_polymer_types,
        chains=("",),
        parse_all_atoms=request.parse_all_atoms,
        parse_atoms_with_zero_occupancy=request.parse_atoms_with_zero_occupancy,
    )

    command = build_ligandmpnn_context_probe_command(request, checkout_root=checkout)
    observed: list[LigandMpnnContextProbeRequest] = []

    def _capture(
        parsed_request: LigandMpnnContextProbeRequest,
        **_kwargs: object,
    ) -> LigandMpnnContextInventoryReference:
        observed.append(parsed_request)
        return LigandMpnnContextInventoryReference(path=parsed_request.output_path, sha256=_DIGEST)

    monkeypatch.setattr(context_probe_module, "materialize_ligandmpnn_context_inventory", _capture)

    assert command.argv[command.argv.index("--chain") + 1] == ""
    assert context_probe_module._main(list(command.argv[3:])) == 0
    assert observed[0].chains == ("",)


def test_reference_rejects_unsafe_paths_and_non_digests() -> None:
    with pytest.raises(ValueError, match="context inventory path"):
        LigandMpnnContextInventoryReference(path=Path("../inventory.json"), sha256=_DIGEST)
    with pytest.raises(ValueError, match="context inventory SHA256"):
        LigandMpnnContextInventoryReference(path=Path("inventory.json"), sha256="not-a-digest")


def test_probe_command_executes_headlessly_and_emits_a_reference(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    command = build_ligandmpnn_context_probe_command(
        request,
        checkout_root=checkout,
        python_executable=sys.executable,
    )

    completed = subprocess.run(command.argv, cwd=tmp_path, check=True, capture_output=True, text=True)

    emitted = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert emitted["path"] == request.output_path.as_posix()
    reference = LigandMpnnContextInventoryReference(
        path=Path(emitted["path"]),
        sha256=emitted["sha256"].removeprefix("sha256:"),
    )
    inventory = load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)
    assert inventory.effective_nucleotide_atom_count == 4


def test_inventory_parser_rejects_nested_schema_drift(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    reference = materialize_ligandmpnn_context_inventory(request, execution_root=tmp_path, checkout_root=checkout)
    path = tmp_path / reference.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parser"]["undocumented_flag"] = True
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="context inventory parser keys mismatch"):
        load_ligandmpnn_context_inventory(
            LigandMpnnContextInventoryReference(path=reference.path, sha256=_sha256(encoded)),
            execution_root=tmp_path,
        )
