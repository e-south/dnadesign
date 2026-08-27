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
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import dnadesign.thread.adapters.ligandmpnn.context_probe as context_probe_module
from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnContextPolymer,
    LigandMpnnContextProbeRequest,
    LigandMpnnContextPublicationUncertainError,
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


def test_public_materializer_anchors_relative_checkout_to_execution_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    reference = materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout.relative_to(tmp_path),
    )

    inventory = load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)
    assert inventory.upstream_commit == commit
    assert (tmp_path / reference.path).is_file()


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


def test_probe_fails_hard_when_working_parser_bytes_are_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    parser_path = checkout / "data_utils.py"
    original_read_bytes = Path.read_bytes

    def _reject_parser(path: Path) -> bytes:
        if path == parser_path:
            raise PermissionError("parser is unreadable during execution")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _reject_parser)

    with pytest.raises(ValueError, match="data_utils.py must be clean at the pinned commit"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert not (tmp_path / request.output_path).exists()


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


def test_probe_rejects_staged_parser_when_worktree_bytes_match_pin(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    parser_path = checkout / "data_utils.py"
    pinned_payload = parser_path.read_bytes()
    parser_path.write_text("def parse_PDB(): return 'staged-modification'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "data_utils.py"], check=True)
    parser_path.write_bytes(pinned_payload)

    with pytest.raises(ValueError, match="data_utils.py Git index does not match the pinned commit"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

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


def test_probe_rejects_ancestor_symlink_race_without_publishing_outside_execution_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = replace(
        _request(tmp_path, checkout, commit),
        output_path=Path("evidence/context/context-inventory.json"),
    )
    output_ancestor = tmp_path / "evidence"
    output_parent = tmp_path / request.output_path.parent
    output_parent.mkdir(parents=True)
    displaced_parent = tmp_path / "displaced-evidence"
    outside_parent = tmp_path / "outside"
    (outside_parent / "context").mkdir(parents=True)
    open_started = threading.Event()
    race_finished = threading.Event()
    original_open = os.open

    def _race_ancestor() -> None:
        assert open_started.wait(timeout=5)
        os.rename(output_ancestor, displaced_parent)
        os.symlink(outside_parent, output_ancestor, target_is_directory=True)
        race_finished.set()

    racer = threading.Thread(target=_race_ancestor, daemon=True)
    racer.start()

    def _synchronized_open(path: os.PathLike[str] | str, flags: int, *args: object, **kwargs: object) -> int:
        candidate = Path(path)
        if not open_started.is_set() and candidate.name in {output_ancestor.name, output_parent.name}:
            open_started.set()
            assert race_finished.wait(timeout=5)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _synchronized_open)

    with pytest.raises(ValueError, match="context probe output directory could not be opened safely"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    racer.join(timeout=5)
    assert not racer.is_alive()
    assert not (outside_parent / "context" / request.output_path.name).exists()


def test_probe_rejects_and_preserves_existing_symlink_receipt(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    outside_path = tmp_path / "outside-receipt.json"
    outside_payload = b"outside sentinel\n"
    outside_path.write_bytes(outside_payload)
    output_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="existing regular file"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert output_path.is_symlink()
    assert output_path.readlink() == outside_path
    assert outside_path.read_bytes() == outside_payload


def test_probe_rejects_and_preserves_existing_fifo_receipt(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    os.mkfifo(output_path)

    with pytest.raises(ValueError, match="existing regular file"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert stat.S_ISFIFO(output_path.lstat().st_mode)


def test_probe_restores_existing_receipt_when_post_replace_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)
    original_fsync = os.fsync
    failed = False

    def _fail_post_replace_directory_fsync(file_descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            assert output_path.read_bytes() != prior_payload
            failed = True
            raise OSError("simulated directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _fail_post_replace_directory_fsync)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert failed
    assert output_path.read_bytes() == prior_payload
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_probe_preserves_concurrent_receipt_when_its_post_replace_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"prior receipt bytes\n")
    concurrent_payload = b"concurrent successful receipt\n"
    concurrent_path = output_path.parent / ".concurrent-receipt.tmp"
    original_fsync = os.fsync
    failed = False

    def _publish_concurrent_then_fail(file_descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            concurrent_path.write_bytes(concurrent_payload)
            concurrent_path.replace(output_path)
            original_fsync(file_descriptor)
            failed = True
            raise OSError("simulated superseded publication fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _publish_concurrent_then_fail)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="receipt changed before publication recovery",
    ):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert failed
    assert output_path.read_bytes() == concurrent_payload
    assert not concurrent_path.exists()


def test_probe_syncs_each_parent_that_receives_a_new_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = replace(
        _request(tmp_path, checkout, commit),
        output_path=Path("new-evidence/nested/context-inventory.json"),
    )
    original_mkdir = os.mkdir
    original_fsync = os.fsync
    created_in_parent: list[tuple[int, int]] = []
    fsynced_directories: list[tuple[int, int]] = []

    def _record_mkdir(
        path: os.PathLike[str] | str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if dir_fd is None:
            original_mkdir(path, mode=mode)
        else:
            parent_status = os.fstat(dir_fd)
            created_in_parent.append((parent_status.st_dev, parent_status.st_ino))
            original_mkdir(path, mode=mode, dir_fd=dir_fd)

    def _record_fsync(file_descriptor: int) -> None:
        status = os.fstat(file_descriptor)
        if stat.S_ISDIR(status.st_mode):
            fsynced_directories.append((status.st_dev, status.st_ino))
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "mkdir", _record_mkdir)
    monkeypatch.setattr(os, "fsync", _record_fsync)

    materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout,
    )

    assert created_in_parent
    assert set(created_in_parent) <= set(fsynced_directories)


def test_probe_removes_new_receipt_when_post_replace_directory_fsync_fails_without_prior_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    original_fsync = os.fsync
    failed = False

    def _fail_post_replace_directory_fsync(file_descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            assert output_path.is_file()
            failed = True
            raise OSError("simulated directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _fail_post_replace_directory_fsync)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert failed
    assert not output_path.exists()
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_probe_reports_typed_uncertainty_when_restoration_directory_fsync_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)
    original_fsync = os.fsync

    def _fail_directory_fsync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError("simulated persistent directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _fail_directory_fsync)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="restoration could not be made durable",
    ):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert output_path.read_bytes() == prior_payload


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


def test_probe_command_preserves_option_looking_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = replace(
        _request(tmp_path, checkout, commit),
        request_id="-control",
        pdb_path=Path("-input.pdb"),
        output_path=Path("-inventory.json"),
        chains=("-",),
    )
    command = build_ligandmpnn_context_probe_command(
        request,
        checkout_root=Path("-checkout"),
    )
    observed: list[tuple[LigandMpnnContextProbeRequest, Path]] = []

    def _capture(
        parsed_request: LigandMpnnContextProbeRequest,
        *,
        checkout_root: Path,
        **_kwargs: object,
    ) -> LigandMpnnContextInventoryReference:
        observed.append((parsed_request, checkout_root))
        return LigandMpnnContextInventoryReference(path=parsed_request.output_path, sha256=_DIGEST)

    monkeypatch.setattr(context_probe_module, "materialize_ligandmpnn_context_inventory", _capture)

    assert context_probe_module._main(list(command.argv[3:])) == 0
    parsed_request, parsed_checkout = observed[0]
    assert parsed_request.request_id == request.request_id
    assert parsed_request.pdb_path == request.pdb_path
    assert parsed_request.output_path == request.output_path
    assert parsed_request.chains == request.chains
    assert parsed_checkout == Path("-checkout")


@pytest.mark.parametrize(
    ("field_name", "path"),
    [
        ("pdb_path", Path("~/target.pdb")),
        ("output_path", Path("~/context-inventory.json")),
    ],
)
def test_context_probe_request_rejects_tilde_prefixed_paths(
    tmp_path: Path,
    field_name: str,
    path: Path,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)

    with pytest.raises(ValueError, match="safe relative file path"):
        replace(request, **{field_name: path})


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


def test_inventory_loader_rejects_ancestor_symlink_swap_before_leaf_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    reference = materialize_ligandmpnn_context_inventory(
        request,
        execution_root=tmp_path,
        checkout_root=checkout,
    )
    inventory_path = tmp_path / reference.path
    safe_parent = inventory_path.parent
    displaced_parent = tmp_path / "displaced-evidence"
    outside_parent = tmp_path / "outside-evidence"
    outside_parent.mkdir()
    (outside_parent / inventory_path.name).write_bytes(inventory_path.read_bytes())
    original_open = os.open
    swapped = False

    def _swap_before_ancestor_open(
        path: os.PathLike[str] | str,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        if str(path) == safe_parent.name and not swapped:
            safe_parent.rename(displaced_parent)
            safe_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _swap_before_ancestor_open)

    with pytest.raises(ValueError, match="context inventory could not be opened safely"):
        load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)

    assert swapped


def test_inventory_loader_rejects_nonregular_leaf_without_blocking(tmp_path: Path) -> None:
    inventory_path = tmp_path / "evidence/context-inventory.json"
    inventory_path.parent.mkdir(parents=True)
    os.mkfifo(inventory_path)
    reference = LigandMpnnContextInventoryReference(path=Path("evidence/context-inventory.json"), sha256=_DIGEST)

    with pytest.raises(ValueError, match="context inventory must be a regular file"):
        load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)


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
