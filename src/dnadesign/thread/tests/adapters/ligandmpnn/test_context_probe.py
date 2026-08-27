"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_context_probe.py

Pinned-upstream LigandMPNN context-inventory contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
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
restype_int_to_str = dict(enumerate("ACDEFGHIKLMNPQRSTVWYX"))


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
    parsed = {
        "Y": _Tensor(xyz),
        "Y_t": _Tensor(types),
        "Y_m": _Tensor([1] * len(types)),
        "R_idx": _Tensor((12, 13, -2)),
        "chain_letters": np.asarray(("A", "A", "B")),
        "S": _Tensor((0, 1, 2)),
        "mask": _Tensor((1, 1, 1)),
    }
    return parsed, None, _Selection(atoms), ("", "B", "A"), None
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


def test_probe_rejects_nonregular_receipt_lock_without_blocking(tmp_path: Path) -> None:
    checkout, commit = _fake_upstream_checkout(tmp_path)
    request = _request(tmp_path, checkout, commit)
    output_path = tmp_path / request.output_path
    output_path.parent.mkdir(parents=True)
    lock_path = output_path.parent / f".{output_path.name}.lock"
    os.mkfifo(lock_path)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        materialize_ligandmpnn_context_inventory(
            request,
            execution_root=tmp_path,
            checkout_root=checkout,
        )

    assert stat.S_ISFIFO(lock_path.lstat().st_mode)
    assert not output_path.exists()


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
        if (
            not failed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and output_path.is_file()
            and output_path.read_bytes() != prior_payload
        ):
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
    original_write_temporary = context_probe_module._write_temporary_receipt
    original_read_quarantined = context_probe_module._read_quarantined_receipt
    published_identity: tuple[int, int] | None = None
    failed = False

    def _capture_published_identity(
        directory_fd: int,
        temporary_name: str,
        payload: bytes,
    ) -> object:
        nonlocal published_identity
        written_receipt = original_write_temporary(directory_fd, temporary_name, payload)
        published_identity = written_receipt.identity
        return written_receipt

    def _simulate_reused_identity(directory_fd: int, name: str) -> tuple[tuple[int, int], bytes]:
        observed_identity, observed_payload = original_read_quarantined(directory_fd, name)
        if name != "publication":
            return observed_identity, observed_payload
        assert observed_identity != published_identity
        assert published_identity is not None
        return published_identity, observed_payload

    def _publish_concurrent_then_fail(file_descriptor: int) -> None:
        nonlocal failed
        try:
            output_status = output_path.stat()
        except FileNotFoundError:
            output_status = None
        if (
            not failed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_identity is not None
            and output_status is not None
            and (output_status.st_dev, output_status.st_ino) == published_identity
        ):
            concurrent_path.write_bytes(concurrent_payload)
            concurrent_path.replace(output_path)
            original_fsync(file_descriptor)
            failed = True
            raise OSError("simulated superseded publication fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _publish_concurrent_then_fail)
    monkeypatch.setattr(context_probe_module, "_write_temporary_receipt", _capture_published_identity)
    monkeypatch.setattr(context_probe_module, "_read_quarantined_receipt", _simulate_reused_identity)

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
    recoveries = list(output_path.parent.glob(f".{output_path.name}.*.recovery"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "prior").read_bytes() == b"prior receipt bytes\n"


@pytest.mark.parametrize("prior_payload", (None, b"prior receipt bytes\n"))
def test_probe_serializes_concurrent_publication_across_prior_snapshot_and_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_payload: bytes | None,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    if prior_payload is not None:
        output_path.write_bytes(prior_payload)
    first_snapshot_read = threading.Event()
    concurrent_completed = threading.Event()
    original_read_prior = context_probe_module._read_prior_receipt
    original_rename_no_replace = context_probe_module._rename_no_replace
    original_fsync = os.fsync
    main_thread = threading.get_ident()
    main_published = False
    failed_main_directory_fsync = False
    snapshot_interleaving_started = False

    def _pause_after_first_snapshot(directory_fd: int, output_name: str) -> bytes | None:
        nonlocal snapshot_interleaving_started
        observed = original_read_prior(directory_fd, output_name)
        if threading.get_ident() == main_thread and not snapshot_interleaving_started:
            snapshot_interleaving_started = True
            first_snapshot_read.set()
            concurrent_completed.wait(timeout=0.2)
        return observed

    def _track_main_publication(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal main_published
        original_rename_no_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if threading.get_ident() == main_thread and str(destination) == output_path.name:
            main_published = True

    def _fail_main_publication_fsync(file_descriptor: int) -> None:
        nonlocal failed_main_directory_fsync
        if (
            threading.get_ident() == main_thread
            and main_published
            and not failed_main_directory_fsync
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
        ):
            failed_main_directory_fsync = True
            raise OSError("simulated first materializer directory fsync failure")
        original_fsync(file_descriptor)

    def _publish_concurrently() -> None:
        assert first_snapshot_read.wait(timeout=5)
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"concurrent durable receipt\n",
        )
        concurrent_completed.set()

    monkeypatch.setattr(context_probe_module, "_read_prior_receipt", _pause_after_first_snapshot)
    monkeypatch.setattr(context_probe_module, "_rename_no_replace", _track_main_publication)
    monkeypatch.setattr(os, "fsync", _fail_main_publication_fsync)
    concurrent = threading.Thread(target=_publish_concurrently)
    concurrent.start()

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"first receipt that cannot be made durable\n",
        )

    concurrent.join(timeout=5)
    assert not concurrent.is_alive()
    assert failed_main_directory_fsync
    assert concurrent_completed.is_set()
    assert output_path.read_bytes() == b"concurrent durable receipt\n"


@pytest.mark.parametrize("prior_payload", (None, b"prior receipt bytes\n"))
def test_probe_never_overwrites_noncooperating_receipt_published_after_prior_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_payload: bytes | None,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    if prior_payload is not None:
        output_path.write_bytes(prior_payload)
    concurrent_payload = b"noncooperating durable receipt\n"
    concurrent_path = output_path.parent / ".noncooperating-receipt.tmp"
    original_read_prior = context_probe_module._read_prior_receipt
    original_fsync = os.fsync
    snapshot_interleaving_started = False

    def _publish_after_snapshot(directory_fd: int, output_name: str) -> bytes | None:
        nonlocal snapshot_interleaving_started
        observed = original_read_prior(directory_fd, output_name)
        if not snapshot_interleaving_started:
            snapshot_interleaving_started = True
            concurrent_path.write_bytes(concurrent_payload)
            os.replace(concurrent_path, output_path)
            original_fsync(directory_fd)
        return observed

    monkeypatch.setattr(context_probe_module, "_read_prior_receipt", _publish_after_snapshot)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="receipt changed before publication",
    ):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt that must not overwrite the concurrent publisher\n",
        )

    assert snapshot_interleaving_started
    assert output_path.read_bytes() == concurrent_payload
    assert not concurrent_path.exists()


@pytest.mark.parametrize("prior_payload", (None, b"prior receipt bytes\n"))
@pytest.mark.parametrize("replacement_kind", ("fifo", "symlink"))
def test_probe_preserves_nonregular_leaf_installed_after_prior_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_payload: bytes | None,
    replacement_kind: str,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    if prior_payload is not None:
        output_path.write_bytes(prior_payload)
    concurrent_path = output_path.parent / ".nonregular-receipt.tmp"
    outside_path = tmp_path / "outside-receipt.json"
    outside_payload = b"outside receipt sentinel\n"
    outside_path.write_bytes(outside_payload)
    original_read_prior = context_probe_module._read_prior_receipt
    replacement_installed = False

    def _install_nonregular_after_snapshot(
        directory_fd: int,
        output_name: str,
    ) -> object:
        nonlocal replacement_installed
        observed = original_read_prior(directory_fd, output_name)
        if replacement_kind == "fifo":
            os.mkfifo(concurrent_path)
        else:
            concurrent_path.symlink_to(outside_path)
        os.rename(concurrent_path, output_path)
        replacement_installed = True
        return observed

    monkeypatch.setattr(context_probe_module, "_read_prior_receipt", _install_nonregular_after_snapshot)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="receipt changed before publication",
    ):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt that must not overwrite the nonregular leaf\n",
        )

    assert replacement_installed
    if replacement_kind == "fifo":
        assert stat.S_ISFIFO(output_path.lstat().st_mode)
    else:
        assert output_path.is_symlink()
        assert output_path.readlink() == outside_path
        assert outside_path.read_bytes() == outside_payload
    assert not list(output_path.parent.glob(f".{output_path.name}.*.recovery"))


def test_probe_preserves_destination_created_after_prior_receipt_is_claimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)
    concurrent_payload = b"receipt published after destination claim\n"
    concurrent_path = output_path.parent / ".post-claim-receipt.tmp"
    original_rename_no_replace = context_probe_module._rename_no_replace
    original_fsync = os.fsync
    collision_installed = False

    def _install_collision_before_publication(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal collision_installed
        if destination == output_path.name and source.endswith(".tmp") and not collision_installed:
            concurrent_path.write_bytes(concurrent_payload)
            os.replace(concurrent_path, output_path)
            original_fsync(dst_dir_fd)
            collision_installed = True
        original_rename_no_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(context_probe_module, "_rename_no_replace", _install_collision_before_publication)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="prior receipt retained",
    ):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt that must lose the no-replace race\n",
        )

    assert collision_installed
    assert output_path.read_bytes() == concurrent_payload
    recoveries = list(output_path.parent.glob(f".{output_path.name}.*.recovery"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "prior").read_bytes() == prior_payload
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_probe_rejects_unavailable_atomic_no_replace_before_receipt_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)

    def _unsupported() -> tuple[object, int]:
        raise OSError(errno.ENOTSUP, "atomic no-replace unavailable")

    monkeypatch.setattr(context_probe_module, "_resolve_rename_no_replace", _unsupported)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt that cannot be safely published\n",
        )

    assert output_path.read_bytes() == prior_payload
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    assert not list(output_path.parent.glob(f".{output_path.name}.*.recovery"))


@pytest.mark.parametrize("error_number", (errno.ENOTSUP, errno.ENOSYS, errno.EINVAL))
@pytest.mark.parametrize("prior_payload", (None, b"prior receipt bytes\n"))
def test_probe_verifies_target_filesystem_no_replace_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    prior_payload: bytes | None,
) -> None:
    relative_output = Path("new-evidence/nested/context-inventory.json")
    output_path = tmp_path / relative_output
    if prior_payload is not None:
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(prior_payload)
    original_rename_no_replace = context_probe_module._rename_no_replace
    operation_attempted = False

    def _unsupported_on_target_filesystem(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal operation_attempted
        if not operation_attempted:
            operation_attempted = True
            raise OSError(error_number, "target filesystem does not support atomic no-replace")
        original_rename_no_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(context_probe_module, "_rename_no_replace", _unsupported_on_target_filesystem)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt that cannot use no-replace on this filesystem\n",
        )

    assert operation_attempted
    if prior_payload is None:
        assert not output_path.parent.exists()
    else:
        assert output_path.read_bytes() == prior_payload
        assert not list(output_path.parent.glob(f".{output_path.name}.*.recovery"))
    assert not list(tmp_path.rglob(".dnadesign-context-noreplace-*"))


def test_probe_restores_claimed_prior_when_publication_no_replace_fails_after_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)
    original_rename_no_replace = context_probe_module._rename_no_replace
    publication_attempted = False

    def _fail_only_publication(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal publication_attempted
        if destination == output_path.name and source.endswith(".tmp"):
            publication_attempted = True
            raise OSError(errno.ENOTSUP, "publication no-replace operation became unsupported")
        original_rename_no_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(context_probe_module, "_rename_no_replace", _fail_only_publication)

    with pytest.raises(ValueError, match="context probe output could not be published atomically"):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt whose final no-replace operation fails\n",
        )

    assert publication_attempted
    assert output_path.read_bytes() == prior_payload
    assert not list(output_path.parent.glob(f".{output_path.name}.*.recovery"))


def test_probe_retains_claimed_prior_when_no_replace_fails_during_publication_and_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    prior_payload = b"prior receipt bytes\n"
    output_path.write_bytes(prior_payload)
    original_rename_no_replace = context_probe_module._rename_no_replace
    filesystem_failure_started = False

    def _fail_publication_and_recovery(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal filesystem_failure_started
        if destination == output_path.name and source.endswith(".tmp"):
            filesystem_failure_started = True
        if filesystem_failure_started:
            raise OSError(errno.ENOTSUP, "target filesystem stopped supporting atomic no-replace")
        original_rename_no_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(context_probe_module, "_rename_no_replace", _fail_publication_and_recovery)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="prior receipt retained",
    ):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            b"receipt whose publication and recovery operations fail\n",
        )

    assert filesystem_failure_started
    assert not output_path.exists()
    recoveries = list(output_path.parent.glob(f".{output_path.name}.*.recovery"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "prior").read_bytes() == prior_payload


@pytest.mark.parametrize("prior_payload", (None, b"prior receipt bytes\n"))
def test_probe_restoration_never_overwrites_replacement_after_ownership_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_payload: bytes | None,
) -> None:
    relative_output = Path("evidence/context-inventory.json")
    output_path = tmp_path / relative_output
    output_path.parent.mkdir(parents=True)
    if prior_payload is not None:
        output_path.write_bytes(prior_payload)
    concurrent_payload = b"replacement after ownership check\n"
    concurrent_path = output_path.parent / ".concurrent-after-check.tmp"
    original_fsync = os.fsync
    original_replace = os.replace
    original_read_quarantined = context_probe_module._read_quarantined_receipt
    publication_fsync_failed = False
    replacement_published = False
    publication_payload = b"publication that cannot be made durable\n"

    def _fail_publication_fsync(file_descriptor: int) -> None:
        nonlocal publication_fsync_failed
        if (
            not publication_fsync_failed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and output_path.is_file()
            and output_path.read_bytes() == publication_payload
        ):
            publication_fsync_failed = True
            raise OSError("simulated publication directory fsync failure")
        original_fsync(file_descriptor)

    def _replace_after_owned_read(directory_fd: int, name: str) -> tuple[tuple[int, int], bytes]:
        nonlocal replacement_published
        observed = original_read_quarantined(directory_fd, name)
        if name == "publication" and publication_fsync_failed and not replacement_published:
            concurrent_path.write_bytes(concurrent_payload)
            original_replace(concurrent_path, output_path)
            parent_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                original_fsync(parent_fd)
            finally:
                os.close(parent_fd)
            replacement_published = True
        return observed

    monkeypatch.setattr(os, "fsync", _fail_publication_fsync)
    monkeypatch.setattr(context_probe_module, "_read_quarantined_receipt", _replace_after_owned_read)

    with pytest.raises(
        LigandMpnnContextPublicationUncertainError,
        match="receipt changed before publication recovery",
    ):
        context_probe_module._publish_context_inventory(
            tmp_path,
            relative_output,
            publication_payload,
        )

    assert publication_fsync_failed
    assert replacement_published
    assert output_path.read_bytes() == concurrent_payload
    assert not concurrent_path.exists()
    recoveries = list(output_path.parent.glob(f".{output_path.name}.*.recovery"))
    if prior_payload is None:
        assert not recoveries
    else:
        assert len(recoveries) == 1
        assert (recoveries[0] / "prior").read_bytes() == prior_payload


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
        if not failed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode) and output_path.is_file():
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
    original_write_temporary = context_probe_module._write_temporary_receipt
    published_identity: tuple[int, int] | None = None
    publication_failure_started = False

    def _capture_published_identity(
        directory_fd: int,
        temporary_name: str,
        payload: bytes,
    ) -> object:
        nonlocal published_identity
        written_receipt = original_write_temporary(directory_fd, temporary_name, payload)
        published_identity = written_receipt.identity
        return written_receipt

    def _fail_directory_fsync(file_descriptor: int) -> None:
        nonlocal publication_failure_started
        try:
            output_status = output_path.stat()
        except FileNotFoundError:
            output_status = None
        is_directory = stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
        if (
            is_directory
            and published_identity is not None
            and output_status is not None
            and (output_status.st_dev, output_status.st_ino) == published_identity
        ):
            publication_failure_started = True
        if is_directory and publication_failure_started:
            raise OSError("simulated persistent directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(context_probe_module, "_write_temporary_receipt", _capture_published_identity)
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
