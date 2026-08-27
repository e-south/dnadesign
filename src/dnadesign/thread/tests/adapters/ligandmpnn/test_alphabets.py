"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_alphabets.py

Residue-specific LigandMPNN alphabet contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnResidueAlphabet,
    LigandMpnnResidueAlphabetSidecar,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_commands,
    build_planned_receipt,
    materialize_residue_alphabet_sidecar,
)
from dnadesign.thread.adapters.ligandmpnn import alphabets as alphabets_module
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import (
    create_pinned_context_checkout,
    write_context_inventory,
)

_DIGEST = "a" * 64
_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"  # pragma: allowlist secret


def _request(*alphabets: LigandMpnnResidueAlphabet) -> LigandMpnnRequest:
    return LigandMpnnRequest(
        request_id="generic_restricted_design",
        pdb_path=Path("inputs/target.pdb"),
        pdb_sha256=_DIGEST,
        output_dir=Path("outputs/designs"),
        upstream=LigandMpnnUpstreamPin(commit=_COMMIT, checkpoint_sha256=_DIGEST),
        context_inventory=LigandMpnnContextInventoryReference(
            path=Path("evidence/context-inventory.json"), sha256=_DIGEST
        ),
        redesigned_residues=(LigandMpnnResidue("B", 2), LigandMpnnResidue("A", 12)),
        residue_alphabets=tuple(alphabets),
    )


def test_alphabet_rejects_noncanonical_empty_or_duplicate_values() -> None:
    residue = LigandMpnnResidue("A", 12)
    with pytest.raises(ValueError, match="must not be empty"):
        LigandMpnnResidueAlphabet(residue, ())
    with pytest.raises(ValueError, match="canonical 20 amino acids"):
        LigandMpnnResidueAlphabet(residue, ("A", "X"))
    with pytest.raises(ValueError, match="duplicate amino acid A"):
        LigandMpnnResidueAlphabet(residue, ("A", "A"))


def test_request_rejects_duplicate_or_nonredesigned_alphabet_residues() -> None:
    a12 = LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G"))
    with pytest.raises(ValueError, match="duplicate residue A12"):
        _request(a12, a12)
    with pytest.raises(ValueError, match="must be redesigned"):
        _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("C", 4), ("A",)))


def test_sidecar_materializes_official_omit_json_deterministically(tmp_path: Path) -> None:
    request = _request(
        LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("W", "A")),
        LigandMpnnResidueAlphabet(LigandMpnnResidue("B", 2), ("G",)),
    )
    path = tmp_path / "inputs" / "omit.json"

    sidecar = materialize_residue_alphabet_sidecar(request, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "A12": "CDEFGHIKLMNPQRSTVYX",
        "B2": "ACDEFHIKLMNPQRSTVWYX",
    }
    assert sidecar.to_dict() == {
        "schema_id": "thread.ligandmpnn.residue_alphabet_sidecar",
        "schema_version": 1,
        "request_id": "generic_restricted_design",
        "path": str(path),
        "sha256": sidecar.sha256,
        "residue_count": 2,
    }
    assert sidecar.sha256.startswith("sha256:")
    assert materialize_residue_alphabet_sidecar(request, path) == sidecar


def test_sidecar_materialization_rejects_dangling_symlink_target(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    outside = tmp_path / "outside/omit.json"
    outside.parent.mkdir()
    target = tmp_path / "omit.json"
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="sidecar target must be a regular file"):
        materialize_residue_alphabet_sidecar(request, target)

    assert target.is_symlink()
    assert not outside.exists()


def test_sidecar_materialization_rejects_nonregular_existing_target(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    target.mkdir()

    with pytest.raises(ValueError, match="sidecar target must be a regular file"):
        materialize_residue_alphabet_sidecar(request, target)


def test_sidecar_materialization_rejects_fifo_target_without_blocking(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    os.mkfifo(target)

    with pytest.raises(ValueError, match="sidecar target must be a regular file"):
        materialize_residue_alphabet_sidecar(request, target)


def test_sidecar_materialization_rejects_socket_target_without_blocking() -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    with tempfile.TemporaryDirectory(prefix="lm-", dir="/private/tmp") as directory:
        target = Path(directory) / "omit.json"
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(target))

            with pytest.raises(ValueError, match="sidecar target must be a regular file"):
                materialize_residue_alphabet_sidecar(request, target)


def test_sidecar_materialization_rejects_symlinked_ancestor_without_writing_outside(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="sidecar target directory could not be opened safely"):
        materialize_residue_alphabet_sidecar(request, alias / "omit.json")

    assert not (outside / "omit.json").exists()


def test_sidecar_validation_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    real_parent = tmp_path / "real"
    target = real_parent / "omit.json"
    sidecar = materialize_residue_alphabet_sidecar(request, target)
    moved_parent = tmp_path / "moved"
    real_parent.replace(moved_parent)
    real_parent.symlink_to(moved_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="residue alphabet sidecar directory could not be opened safely"):
        sidecar.validate_for(request)


def test_sidecar_partial_private_write_never_exposes_public_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    real_write = os.write
    calls = 0

    def _partial_then_fail(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload[:1]))
        raise OSError("injected partial write failure")

    monkeypatch.setattr(os, "write", _partial_then_fail)

    with pytest.raises(ValueError, match="sidecar could not be written completely"):
        materialize_residue_alphabet_sidecar(request, target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing_matches", [True, False])
def test_sidecar_concurrent_publication_validates_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_matches: bool,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    canonical = alphabets_module._canonical_bytes(request)
    real_link = os.link
    attempted = False

    def _race_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal attempted
        attempted = True
        payload = canonical if existing_matches else b"{}\n"
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dst_dir_fd)
        try:
            real_write = os.write
            real_write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(dst_dir_fd)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", _race_link)

    if existing_matches:
        materialize_residue_alphabet_sidecar(request, target)
        assert target.read_bytes() == canonical
    else:
        with pytest.raises(FileExistsError, match="refusing to overwrite different"):
            materialize_residue_alphabet_sidecar(request, target)
        assert target.read_bytes() == b"{}\n"
    assert attempted
    assert not any(path.name.startswith(".omit.json.") for path in tmp_path.iterdir())


def test_sidecar_matching_existing_file_syncs_file_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    expected = materialize_residue_alphabet_sidecar(request, target)
    target_status = target.stat()
    target_identity = (target_status.st_dev, target_status.st_ino)
    real_fsync = os.fsync
    syncs: list[str] = []

    def _record_fsync(descriptor: int) -> None:
        status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) == target_identity:
            syncs.append("target")
        elif stat.S_ISDIR(status.st_mode):
            syncs.append("directory")
        else:
            syncs.append("private")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", _record_fsync)

    assert materialize_residue_alphabet_sidecar(request, target) == expected
    assert syncs == ["private", "target", "directory"]


@pytest.mark.parametrize("failure_target", ["target", "directory"])
def test_sidecar_matching_existing_durability_failure_does_not_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    materialize_residue_alphabet_sidecar(request, target)
    expected = target.read_bytes()
    target_status = target.stat()
    target_identity = (target_status.st_dev, target_status.st_ino)
    real_fsync = os.fsync

    def _fail_selected_fsync(descriptor: int) -> None:
        status = os.fstat(descriptor)
        is_target = (status.st_dev, status.st_ino) == target_identity
        if (failure_target == "target" and is_target) or (
            failure_target == "directory" and stat.S_ISDIR(status.st_mode)
        ):
            raise OSError(f"injected existing {failure_target} fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", _fail_selected_fsync)

    with pytest.raises(ValueError, match="matching sidecar could not be made durable"):
        materialize_residue_alphabet_sidecar(request, target)

    assert target.read_bytes() == expected
    assert not any(path.name.startswith(".omit.json.") for path in tmp_path.iterdir())


def test_sidecar_parent_fsync_failure_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    real_fsync = os.fsync
    failed = False

    def _fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and target.exists() and not failed:
            failed = True
            raise OSError("injected publication directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", _fail_first_directory_fsync)

    with pytest.raises(ValueError, match="sidecar publication could not be made durable"):
        materialize_residue_alphabet_sidecar(request, target)

    assert failed
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_sidecar_persistent_parent_fsync_failure_is_typed_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    real_fsync = os.fsync
    publication_observed = False

    def _fail_directory_fsync_after_publication(descriptor: int) -> None:
        nonlocal publication_observed
        if target.exists():
            publication_observed = True
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and publication_observed:
            raise OSError("injected persistent directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", _fail_directory_fsync_after_publication)

    with pytest.raises(
        alphabets_module.LigandMpnnSidecarPublicationUncertainError,
        match="rollback durability is uncertain",
    ):
        materialize_residue_alphabet_sidecar(request, target)

    assert not target.exists()


def test_sidecar_validation_rejects_symlink_replacement(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    target = tmp_path / "omit.json"
    sidecar = materialize_residue_alphabet_sidecar(request, target)
    outside = tmp_path / "outside.json"
    target.replace(outside)
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="residue alphabet sidecar must be a regular file"):
        sidecar.validate_for(request)


@pytest.mark.parametrize(
    ("path", "materialized_path"),
    [
        (Path("~/omit.json"), None),
        (Path("final/omit.json"), Path("~/staged.json")),
    ],
)
def test_sidecar_receipt_rejects_tilde_prefixed_paths(path: Path, materialized_path: Path | None) -> None:
    with pytest.raises(ValueError, match="must not begin with '~'"):
        LigandMpnnResidueAlphabetSidecar(
            request_id="generic_restricted_design",
            path=path,
            sha256=f"sha256:{_DIGEST}",
            residue_count=1,
            materialized_path=materialized_path,
        )


@pytest.mark.parametrize(
    ("path", "write_path"),
    [
        (Path("~/omit.json"), None),
        (Path("final/omit.json"), Path("~/staged.json")),
    ],
)
def test_sidecar_materialization_rejects_tilde_prefixed_paths_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    write_path: Path | None,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must not begin with '~'"):
        materialize_residue_alphabet_sidecar(request, path, write_path=write_path)

    assert not (tmp_path / "~").exists()


@pytest.mark.parametrize(
    ("path", "materialized_path"),
    [
        (Path("-omit.json"), None),
        (Path("final/omit.json"), Path("-staged.json")),
    ],
)
def test_sidecar_receipt_rejects_option_looking_relative_paths(
    path: Path,
    materialized_path: Path | None,
) -> None:
    with pytest.raises(ValueError, match="must not begin with '-'"):
        LigandMpnnResidueAlphabetSidecar(
            request_id="generic_restricted_design",
            path=path,
            sha256=f"sha256:{_DIGEST}",
            residue_count=1,
            materialized_path=materialized_path,
        )


@pytest.mark.parametrize(
    ("path", "write_path"),
    [
        (Path("-omit.json"), None),
        (Path("final/omit.json"), Path("-staged.json")),
    ],
)
def test_sidecar_materialization_rejects_option_looking_paths_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    write_path: Path | None,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must not begin with '-'"):
        materialize_residue_alphabet_sidecar(request, path, write_path=write_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "materialized_path"),
    [
        (Path("stage/../omit.json"), None),
        (Path("final/omit.json"), Path("stage/../staged.json")),
    ],
)
def test_sidecar_receipt_rejects_traversal_paths(path: Path, materialized_path: Path | None) -> None:
    with pytest.raises(ValueError, match="must not contain traversal"):
        LigandMpnnResidueAlphabetSidecar(
            request_id="generic_restricted_design",
            path=path,
            sha256=f"sha256:{_DIGEST}",
            residue_count=1,
            materialized_path=materialized_path,
        )


@pytest.mark.parametrize(
    ("path", "write_path"),
    [
        (Path("stage/../omit.json"), None),
        (Path("final/omit.json"), Path("stage/../staged.json")),
    ],
)
def test_sidecar_materialization_rejects_traversal_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    write_path: Path | None,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must not contain traversal"):
        materialize_residue_alphabet_sidecar(request, path, write_path=write_path)

    assert list(tmp_path.iterdir()) == []


def test_valid_relative_sidecar_path_emits_argparse_safe_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    monkeypatch.chdir(tmp_path)
    sidecar = materialize_residue_alphabet_sidecar(request, Path("inputs/omit.json"))
    argv = build_ligandmpnn_commands(
        request,
        checkout_root=Path("tool"),
        residue_alphabet_sidecar=sidecar,
    )[0].argv
    option_index = argv.index("--omit_AA_per_residue")
    emitted_pair = argv[option_index : option_index + 2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--omit_AA_per_residue", required=True)

    parsed = parser.parse_args(emitted_pair)

    assert parsed.omit_AA_per_residue == "inputs/omit.json"


def test_command_requires_and_verifies_typed_sidecar(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    with pytest.raises(ValueError, match="typed residue alphabet sidecar"):
        build_ligandmpnn_commands(request, checkout_root=Path("tool"))

    sidecar = materialize_residue_alphabet_sidecar(request, tmp_path / "omit.json")
    argv = build_ligandmpnn_commands(
        request,
        checkout_root=Path("tool"),
        residue_alphabet_sidecar=sidecar,
    )[0].argv
    assert argv[argv.index("--omit_AA_per_residue") + 1] == str(sidecar.path)
    assert argv[argv.index("--residue-alphabet-sha256") + 1] == sidecar.sha256.removeprefix("sha256:")

    sidecar.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 does not match"):
        build_ligandmpnn_commands(
            request,
            checkout_root=Path("tool"),
            residue_alphabet_sidecar=sidecar,
        )


def test_sidecar_can_stage_bytes_while_binding_final_execution_path(tmp_path: Path) -> None:
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    final_path = tmp_path / "final" / "omit.json"
    staging_path = tmp_path / "staging" / "omit.json"

    sidecar = materialize_residue_alphabet_sidecar(request, final_path, write_path=staging_path)
    commands = build_ligandmpnn_commands(
        request,
        checkout_root=Path("tool"),
        residue_alphabet_sidecar=sidecar,
    )

    assert staging_path.is_file()
    assert not final_path.exists()
    assert sidecar.to_dict()["path"] == str(final_path)
    assert "materialized_path" not in sidecar.to_dict()
    assert commands[0].argv[commands[0].argv.index("--omit_AA_per_residue") + 1] == str(final_path)

    final_path.parent.mkdir(parents=True)
    staging_path.replace(final_path)
    sidecar.validate_execution_file(request)


def test_planned_receipt_binds_the_sidecar_digest(tmp_path: Path) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_payload = b"ATOM pinned context input\n"
    pdb_path = tmp_path / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    pdb_sha256 = hashlib.sha256(pdb_payload).hexdigest()
    request = _request(LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 12), ("A", "G")))
    request = replace(
        request,
        pdb_sha256=pdb_sha256,
        upstream=LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_DIGEST),
        context_inventory=write_context_inventory(
            tmp_path,
            input_path=request.pdb_path,
            input_sha256=pdb_sha256,
            upstream_commit=commit,
            parse_all_atoms=request.use_side_chain_context,
            parser_sha256=parser_sha256,
        ),
    )
    sidecar = materialize_residue_alphabet_sidecar(request, tmp_path / "omit.json")
    commands = build_ligandmpnn_commands(
        request,
        checkout_root=checkout_root,
        residue_alphabet_sidecar=sidecar,
    )

    payload = build_planned_receipt(
        request,
        commands,
        execution_root=tmp_path,
        checkout_root=checkout_root,
        residue_alphabet_sidecar=sidecar,
    ).to_dict()

    assert payload["residue_alphabet_sidecar"] == sidecar.to_dict()

    with pytest.raises(ValueError, match="commands do not match the deterministic request command set"):
        build_planned_receipt(
            request,
            (),
            execution_root=tmp_path,
            checkout_root=checkout_root,
            residue_alphabet_sidecar=sidecar,
        )
