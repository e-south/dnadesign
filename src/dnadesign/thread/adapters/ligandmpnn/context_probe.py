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
import ast
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from dnadesign.thread.adapters.ligandmpnn.context_inventory import (
    LigandMpnnContextAtom,
    LigandMpnnContextInventory,
    LigandMpnnContextPolymer,
    LigandMpnnProteinStructureEvidence,
    _read_descriptor_relative_regular_bytes,
)
from dnadesign.thread.adapters.ligandmpnn.models import (
    LigandMpnnContextInventoryReference,
    LigandMpnnUpstreamPin,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import (
    attested_working_tree_path_bytes,
    index_path_matches_commit,
    materialize_pinned_tree,
)

_DNA_RESIDUE_NAMES = frozenset({"DA", "DC", "DG", "DI", "DT", "DU"})
_RNA_RESIDUE_NAMES = frozenset({"A", "C", "G", "I", "U", "RA", "RC", "RG", "RI", "RU"})
_LINUX_RENAME_NOREPLACE = 1
_MACOS_RENAME_EXCL = 0x00000004
_ReceiptIdentity = tuple[int, int]


@dataclass(frozen=True)
class _PinnedParserResult:
    parsed: Any
    other_atoms: Any
    insertion_codes: Any
    protein_evidence: LigandMpnnProteinStructureEvidence
    preserved_nonprotein_atoms: tuple[tuple[object, ...], ...] = ()
    water_atom_count: int = 0
    canonical_heavy_atom_names: tuple[tuple[str, str, tuple[str, ...]], ...] = ()


class LigandMpnnContextPublicationUncertainError(RuntimeError):
    """Receipt rollback could not establish a durable pre-publication state."""


@dataclass(frozen=True)
class _ReceiptSnapshot:
    descriptor: int
    identity: _ReceiptIdentity
    payload: bytes


@dataclass(frozen=True)
class _WrittenReceipt:
    descriptor: int
    identity: _ReceiptIdentity


@dataclass(frozen=True)
class _ClaimedReceipt:
    quarantine_name: str
    quarantine_fd: int
    leaf_name: str


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
    ]
    _append_cli_option(argv, "--request-id", request.request_id)
    _append_cli_option(argv, "--checkout-root", str(checkout_root))
    _append_cli_option(argv, "--upstream-commit", request.upstream.commit)
    _append_cli_option(argv, "--pdb-path", request.pdb_path.as_posix())
    _append_cli_option(argv, "--pdb-sha256", request.pdb_sha256)
    _append_cli_option(argv, "--output-path", request.output_path.as_posix())
    _append_cli_option(argv, "--minimum-nucleotide-atoms", str(request.minimum_nucleotide_atoms))
    _append_cli_option(
        argv,
        "--required-polymer-types",
        ",".join(item.value for item in request.required_polymer_types),
    )
    _append_cli_option(argv, "--parse-all-atoms", _flag(request.parse_all_atoms))
    _append_cli_option(
        argv,
        "--parse-atoms-with-zero-occupancy",
        _flag(request.parse_atoms_with_zero_occupancy),
    )
    for chain in request.chains:
        _append_cli_option(argv, "--chain", chain)
    return LigandMpnnContextProbeCommand(output_path=request.output_path, argv=tuple(argv))


def _append_cli_option(argv: list[str], option: str, value: str) -> None:
    if value.startswith("-"):
        argv.append(f"{option}={value}")
    else:
        argv.extend([option, value])


def materialize_ligandmpnn_context_inventory(
    request: LigandMpnnContextProbeRequest,
    *,
    execution_root: Path,
    checkout_root: Path,
) -> LigandMpnnContextInventoryReference:
    """Run upstream ``parse_PDB`` and persist its effective context inventory."""

    inventory = _derive_ligandmpnn_context_inventory(
        request,
        execution_root=execution_root,
        checkout_root=checkout_root,
    )
    root = execution_root.expanduser().resolve()
    payload = (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_context_inventory(root, request.output_path, payload)
    return LigandMpnnContextInventoryReference(path=request.output_path, sha256=hashlib.sha256(payload).hexdigest())


def _derive_ligandmpnn_context_inventory(
    request: LigandMpnnContextProbeRequest,
    *,
    execution_root: Path,
    checkout_root: Path,
    require_clean_parser_checkout: bool = True,
) -> LigandMpnnContextInventory:
    """Derive one inventory from exact input bytes and the pinned parser without publishing it."""

    inventory, _protein_residue_ids = _derive_ligandmpnn_context_evidence(
        request,
        execution_root=execution_root,
        checkout_root=checkout_root,
        require_clean_parser_checkout=require_clean_parser_checkout,
    )
    return inventory


def _derive_ligandmpnn_context_evidence(
    request: LigandMpnnContextProbeRequest,
    *,
    execution_root: Path,
    checkout_root: Path,
    require_clean_parser_checkout: bool = True,
) -> tuple[LigandMpnnContextInventory, LigandMpnnProteinStructureEvidence]:
    """Derive context and exact protein selector identities from one pinned parse."""

    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("execution_root must be an existing directory")
    checkout_root = _resolve_context_probe_checkout_root(checkout_root, execution_root=root)
    input_bytes = _read_descriptor_relative_regular_bytes(
        root,
        request.pdb_path,
        label="context probe input",
    )
    observed_input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if observed_input_sha256 != request.pdb_sha256:
        raise ValueError(
            f"context probe input SHA256 mismatch: expected {request.pdb_sha256}, observed {observed_input_sha256}"
        )
    parsed, other_atoms, element_dict_rev, parser_sha256, protein_evidence = _run_pinned_upstream_parser(
        checkout_root,
        expected_commit=request.upstream.commit,
        input_bytes=input_bytes,
        input_name=request.pdb_path.name,
        chains=request.chains,
        parse_all_atoms=request.parse_all_atoms,
        parse_atoms_with_zero_occupancy=request.parse_atoms_with_zero_occupancy,
        require_clean_checkout=require_clean_parser_checkout,
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
    return inventory, protein_evidence


def _resolve_context_probe_checkout_root(checkout_root: Path, *, execution_root: Path) -> Path:
    if checkout_root.is_absolute():
        return checkout_root
    if ".." in checkout_root.parts:
        raise ValueError("relative context probe checkout_root must not contain traversal")
    if str(checkout_root).startswith("~"):
        raise ValueError("relative context probe checkout_root must not begin with '~'")
    return execution_root / checkout_root


def _publish_context_inventory(execution_root: Path, output_path: Path, payload: bytes) -> None:
    """Publish one receipt without overwriting a concurrent materializer."""

    temporary_name = f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    directory_fd = _open_verified_output_directory(execution_root, output_path.parent)
    lock_fd: int | None = None
    claim: _ClaimedReceipt | None = None
    prior_snapshot: _ReceiptSnapshot | None = None
    written_receipt: _WrittenReceipt | None = None
    try:
        lock_fd = _lock_context_receipt(directory_fd, output_path.name)
        prior_snapshot = _read_prior_receipt(directory_fd, output_path.name)
        written_receipt = _write_temporary_receipt(directory_fd, temporary_name, payload)
        if prior_snapshot is not None:
            claim = _claim_prior_receipt(directory_fd, output_path.name, prior_snapshot)
        try:
            _rename_no_replace(
                temporary_name,
                output_path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        except FileExistsError as error:
            recovery = (
                f"; prior receipt retained in {claim.quarantine_name}/{claim.leaf_name}" if claim is not None else ""
            )
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt changed before publication" + recovery
            ) from error
        except OSError as publication_error:
            if claim is not None:
                _restore_claimed_prior_receipt(directory_fd, output_path.name, claim)
                os.close(claim.quarantine_fd)
                claim = None
            raise ValueError("context probe output could not be published atomically") from publication_error
        try:
            os.fsync(directory_fd)
        except OSError as durability_error:
            try:
                _restore_prior_receipt(
                    directory_fd,
                    output_path.name,
                    claim,
                    published_identity=written_receipt.identity,
                    published_payload=payload,
                )
                if claim is not None:
                    os.close(claim.quarantine_fd)
                claim = None
            except OSError as restoration_error:
                raise LigandMpnnContextPublicationUncertainError(
                    "context probe receipt restoration could not be made durable after publication failure"
                ) from restoration_error
            raise ValueError("context probe output could not be published atomically") from durability_error
        if claim is not None:
            _discard_claimed_prior_receipt(directory_fd, claim)
            os.close(claim.quarantine_fd)
            claim = None
    except LigandMpnnContextPublicationUncertainError:
        raise
    except OSError as error:
        raise ValueError("context probe output could not be published atomically") from error
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            if claim is not None:
                os.close(claim.quarantine_fd)
            if prior_snapshot is not None:
                os.close(prior_snapshot.descriptor)
            if written_receipt is not None:
                os.close(written_receipt.descriptor)
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(directory_fd)


def _lock_context_receipt(directory_fd: int, output_name: str) -> int:
    """Serialize cooperating publishers across snapshot, replacement, and recovery."""

    lock_name = f".{output_name}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    lock_fd = os.open(lock_name, flags, 0o600, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise OSError("context probe receipt lock must be a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd
    except BaseException:
        os.close(lock_fd)
        raise


def _read_prior_receipt(directory_fd: int, output_name: str) -> _ReceiptSnapshot | None:
    """Read restorable regular-file bytes without following a receipt symlink."""

    try:
        status = os.stat(output_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(status.st_mode):
        raise ValueError("context probe output must be absent or an existing regular file")
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_descriptor = os.open(output_name, file_flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("context probe output must be absent or an existing regular file")
        handle = os.fdopen(file_descriptor, "rb", closefd=False)
        with handle:
            payload = handle.read()
        return _ReceiptSnapshot(
            descriptor=file_descriptor,
            identity=(opened.st_dev, opened.st_ino),
            payload=payload,
        )
    except BaseException:
        os.close(file_descriptor)
        raise


def _write_temporary_receipt(directory_fd: int, temporary_name: str, payload: bytes) -> _WrittenReceipt:
    """Write and sync one no-follow temporary receipt in an opened directory."""

    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    file_descriptor = os.open(temporary_name, file_flags, 0o600, dir_fd=directory_fd)
    try:
        handle = os.fdopen(file_descriptor, "wb", closefd=False)
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        status = os.fstat(file_descriptor)
        return _WrittenReceipt(
            descriptor=file_descriptor,
            identity=(status.st_dev, status.st_ino),
        )
    except BaseException:
        os.close(file_descriptor)
        raise


def _claim_prior_receipt(
    directory_fd: int,
    output_name: str,
    snapshot: _ReceiptSnapshot,
) -> _ClaimedReceipt:
    """Move and verify the actual destination leaf before replacing it."""

    quarantine_name = f".{output_name}.{uuid.uuid4().hex}.recovery"
    quarantine_leaf = "prior"
    quarantine_fd: int | None = None
    quarantine_created = False
    receipt_displaced = False
    claim_returned = False
    try:
        os.mkdir(quarantine_name, mode=0o700, dir_fd=directory_fd)
        quarantine_created = True
        quarantine_fd = os.open(
            quarantine_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        try:
            os.rename(
                output_name,
                quarantine_leaf,
                src_dir_fd=directory_fd,
                dst_dir_fd=quarantine_fd,
            )
            receipt_displaced = True
        except FileNotFoundError as error:
            os.rmdir(quarantine_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt changed before publication"
            ) from error
        try:
            observed_identity, observed_payload = _read_quarantined_receipt(quarantine_fd, quarantine_leaf)
        except OSError as error:
            _restore_quarantined_receipt_without_overwrite(
                directory_fd,
                output_name,
                quarantine_fd,
                quarantine_leaf,
                quarantine_name=quarantine_name,
            )
            receipt_displaced = False
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt changed before publication"
            ) from error
        if observed_identity != snapshot.identity or observed_payload != snapshot.payload:
            _restore_quarantined_receipt_without_overwrite(
                directory_fd,
                output_name,
                quarantine_fd,
                quarantine_leaf,
                quarantine_name=quarantine_name,
            )
            receipt_displaced = False
            raise LigandMpnnContextPublicationUncertainError("context probe receipt changed before publication")
        try:
            os.fsync(quarantine_fd)
            os.fsync(directory_fd)
        except OSError as error:
            _restore_quarantined_receipt_without_overwrite(
                directory_fd,
                output_name,
                quarantine_fd,
                quarantine_leaf,
                quarantine_name=quarantine_name,
            )
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt claim could not be made durable"
            ) from error
        claim = _ClaimedReceipt(
            quarantine_name=quarantine_name,
            quarantine_fd=quarantine_fd,
            leaf_name=quarantine_leaf,
        )
        claim_returned = True
        return claim
    except LigandMpnnContextPublicationUncertainError:
        raise
    except OSError as error:
        if quarantine_created and not receipt_displaced:
            try:
                os.rmdir(quarantine_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError:
                pass
        recovery = f"; displaced receipt retained in {quarantine_name}/{quarantine_leaf}" if receipt_displaced else ""
        raise LigandMpnnContextPublicationUncertainError("context probe receipt claim failed" + recovery) from error
    finally:
        if quarantine_fd is not None and not claim_returned:
            os.close(quarantine_fd)


def _restore_prior_receipt(
    directory_fd: int,
    output_name: str,
    claim: _ClaimedReceipt | None,
    *,
    published_identity: _ReceiptIdentity,
    published_payload: bytes,
) -> None:
    """Quarantine this publication and restore the actually claimed prior leaf."""

    owns_quarantine = claim is None
    if claim is None:
        quarantine_name = f".{output_name}.{uuid.uuid4().hex}.recovery"
        os.mkdir(quarantine_name, mode=0o700, dir_fd=directory_fd)
        quarantine_fd = os.open(
            quarantine_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    else:
        quarantine_name = claim.quarantine_name
        quarantine_fd = claim.quarantine_fd
    publication_leaf = "publication"
    publication_quarantined = False
    try:
        os.rename(
            output_name,
            publication_leaf,
            src_dir_fd=directory_fd,
            dst_dir_fd=quarantine_fd,
        )
        publication_quarantined = True
        observed_identity, observed_payload = _read_quarantined_receipt(quarantine_fd, publication_leaf)
        if observed_identity != published_identity or observed_payload != published_payload:
            _restore_quarantined_receipt_without_overwrite(
                directory_fd,
                output_name,
                quarantine_fd,
                publication_leaf,
                quarantine_name=quarantine_name,
                remove_quarantine=False,
            )
            publication_quarantined = False
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt changed before publication recovery"
            )
        os.unlink(publication_leaf, dir_fd=quarantine_fd)
        publication_quarantined = False
        if claim is not None:
            _restore_claimed_prior_receipt(directory_fd, output_name, claim)
        else:
            os.fsync(quarantine_fd)
            os.rmdir(quarantine_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            try:
                os.stat(output_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise LigandMpnnContextPublicationUncertainError(
                "context probe receipt changed before publication recovery"
            )
    except LigandMpnnContextPublicationUncertainError:
        raise
    except OSError as error:
        recovery = (
            f"; displaced publication retained in {quarantine_name}/{publication_leaf}"
            if publication_quarantined
            else ""
        )
        raise LigandMpnnContextPublicationUncertainError(
            "context probe receipt restoration could not be made durable after publication failure" + recovery
        ) from error
    finally:
        if owns_quarantine:
            os.close(quarantine_fd)


def _read_quarantined_receipt(directory_fd: int, name: str) -> tuple[_ReceiptIdentity, bytes]:
    """Read identity and bytes from one no-follow regular recovery leaf."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("context probe recovery leaf is not a regular file")
        handle = os.fdopen(descriptor, "rb", closefd=False)
        with handle:
            payload = handle.read()
        return (status.st_dev, status.st_ino), payload
    finally:
        os.close(descriptor)


def _restore_quarantined_receipt_without_overwrite(
    directory_fd: int,
    output_name: str,
    quarantine_fd: int,
    quarantine_leaf: str,
    *,
    quarantine_name: str,
    remove_quarantine: bool = True,
) -> None:
    """Restore a displaced foreign receipt only when its public name is absent."""

    try:
        _rename_no_replace(
            quarantine_leaf,
            output_name,
            src_dir_fd=quarantine_fd,
            dst_dir_fd=directory_fd,
        )
    except FileExistsError as error:
        raise LigandMpnnContextPublicationUncertainError(
            f"context probe receipt changed before publication recovery; displaced receipt retained in "
            f"{quarantine_name}/{quarantine_leaf}"
        ) from error
    try:
        os.fsync(quarantine_fd)
        os.fsync(directory_fd)
        if remove_quarantine:
            os.rmdir(quarantine_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except OSError as error:
        raise LigandMpnnContextPublicationUncertainError(
            "context probe receipt restoration could not be made durable after publication failure; "
            "restored concurrent receipt durability is uncertain"
        ) from error


def _restore_claimed_prior_receipt(
    directory_fd: int,
    output_name: str,
    claim: _ClaimedReceipt,
) -> None:
    """Restore the exact displaced prior leaf without overwriting a replacement."""

    try:
        _rename_no_replace(
            claim.leaf_name,
            output_name,
            src_dir_fd=claim.quarantine_fd,
            dst_dir_fd=directory_fd,
        )
    except FileExistsError as error:
        raise LigandMpnnContextPublicationUncertainError(
            f"context probe receipt changed before publication recovery; prior receipt retained in "
            f"{claim.quarantine_name}/{claim.leaf_name}"
        ) from error
    except OSError as error:
        raise LigandMpnnContextPublicationUncertainError(
            f"context probe receipt restoration could not use atomic no-replace; prior receipt retained in "
            f"{claim.quarantine_name}/{claim.leaf_name}"
        ) from error
    try:
        os.fsync(claim.quarantine_fd)
        os.fsync(directory_fd)
        os.rmdir(claim.quarantine_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        raise LigandMpnnContextPublicationUncertainError(
            "context probe receipt restoration could not be made durable after publication failure"
        ) from error


def _discard_claimed_prior_receipt(directory_fd: int, claim: _ClaimedReceipt) -> None:
    """Discard one superseded claimed leaf after the replacement is durable."""

    prior_retained = True
    recovery_directory_exists = True
    try:
        os.unlink(claim.leaf_name, dir_fd=claim.quarantine_fd)
        prior_retained = False
        os.fsync(claim.quarantine_fd)
        os.rmdir(claim.quarantine_name, dir_fd=directory_fd)
        recovery_directory_exists = False
        os.fsync(directory_fd)
    except OSError as error:
        if prior_retained:
            detail = f"; recovery at {claim.quarantine_name}/{claim.leaf_name}"
        elif recovery_directory_exists:
            detail = f"; empty recovery directory at {claim.quarantine_name}"
        else:
            detail = "; recovery-directory removal durability is uncertain"
        raise LigandMpnnContextPublicationUncertainError(
            "context probe receipt is durable but superseded prior cleanup is uncertain" + detail
        ) from error


def _resolve_rename_no_replace() -> tuple[ctypes._CFuncPtr, int]:
    """Resolve the platform-native atomic no-replace operation before mutation."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        if sys.platform.startswith("linux"):
            rename_function = libc.renameat2
            flags = _LINUX_RENAME_NOREPLACE
        elif sys.platform == "darwin":
            rename_function = libc.renameatx_np
            flags = _MACOS_RENAME_EXCL
        else:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is not supported on this platform")
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from error
    rename_function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_function.restype = ctypes.c_int
    return rename_function, flags


def _rename_no_replace(
    source_name: str,
    destination_name: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
) -> None:
    """Atomically rename one leaf without replacing any destination type."""

    rename_function, flags = _resolve_rename_no_replace()
    result = rename_function(
        src_dir_fd,
        os.fsencode(source_name),
        dst_dir_fd,
        os.fsencode(destination_name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _open_verified_output_directory(execution_root: Path, relative_parent: Path) -> int:
    """Prove no-replace support before creating or mutating the output tree."""

    try:
        preflight_fd, target_exists = _open_nearest_existing_output_directory(execution_root, relative_parent)
    except OSError as error:
        raise ValueError("context probe output directory could not be opened safely") from error
    try:
        _probe_atomic_no_replace(preflight_fd)
    except OSError as error:
        os.close(preflight_fd)
        raise ValueError("context probe output could not be published atomically") from error
    if target_exists:
        return preflight_fd
    os.close(preflight_fd)

    try:
        directory_fd = _open_output_directory(execution_root, relative_parent)
    except OSError as error:
        raise ValueError("context probe output directory could not be opened safely") from error
    try:
        _probe_atomic_no_replace(directory_fd)
        return directory_fd
    except OSError as error:
        os.close(directory_fd)
        raise ValueError("context probe output could not be published atomically") from error


def _probe_atomic_no_replace(directory_fd: int) -> None:
    """Exercise collision and round-trip semantics in the target filesystem."""

    _resolve_rename_no_replace()
    probe_id = uuid.uuid4().hex
    source_name = f".dnadesign-context-noreplace-{probe_id}.source"
    destination_name = f".dnadesign-context-noreplace-{probe_id}.destination"
    source_payload = b"dnadesign no-replace source\n"
    destination_payload = b"dnadesign no-replace destination\n"
    source_receipt: _WrittenReceipt | None = None
    destination_receipt: _WrittenReceipt | None = None
    try:
        source_receipt = _write_temporary_receipt(directory_fd, source_name, source_payload)
        destination_receipt = _write_temporary_receipt(directory_fd, destination_name, destination_payload)
        try:
            _rename_no_replace(
                source_name,
                destination_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        except FileExistsError:
            pass
        else:
            raise OSError(errno.EIO, "atomic no-replace probe overwrote an existing destination")
        _require_probe_leaf(
            directory_fd,
            source_name,
            expected_identity=source_receipt.identity,
            expected_payload=source_payload,
        )
        _require_probe_leaf(
            directory_fd,
            destination_name,
            expected_identity=destination_receipt.identity,
            expected_payload=destination_payload,
        )

        os.unlink(destination_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        _rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        _require_probe_leaf(
            directory_fd,
            destination_name,
            expected_identity=source_receipt.identity,
            expected_payload=source_payload,
        )
        _rename_no_replace(
            destination_name,
            source_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        _require_probe_leaf(
            directory_fd,
            source_name,
            expected_identity=source_receipt.identity,
            expected_payload=source_payload,
        )
    except BaseException:
        _cleanup_no_replace_probe(directory_fd, source_name, destination_name)
        raise
    else:
        _cleanup_no_replace_probe(directory_fd, source_name, destination_name)
    finally:
        if source_receipt is not None:
            os.close(source_receipt.descriptor)
        if destination_receipt is not None:
            os.close(destination_receipt.descriptor)


def _require_probe_leaf(
    directory_fd: int,
    name: str,
    *,
    expected_identity: _ReceiptIdentity,
    expected_payload: bytes,
) -> None:
    observed_identity, observed_payload = _read_quarantined_receipt(directory_fd, name)
    if observed_identity != expected_identity or observed_payload != expected_payload:
        raise OSError(errno.EIO, "atomic no-replace probe changed a scratch leaf")


def _cleanup_no_replace_probe(directory_fd: int, *names: str) -> None:
    changed = False
    for name in names:
        try:
            os.unlink(name, dir_fd=directory_fd)
            changed = True
        except FileNotFoundError:
            pass
    if changed:
        os.fsync(directory_fd)


def _open_nearest_existing_output_directory(
    execution_root: Path,
    relative_parent: Path,
) -> tuple[int, bool]:
    """Open the nearest existing output ancestor without creating components."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_parts = execution_root.parts
    if not execution_root.is_absolute() or not root_parts:
        raise OSError("execution_root must be absolute")
    current_fd = os.open(execution_root.anchor, directory_flags)
    try:
        for component in root_parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        for component in relative_parent.parts:
            if component in {"", "."}:
                continue
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                return current_fd, False
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, True
    except BaseException:
        os.close(current_fd)
        raise


def _open_output_directory(execution_root: Path, relative_parent: Path) -> int:
    """Open or create an output parent without following any path component."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_parts = execution_root.parts
    if not execution_root.is_absolute() or not root_parts:
        raise OSError("execution_root must be absolute")
    current_fd = os.open(execution_root.anchor, directory_flags)
    try:
        for component in root_parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        for component in relative_parent.parts:
            if component in {"", "."}:
                continue
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                created = False
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(current_fd)
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _run_pinned_upstream_parser(
    checkout_root: Path,
    *,
    expected_commit: str,
    input_bytes: bytes,
    input_name: str,
    chains: tuple[str, ...],
    parse_all_atoms: bool,
    parse_atoms_with_zero_occupancy: bool,
    require_clean_checkout: bool = True,
) -> tuple[Any, Any, dict[int, str], str, LigandMpnnProteinStructureEvidence]:
    results, element_dict_rev, parser_sha256 = _run_pinned_upstream_parser_batch(
        checkout_root,
        expected_commit=expected_commit,
        inputs=((input_name, input_bytes),),
        chains=chains,
        parse_all_atoms=parse_all_atoms,
        parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy,
        require_clean_checkout=require_clean_checkout,
    )
    result = results[0]
    return result.parsed, result.other_atoms, element_dict_rev, parser_sha256, result.protein_evidence


def _derive_pinned_packing_structure_evidence_for_payloads(
    checkout_root: Path,
    *,
    expected_commit: str,
    inputs: tuple[tuple[str, bytes], ...],
) -> tuple[_PinnedParserResult, ...]:
    """Parse a bounded group of packed PDB payloads through one attested snapshot."""

    results, _element_dict_rev, _parser_sha256 = _run_pinned_upstream_parser_batch(
        checkout_root,
        expected_commit=expected_commit,
        inputs=inputs,
        chains=(),
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=False,
        retain_parser_outputs=False,
        capture_packing_contract=True,
    )
    return results


def _run_pinned_upstream_parser_batch(
    checkout_root: Path,
    *,
    expected_commit: str,
    inputs: tuple[tuple[str, bytes], ...],
    chains: tuple[str, ...],
    parse_all_atoms: bool,
    parse_atoms_with_zero_occupancy: bool,
    require_clean_checkout: bool = True,
    retain_parser_outputs: bool = True,
    capture_packing_contract: bool = False,
) -> tuple[tuple[_PinnedParserResult, ...], dict[int, str], str]:
    if not inputs:
        raise ValueError("pinned parser inputs must not be empty")
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    head = _git(checkout, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {expected_commit}, observed {head}")
    tracked = _git(checkout, "ls-files", "--error-unmatch", "data_utils.py")
    if tracked != "data_utils.py":
        raise ValueError("pinned LigandMPNN checkout does not track data_utils.py")
    if index_path_matches_commit(checkout, expected_commit, "data_utils.py") is not True:
        raise ValueError("data_utils.py Git index does not match the pinned commit")
    if require_clean_checkout:
        source_bytes = attested_working_tree_path_bytes(checkout, expected_commit, "data_utils.py")
        if source_bytes is None:
            raise ValueError("data_utils.py must be clean at the pinned commit")
        working_source_path = checkout / "data_utils.py"
        if working_source_path.is_symlink() or not working_source_path.is_file():
            raise ValueError("pinned LigandMPNN data_utils.py must be a regular file")
    else:
        try:
            source_bytes = subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(checkout), "show", f"{expected_commit}:data_utils.py"],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("pinned LigandMPNN parser blob could not be read") from error
    with tempfile.TemporaryDirectory(prefix="dnadesign-ligandmpnn-context-") as temporary:
        snapshot = Path(temporary) / "source"
        snapshot.mkdir()
        materialize_pinned_tree(checkout, expected_commit, snapshot)
        source_path = snapshot / "data_utils.py"
        snapshot_source_bytes = source_path.read_bytes()
        if snapshot_source_bytes != source_bytes:
            raise ValueError("materialized data_utils.py does not match the pinned commit")
        module = _import_upstream_module(
            source_bytes,
            source_path=source_path,
            source_root=snapshot,
            excluded_root=checkout,
        )
        parser = getattr(module, "parse_PDB", None)
        if not callable(parser):
            raise ValueError("pinned LigandMPNN data_utils.py does not expose parse_PDB")
        element_dict_rev = getattr(module, "element_dict_rev", None)
        if not isinstance(element_dict_rev, dict) or any(
            not isinstance(key, int) or not isinstance(value, str) for key, value in element_dict_rev.items()
        ):
            raise ValueError("pinned LigandMPNN data_utils.py does not expose element_dict_rev")
        restype_int_to_str = getattr(module, "restype_int_to_str", None)
        if not isinstance(restype_int_to_str, dict) or any(
            not isinstance(key, int) or not isinstance(value, str) for key, value in restype_int_to_str.items()
        ):
            raise ValueError("pinned LigandMPNN data_utils.py does not expose restype_int_to_str")
        canonical_heavy_atom_names = (
            _pinned_write_full_pdb_atom_names(source_bytes, module) if capture_packing_contract else ()
        )
        results: list[_PinnedParserResult] = []
        for input_index, (input_name, input_bytes) in enumerate(inputs):
            if not input_name or Path(input_name).name != input_name:
                raise ValueError("pinned parser input names must be plain file names")
            input_path = snapshot / ".dnadesign-inputs" / str(input_index) / input_name
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(input_bytes)
            input_path.chmod(0o400)
            parsed, _, other_atoms, insertion_codes, water_atoms = parser(
                str(input_path),
                device="cpu",
                chains=list(chains),
                parse_all_atoms=parse_all_atoms,
                parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy,
            )
            results.append(
                _PinnedParserResult(
                    parsed=parsed if retain_parser_outputs else None,
                    other_atoms=other_atoms if retain_parser_outputs else None,
                    insertion_codes=insertion_codes if retain_parser_outputs else None,
                    protein_evidence=_pinned_parser_protein_evidence(
                        parsed,
                        insertion_codes,
                        restype_int_to_str=restype_int_to_str,
                    ),
                    preserved_nonprotein_atoms=(
                        _pinned_preserved_nonprotein_atoms(other_atoms) if capture_packing_contract else ()
                    ),
                    water_atom_count=(_pinned_atom_count(water_atoms) if capture_packing_contract else 0),
                    canonical_heavy_atom_names=canonical_heavy_atom_names,
                )
            )
    return tuple(results), element_dict_rev, hashlib.sha256(source_bytes).hexdigest()


def _pinned_parser_protein_evidence(
    parsed: Any,
    insertion_codes: Any,
    *,
    restype_int_to_str: dict[int, str],
) -> LigandMpnnProteinStructureEvidence:
    """Encode exact identities and native sequence consumed by pinned entrypoints."""

    if not isinstance(parsed, dict) or not {"R_idx", "chain_letters", "S", "mask"}.issubset(parsed):
        raise ValueError("upstream parse_PDB did not return R_idx, chain_letters, S, and mask")
    residue_numbers = _to_numpy(parsed["R_idx"]).reshape(-1)
    chain_letters = _to_numpy(parsed["chain_letters"]).reshape(-1)
    insertion_codes_array = _to_numpy(insertion_codes).reshape(-1)
    native_indices = _to_numpy(parsed["S"]).reshape(-1)
    residue_validity_mask = _to_numpy(parsed["mask"]).reshape(-1)
    if not (
        len(residue_numbers)
        == len(chain_letters)
        == len(insertion_codes_array)
        == len(native_indices)
        == len(residue_validity_mask)
    ):
        raise ValueError("upstream protein residue identity lengths differ")
    identities: list[str] = []
    native_sequence: list[str] = []
    for chain_id, residue_number, insertion_code, native_index in zip(
        chain_letters,
        residue_numbers,
        insertion_codes_array,
        native_indices,
        strict=True,
    ):
        if not isinstance(chain_id, str) or not isinstance(insertion_code, str):
            raise ValueError("upstream protein residue identities must use text chain and insertion codes")
        if isinstance(residue_number, (bool, np.bool_)) or not np.issubdtype(type(residue_number), np.integer):
            raise ValueError("upstream protein residue numbers must be integers")
        identities.append(f"{chain_id}{int(residue_number)}{insertion_code}")
        if isinstance(native_index, (bool, np.bool_)) or not np.issubdtype(type(native_index), np.integer):
            raise ValueError("upstream native sequence indices must be integers")
        native_residue = restype_int_to_str.get(int(native_index))
        if native_residue is None:
            raise ValueError("upstream native sequence contains an unknown residue index")
        native_sequence.append(native_residue)
    return LigandMpnnProteinStructureEvidence(
        residue_ids=tuple(identities),
        chain_ids=tuple(str(item) for item in chain_letters),
        native_sequence=tuple(native_sequence),
        residue_validity_mask=_pinned_parser_binary_mask(residue_validity_mask),
    )


def _pinned_parser_binary_mask(values: np.ndarray) -> tuple[int, ...]:
    mask: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not np.issubdtype(type(value), np.integer):
            raise ValueError("upstream protein residue validity mask must contain integers")
        integer = int(value)
        if integer not in {0, 1}:
            raise ValueError("upstream protein residue validity mask must contain only zero or one")
        mask.append(integer)
    return tuple(mask)


def _pinned_write_full_pdb_atom_names(
    source_bytes: bytes,
    module: ModuleType,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Extract the exact amino-acid atom14 table used by pinned ``write_full_PDB``."""

    try:
        tree = ast.parse(source_bytes)
    except (SyntaxError, ValueError) as error:
        raise ValueError("pinned LigandMPNN data_utils.py could not be parsed for packing semantics") from error
    assignments: list[ast.AST] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "write_full_PDB":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target = child.targets[0]
            if isinstance(target, ast.Name) and target.id == "restype_name_to_atom14_names":
                assignments.append(child.value)
    if len(assignments) != 1:
        raise ValueError("pinned write_full_PDB must define exactly one atom14 name table")
    try:
        by_three_letter = ast.literal_eval(assignments[0])
    except (TypeError, ValueError, SyntaxError) as error:
        raise ValueError("pinned write_full_PDB atom14 name table must be literal") from error
    one_to_three = getattr(module, "restype_1to3", None)
    if not isinstance(by_three_letter, dict) or not isinstance(one_to_three, dict):
        raise ValueError("pinned write_full_PDB amino-acid atom tables are invalid")
    observed: list[tuple[str, str, tuple[str, ...]]] = []
    for amino_acid, residue_name in sorted(one_to_three.items()):
        names = by_three_letter.get(residue_name)
        if (
            not isinstance(amino_acid, str)
            or len(amino_acid) != 1
            or not isinstance(residue_name, str)
            or not isinstance(names, list)
            or len(names) != 14
            or any(not isinstance(name, str) for name in names)
        ):
            raise ValueError("pinned write_full_PDB atom14 name table is invalid")
        nonempty = tuple(name for name in names if name)
        if len(set(nonempty)) != len(nonempty):
            raise ValueError("pinned write_full_PDB atom14 name table contains duplicate atoms")
        observed.append((amino_acid, residue_name, nonempty))
    return tuple(observed)


def _pinned_preserved_nonprotein_atoms(other_atoms: Any) -> tuple[tuple[object, ...], ...]:
    """Normalize fields that pinned ``write_full_PDB`` copies into packed output."""

    if other_atoms is None:
        return ()
    observed: list[tuple[object, ...]] = []
    for atom in other_atoms.iterAtoms():
        coordinates = np.asarray(atom.getCoords(), dtype=np.float64).reshape(-1)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise ValueError("pinned parser nonprotein atom coordinates are invalid")
        occupancy = atom.getOccupancy()
        if isinstance(occupancy, (bool, np.bool_)) or not isinstance(occupancy, (int, float, np.number)):
            raise ValueError("pinned parser nonprotein atom occupancy is invalid")
        observed.append(
            (
                str(atom.getName()).strip(),
                str(atom.getElement() or "").strip().upper(),
                str(atom.getChid()).strip(),
                str(atom.getResname()).strip().upper(),
                int(atom.getResnum()),
                tuple(round(float(value), 3) for value in coordinates),
                round(float(occupancy), 2),
            )
        )
    return tuple(observed)


def _pinned_atom_count(selection: Any) -> int:
    return 0 if selection is None else sum(1 for _atom in selection.iterAtoms())


def _import_upstream_module(
    source_bytes: bytes,
    *,
    source_path: Path,
    source_root: Path,
    excluded_root: Path,
) -> ModuleType:
    module_name = f"_dnadesign_ligandmpnn_data_utils_{hashlib.sha256(source_bytes).hexdigest()[:12]}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    previous_path = sys.path[:]
    loaded_before = set(sys.modules)
    excluded = excluded_root.resolve()
    for loaded_name, loaded_module in tuple(sys.modules.items()):
        if _module_is_within(loaded_module, excluded):
            raise ValueError(f"mutable LigandMPNN checkout module is already loaded: {loaded_name}")
    safe_path = [entry for entry in previous_path if entry and not _path_is_within(entry, excluded)]
    sys.path[:] = [str(source_root), *safe_path]
    try:
        exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)
    except Exception as error:
        raise ValueError(f"could not import pinned LigandMPNN data_utils.py: {error}") from error
    finally:
        sys.path[:] = previous_path
        for loaded_name in set(sys.modules) - loaded_before:
            if _module_is_within(sys.modules.get(loaded_name), source_root):
                sys.modules.pop(loaded_name, None)
    return module


def _path_is_within(value: str, root: Path) -> bool:
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return path == root or root in path.parents


def _module_is_within(module: object, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    return isinstance(module_file, str) and _path_is_within(module_file, root)


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
    if (
        not isinstance(path, Path)
        or path.is_absolute()
        or not path.name
        or ".." in path.parts
        or str(path).startswith("~")
    ):
        raise ValueError(f"{field_name} must be a safe relative file path")
    if suffix is not None and path.suffix.lower() != suffix:
        raise ValueError(f"{field_name} must end in {suffix}")


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
    materialize.add_argument("--chain", action="append", dest="chains")
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
        chains=tuple(args.chains or ()),
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
