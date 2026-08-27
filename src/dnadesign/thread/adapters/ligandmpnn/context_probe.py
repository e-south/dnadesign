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


class LigandMpnnContextPublicationUncertainError(RuntimeError):
    """Receipt rollback could not establish a durable pre-publication state."""


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

    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("execution_root must be an existing directory")
    checkout_root = _resolve_context_probe_checkout_root(checkout_root, execution_root=root)
    input_path = _within_root(root, request.pdb_path, field_name="context probe pdb_path")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("context probe input must be an existing regular file, not a symlink")
    try:
        input_bytes = input_path.read_bytes()
    except OSError as error:
        raise ValueError("context probe input could not be read") from error
    observed_input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if observed_input_sha256 != request.pdb_sha256:
        raise ValueError(
            f"context probe input SHA256 mismatch: expected {request.pdb_sha256}, observed {observed_input_sha256}"
        )
    parsed, other_atoms, element_dict_rev, parser_sha256 = _run_pinned_upstream_parser(
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
    return LigandMpnnContextInventory(
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


def _resolve_context_probe_checkout_root(checkout_root: Path, *, execution_root: Path) -> Path:
    if checkout_root.is_absolute():
        return checkout_root
    if ".." in checkout_root.parts:
        raise ValueError("relative context probe checkout_root must not contain traversal")
    if str(checkout_root).startswith("~"):
        raise ValueError("relative context probe checkout_root must not begin with '~'")
    return execution_root / checkout_root


def _publish_context_inventory(execution_root: Path, output_path: Path, payload: bytes) -> None:
    """Atomically replace one receipt through a descriptor-pinned directory chain."""

    temporary_name = f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        directory_fd = _open_output_directory(execution_root, output_path.parent)
    except OSError as error:
        raise ValueError("context probe output directory could not be opened safely") from error
    try:
        prior_payload = _read_prior_receipt(directory_fd, output_path.name)
        published_identity = _write_temporary_receipt(directory_fd, temporary_name, payload)
        os.replace(
            temporary_name,
            output_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        try:
            os.fsync(directory_fd)
        except OSError as durability_error:
            try:
                _restore_prior_receipt(
                    directory_fd,
                    output_path.name,
                    prior_payload,
                    published_identity=published_identity,
                )
            except OSError as restoration_error:
                raise LigandMpnnContextPublicationUncertainError(
                    "context probe receipt restoration could not be made durable after publication failure"
                ) from restoration_error
            raise ValueError("context probe output could not be published atomically") from durability_error
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
            os.close(directory_fd)


def _read_prior_receipt(directory_fd: int, output_name: str) -> bytes | None:
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
        handle = os.fdopen(file_descriptor, "rb")
    except BaseException:
        os.close(file_descriptor)
        raise
    with handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("context probe output must be absent or an existing regular file")
        return handle.read()


def _write_temporary_receipt(directory_fd: int, temporary_name: str, payload: bytes) -> tuple[int, int]:
    """Write and sync one no-follow temporary receipt in an opened directory."""

    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    file_descriptor = os.open(temporary_name, file_flags, 0o600, dir_fd=directory_fd)
    try:
        handle = os.fdopen(file_descriptor, "wb")
    except BaseException:
        os.close(file_descriptor)
        raise
    with handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        status = os.fstat(handle.fileno())
        return status.st_dev, status.st_ino


def _restore_prior_receipt(
    directory_fd: int,
    output_name: str,
    prior_payload: bytes | None,
    *,
    published_identity: tuple[int, int],
) -> None:
    """Restore the descriptor-relative pre-publication state and sync it."""

    try:
        current_status = os.stat(output_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise LigandMpnnContextPublicationUncertainError(
            "context probe receipt changed before publication recovery"
        ) from error
    if (current_status.st_dev, current_status.st_ino) != published_identity:
        raise LigandMpnnContextPublicationUncertainError("context probe receipt changed before publication recovery")
    if prior_payload is None:
        try:
            os.unlink(output_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    else:
        restoration_name = f".{output_name}.{uuid.uuid4().hex}.restore.tmp"
        try:
            _write_temporary_receipt(directory_fd, restoration_name, prior_payload)
            os.replace(
                restoration_name,
                output_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        finally:
            try:
                os.unlink(restoration_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    os.fsync(directory_fd)


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
) -> tuple[Any, Any, dict[int, str], str]:
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
        input_path = snapshot / ".dnadesign-inputs" / input_name
        input_path.parent.mkdir()
        input_path.write_bytes(input_bytes)
        input_path.chmod(0o400)
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
        parsed, _, other_atoms, _, _ = parser(
            str(input_path),
            device="cpu",
            chains=list(chains),
            parse_all_atoms=parse_all_atoms,
            parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy,
        )
    return parsed, other_atoms, element_dict_rev, hashlib.sha256(source_bytes).hexdigest()


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
