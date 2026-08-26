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
)


def write_context_inventory(
    root: Path,
    *,
    input_path: Path,
    input_sha256: str,
    upstream_commit: str,
    parse_all_atoms: bool,
    parser_sha256: str = "c" * 64,
) -> LigandMpnnContextInventoryReference:
    """Write one valid inventory and return its exact relative reference."""

    inventory = LigandMpnnContextInventory(
        request_id="generic_context_inventory",
        request_sha256="b" * 64,
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
        required_polymer_types=(LigandMpnnContextPolymer.DNA,),
        atoms=(LigandMpnnContextAtom(1, "P", "P", 15, "D", "DC", 12, "", LigandMpnnContextPolymer.DNA),),
    )
    payload = (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    relative_path = Path("evidence/context-inventory.json")
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
    parser_payload = b"def parse_PDB(*args, **kwargs):\n    return {}\n"
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


__all__ = ["create_pinned_context_checkout", "write_context_inventory"]
