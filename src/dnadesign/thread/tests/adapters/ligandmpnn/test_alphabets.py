"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_alphabets.py

Residue-specific LigandMPNN alphabet contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnResidueAlphabet,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_commands,
    build_planned_receipt,
    materialize_residue_alphabet_sidecar,
)
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
