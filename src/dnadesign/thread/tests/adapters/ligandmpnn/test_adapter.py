"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_adapter.py

Behavior tests for LigandMPNN request and command adaptation.

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
    LigandMpnnPackingConfig,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_commands,
    build_planned_receipt,
    load_ligandmpnn_context_inventory,
)
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import (
    create_pinned_context_checkout,
    write_context_inventory,
)

_DIGEST = "a" * 64
_PACKING_DIGEST = "b" * 64
_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"  # pragma: allowlist secret
_CONTEXT_PDB_PAYLOAD = b"ATOM pinned context input\n"


def _write_context_input(root: Path) -> str:
    path = root / "inputs/target.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CONTEXT_PDB_PAYLOAD)
    return hashlib.sha256(_CONTEXT_PDB_PAYLOAD).hexdigest()


def _request(**overrides: object) -> LigandMpnnRequest:
    values: dict[str, object] = {
        "request_id": "generic_binding_site_v1",
        "pdb_path": Path("inputs/target.pdb"),
        "pdb_sha256": _DIGEST,
        "output_dir": Path("outputs/designs"),
        "upstream": LigandMpnnUpstreamPin(
            commit=_COMMIT,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
        "context_inventory": LigandMpnnContextInventoryReference(
            path=Path("evidence/context-inventory.json"), sha256=_DIGEST
        ),
        "fixed_residues": (
            LigandMpnnResidue(chain_id="A", residue_number=12),
            LigandMpnnResidue(chain_id="A", residue_number=13, insertion_code="B"),
        ),
        "seeds": (7, 11),
        "temperature": 0.2,
        "batch_size": 2,
        "number_of_batches": 3,
        "use_atom_context": False,
        "use_side_chain_context": True,
        "packing": LigandMpnnPackingConfig(
            enabled=True,
            number_of_packs_per_design=4,
            repack_everything=False,
            use_ligand_context=True,
        ),
    }
    values.update(overrides)
    return LigandMpnnRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("pdb_path", Path("/tmp/target.pdb"), "safe non-option relative"),
        ("pdb_path", Path("~/target.pdb"), "safe non-option relative"),
        ("pdb_path", Path("-option-like-input.pdb"), "safe non-option relative"),
        ("output_dir", Path("/tmp/designs"), "safe non-option relative"),
        ("output_dir", Path("~/designs"), "safe non-option relative"),
        ("output_dir", Path("-option-like-output"), "must not begin with a hyphen"),
        ("output_dir", Path("results/../designs"), "must not contain traversal"),
    ],
)
def test_request_rejects_paths_that_cannot_round_trip_through_runtime_argv(
    field_name: str,
    value: Path,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**{field_name: value})


def test_request_preserves_valid_nested_relative_output_directory() -> None:
    request = _request(output_dir=Path("results/nested/designs"))

    command = build_ligandmpnn_commands(request, checkout_root=Path("LigandMPNN"))[0]

    assert command.output_dir == Path("results/nested/designs/seed_7")
    assert command.argv[command.argv.index("--out_folder") + 1] == "results/nested/designs/seed_7"


def test_build_commands_declares_exact_official_ligandmpnn_flags_per_seed() -> None:
    request = _request()

    commands = build_ligandmpnn_commands(
        request,
        checkout_root=Path("/opt/LigandMPNN"),
        python_executable="python3",
    )

    assert len(commands) == 2
    planned_execution_sha256 = commands[0].argv[commands[0].argv.index("--planned-execution-sha256") + 1]
    assert len(planned_execution_sha256) == 64
    assert commands[0].argv == (
        "python3",
        "-m",
        "dnadesign.thread.adapters.ligandmpnn.pinned_runtime",
        "--checkout-root",
        "/opt/LigandMPNN",
        "--upstream-commit",
        _COMMIT,
        "--checkpoint-sha256",
        _DIGEST,
        "--pdb-sha256",
        _DIGEST,
        "--packing-checkpoint-sha256",
        _PACKING_DIGEST,
        "--planned-execution-sha256",
        planned_execution_sha256,
        "--completion-record",
        "outputs/designs/seed_7/.dnadesign-ligandmpnn-execution.json",
        "--entrypoint",
        "run.py",
        "--",
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        "/opt/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        "--pdb_path",
        "inputs/target.pdb",
        "--out_folder",
        "outputs/designs/seed_7",
        "--seed",
        "7",
        "--temperature",
        "0.2",
        "--batch_size",
        "2",
        "--number_of_batches",
        "3",
        "--ligand_mpnn_use_atom_context",
        "0",
        "--ligand_mpnn_use_side_chain_context",
        "1",
        "--fixed_residues",
        "A12 A13B",
        "--pack_side_chains",
        "1",
        "--number_of_packs_per_design",
        "4",
        "--repack_everything",
        "0",
        "--pack_with_ligand_context",
        "1",
        "--checkpoint_path_sc",
        "/opt/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt",
    )
    assert commands[1].seed == 11
    assert commands[1].argv[commands[1].argv.index("--seed") + 1] == "11"
    assert request.expected_sequence_count == 12


def test_redesigned_residues_use_the_distinct_official_flag() -> None:
    request = _request(
        fixed_residues=(),
        redesigned_residues=(LigandMpnnResidue(chain_id="B", residue_number=-2, insertion_code="A"),),
        packing=LigandMpnnPackingConfig(),
    )

    argv = build_ligandmpnn_commands(request, checkout_root=Path("tool"))[0].argv

    assert "--fixed_residues" not in argv
    assert argv[argv.index("--redesigned_residues") + 1] == "B-2A"
    assert argv[argv.index("--pack_side_chains") + 1] == "0"


def test_command_preserves_requested_temperature_precision() -> None:
    request = _request(temperature=0.123456789)

    argv = build_ligandmpnn_commands(request, checkout_root=Path("tool"))[0].argv

    assert argv[argv.index("--temperature") + 1] == "0.123456789"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seeds": (0,)}, "seeds must contain positive integers"),
        ({"temperature": float("nan")}, "temperature must be finite and positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        (
            {"redesigned_residues": (LigandMpnnResidue("B", 2),)},
            "fixed_residues and redesigned_residues are mutually exclusive",
        ),
        (
            {"fixed_residues": (LigandMpnnResidue("A", 12), LigandMpnnResidue("A", 12))},
            "fixed_residues contains duplicate residue A12",
        ),
    ],
)
def test_request_rejects_ambiguous_or_nondeterministic_inputs(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


def test_residue_identifier_rejects_non_pdb_chain_or_insertion_codes() -> None:
    with pytest.raises(ValueError, match="chain_id must be one ASCII alphanumeric character"):
        LigandMpnnResidue(chain_id="AA", residue_number=1)
    with pytest.raises(ValueError, match="insertion_code must be one ASCII letter"):
        LigandMpnnResidue(chain_id="A", residue_number=1, insertion_code="BC")
    with pytest.raises(ValueError, match="insertion_code must be one ASCII letter"):
        LigandMpnnResidue(chain_id="A", residue_number=1, insertion_code="2")


@pytest.mark.parametrize("chain_id", ["é", "１", "١"])
def test_residue_identifier_rejects_non_ascii_alphanumeric_chain_ids(chain_id: str) -> None:
    with pytest.raises(ValueError, match="chain_id must be one ASCII alphanumeric character"):
        LigandMpnnResidue(chain_id=chain_id, residue_number=12)


@pytest.mark.parametrize("chain_id", ["A", "z", "0"])
def test_residue_identifier_preserves_ascii_alphanumeric_chain_ids(chain_id: str) -> None:
    assert LigandMpnnResidue(chain_id=chain_id, residue_number=12).upstream_id == f"{chain_id}12"


def test_planned_receipt_is_normalized_and_records_no_execution_claim(tmp_path: Path) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_sha256 = _write_context_input(tmp_path)
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=True,
        parser_sha256=parser_sha256,
    )
    request = _request(
        pdb_sha256=pdb_sha256,
        context_inventory=context_inventory,
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
    )
    commands = build_ligandmpnn_commands(request, checkout_root=checkout_root)

    receipt = build_planned_receipt(
        request,
        commands,
        execution_root=tmp_path,
        checkout_root=checkout_root,
    )

    payload = receipt.to_dict()
    assert payload["schema_id"] == "thread.ligandmpnn.run_receipt"
    assert payload["status"] == "planned_not_run"
    assert payload["model_type"] == "ligand_mpnn"
    assert payload["schema_version"] == 2
    assert payload["expected_sequence_count"] == 12
    assert payload["provenance"] == {
        "upstream_repository": "https://github.com/dauparas/LigandMPNN",
        "upstream_commit": commit,
        "checkpoint_sha256": f"sha256:{_DIGEST}",
        "packing_checkpoint_sha256": f"sha256:{_PACKING_DIGEST}",
    }
    assert payload["commands"][0]["argv"][0] == "python"
    assert payload["input"] == {"path": "inputs/target.pdb", "sha256": f"sha256:{pdb_sha256}"}
    assert payload["context_inventory"] == {
        "path": "evidence/context-inventory.json",
        "sha256": f"sha256:{context_inventory.sha256}",
    }


def test_planned_receipt_rejects_missing_or_partial_command_sets(tmp_path: Path) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_sha256 = _write_context_input(tmp_path)
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=True,
        parser_sha256=parser_sha256,
    )
    request = _request(
        pdb_sha256=pdb_sha256,
        context_inventory=context_inventory,
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
    )
    commands = build_ligandmpnn_commands(request, checkout_root=checkout_root)

    for supplied in ((), commands[:-1]):
        with pytest.raises(ValueError, match="commands do not match the deterministic request command set"):
            build_planned_receipt(
                request,
                supplied,
                execution_root=tmp_path,
                checkout_root=checkout_root,
            )


@pytest.mark.parametrize(
    ("input_path", "input_sha256", "upstream_commit", "parse_all_atoms", "message"),
    [
        (
            Path("inputs/other.pdb"),
            "d" * 64,
            _COMMIT,
            True,
            "context inventory input identity does not match request",
        ),
        (
            Path("inputs/target.pdb"),
            _DIGEST,
            "d" * 40,
            True,
            "context inventory upstream commit does not match request",
        ),
        (
            Path("inputs/target.pdb"),
            _DIGEST,
            _COMMIT,
            False,
            "context inventory parse_all_atoms does not match side-chain-context mode",
        ),
    ],
)
def test_planned_receipt_rejects_context_inventory_for_different_request(
    tmp_path: Path,
    input_path: Path,
    input_sha256: str,
    upstream_commit: str,
    parse_all_atoms: bool,
    message: str,
) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    inventory_commit = commit if upstream_commit == _COMMIT else upstream_commit
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=input_path,
        input_sha256=input_sha256,
        upstream_commit=inventory_commit,
        parse_all_atoms=parse_all_atoms,
        parser_sha256=parser_sha256,
    )
    request = _request(
        context_inventory=context_inventory,
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
    )
    commands = build_ligandmpnn_commands(request, checkout_root=checkout_root)

    with pytest.raises(ValueError, match=message):
        build_planned_receipt(
            request,
            commands,
            execution_root=tmp_path,
            checkout_root=checkout_root,
        )


def test_planned_receipt_rejects_inventory_from_different_parser_blob(tmp_path: Path) -> None:
    checkout_root, commit, _parser_sha256 = create_pinned_context_checkout(tmp_path)
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=_DIGEST,
        upstream_commit=commit,
        parse_all_atoms=True,
        parser_sha256="d" * 64,
    )
    request = _request(
        context_inventory=context_inventory,
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
    )
    commands = build_ligandmpnn_commands(request, checkout_root=checkout_root)

    with pytest.raises(ValueError, match="parser digest does not match pinned upstream commit"):
        build_planned_receipt(
            request,
            commands,
            execution_root=tmp_path,
            checkout_root=checkout_root,
        )


def test_planned_receipt_rejects_self_asserted_context_atoms(tmp_path: Path) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_payload = b"ATOM pinned context input\n"
    pdb_path = tmp_path / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    pdb_sha256 = hashlib.sha256(pdb_payload).hexdigest()
    reference = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=True,
        parser_sha256=parser_sha256,
    )
    inventory = load_ligandmpnn_context_inventory(reference, execution_root=tmp_path)
    forged_atom = replace(
        inventory.atoms[0],
        serial=2,
        atom_name="O5'",
        element="O",
        upstream_element_type=8,
    )
    forged_inventory = replace(inventory, atoms=(*inventory.atoms, forged_atom))
    forged_payload = (json.dumps(forged_inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    (tmp_path / reference.path).write_bytes(forged_payload)
    forged_reference = LigandMpnnContextInventoryReference(
        path=reference.path,
        sha256=hashlib.sha256(forged_payload).hexdigest(),
    )
    request = _request(
        pdb_sha256=pdb_sha256,
        context_inventory=forged_reference,
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=_DIGEST,
            packing_checkpoint_sha256=_PACKING_DIGEST,
        ),
    )
    commands = build_ligandmpnn_commands(request, checkout_root=checkout_root)

    with pytest.raises(ValueError, match="context inventory does not match pinned parser derivation"):
        build_planned_receipt(
            request,
            commands,
            execution_root=tmp_path,
            checkout_root=checkout_root,
        )
