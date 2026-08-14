"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_adapter.py

Behavior tests for LigandMPNN request and command adaptation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
)

_DIGEST = "a" * 64
_PACKING_DIGEST = "b" * 64
_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"  # pragma: allowlist secret


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


def test_build_commands_declares_exact_official_ligandmpnn_flags_per_seed() -> None:
    request = _request()

    commands = build_ligandmpnn_commands(
        request,
        checkout_root=Path("/opt/LigandMPNN"),
        python_executable="python3",
    )

    assert len(commands) == 2
    assert commands[0].argv == (
        "python3",
        "/opt/LigandMPNN/run.py",
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
    with pytest.raises(ValueError, match="chain_id must be one alphanumeric character"):
        LigandMpnnResidue(chain_id="AA", residue_number=1)
    with pytest.raises(ValueError, match="insertion_code must be one alphanumeric character"):
        LigandMpnnResidue(chain_id="A", residue_number=1, insertion_code="BC")


def test_planned_receipt_is_normalized_and_records_no_execution_claim() -> None:
    request = _request()
    commands = build_ligandmpnn_commands(request, checkout_root=Path("tool"))

    receipt = build_planned_receipt(request, commands)

    payload = receipt.to_dict()
    assert payload["schema_id"] == "thread.ligandmpnn.run_receipt"
    assert payload["status"] == "planned_not_run"
    assert payload["model_type"] == "ligand_mpnn"
    assert payload["schema_version"] == 2
    assert payload["expected_sequence_count"] == 12
    assert payload["provenance"] == {
        "upstream_repository": "https://github.com/dauparas/LigandMPNN",
        "upstream_commit": _COMMIT,
        "checkpoint_sha256": f"sha256:{_DIGEST}",
        "packing_checkpoint_sha256": f"sha256:{_PACKING_DIGEST}",
    }
    assert payload["commands"][0]["argv"][0] == "python"
    assert payload["input"] == {"path": "inputs/target.pdb", "sha256": f"sha256:{_DIGEST}"}
    assert payload["context_inventory"] == {
        "path": "evidence/context-inventory.json",
        "sha256": f"sha256:{_DIGEST}",
    }
