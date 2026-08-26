"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_scoring.py

Official LigandMPNN probability-scoring command tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnScoreMode,
    LigandMpnnScoreRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_score_commands,
)

_DIGEST = "a" * 64
_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"  # pragma: allowlist secret


def _request(**overrides: object) -> LigandMpnnScoreRequest:
    values: dict[str, object] = {
        "request_id": "generic_context_probe",
        "pdb_path": Path("inputs/target.pdb"),
        "pdb_sha256": _DIGEST,
        "output_dir": Path("outputs/scores"),
        "upstream": LigandMpnnUpstreamPin(commit=_COMMIT, checkpoint_sha256=_DIGEST),
        "context_inventory": LigandMpnnContextInventoryReference(
            path=Path("evidence/context-inventory.json"), sha256=_DIGEST
        ),
        "seeds": (7,),
        "batch_size": 2,
        "number_of_batches": 10,
        "mode": LigandMpnnScoreMode.SINGLE_AA,
        "use_sequence": False,
        "use_atom_context": False,
        "use_side_chain_context": True,
    }
    values.update(overrides)
    return LigandMpnnScoreRequest(**values)  # type: ignore[arg-type]


def test_single_aa_probability_command_is_explicit() -> None:
    command = build_ligandmpnn_score_commands(
        _request(), checkout_root=Path("/opt/LigandMPNN"), python_executable="python3"
    )[0]

    assert command.argv == (
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
        "--entrypoint",
        "score.py",
        "--",
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        "/opt/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        "--pdb_path",
        "inputs/target.pdb",
        "--out_folder",
        "outputs/scores/seed_7",
        "--seed",
        "7",
        "--batch_size",
        "2",
        "--number_of_batches",
        "10",
        "--ligand_mpnn_use_atom_context",
        "0",
        "--ligand_mpnn_use_side_chain_context",
        "1",
        "--use_sequence",
        "0",
        "--autoregressive_score",
        "0",
        "--single_aa_score",
        "1",
    )


def test_autoregressive_probability_mode_sets_exclusive_official_flags() -> None:
    argv = build_ligandmpnn_score_commands(
        _request(mode=LigandMpnnScoreMode.AUTOREGRESSIVE), checkout_root=Path("tool")
    )[0].argv
    assert argv[argv.index("--autoregressive_score") + 1] == "1"
    assert argv[argv.index("--single_aa_score") + 1] == "0"


def test_score_request_enforces_upstream_minimum_batch_policy() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        _request(number_of_batches=9)
    with pytest.raises(ValueError, match="LigandMpnnScoreMode"):
        _request(mode="single_aa")
    with pytest.raises(ValueError, match="LigandMpnnUpstreamPin"):
        _request(upstream="unpinned")
    with pytest.raises(ValueError, match="pdb_sha256"):
        _request(pdb_sha256="not-a-digest")
