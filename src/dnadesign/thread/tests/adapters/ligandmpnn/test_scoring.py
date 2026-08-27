"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_scoring.py

Official LigandMPNN probability-scoring command tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnContextInventoryReference,
    LigandMpnnScoreMode,
    LigandMpnnScoreRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_score_commands,
)
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import (
    create_pinned_context_checkout,
    write_context_inventory,
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


def _validated_request(tmp_path: Path, **overrides: object) -> tuple[LigandMpnnScoreRequest, Path]:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_payload = b"ATOM pinned score input\n"
    pdb_path = tmp_path / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    pdb_sha256 = hashlib.sha256(pdb_payload).hexdigest()
    use_side_chain_context = bool(overrides.get("use_side_chain_context", True))
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=use_side_chain_context,
        parser_sha256=parser_sha256,
    )
    values: dict[str, object] = {
        "pdb_sha256": pdb_sha256,
        "upstream": LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_DIGEST),
        "context_inventory": context_inventory,
    }
    values.update(overrides)
    return _request(**values), checkout_root


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("pdb_path", Path("/tmp/target.pdb"), "safe non-option relative"),
        ("pdb_path", Path("~/target.pdb"), "safe non-option relative"),
        ("pdb_path", Path("-option-like-input.pdb"), "safe non-option relative"),
        ("output_dir", Path("/tmp/scores"), "safe non-option relative"),
        ("output_dir", Path("~/scores"), "safe non-option relative"),
        ("output_dir", Path("-option-like-output"), "must not begin with a hyphen"),
        ("output_dir", Path("results/../scores"), "must not contain traversal"),
    ],
)
def test_score_request_rejects_paths_that_cannot_round_trip_through_runtime_argv(
    field_name: str,
    value: Path,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**{field_name: value})


def test_score_request_preserves_valid_nested_relative_output_directory(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path, output_dir=Path("results/nested/scores"))

    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
    )[0]

    assert command.output_dir == Path("results/nested/scores/seed_7")
    assert command.argv[command.argv.index("--out_folder") + 1] == "results/nested/scores/seed_7"


def test_score_request_preserves_dot_output_as_an_execution_root_seed_directory(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path, output_dir=Path("."))

    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
    )[0]

    assert command.output_dir == Path("seed_7")
    assert command.argv[command.argv.index("--out_folder") + 1] == "seed_7"


def test_single_aa_probability_command_is_explicit(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path)
    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
        python_executable="python3",
    )[0]

    planned_execution_sha256 = command.argv[command.argv.index("--planned-execution-sha256") + 1]
    assert len(planned_execution_sha256) == 64
    assert command.argv == (
        "python3",
        "-m",
        "dnadesign.thread.adapters.ligandmpnn.pinned_runtime",
        "--checkout-root",
        str(checkout_root),
        "--upstream-commit",
        request.upstream.commit,
        "--checkpoint-sha256",
        _DIGEST,
        "--pdb-sha256",
        request.pdb_sha256,
        "--execution-root",
        str(tmp_path),
        "--context-inventory-path",
        request.context_inventory.path.as_posix(),
        "--context-inventory-sha256",
        request.context_inventory.sha256,
        "--planned-execution-sha256",
        planned_execution_sha256,
        "--completion-record",
        "outputs/scores/seed_7/.dnadesign-ligandmpnn-execution.json",
        "--entrypoint",
        "score.py",
        "--",
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        str(checkout_root / "model_params/ligandmpnn_v_32_010_25.pt"),
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


def test_autoregressive_probability_mode_sets_exclusive_official_flags(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path, mode=LigandMpnnScoreMode.AUTOREGRESSIVE)
    argv = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
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


@pytest.mark.parametrize("seeds", [(-1,), (0,), (2**32,), (True,), (1.5,)])
def test_score_request_rejects_seeds_outside_upstream_deterministic_domain(seeds: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="integers from 1 through 4294967295"):
        _request(seeds=seeds)


@pytest.mark.parametrize("seeds", [(), [1]])
def test_score_request_requires_nonempty_seed_tuple(seeds: object) -> None:
    with pytest.raises(ValueError, match="nonempty tuple"):
        _request(seeds=seeds)


def test_score_request_emits_deterministic_seed_boundaries(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path, seeds=(1, 2**32 - 1))
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
    )

    assert [command.seed for command in commands] == [1, 4294967295]
    assert [command.argv[command.argv.index("--seed") + 1] for command in commands] == ["1", "4294967295"]
