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
    LigandMpnnResidue,
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


def test_score_plan_binds_final_root_after_validating_staged_inputs(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout_root = _validated_request(staging)
    # The producer checkout is external to the directory being promoted.
    checkout = tmp_path / "checkout"
    checkout_root.rename(checkout)
    final = tmp_path / "final"
    planned = build_ligandmpnn_score_commands(request, checkout_root=checkout, execution_root=final, input_root=staging)
    assert str(staging) not in planned[0].argv
    staging.rename(final)
    replayed = build_ligandmpnn_score_commands(request, checkout_root=checkout, execution_root=final)
    assert planned == replayed


@pytest.mark.parametrize("python_executable", ["venv/bin/python", "./python"])
def test_staged_score_plan_rejects_relative_interpreter_paths_independent_of_planner_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python_executable: str
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    interpreter = staging / python_executable
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("placeholder")
    planner = tmp_path / "planner"
    planner.mkdir()
    monkeypatch.chdir(planner)

    with pytest.raises(ValueError, match="staged planning requires an absolute interpreter"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=external_checkout,
            execution_root=tmp_path / "final",
            input_root=staging,
            python_executable=python_executable,
        )


@pytest.mark.parametrize("interpreter_kind", ["path_command", "absolute_external"])
def test_staged_score_plan_preserves_explicit_external_interpreter_after_promotion(
    tmp_path: Path, interpreter_kind: str
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    python_executable = "python3"
    if interpreter_kind == "absolute_external":
        interpreter = tmp_path / "external-venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("placeholder")
        python_executable = str(interpreter)
    final = tmp_path / "final"
    planned = build_ligandmpnn_score_commands(
        request,
        checkout_root=external_checkout,
        execution_root=final,
        input_root=staging,
        python_executable=python_executable,
    )
    assert planned[0].argv[0] == python_executable
    staging.rename(final)
    replayed = build_ligandmpnn_score_commands(
        request,
        checkout_root=external_checkout,
        execution_root=final,
        python_executable=python_executable,
    )
    assert planned == replayed


@pytest.mark.parametrize("explicit_input_root", [False, True])
def test_unstaged_score_plan_retains_relative_interpreter_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_input_root: bool
) -> None:
    request, checkout = _validated_request(tmp_path)
    monkeypatch.chdir(tmp_path)
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout,
        execution_root=tmp_path,
        input_root=tmp_path if explicit_input_root else None,
        python_executable="venv/bin/python",
    )
    assert commands[0].argv[0] == "venv/bin/python"


def test_score_staging_does_not_bypass_input_digest_validation(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    (staging / request.pdb_path).write_text("changed\n")
    with pytest.raises(ValueError, match="SHA256|digest"):
        build_ligandmpnn_score_commands(
            request, checkout_root=external_checkout, execution_root=tmp_path / "final", input_root=staging
        )


@pytest.mark.parametrize("relative", [True, False])
def test_staged_score_plan_rejects_interpreter_under_moving_input_root(tmp_path, monkeypatch, relative):
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    executable = staging / "bin/python"
    executable.parent.mkdir()
    executable.write_text("placeholder")
    monkeypatch.chdir(tmp_path)
    python_executable = str(executable.relative_to(tmp_path) if relative else executable)
    with pytest.raises(ValueError, match="interpreter.*input_root"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=external_checkout,
            execution_root=tmp_path / "final",
            input_root=staging,
            python_executable=python_executable,
        )


@pytest.mark.parametrize("field", ["checkout", "interpreter"])
def test_staged_score_plan_rejects_external_target_reached_through_moving_symlink(tmp_path, field):
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    alias_root = tmp_path / "stage-alias"
    alias_root.symlink_to(staging, target_is_directory=True)
    link = alias_root / "external-link"
    link.symlink_to(external_checkout, target_is_directory=True)
    kwargs = {"checkout_root": external_checkout, "python_executable": "python"}
    if field == "checkout":
        kwargs["checkout_root"] = link
    else:
        kwargs["python_executable"] = str(link / "python")
    with pytest.raises(ValueError, match="outside input_root"):
        build_ligandmpnn_score_commands(request, execution_root=tmp_path / "final", input_root=staging, **kwargs)


@pytest.mark.parametrize("field", ["checkout", "interpreter"])
@pytest.mark.parametrize("route", ["direct", "root_alias", "external_target"])
def test_staged_score_plan_rejects_runtime_paths_through_final_root(tmp_path, field, route):
    staging = tmp_path / "staging"
    staging.mkdir()
    request, checkout = _validated_request(staging)
    external_checkout = tmp_path / "checkout"
    checkout.rename(external_checkout)
    final = tmp_path / "final"
    runtime_path = final / ("checkout" if field == "checkout" else "venv/bin/python")
    if field == "checkout":
        final.mkdir()
        if route == "external_target":
            runtime_path.symlink_to(external_checkout, target_is_directory=True)
        else:
            external_checkout.rename(runtime_path)
    elif route == "external_target":
        final.mkdir()
        external_venv = tmp_path / "external-venv"
        external_venv.mkdir()
        (final / "venv").symlink_to(external_venv, target_is_directory=True)
    else:
        moving_interpreter = staging / "venv/bin/python"
        moving_interpreter.parent.mkdir(parents=True)
        moving_interpreter.write_text("placeholder")
    if route == "root_alias":
        alias = tmp_path / "final-alias"
        alias.symlink_to(final, target_is_directory=True)
        runtime_path = alias / runtime_path.relative_to(final)
    kwargs = {"checkout_root": external_checkout, "python_executable": "python"}
    kwargs["checkout_root" if field == "checkout" else "python_executable"] = (
        runtime_path if field == "checkout" else str(runtime_path)
    )
    with pytest.raises(ValueError, match=f"{field}.*outside input_root and execution_root"):
        build_ligandmpnn_score_commands(request, execution_root=final, input_root=staging, **kwargs)


def test_score_request_preserves_dot_output_as_an_execution_root_seed_directory(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path, output_dir=Path("."))

    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
    )[0]

    assert command.output_dir == Path("seed_7")
    assert command.argv[command.argv.index("--out_folder") + 1] == "seed_7"


def test_score_commands_reject_checkout_nested_inside_per_seed_output(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path)
    nested_checkout = tmp_path / "outputs/scores/seed_7/LigandMPNN"
    nested_checkout.parent.mkdir(parents=True)
    checkout_root.rename(nested_checkout)

    with pytest.raises(ValueError, match="checkout_root.*per-seed output"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=nested_checkout,
            execution_root=tmp_path,
        )


def test_score_commands_reject_checkout_symlinked_inside_per_seed_output(tmp_path: Path) -> None:
    request, checkout_root = _validated_request(tmp_path)
    nested_checkout = tmp_path / "outputs/scores/seed_7/LigandMPNN"
    nested_checkout.parent.mkdir(parents=True)
    checkout_root.rename(nested_checkout)
    checkout_alias = tmp_path / "checkout-alias"
    checkout_alias.symlink_to(nested_checkout, target_is_directory=True)

    with pytest.raises(ValueError, match="checkout_root.*per-seed output"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=checkout_alias,
            execution_root=tmp_path,
        )


def test_score_commands_reject_context_inventory_at_planned_completion_leaf(tmp_path: Path) -> None:
    checkout_root, commit, parser_sha256 = create_pinned_context_checkout(tmp_path)
    pdb_payload = b"ATOM pinned score input\n"
    pdb_path = tmp_path / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    pdb_sha256 = hashlib.sha256(pdb_payload).hexdigest()
    completion_path = Path("outputs/scores/seed_7/.dnadesign-ligandmpnn-execution.json")
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=True,
        parser_sha256=parser_sha256,
        relative_path=completion_path,
    )
    request = _request(
        pdb_sha256=pdb_sha256,
        upstream=LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_DIGEST),
        context_inventory=context_inventory,
    )

    with pytest.raises(ValueError, match="context inventory path.*per-seed output"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=checkout_root,
            execution_root=tmp_path,
        )


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
        "--request-id",
        request.request_id,
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


@pytest.mark.parametrize("field_name", ["fixed_residues", "redesigned_residues"])
def test_score_commands_reject_selectors_absent_from_pinned_parser_protein_identities(
    tmp_path: Path,
    field_name: str,
) -> None:
    request, checkout_root = _validated_request(
        tmp_path,
        **{field_name: (LigandMpnnResidue("A", 13, "A"),)},
    )

    with pytest.raises(ValueError, match=rf"{field_name}.*A13A.*not present"):
        build_ligandmpnn_score_commands(request, checkout_root=checkout_root, execution_root=tmp_path)


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
