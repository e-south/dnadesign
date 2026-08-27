"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_score_results.py

Executed LigandMPNN probability-result boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dnadesign.thread.adapters.ligandmpnn.score_results as score_results_module
from dnadesign.thread.adapters.ligandmpnn import (
    EXPECTED_LIGANDMPNN_SCORE_ALPHABET,
    LigandMpnnCanonical20Policy,
    LigandMpnnContextInventoryReference,
    LigandMpnnResidue,
    LigandMpnnScoreMode,
    LigandMpnnScoreOutputTrust,
    LigandMpnnScoreRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_score_commands,
    parse_ligandmpnn_score_outputs,
    score_request_sha256,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import pinned_runtime_completion_contract
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import (
    create_pinned_context_checkout,
    write_context_inventory,
)

_CHECKPOINT_SHA256 = "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepare_request(
    root: Path,
    *,
    seeds: tuple[int, ...] = (7, 11),
    fixed_residues: tuple[LigandMpnnResidue, ...] = (),
    redesigned_residues: tuple[LigandMpnnResidue, ...] = (),
) -> LigandMpnnScoreRequest:
    _checkout, commit, parser_sha256 = create_pinned_context_checkout(root)
    pdb_payload = b"ATOM      1  N   ALA A   1      0.000   0.000   0.000\n"
    pdb_path = root / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    context_inventory = _write_context_inventory(
        root,
        pdb_sha256=_sha256(pdb_payload),
        upstream_commit=commit,
        parser_sha256=parser_sha256,
    )
    return LigandMpnnScoreRequest(
        request_id="generic_context_probe",
        pdb_path=Path("inputs/target.pdb"),
        pdb_sha256=_sha256(pdb_payload),
        output_dir=Path("outputs/scores"),
        upstream=LigandMpnnUpstreamPin(commit=commit, checkpoint_sha256=_CHECKPOINT_SHA256),
        context_inventory=context_inventory,
        fixed_residues=fixed_residues,
        redesigned_residues=redesigned_residues,
        seeds=seeds,
        batch_size=2,
        number_of_batches=10,
        mode=LigandMpnnScoreMode.SINGLE_AA,
        use_sequence=False,
        use_atom_context=True,
        use_side_chain_context=False,
    )


def _prepare_executable_dot_output_request(root: Path) -> tuple[LigandMpnnScoreRequest, Path]:
    vendor_root = root / "vendor"
    vendor_root.mkdir()
    checkout, _initial_commit, parser_sha256 = create_pinned_context_checkout(vendor_root)
    checkpoint = checkout / "model_params/ligandmpnn_v_32_010_25.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("checkpoint-v1", encoding="utf-8")
    (checkout / "score.py").write_text(
        "import argparse\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--pdb_path', required=True)\n"
        "parser.add_argument('--out_folder', required=True)\n"
        "args, _ = parser.parse_known_args()\n"
        "output = Path(args.out_folder) / f'{Path(args.pdb_path).stem}.pt'\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copyfile(args.pdb_path, output)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(checkout), "add", "score.py", checkpoint.relative_to(checkout)], check=True)
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
            "score fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    pdb_path = root / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    torch.save(_score_payload(7), pdb_path)
    pdb_sha256 = _sha256(pdb_path.read_bytes())
    context_inventory = _write_context_inventory(
        root,
        pdb_sha256=pdb_sha256,
        upstream_commit=commit,
        parser_sha256=parser_sha256,
    )
    return (
        LigandMpnnScoreRequest(
            request_id="dot_output_score",
            pdb_path=pdb_path.relative_to(root),
            pdb_sha256=pdb_sha256,
            output_dir=Path("."),
            upstream=LigandMpnnUpstreamPin(
                commit=commit,
                checkpoint_sha256=_sha256(checkpoint.read_bytes()),
                checkpoint_path=checkpoint.relative_to(checkout),
            ),
            context_inventory=context_inventory,
            seeds=(7,),
            batch_size=2,
            number_of_batches=10,
            mode=LigandMpnnScoreMode.SINGLE_AA,
            use_sequence=False,
            use_atom_context=True,
            use_side_chain_context=False,
        ),
        checkout,
    )


def _write_context_inventory(
    root: Path,
    *,
    pdb_sha256: str,
    upstream_commit: str,
    parser_sha256: str,
) -> LigandMpnnContextInventoryReference:
    return write_context_inventory(
        root,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=upstream_commit,
        parser_sha256=parser_sha256,
        parse_all_atoms=False,
    )


def _score_payload(
    seed: int,
    *,
    draws: int = 20,
    mode: LigandMpnnScoreMode = LigandMpnnScoreMode.SINGLE_AA,
) -> dict[str, object]:
    residue_names = {0: "A12", 1: "A13B", 2: "B-2A", 3: "B2"}
    residue_count = len(residue_names)
    raw_probabilities = np.full((draws, residue_count, 21), 0.95 / 20.0, dtype=np.float32)
    raw_probabilities[..., -1] = 0.05
    log_probabilities = np.log(raw_probabilities).astype(np.float32)
    means = np.mean(raw_probabilities, axis=0)
    standard_deviations = np.std(raw_probabilities, axis=0)
    if mode is LigandMpnnScoreMode.SINGLE_AA:
        decoding_order = np.tile(
            np.arange(residue_count, dtype=np.float32),
            (draws, residue_count, 1),
        )
    else:
        decoding_order = np.tile(np.arange(residue_count, dtype=np.int64), (draws, 1))
    return {
        "logits": log_probabilities.copy(),
        "probs": raw_probabilities,
        "log_probs": log_probabilities,
        "decoding_order": decoding_order,
        "native_sequence": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "mask": np.asarray([1, 0, 1, 1], dtype=np.float32),
        "chain_mask": np.ones(residue_count, dtype=np.int64),
        "seed": seed,
        "alphabet": list(EXPECTED_LIGANDMPNN_SCORE_ALPHABET),
        "residue_names": residue_names,
        "sequence": ["A", "C", "D", "E"],
        "mean_of_probs": {
            residue_names[index]: dict(zip(EXPECTED_LIGANDMPNN_SCORE_ALPHABET, means[index], strict=True))
            for index in range(residue_count)
        },
        "std_of_probs": {
            residue_names[index]: dict(zip(EXPECTED_LIGANDMPNN_SCORE_ALPHABET, standard_deviations[index], strict=True))
            for index in range(residue_count)
        },
    }


def _score_payload_with_residue_names(seed: int, names: tuple[str, ...]) -> dict[str, object]:
    payload = _score_payload(seed)
    original_names = tuple(payload["residue_names"].values())  # type: ignore[union-attr]
    payload["residue_names"] = dict(enumerate(names))
    for field_name in ("mean_of_probs", "std_of_probs"):
        original = payload[field_name]
        assert isinstance(original, dict)
        payload[field_name] = {
            name: original[original_name] for name, original_name in zip(names, original_names, strict=True)
        }
    return payload


def _truncated_score_payload(seed: int) -> dict[str, object]:
    payload = _score_payload(seed)
    residue_count = 2
    for field_name in ("logits", "probs", "log_probs"):
        payload[field_name] = payload[field_name][:, :residue_count, :]  # type: ignore[index]
    payload["decoding_order"] = np.tile(
        np.arange(residue_count, dtype=np.float32),
        (20, residue_count, 1),
    )
    for field_name in ("native_sequence", "mask", "chain_mask"):
        payload[field_name] = payload[field_name][:residue_count]  # type: ignore[index]
    payload["sequence"] = payload["sequence"][:residue_count]  # type: ignore[index]
    names = tuple(payload["residue_names"][index] for index in range(residue_count))  # type: ignore[index]
    payload["residue_names"] = dict(enumerate(names))
    for field_name in ("mean_of_probs", "std_of_probs"):
        payload[field_name] = {name: payload[field_name][name] for name in names}  # type: ignore[index]
    return payload


def _write_output(
    root: Path,
    request: LigandMpnnScoreRequest,
    expected_seed: int,
    **payload_overrides: object,
) -> Path:
    payload = _score_payload(expected_seed, mode=request.mode)
    payload.update(payload_overrides)
    path = root / request.output_dir / f"seed_{expected_seed}" / f"{request.pdb_path.stem}.pt"
    path.parent.mkdir(parents=True)
    torch.save(payload, path)
    command = next(
        command
        for command in build_ligandmpnn_score_commands(
            request,
            checkout_root=root / "LigandMPNN",
            execution_root=root,
        )
        if command.seed == expected_seed
    )
    completion_path, completion = pinned_runtime_completion_contract(
        command.argv,
        upstream_commit=request.upstream.commit,
        checkpoint_sha256=request.upstream.checkpoint_sha256,
        pdb_sha256=request.pdb_sha256,
        request_id=request.request_id,
        context_inventory_path=request.context_inventory.path,
        context_inventory_sha256=request.context_inventory.sha256,
        execution_root=root,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        score_output_sha256=f"sha256:{_sha256(path.read_bytes())}",
    )
    absolute_completion_path = completion_path if completion_path.is_absolute() else root / completion_path
    absolute_completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse(root: Path, request: LigandMpnnScoreRequest):
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=root / "LigandMPNN",
        execution_root=root,
    )
    return parse_ligandmpnn_score_outputs(
        request,
        commands,
        execution_root=root,
        trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
    )


@pytest.mark.parametrize("mutation", ["missing", "forged", "mismatched", "tampered"])
def test_score_builder_rejects_invalid_context_before_command_emission(tmp_path: Path, mutation: str) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    inventory_path = tmp_path / request.context_inventory.path
    if mutation == "missing":
        inventory_path.unlink()
    elif mutation == "forged":
        request = replace(
            request,
            context_inventory=LigandMpnnContextInventoryReference(
                path=request.context_inventory.path,
                sha256="f" * 64,
            ),
        )
    elif mutation == "mismatched":
        request = replace(request, pdb_sha256="e" * 64)
    else:
        inventory_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="context inventory|input SHA256"):
        build_ligandmpnn_score_commands(
            request,
            checkout_root=tmp_path / "LigandMPNN",
            execution_root=tmp_path,
        )


def test_score_relative_checkout_is_anchored_for_foreign_cwd_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=Path("LigandMPNN"),
        execution_root=tmp_path,
    )
    _write_output(tmp_path, request, 7)

    result = parse_ligandmpnn_score_outputs(
        request,
        commands,
        execution_root=tmp_path,
        trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
    )

    assert result.outputs[0].seed == 7
    assert commands[0].argv[commands[0].argv.index("--checkout-root") + 1] == str(tmp_path / "LigandMPNN")


@pytest.mark.parametrize("checkout_form", ["relative", "absolute"])
@pytest.mark.parametrize("caller_cwd", ["execution-root", "foreign"])
@pytest.mark.parametrize("workspace_artifact", ["checkout-checkpoint", "regular", "fifo"])
def test_dot_output_admission_ignores_workspace_pt_outside_command_seed_directories(
    tmp_path: Path,
    checkout_form: str,
    caller_cwd: str,
    workspace_artifact: str,
) -> None:
    request, checkout = _prepare_executable_dot_output_request(tmp_path)
    checkout_root = checkout.relative_to(tmp_path) if checkout_form == "relative" else checkout
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=tmp_path,
        python_executable=sys.executable,
    )
    assert (checkout / request.upstream.checkpoint_path).is_file()
    if workspace_artifact == "regular":
        (tmp_path / "unrelated-workspace.pt").write_text("not a score", encoding="utf-8")
    elif workspace_artifact == "fifo":
        os.mkfifo(tmp_path / "unrelated-workspace-fifo.pt")
    cwd = tmp_path
    if caller_cwd == "foreign":
        cwd = tmp_path / "foreign-cwd"
        cwd.mkdir()

    subprocess.run(commands[0].argv, cwd=cwd, check=True)
    result = parse_ligandmpnn_score_outputs(
        request,
        commands,
        execution_root=tmp_path,
        trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
    )

    assert [output.artifact_path for output in result.outputs] == [Path("seed_7/target.pt")]


def test_parser_binds_exact_request_commands_inputs_and_raw_probabilities(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path)
    for seed in request.seeds:
        _write_output(tmp_path, request, seed)

    result = _parse(tmp_path, request)

    assert result.request_sha256 == score_request_sha256(request)
    assert result.input_sha256 == f"sha256:{request.pdb_sha256}"
    assert result.provenance.upstream_commit == request.upstream.commit
    assert result.provenance.checkpoint_sha256 == f"sha256:{_CHECKPOINT_SHA256}"
    assert result.atom_context_requested is True
    assert result.atom_context_status == "enabled_with_observed_nucleotide_context"
    assert result.context_inventory.effective_nucleotide_atom_count == 2
    assert result.expected_draws_per_seed == 20
    assert [output.seed for output in result.outputs] == [7, 11]
    assert result.outputs[0].residue_names == ("A12", "A13B", "B-2A", "B2")
    assert all(output.raw_probabilities.shape == (20, 4, 21) for output in result.outputs)
    assert np.allclose(result.outputs[0].raw_x_probabilities, 0.05)
    assert not result.outputs[0].raw_probabilities.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.outputs[0].raw_probabilities.setflags(write=True)
    assert result.outputs[0].artifact_path == Path("outputs/scores/seed_7/target.pt")

    policy = LigandMpnnCanonical20Policy(minimum_canonical_mass=0.90)
    canonical = result.outputs[0].canonical20_probabilities(policy)
    assert canonical.shape == (20, 4, 20)
    assert np.allclose(canonical.sum(axis=-1), 1.0)
    assert not canonical.flags.writeable
    with pytest.raises(ValueError, match="minimum canonical mass"):
        result.outputs[0].canonical20_probabilities(LigandMpnnCanonical20Policy(minimum_canonical_mass=0.96))

    receipt = result.to_dict()
    assert receipt["schema_id"] == "thread.ligandmpnn.score_result"
    assert receipt["schema_version"] == 3
    assert receipt["status"] == "completed_validated"
    assert receipt["input"] == {
        "path": "inputs/target.pdb",
        "sha256": f"sha256:{request.pdb_sha256}",
    }
    assert receipt["outputs"][0]["command_sha256"].startswith("sha256:")
    assert receipt["outputs"][0]["execution_sha256"].startswith("sha256:")
    assert receipt["outputs"][0]["output_sha256"].startswith("sha256:")
    assert "raw_x_probability" in receipt["outputs"][0]
    assert receipt["context"]["atom_context_status"] == "enabled_with_observed_nucleotide_context"
    assert receipt["context"]["inventory_reference"] == request.context_inventory.to_dict()
    assert receipt["context"]["observed_inventory"]["observed"]["effective_nucleotide_atom_count"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        _truncated_score_payload(7),
        _score_payload_with_residue_names(7, ("A13B", "A12", "B-2A", "B2")),
        _score_payload_with_residue_names(7, ("A12", "A13B", "B-2A", "Z999")),
    ],
    ids=["truncated", "reordered", "mislabeled"],
)
def test_score_admission_rejects_residue_axis_not_identical_to_pinned_parser(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7, **payload)

    with pytest.raises(ValueError, match="pinned parser protein residue identities"):
        _parse(tmp_path, request)


def test_score_admission_rejects_native_sequence_not_matching_pinned_parser(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(
        tmp_path,
        request,
        7,
        native_sequence=np.asarray([1, 1, 2, 3], dtype=np.int64),
        sequence=["C", "C", "D", "E"],
    )

    with pytest.raises(ValueError, match="native sequence does not match pinned parser"):
        _parse(tmp_path, request)


@pytest.mark.parametrize(
    ("fixed_residues", "redesigned_residues", "observed_chain_mask"),
    [
        ((LigandMpnnResidue("A", 13, "B"),), (), np.asarray([1, 1, 1, 1], dtype=np.int64)),
        ((), (LigandMpnnResidue("B", -2, "A"),), np.asarray([1, 1, 1, 1], dtype=np.int64)),
    ],
    ids=["fixed", "redesigned"],
)
def test_score_admission_rejects_chain_mask_not_matching_bound_selectors(
    tmp_path: Path,
    fixed_residues: tuple[LigandMpnnResidue, ...],
    redesigned_residues: tuple[LigandMpnnResidue, ...],
    observed_chain_mask: np.ndarray,
) -> None:
    request = _prepare_request(
        tmp_path,
        seeds=(7,),
        fixed_residues=fixed_residues,
        redesigned_residues=redesigned_residues,
    )
    _write_output(tmp_path, request, 7, chain_mask=observed_chain_mask)

    with pytest.raises(ValueError, match="chain_mask does not match pinned parser and requested selectors"):
        _parse(tmp_path, request)


@pytest.mark.parametrize(
    ("fixed_residues", "redesigned_residues", "expected_chain_mask"),
    [
        ((LigandMpnnResidue("A", 13, "B"),), (), np.asarray([1, 0, 1, 1], dtype=np.int64)),
        ((), (LigandMpnnResidue("B", -2, "A"),), np.asarray([0, 0, 1, 0], dtype=np.int64)),
    ],
    ids=["fixed", "redesigned"],
)
def test_score_admission_accepts_chain_mask_matching_bound_selectors(
    tmp_path: Path,
    fixed_residues: tuple[LigandMpnnResidue, ...],
    redesigned_residues: tuple[LigandMpnnResidue, ...],
    expected_chain_mask: np.ndarray,
) -> None:
    request = _prepare_request(
        tmp_path,
        seeds=(7,),
        fixed_residues=fixed_residues,
        redesigned_residues=redesigned_residues,
    )
    _write_output(tmp_path, request, 7, chain_mask=expected_chain_mask)

    result = _parse(tmp_path, request)

    assert result.outputs[0].residue_names == ("A12", "A13B", "B-2A", "B2")


def test_score_admission_rejects_residue_mask_not_matching_pinned_parser(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7, mask=np.ones(4, dtype=np.int64))

    with pytest.raises(ValueError, match="mask does not match pinned parser residue validity"):
        _parse(tmp_path, request)


def test_score_admission_rejects_completed_execution_for_different_request_id(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=tmp_path / "LigandMPNN",
        execution_root=tmp_path,
    )
    _write_output(tmp_path, request, 7)
    replayed_request = replace(request, request_id="different_score_request")

    with pytest.raises(ValueError, match="commands do not exactly match score request"):
        parse_ligandmpnn_score_outputs(
            replayed_request,
            commands,
            execution_root=tmp_path,
            trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
        )


def test_parser_requires_exact_actual_execution_completion(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=tmp_path / "LigandMPNN",
        execution_root=tmp_path,
    )[0]
    completion_path, completion = pinned_runtime_completion_contract(
        command.argv,
        upstream_commit=request.upstream.commit,
        checkpoint_sha256=request.upstream.checkpoint_sha256,
        pdb_sha256=request.pdb_sha256,
        request_id=request.request_id,
        context_inventory_path=request.context_inventory.path,
        context_inventory_sha256=request.context_inventory.sha256,
        execution_root=tmp_path,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        score_output_sha256=f"sha256:{_sha256((tmp_path / request.output_dir / 'seed_7/target.pt').read_bytes())}",
    )
    assert completion["schema_version"] == 3
    assert completion["execution"]["request_id"] == request.request_id
    absolute_completion_path = tmp_path / completion_path
    absolute_completion_path.unlink()
    with pytest.raises(ValueError, match="execution completion record does not exist"):
        _parse(tmp_path, request)

    absolute_completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = json.loads(absolute_completion_path.read_text(encoding="utf-8"))
    payload["execution"]["arguments"].extend(["--unplanned_unique_override", "1"])
    absolute_completion_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution completion does not match planned command"):
        _parse(tmp_path, request)


def test_parser_rejects_valid_score_replaced_after_execution_completion(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    output_path = _write_output(tmp_path, request, 7)
    original_sha256 = _sha256(output_path.read_bytes())
    foreign_payload = _score_payload(7, mode=request.mode)
    foreign_payload["chain_mask"] = np.asarray([0, 1, 0, 1], dtype=np.int64)
    torch.save(foreign_payload, output_path)
    assert _sha256(output_path.read_bytes()) != original_sha256

    with pytest.raises(ValueError, match="score output SHA256 does not match execution completion"):
        _parse(tmp_path, request)


def test_parser_rejects_symlinked_execution_completion_leaf(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=tmp_path / "LigandMPNN",
        execution_root=tmp_path,
    )[0]
    completion_path, _completion = pinned_runtime_completion_contract(
        command.argv,
        upstream_commit=request.upstream.commit,
        checkpoint_sha256=request.upstream.checkpoint_sha256,
        pdb_sha256=request.pdb_sha256,
        request_id=request.request_id,
        context_inventory_path=request.context_inventory.path,
        context_inventory_sha256=request.context_inventory.sha256,
        execution_root=tmp_path,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
    )
    absolute_completion_path = tmp_path / completion_path
    matching_payload = tmp_path / "matching-completion.json"
    absolute_completion_path.replace(matching_payload)
    absolute_completion_path.symlink_to(matching_payload)

    with pytest.raises(ValueError, match="execution completion record could not be opened safely"):
        _parse(tmp_path, request)


def test_parser_rejects_symlinked_execution_completion_ancestor(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    seed_directory = tmp_path / request.output_dir / "seed_7"
    matching_directory = seed_directory.with_name("matching-seed-7")
    seed_directory.replace(matching_directory)
    seed_directory.symlink_to(matching_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="execution completion record could not be opened safely"):
        _parse(tmp_path, request)


def test_request_digest_is_path_portable_and_context_off_is_explicit(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    context_off = replace(request, use_atom_context=False)
    _write_output(tmp_path, context_off, 7)

    result = _parse(tmp_path, context_off)

    assert result.atom_context_requested is False
    assert result.atom_context_status == "disabled_control_with_observed_nucleotide_context"
    relocated = replace(
        context_off,
        pdb_path=Path("different/host/input.pdb"),
        output_dir=Path("different/host/output"),
    )
    assert score_request_sha256(relocated) == score_request_sha256(context_off)


def test_parser_accepts_the_distinct_autoregressive_decoding_order_shape(tmp_path: Path) -> None:
    request = replace(
        _prepare_request(tmp_path, seeds=(7,)),
        mode=LigandMpnnScoreMode.AUTOREGRESSIVE,
    )
    _write_output(tmp_path, request, 7)

    result = _parse(tmp_path, request)

    assert result.mode == "autoregressive"


def test_parser_fails_closed_on_missing_and_extra_output_files(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path)
    _write_output(tmp_path, request, 7)

    with pytest.raises(ValueError, match="missing expected LigandMPNN score outputs"):
        _parse(tmp_path, request)

    _write_output(tmp_path, request, 11)
    extra = tmp_path / request.output_dir / "seed_7/extra.pt"
    torch.save(_score_payload(99), extra)
    with pytest.raises(ValueError, match="unexpected LigandMPNN score outputs"):
        _parse(tmp_path, request)


def test_parser_rejects_private_named_artifact_inside_owned_seed_directory(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    abandoned = tmp_path / request.output_dir / "seed_7/.dnadesign-score-killed/partial.pt"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"killed-attempt")

    with pytest.raises(ValueError, match="unexpected LigandMPNN score outputs"):
        _parse(tmp_path, request)


def test_parser_ignores_sibling_abandoned_private_score_attempt(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    abandoned = tmp_path / request.output_dir / ".dnadesign-score-killed/partial.pt"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"killed-attempt")

    result = _parse(tmp_path, request)

    assert [output.seed for output in result.outputs] == [7]
    assert abandoned.read_bytes() == b"killed-attempt"


def test_parser_rejects_symlinked_output_artifacts(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    source = _write_output(tmp_path, request, 7)
    linked = source.with_name("linked.pt")
    linked.symlink_to(source)

    with pytest.raises(ValueError, match="must not be symlinks"):
        _parse(tmp_path, request)


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory"])
def test_parser_rejects_extra_nonregular_score_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    valid_output = _write_output(tmp_path, request, 7)
    extra = valid_output.with_name(f"extra-{kind}.pt")
    if kind == "fifo":
        os.mkfifo(extra)
    elif kind == "socket":
        monkeypatch.chdir(extra.parent)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix_socket:
            unix_socket.bind(extra.name)
    else:
        extra.mkdir()

    with pytest.raises(ValueError, match="must be regular files"):
        _parse(tmp_path, request)


def test_parser_rejects_expected_score_replaced_by_fifo_after_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    output = _write_output(tmp_path, request, 7)
    payload = output.read_bytes()
    original_lstat = Path.lstat
    target_lstat_calls = 0
    writer: threading.Thread | None = None

    def swap_after_regular_lstat(path: Path) -> os.stat_result:
        nonlocal target_lstat_calls, writer
        status = original_lstat(path)
        if path == output:
            target_lstat_calls += 1
            if target_lstat_calls == 2:
                path.unlink()
                os.mkfifo(path)

                def stream_original_bytes() -> None:
                    descriptor = os.open(path, os.O_WRONLY)
                    try:
                        os.write(descriptor, payload)
                    except OSError:
                        pass
                    finally:
                        os.close(descriptor)

                writer = threading.Thread(target=stream_original_bytes, daemon=True)
                writer.start()
        return status

    monkeypatch.setattr(Path, "lstat", swap_after_regular_lstat)
    try:
        with pytest.raises(ValueError, match="LigandMPNN score output must be a regular file"):
            _parse(tmp_path, request)
    finally:
        assert writer is not None
        writer.join(timeout=2)
        assert not writer.is_alive()


def test_parser_rejects_expected_score_replaced_by_symlink_after_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    output = _write_output(tmp_path, request, 7)
    foreign = tmp_path / "foreign-score.pt"
    foreign.write_bytes(output.read_bytes())
    original_resolve = Path.resolve
    target_resolve_calls = 0

    def swap_after_path_resolution(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal target_resolve_calls
        resolved = original_resolve(path, *args, **kwargs)
        if path == output:
            target_resolve_calls += 1
            if target_resolve_calls == 2:
                path.unlink()
                path.symlink_to(foreign)
        return resolved

    monkeypatch.setattr(Path, "resolve", swap_after_path_resolution)

    with pytest.raises(ValueError, match="LigandMPNN score output could not be opened safely"):
        _parse(tmp_path, request)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alphabet": list("XCDEFGHIKLMNPQRSTVWYA")}, "raw alphabet"),
        ({"seed": 999}, "seed"),
        ({"probs": np.ones((19, 4, 21), dtype=np.float32) / 21.0}, "expected 20 draws"),
        ({"extra": "schema drift"}, "unexpected keys"),
    ],
)
def test_parser_rejects_mismatched_payloads(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7, **overrides)

    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, request)


def test_parser_rejects_input_or_context_command_drift(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    original_input = (tmp_path / request.pdb_path).read_bytes()
    (tmp_path / request.pdb_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="input SHA256"):
        _parse(tmp_path, request)

    corrected = replace(request, pdb_sha256=_sha256(b"tampered"))
    with pytest.raises(ValueError, match="context inventory input identity"):
        build_ligandmpnn_score_commands(
            corrected,
            checkout_root=tmp_path / "LigandMPNN",
            execution_root=tmp_path,
        )

    (tmp_path / request.pdb_path).write_bytes(original_input)
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=tmp_path / "LigandMPNN",
        execution_root=tmp_path,
    )
    argv = list(commands[0].argv)
    context_index = argv.index("--ligand_mpnn_use_atom_context") + 1
    argv[context_index] = "0"
    drifted = (replace(commands[0], argv=tuple(argv)),)
    with pytest.raises(ValueError, match="commands do not exactly match"):
        parse_ligandmpnn_score_outputs(
            request,
            drifted,
            execution_root=tmp_path,
            trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
        )


def test_parser_rejects_missing_or_non_nucleotide_observed_context(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    (tmp_path / request.context_inventory.path).unlink()
    with pytest.raises(ValueError, match="context inventory does not exist"):
        _parse(tmp_path, request)

    inventory_path = tmp_path / request.context_inventory.path
    payload = {
        "schema_id": "thread.ligandmpnn.context_inventory",
        "schema_version": 1,
        "status": "completed_validated",
        "request_id": "generic_context_inventory",
        "request_sha256": f"sha256:{'b' * 64}",
        "input": {"path": "inputs/target.pdb", "sha256": f"sha256:{request.pdb_sha256}"},
        "upstream": {
            "repository": "https://github.com/dauparas/LigandMPNN",
            "commit": request.upstream.commit,
        },
        "parser": {
            "path": "data_utils.py",
            "sha256": f"sha256:{'c' * 64}",
            "callable": "parse_PDB",
            "chains": [],
            "parse_all_atoms": False,
            "parse_atoms_with_zero_occupancy": False,
        },
        "requirements": {"minimum_nucleotide_atoms": 1, "required_polymer_types": []},
        "observed": {
            "effective_nonprotein_atom_count": 1,
            "effective_nucleotide_atom_count": 0,
            "polymer_atom_counts": {"dna": 0, "rna": 0, "other": 1},
            "element_counts": {"ZN": 1},
            "chain_ids": ["Z"],
            "residues": [
                {
                    "chain_id": "Z",
                    "residue_name": "ZN",
                    "residue_number": 1,
                    "insertion_code": "",
                    "polymer_type": "other",
                    "effective_atom_count": 1,
                    "elements": {"ZN": 1},
                }
            ],
            "atoms": [
                {
                    "serial": 1,
                    "atom_name": "ZN",
                    "element": "ZN",
                    "upstream_element_type": 30,
                    "chain_id": "Z",
                    "residue_name": "ZN",
                    "residue_number": 1,
                    "insertion_code": "",
                    "polymer_type": "other",
                }
            ],
        },
    }
    inventory_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_path.write_bytes(inventory_bytes)
    zero_context = replace(
        request,
        context_inventory=LigandMpnnContextInventoryReference(
            path=request.context_inventory.path,
            sha256=_sha256(inventory_bytes),
        ),
    )
    with pytest.raises(ValueError, match="expected at least 1 effective DNA/RNA context atoms"):
        _parse(tmp_path, zero_context)


def test_parser_requires_explicit_trust_and_still_uses_weights_only_loading(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=tmp_path / "LigandMPNN",
        execution_root=tmp_path,
    )
    _write_output(tmp_path, request, 7)

    with pytest.raises(ValueError, match="explicit pinned-local-execution trust"):
        parse_ligandmpnn_score_outputs(request, commands, execution_root=tmp_path, trust="trusted")  # type: ignore[arg-type]

    payload = _score_payload(7)
    payload["untrusted_global"] = Path("not-allowlisted")
    output_path = tmp_path / request.output_dir / "seed_7/target.pt"
    torch.save(payload, output_path)
    completion_path, completion = pinned_runtime_completion_contract(
        commands[0].argv,
        upstream_commit=request.upstream.commit,
        checkpoint_sha256=request.upstream.checkpoint_sha256,
        pdb_sha256=request.pdb_sha256,
        request_id=request.request_id,
        context_inventory_path=request.context_inventory.path,
        context_inventory_sha256=request.context_inventory.sha256,
        execution_root=tmp_path,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        score_output_sha256=f"sha256:{_sha256(output_path.read_bytes())}",
    )
    (tmp_path / completion_path).write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="weights-only loader rejected"):
        parse_ligandmpnn_score_outputs(
            request,
            commands,
            execution_root=tmp_path,
            trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
        )


def test_weights_only_loader_supports_numpy_without_private_core_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    torch.save({"array": np.asarray([1.0], dtype=np.float32)}, buffer)
    numpy_1_compat = SimpleNamespace(
        core=SimpleNamespace(multiarray=importlib.import_module("numpy.core.multiarray")),
        ndarray=np.ndarray,
        dtype=np.dtype,
        float32=np.float32,
        float64=np.float64,
        int32=np.int32,
        int64=np.int64,
        bool_=np.bool_,
    )
    monkeypatch.setattr(score_results_module, "np", numpy_1_compat)

    loaded = score_results_module._load_weights_only_payload(  # noqa: SLF001
        buffer.getvalue(),
        artifact_path=Path("score.pt"),
    )

    assert np.array_equal(loaded["array"], np.asarray([1.0], dtype=np.float32))
