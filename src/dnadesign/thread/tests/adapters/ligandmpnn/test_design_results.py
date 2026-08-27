"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_design_results.py

Admission tests for digest-bound LigandMPNN design output trees.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import dnadesign.thread.adapters.ligandmpnn.design_results as design_results_module
from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnCommand,
    LigandMpnnPackingConfig,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnResidueAlphabet,
    LigandMpnnResidueAlphabetSidecar,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_commands,
    materialize_residue_alphabet_sidecar,
    parse_ligandmpnn_design_outputs,
)
from dnadesign.thread.adapters.ligandmpnn.design_manifest import build_design_output_manifest
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import write_context_inventory
from dnadesign.thread.tests.adapters.ligandmpnn.test_pinned_runtime import _checkout


def _execute_design(
    tmp_path: Path,
    *,
    batch_size: int = 1,
    number_of_batches: int = 1,
    seeds: tuple[int, ...] = (7,),
    pdb_name: str = "input.pdb",
    packing: LigandMpnnPackingConfig | None = None,
    fixed_residues: tuple[LigandMpnnResidue, ...] = (),
    redesigned_residues: tuple[LigandMpnnResidue, ...] = (),
    residue_alphabets: tuple[LigandMpnnResidueAlphabet, ...] = (),
) -> tuple[
    LigandMpnnRequest,
    tuple[LigandMpnnCommand, ...],
    Path,
    LigandMpnnResidueAlphabetSidecar | None,
]:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    if pdb.name != pdb_name:
        renamed_pdb = pdb.with_name(pdb_name)
        pdb.rename(renamed_pdb)
        pdb = renamed_pdb
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    packing = packing or LigandMpnnPackingConfig()
    packing_checkpoint = checkout / "packing.pt"
    packing_checkpoint.write_text("packing-checkpoint-v1", encoding="utf-8")
    packing_checkpoint_sha256 = hashlib.sha256(packing_checkpoint.read_bytes()).hexdigest()
    request = LigandMpnnRequest(
        request_id="admit_design",
        pdb_path=pdb.relative_to(tmp_path),
        pdb_sha256=pdb_sha256,
        output_dir=Path("designs"),
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_path=checkpoint.relative_to(checkout),
            packing_checkpoint_sha256=(packing_checkpoint_sha256 if packing.enabled else None),
            packing_checkpoint_path=packing_checkpoint.relative_to(checkout),
        ),
        context_inventory=reference,
        fixed_residues=fixed_residues,
        redesigned_residues=redesigned_residues,
        residue_alphabets=residue_alphabets,
        seeds=seeds,
        batch_size=batch_size,
        number_of_batches=number_of_batches,
        packing=packing,
    )
    sidecar = None
    if residue_alphabets:
        sidecar_path = Path("evidence/design-residue-alphabets.json")
        sidecar = materialize_residue_alphabet_sidecar(
            request,
            sidecar_path,
            write_path=tmp_path / sidecar_path,
        )
    commands = build_ligandmpnn_commands(
        request,
        checkout_root=checkout,
        execution_root=tmp_path,
        python_executable=sys.executable,
        residue_alphabet_sidecar=sidecar,
    )
    for command in commands:
        subprocess.run(command.argv, cwd=tmp_path, check=True)
    return request, commands, tmp_path / commands[0].output_dir, sidecar


def _rebind_completion_to_current_tree(output_root: Path) -> None:
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["design_output_manifest"] = build_design_output_manifest(output_root)
    completion_path.write_text(json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8")


def _replace_official_fasta(output_root: Path, *, native: str, designs: tuple[str, ...]) -> None:
    records = [f">input, T=0.1, seed=7\n{native}"]
    records.extend(f">input, id={index}, T=0.1, seed=7\n{sequence}" for index, sequence in enumerate(designs, start=1))
    (output_root / "seqs/input.fa").write_text("\n".join(records), encoding="utf-8")
    _rebind_completion_to_current_tree(output_root)


def test_design_admission_binds_exact_published_tree(tmp_path: Path) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)

    result = parse_ligandmpnn_design_outputs(
        request,
        commands,
        execution_root=tmp_path,
    )

    completion = json.loads((output_root / ".dnadesign-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == 4
    assert completion["design_output_manifest"] == result.outputs[0].manifest
    assert result.outputs[0].sequence_count == 1
    assert result.sequence_count == request.expected_sequence_count == 1
    assert result.to_dict()["expected_sequence_count"] == 1
    assert result.to_dict()["sequence_count"] == 1
    assert any(entry["path"] == "seqs/input.fa" for entry in result.outputs[0].manifest["entries"])


def test_design_admission_accepts_complete_official_packed_artifact_set(tmp_path: Path) -> None:
    request, commands, _output_root, _sidecar = _execute_design(
        tmp_path,
        batch_size=2,
        packing=LigandMpnnPackingConfig(enabled=True, number_of_packs_per_design=2),
    )

    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)

    packed_paths = {
        entry["path"]
        for entry in result.outputs[0].manifest["entries"]
        if entry.get("type") == "file" and str(entry["path"]).startswith("packed/")
    }
    assert packed_paths == {
        "packed/input_packed_1_1.pdb",
        "packed/input_packed_1_2.pdb",
        "packed/input_packed_2_1.pdb",
        "packed/input_packed_2_2.pdb",
    }


@pytest.mark.parametrize("mutation", ["missing", "partial", "extra", "wrong"])
def test_design_admission_rejects_incomplete_or_nonofficial_packed_artifact_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    request, commands, output_root, _sidecar = _execute_design(
        tmp_path,
        batch_size=2,
        packing=LigandMpnnPackingConfig(enabled=True, number_of_packs_per_design=2),
    )
    packed_paths = sorted((output_root / "packed").glob("*.pdb"))
    if mutation == "missing":
        for packed_path in packed_paths:
            packed_path.unlink()
    elif mutation == "partial":
        packed_paths[-1].unlink()
    elif mutation == "extra":
        (output_root / "packed/input_packed_3_1.pdb").write_text("extra", encoding="utf-8")
    else:
        packed_paths[-1].rename(output_root / "packed/input_packed_0_1.pdb")
    _rebind_completion_to_current_tree(output_root)

    with pytest.raises(ValueError, match="packed artifacts do not exactly match"):
        parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)


def test_design_admission_rejects_completed_execution_for_different_request_id(tmp_path: Path) -> None:
    request, commands, _output_root, _sidecar = _execute_design(tmp_path)
    replayed_request = replace(request, request_id="different_design_request")

    with pytest.raises(ValueError, match="commands do not exactly match the design request"):
        parse_ligandmpnn_design_outputs(
            replayed_request,
            commands,
            execution_root=tmp_path,
        )


def test_design_builder_and_admission_normalize_symlinked_execution_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-workspace"
    real_root.mkdir()
    linked_root = tmp_path / "linked-workspace"
    linked_root.symlink_to(real_root, target_is_directory=True)

    request, commands, _output_root, _sidecar = _execute_design(linked_root)
    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=linked_root)

    assert commands[0].argv[commands[0].argv.index("--execution-root") + 1] == str(real_root)
    assert result.sequence_count == request.expected_sequence_count == 1


def test_design_admission_preserves_uppercase_pdb_extension_in_official_fasta_name(tmp_path: Path) -> None:
    request, commands, _output_root, _sidecar = _execute_design(tmp_path, pdb_name="input.PDB")

    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)

    assert result.sequence_count == request.expected_sequence_count == 1


def test_design_admission_rejects_completed_tree_without_official_fasta(tmp_path: Path) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    (output_root / "seqs/input.fa").unlink()
    _rebind_completion_to_current_tree(output_root)

    with pytest.raises(ValueError, match="official LigandMPNN FASTA"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([">input, T=0.1, seed=7\nACD"], "expected 2 designed records; observed 0"),
        (
            [
                ">input, T=0.1, seed=7\nACD",
                ">input, id=1, T=0.1, seed=7\nACD",
            ],
            "expected 2 designed records; observed 1",
        ),
        (
            [
                ">input, T=0.1, seed=7\nACD",
                ">input, id=1, T=0.1, seed=7\nACD",
                ">input, id=2, T=0.1, seed=7\nACD",
                ">input, id=3, T=0.1, seed=7\nACD",
            ],
            "expected 2 designed records; observed 3",
        ),
        (
            [
                ">input, T=0.1, seed=7\nACD",
                ">input, id=1, T=0.1, seed=7\nACD",
                ">input, id=1, T=0.1, seed=7\nACD",
            ],
            "design record ids must be exactly",
        ),
        (
            [
                ">input, T=0.1, seed=7\nACD",
                ">input, id=1, T=0.1, seed=7\nAC*",
                ">input, id=2, T=0.1, seed=7\nACD",
            ],
            "invalid amino-acid sequence",
        ),
    ],
)
def test_design_admission_rejects_invalid_official_fasta_records(
    tmp_path: Path,
    records: list[str],
    message: str,
) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path, batch_size=2)
    (output_root / "seqs/input.fa").write_text("\n".join(records), encoding="utf-8")
    _rebind_completion_to_current_tree(output_root)

    with pytest.raises(ValueError, match=message):
        parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)


@pytest.mark.parametrize(
    "designed_sequence",
    [
        "A",
        "ACDEF",
        "AC:D:E",
        "DEF:AC",
    ],
)
def test_design_admission_rejects_designs_that_change_ordered_native_chain_shape(
    tmp_path: Path,
    designed_sequence: str,
) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    (output_root / "seqs/input.fa").write_text(
        "\n".join(
            [
                ">input, T=0.1, seed=7\nAC:DE",
                f">input, id=1, T=0.1, seed=7\n{designed_sequence}",
            ]
        ),
        encoding="utf-8",
    )
    _rebind_completion_to_current_tree(output_root)

    with pytest.raises(ValueError, match="ordered chain-segment lengths"):
        parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)


def test_design_admission_accepts_mutations_that_preserve_ordered_native_chain_shape(
    tmp_path: Path,
) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    (output_root / "seqs/input.fa").write_text(
        "\n".join(
            [
                ">input, T=0.1, seed=7\nAC:DE",
                ">input, id=1, T=0.1, seed=7\nYY:XX",
            ]
        ),
        encoding="utf-8",
    )
    _rebind_completion_to_current_tree(output_root)

    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)

    assert result.sequence_count == 1


def test_design_admission_rejects_native_chain_order_not_matching_pinned_parser(
    tmp_path: Path,
) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    _replace_official_fasta(output_root, native="DE:AC", designs=("DE:AC",))

    with pytest.raises(ValueError, match="native sequence does not match pinned parser"):
        parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)


@pytest.mark.parametrize(
    ("fixed_residue", "designed_sequence"),
    [
        (LigandMpnnResidue("A", 12), "YC:DE"),
        (LigandMpnnResidue("A", 13, "B"), "AY:DE"),
        (LigandMpnnResidue("B", -2, "A"), "AC:YE"),
        (LigandMpnnResidue("B", 2), "AC:DY"),
    ],
)
def test_design_admission_rejects_mutation_of_exact_fixed_parser_residue(
    tmp_path: Path,
    fixed_residue: LigandMpnnResidue,
    designed_sequence: str,
) -> None:
    request, commands, output_root, sidecar = _execute_design(
        tmp_path,
        fixed_residues=(fixed_residue,),
    )
    _replace_official_fasta(output_root, native="AC:DE", designs=(designed_sequence,))

    with pytest.raises(ValueError, match=rf"fixed residue {fixed_residue.upstream_id} was mutated"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
            residue_alphabet_sidecar=sidecar,
        )


def test_design_admission_checks_every_designed_record_against_exact_mask(tmp_path: Path) -> None:
    fixed = LigandMpnnResidue("A", 12)
    request, commands, output_root, sidecar = _execute_design(
        tmp_path,
        batch_size=2,
        fixed_residues=(fixed,),
    )
    _replace_official_fasta(output_root, native="AC:DE", designs=("AC:DE", "YC:DE"))

    with pytest.raises(ValueError, match="design 2 fixed residue A12 was mutated"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
            residue_alphabet_sidecar=sidecar,
        )


def test_design_admission_rejects_mutation_outside_exact_redesigned_parser_residues(tmp_path: Path) -> None:
    redesigned = LigandMpnnResidue("B", -2, "A")
    request, commands, output_root, sidecar = _execute_design(
        tmp_path,
        redesigned_residues=(redesigned,),
    )
    _replace_official_fasta(output_root, native="AC:DE", designs=("YC:YE",))

    with pytest.raises(ValueError, match="outside redesigned_residues"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
            residue_alphabet_sidecar=sidecar,
        )


def test_design_admission_rejects_disallowed_residue_alphabet_output(tmp_path: Path) -> None:
    redesigned = LigandMpnnResidue("A", 13, "B")
    alphabet = LigandMpnnResidueAlphabet(redesigned, ("A", "G"))
    request, commands, output_root, sidecar = _execute_design(
        tmp_path,
        redesigned_residues=(redesigned,),
        residue_alphabets=(alphabet,),
    )
    _replace_official_fasta(output_root, native="AC:DE", designs=("AY:DE",))

    with pytest.raises(ValueError, match="residue alphabet constraint A13B"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
            residue_alphabet_sidecar=sidecar,
        )


@pytest.mark.parametrize(
    ("fixed_residues", "redesigned_residues", "residue_alphabets", "designed_sequence"),
    [
        ((LigandMpnnResidue("A", 12),), (), (), "AC:YY"),
        (
            (),
            (LigandMpnnResidue("A", 13, "B"),),
            (LigandMpnnResidueAlphabet(LigandMpnnResidue("A", 13, "B"), ("G",)),),
            "AG:DE",
        ),
    ],
)
def test_design_admission_accepts_outputs_obeying_exact_parser_mask_and_alphabet(
    tmp_path: Path,
    fixed_residues: tuple[LigandMpnnResidue, ...],
    redesigned_residues: tuple[LigandMpnnResidue, ...],
    residue_alphabets: tuple[LigandMpnnResidueAlphabet, ...],
    designed_sequence: str,
) -> None:
    request, commands, output_root, sidecar = _execute_design(
        tmp_path,
        fixed_residues=fixed_residues,
        redesigned_residues=redesigned_residues,
        residue_alphabets=residue_alphabets,
    )
    _replace_official_fasta(output_root, native="AC:DE", designs=(designed_sequence,))

    result = parse_ligandmpnn_design_outputs(
        request,
        commands,
        execution_root=tmp_path,
        residue_alphabet_sidecar=sidecar,
    )

    assert result.sequence_count == 1


def test_design_admission_counts_only_the_official_input_fasta(tmp_path: Path) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path, batch_size=2)
    (output_root / "unrelated.fasta").write_text(">foreign\nAAAA\n", encoding="utf-8")
    (output_root / "seqs/unrelated.fa").write_text(">foreign\nAAAA\n", encoding="utf-8")
    _rebind_completion_to_current_tree(output_root)

    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)

    assert result.sequence_count == request.expected_sequence_count == 2


def test_design_admission_binds_parsed_fasta_bytes_to_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, commands, _output_root, _sidecar = _execute_design(tmp_path)
    original_reader = design_results_module._read_descriptor_relative_regular_bytes

    def _replace_fasta_after_manifest(root: Path, path: Path, *, label: str) -> bytes:
        if label == "official LigandMPNN FASTA":
            return b">input, T=0.1, seed=7\nAAA\n>input, id=1, T=0.1, seed=7\nAAA"
        return original_reader(root, path, label=label)

    monkeypatch.setattr(
        design_results_module,
        "_read_descriptor_relative_regular_bytes",
        _replace_fasta_after_manifest,
    )

    with pytest.raises(ValueError, match="does not match admitted manifest"):
        parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)


def test_design_admission_enforces_seed_batch_total(tmp_path: Path) -> None:
    request, commands, _output_root, _sidecar = _execute_design(
        tmp_path,
        seeds=(7, 11),
        batch_size=2,
        number_of_batches=3,
    )

    result = parse_ligandmpnn_design_outputs(request, commands, execution_root=tmp_path)

    assert tuple(output.sequence_count for output in result.outputs) == (6, 6)
    assert result.sequence_count == request.expected_sequence_count == 12


@pytest.mark.parametrize("mutation", ["edit", "replace", "add", "delete"])
def test_design_admission_rejects_artifact_tree_mutation(tmp_path: Path, mutation: str) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    design_path = output_root / "design.txt"
    if mutation == "edit":
        design_path.write_text("edited", encoding="utf-8")
    elif mutation == "replace":
        design_path.unlink()
        design_path.write_text("replacement", encoding="utf-8")
    elif mutation == "add":
        (output_root / "supplemental.fasta").write_text(">foreign\nAA\n", encoding="utf-8")
    else:
        design_path.unlink()

    with pytest.raises(ValueError, match="design output manifest mismatch"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
        )


def test_design_admission_rejects_nonregular_artifact(tmp_path: Path) -> None:
    request, commands, output_root, _sidecar = _execute_design(tmp_path)
    (output_root / "foreign-link").symlink_to(output_root / "design.txt")

    with pytest.raises(ValueError, match="design output entry must be regular"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
        )
