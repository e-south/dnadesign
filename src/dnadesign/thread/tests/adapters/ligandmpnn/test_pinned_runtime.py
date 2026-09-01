"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_pinned_runtime.py

Tests attested LigandMPNN entrypoint execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import py_compile
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dnadesign.thread.adapters.ligandmpnn.pinned_runtime as pinned_runtime_module
from dnadesign.thread.adapters.ligandmpnn.alphabets import (
    LigandMpnnResidueAlphabetSidecar,
    materialize_residue_alphabet_sidecar,
)
from dnadesign.thread.adapters.ligandmpnn.commands import build_ligandmpnn_commands
from dnadesign.thread.adapters.ligandmpnn.design_results import parse_ligandmpnn_design_outputs
from dnadesign.thread.adapters.ligandmpnn.models import (
    LigandMpnnCommand,
    LigandMpnnRequest,
    LigandMpnnResidue,
    LigandMpnnResidueAlphabet,
    LigandMpnnUpstreamPin,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import (
    _validate_runtime_option_contract,
    build_pinned_runtime_command,
    pinned_execution_sha256,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import (
    execute_pinned_entrypoint as _execute_pinned_entrypoint,
)
from dnadesign.thread.adapters.ligandmpnn.scoring import (
    LigandMpnnScoreRequest,
    build_ligandmpnn_score_commands,
)
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import (
    PINNED_CONTEXT_PARSER_PAYLOAD,
    write_context_inventory,
)


def execute_pinned_entrypoint(**kwargs: object) -> None:
    """Exercise the direct runtime boundary with a canonical planned execution."""

    arguments = kwargs["arguments"]
    assert isinstance(arguments, tuple)
    if "--seed" not in arguments and not any(value.startswith("--seed=") for value in arguments):
        arguments = (*arguments, "--seed", "1")
        kwargs["arguments"] = arguments
    checkout_root = kwargs["checkout_root"]
    assert isinstance(checkout_root, Path)
    completion_record_path = kwargs.pop(
        "completion_record_path",
        checkout_root.parent / ".test-ligandmpnn-execution.json",
    )
    assert isinstance(completion_record_path, Path)
    planned_arguments = kwargs.pop("planned_arguments", arguments)
    assert isinstance(planned_arguments, tuple)
    if "--seed" not in planned_arguments and not any(value.startswith("--seed=") for value in planned_arguments):
        planned_arguments = (*planned_arguments, "--seed", "1")
    entrypoint = str(kwargs["entrypoint"])
    context_inventory_path = kwargs.pop("context_inventory_path", None)
    context_inventory_sha256 = kwargs.pop("context_inventory_sha256", None)
    execution_root = kwargs.pop("execution_root", None)
    request_id = kwargs.pop("request_id", "test_request")
    if entrypoint in {"run.py", "score.py"} and context_inventory_path is None:
        execution_root = checkout_root.parent
        pdb_value = planned_arguments[planned_arguments.index("--pdb_path") + 1]
        pdb_path = Path(pdb_value)
        relative_pdb_path = pdb_path.relative_to(execution_root) if pdb_path.is_absolute() else pdb_path
        use_side_chain_context = (
            planned_arguments[planned_arguments.index("--ligand_mpnn_use_side_chain_context") + 1] == "1"
            if "--ligand_mpnn_use_side_chain_context" in planned_arguments
            else False
        )
        parser_payload = subprocess.check_output(
            ["git", "-C", str(checkout_root), "show", f"{kwargs['upstream_commit']}:data_utils.py"]
        )
        parser_sha256 = hashlib.sha256(parser_payload).hexdigest()
        reference = write_context_inventory(
            execution_root,
            input_path=relative_pdb_path,
            input_sha256=str(kwargs["pdb_sha256"]),
            upstream_commit=str(kwargs["upstream_commit"]),
            parse_all_atoms=use_side_chain_context,
            parser_sha256=parser_sha256,
            relative_path=Path("evidence") / f"context-{relative_pdb_path.stem}-{kwargs['pdb_sha256']}.json",
        )
        context_inventory_path = reference.path
        context_inventory_sha256 = reference.sha256
    planned_execution_sha256 = pinned_execution_sha256(
        checkout_root=checkout_root,
        upstream_commit=str(kwargs["upstream_commit"]),
        checkpoint_sha256=str(kwargs["checkpoint_sha256"]),
        pdb_sha256=str(kwargs["pdb_sha256"]),
        request_id=str(request_id),
        packing_checkpoint_sha256=kwargs["packing_checkpoint_sha256"],  # type: ignore[arg-type]
        residue_alphabet_sha256=kwargs["residue_alphabet_sha256"],  # type: ignore[arg-type]
        context_inventory_path=context_inventory_path,  # type: ignore[arg-type]
        context_inventory_sha256=context_inventory_sha256,  # type: ignore[arg-type]
        execution_root=execution_root,  # type: ignore[arg-type]
        entrypoint=entrypoint,
        completion_record_path=completion_record_path,
        arguments=planned_arguments,
    )
    _execute_pinned_entrypoint(
        **kwargs,  # type: ignore[arg-type]
        request_id=str(request_id),
        context_inventory_path=context_inventory_path,  # type: ignore[arg-type]
        context_inventory_sha256=context_inventory_sha256,  # type: ignore[arg-type]
        execution_root=execution_root,  # type: ignore[arg-type]
        planned_execution_sha256=planned_execution_sha256,
        completion_record_path=completion_record_path,
    )


def _checkout(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    root = tmp_path / "LigandMPNN"
    root.mkdir()
    (root / "data_utils.py").write_bytes(PINNED_CONTEXT_PARSER_PAYLOAD)
    (root / "model_utils.py").write_text("HELPER = 'helper-attested'\n", encoding="utf-8")
    (root / "run.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "from data_utils import VALUE, packed_pdb_payload\n"
        "from model_utils import HELPER\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--model_type', required=True)\n"
        "parser.add_argument('--checkpoint_ligand_mpnn', required=True)\n"
        "parser.add_argument('--pdb_path', required=True)\n"
        "parser.add_argument('--omit_AA_per_residue')\n"
        "parser.add_argument('--fixed_residues')\n"
        "parser.add_argument('--redesigned_residues')\n"
        "parser.add_argument('--ligand_mpnn_use_atom_context')\n"
        "parser.add_argument('--output')\n"
        "parser.add_argument('--out_folder')\n"
        "parser.add_argument('--seed', type=int, default=0)\n"
        "parser.add_argument('--temperature', type=float, default=0.1)\n"
        "parser.add_argument('--batch_size', type=int, default=1)\n"
        "parser.add_argument('--number_of_batches', type=int, default=1)\n"
        "parser.add_argument('--pack_side_chains', type=int, default=0)\n"
        "parser.add_argument('--number_of_packs_per_design', type=int, default=4)\n"
        "args, _ = parser.parse_known_args()\n"
        "checkpoint = Path(args.checkpoint_ligand_mpnn).read_text(encoding='utf-8')\n"
        "pdb = Path(args.pdb_path).read_text(encoding='utf-8')\n"
        "sidecar = (\n"
        "    Path(args.omit_AA_per_residue).read_text(encoding='utf-8')\n"
        "    if args.omit_AA_per_residue\n"
        "    else 'no-sidecar'\n"
        ")\n"
        "if args.output:\n"
        "    Path(args.output).write_text(\n"
        "        f'{VALUE}:{HELPER}:{checkpoint}:{pdb}:{sidecar}', encoding='utf-8'\n"
        "    )\n"
        "if args.out_folder:\n"
        "    output_root = Path(args.out_folder)\n"
        "    output_root.mkdir(parents=True, exist_ok=True)\n"
        "    (output_root / 'design.txt').write_text(pdb, encoding='utf-8')\n"
        "    seqs = output_root / 'seqs'\n"
        "    seqs.mkdir()\n"
        "    name = Path(args.pdb_path).name\n"
        "    if name.endswith('.pdb'):\n"
        "        name = name[:-4]\n"
        "    records = [(\n"
        "        f'>{name}, T={args.temperature}, seed={args.seed}, batch_size={args.batch_size}, ' \n"
        "        f'number_of_batches={args.number_of_batches}, model_path={args.checkpoint_ligand_mpnn}\\nAC:DE'\n"
        "    )]\n"
        "    for design_id in range(1, args.batch_size * args.number_of_batches + 1):\n"
        "        records.append(\n"
        "            f'>{name}, id={design_id}, T={args.temperature}, seed={args.seed}, ' \n"
        "            'overall_confidence=0.1, ligand_confidence=0.2, seq_rec=0.3\\nAC:DE'\n"
        "        )\n"
        "    (seqs / f'{name}.fa').write_text('\\n'.join(records), encoding='utf-8')\n"
        "    if args.pack_side_chains:\n"
        "        packed = output_root / 'packed'\n"
        "        packed.mkdir()\n"
        "        for design_id in range(1, args.batch_size * args.number_of_batches + 1):\n"
        "            for pack_id in range(1, args.number_of_packs_per_design + 1):\n"
        "                (packed / f'{name}_packed_{design_id}_{pack_id}.pdb').write_text(\n"
        "                    packed_pdb_payload('ACDE'), encoding='utf-8'\n"
        "                )\n",
        encoding="utf-8",
    )
    (root / "score.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--model_type', required=True)\n"
        "parser.add_argument('--checkpoint_ligand_mpnn', required=True)\n"
        "parser.add_argument('--pdb_path', required=True)\n"
        "parser.add_argument('--out_folder', required=True)\n"
        "args, _ = parser.parse_known_args()\n"
        "output = Path(args.out_folder) / f'{Path(args.pdb_path).stem}.pt'\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_text(Path(args.pdb_path).read_text(encoding='utf-8'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
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
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_text("checkpoint-v1", encoding="utf-8")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pdb = root / "input.pdb"
    pdb.write_text("input-v1", encoding="utf-8")
    pdb_sha256 = hashlib.sha256(pdb.read_bytes()).hexdigest()
    return root, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256


def _public_runtime_command(
    tmp_path: Path,
    *,
    entrypoint: str,
    with_residue_sidecar: bool = False,
) -> tuple[LigandMpnnRequest | LigandMpnnScoreRequest, LigandMpnnCommand, Path]:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    upstream = LigandMpnnUpstreamPin(
        commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_path=checkpoint.relative_to(checkout),
    )
    if entrypoint == "run.py":
        residue = LigandMpnnResidue("A", 12)
        request: LigandMpnnRequest | LigandMpnnScoreRequest = LigandMpnnRequest(
            request_id="foreign_cwd_design",
            pdb_path=pdb.relative_to(tmp_path),
            pdb_sha256=pdb_sha256,
            output_dir=Path("designs"),
            upstream=upstream,
            context_inventory=reference,
            redesigned_residues=((residue,) if with_residue_sidecar else ()),
            residue_alphabets=(
                (LigandMpnnResidueAlphabet(residue=residue, allowed_amino_acids=("A", "C")),)
                if with_residue_sidecar
                else ()
            ),
            seeds=(7,),
        )
        sidecar = None
        if with_residue_sidecar:
            sidecar_path = Path("evidence/residue-alphabet.json")
            sidecar = materialize_residue_alphabet_sidecar(
                request,
                sidecar_path,
                write_path=tmp_path / sidecar_path,
            )
        command = build_ligandmpnn_commands(
            request,
            checkout_root=checkout,
            execution_root=tmp_path,
            python_executable=sys.executable,
            residue_alphabet_sidecar=sidecar,
        )[0]
    else:
        request = LigandMpnnScoreRequest(
            request_id="foreign_cwd_score",
            pdb_path=pdb.relative_to(tmp_path),
            pdb_sha256=pdb_sha256,
            output_dir=Path("scores"),
            upstream=upstream,
            context_inventory=reference,
            seeds=(7,),
            number_of_batches=10,
            use_atom_context=False,
        )
        command = build_ligandmpnn_score_commands(
            request,
            checkout_root=checkout,
            execution_root=tmp_path,
            python_executable=sys.executable,
        )[0]
    return request, command, pdb


@pytest.mark.parametrize("entrypoint", ["run.py", "score.py"])
def test_public_runtime_anchors_relative_paths_to_execution_root_from_foreign_cwd(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    request, command, pdb = _public_runtime_command(tmp_path, entrypoint=entrypoint)
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_pdb = foreign_cwd / request.pdb_path
    foreign_pdb.parent.mkdir(parents=True)
    foreign_pdb.write_bytes(pdb.read_bytes())

    subprocess.run(command.argv, cwd=foreign_cwd, check=True)

    output_root = tmp_path / command.output_dir
    assert (output_root / ".dnadesign-ligandmpnn-execution.json").is_file()
    assert not (foreign_cwd / command.output_dir).exists()
    if isinstance(request, LigandMpnnRequest):
        result = parse_ligandmpnn_design_outputs(request, (command,), execution_root=tmp_path)
        assert result.sequence_count == request.expected_sequence_count
    else:
        assert (output_root / "input.pt").read_text(encoding="utf-8") == "input-v1"


@pytest.mark.parametrize("entrypoint", ["run.py", "score.py"])
@pytest.mark.parametrize("foreign_input", ["missing", "tampered"])
def test_public_runtime_ignores_non_authoritative_foreign_relative_input(
    tmp_path: Path,
    entrypoint: str,
    foreign_input: str,
) -> None:
    request, command, _pdb = _public_runtime_command(tmp_path, entrypoint=entrypoint)
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    if foreign_input == "tampered":
        foreign_pdb = foreign_cwd / request.pdb_path
        foreign_pdb.parent.mkdir(parents=True)
        foreign_pdb.write_text("tampered-foreign-input", encoding="utf-8")

    subprocess.run(command.argv, cwd=foreign_cwd, check=True)

    output_root = tmp_path / command.output_dir
    assert (output_root / ".dnadesign-ligandmpnn-execution.json").is_file()
    assert not (foreign_cwd / command.output_dir).exists()


@pytest.mark.parametrize("foreign_sidecar", ["missing", "tampered"])
def test_public_design_runtime_ignores_non_authoritative_foreign_relative_sidecar(
    tmp_path: Path,
    foreign_sidecar: str,
) -> None:
    request, command, pdb = _public_runtime_command(
        tmp_path,
        entrypoint="run.py",
        with_residue_sidecar=True,
    )
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_pdb = foreign_cwd / request.pdb_path
    foreign_pdb.parent.mkdir(parents=True)
    foreign_pdb.write_bytes(pdb.read_bytes())
    sidecar_value = command.argv[command.argv.index("--omit_AA_per_residue") + 1]
    if foreign_sidecar == "tampered":
        foreign_sidecar_path = foreign_cwd / sidecar_value
        foreign_sidecar_path.parent.mkdir(parents=True)
        foreign_sidecar_path.write_text("tampered-foreign-sidecar", encoding="utf-8")

    subprocess.run(command.argv, cwd=foreign_cwd, check=True)

    assert (tmp_path / command.output_dir / ".dnadesign-ligandmpnn-execution.json").is_file()
    assert not (foreign_cwd / command.output_dir).exists()


@pytest.mark.parametrize(
    ("authoritative_sidecar", "foreign_sidecar", "succeeds"),
    [
        ("matching", "missing", True),
        ("matching", "tampered", True),
        ("missing", "matching", False),
        ("tampered", "matching", False),
    ],
)
def test_design_admission_validates_unstaged_relative_sidecar_at_execution_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authoritative_sidecar: str,
    foreign_sidecar: str,
    succeeds: bool,
) -> None:
    request, command, _pdb = _public_runtime_command(
        tmp_path,
        entrypoint="run.py",
        with_residue_sidecar=True,
    )
    subprocess.run(command.argv, cwd=tmp_path, check=True)
    relative_path = Path(command.argv[command.argv.index("--omit_AA_per_residue") + 1])
    canonical_bytes = (tmp_path / relative_path).read_bytes()
    sidecar = LigandMpnnResidueAlphabetSidecar(
        request_id=request.request_id,
        path=relative_path,
        sha256=f"sha256:{command.argv[command.argv.index('--residue-alphabet-sha256') + 1]}",
        residue_count=1,
    )
    if authoritative_sidecar == "missing":
        (tmp_path / relative_path).unlink()
    elif authoritative_sidecar == "tampered":
        (tmp_path / relative_path).write_text("{}\n", encoding="utf-8")

    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    if foreign_sidecar != "missing":
        foreign_path = foreign_cwd / relative_path
        foreign_path.parent.mkdir(parents=True)
        foreign_path.write_bytes(canonical_bytes if foreign_sidecar == "matching" else b"{}\n")
    monkeypatch.chdir(foreign_cwd)

    if succeeds:
        result = parse_ligandmpnn_design_outputs(
            request,
            (command,),
            execution_root=tmp_path,
            residue_alphabet_sidecar=sidecar,
        )
        assert result.sequence_count == request.expected_sequence_count
    else:
        with pytest.raises(ValueError, match="sidecar file SHA256 does not match receipt"):
            parse_ligandmpnn_design_outputs(
                request,
                (command,),
                execution_root=tmp_path,
                residue_alphabet_sidecar=sidecar,
            )


@pytest.mark.parametrize("entrypoint", ["run.py", "score.py"])
def test_pinned_runtime_command_preserves_option_looking_checkout_roots(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    context_binding: dict[str, object] = {
        "execution_root": Path("-execution"),
        "context_inventory_path": Path("evidence/context.json"),
        "context_inventory_sha256": "c" * 64,
    }
    command = build_pinned_runtime_command(
        checkout_root=Path("-checkout"),
        upstream_commit="1" * 40,
        checkpoint_sha256="a" * 64,
        pdb_sha256="b" * 64,
        request_id="test_request",
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint=entrypoint,
        python_executable=sys.executable,
        output_dir=Path("outputs/seed_1"),
        arguments=(),
        **context_binding,  # type: ignore[arg-type]
    )
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(sys, "argv", ["pinned-runtime", *command[3:]])
    monkeypatch.setattr(
        pinned_runtime_module,
        "execute_pinned_entrypoint",
        lambda **kwargs: observed.append(kwargs),
    )

    assert "--checkout-root=-checkout" in command
    pinned_runtime_module.main()
    parsed_checkout, _python, _completion, _digest = pinned_runtime_module.parse_pinned_runtime_prefix(
        command,
        upstream_commit="1" * 40,
        checkpoint_sha256="a" * 64,
        pdb_sha256="b" * 64,
        request_id="test_request",
        context_inventory_path=Path("evidence/context.json"),
        context_inventory_sha256="c" * 64,
        execution_root=Path("-execution"),
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint=entrypoint,
    )

    assert observed[0]["checkout_root"] == Path("-checkout")
    assert parsed_checkout == Path("-checkout")


def test_pinned_runtime_ignores_timestamp_valid_poisoned_parser_bytecode(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    parser_path = checkout / "data_utils.py"
    malicious_source = tmp_path / "data_utils.py"
    malicious_source.write_bytes(parser_path.read_bytes().replace(b'VALUE = "attested"', b'VALUE = "poisoned"'))
    assert malicious_source.stat().st_size == parser_path.stat().st_size
    parser_mtime = parser_path.stat().st_mtime
    os.utime(malicious_source, (parser_mtime, parser_mtime))
    cache_path = Path(importlib.util.cache_from_source(str(parser_path)))
    cache_path.parent.mkdir()
    py_compile.compile(
        str(malicious_source),
        cfile=str(cache_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )

    poisoned_output = tmp_path / "poisoned.txt"
    subprocess.run(
        [
            sys.executable,
            str(checkout / "run.py"),
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--output",
            str(poisoned_output),
        ],
        check=True,
    )
    assert poisoned_output.read_text(encoding="utf-8") == "poisoned:helper-attested:checkpoint-v1:input-v1:no-sidecar"

    attested_output = tmp_path / "attested.txt"
    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        request_id="test_request",
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--seed",
            "1",
            "--output",
            str(attested_output),
        ),
    )

    assert attested_output.read_text(encoding="utf-8") == "attested:helper-attested:checkpoint-v1:input-v1:no-sidecar"
    completion = json.loads((tmp_path / ".test-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert completion["status"] == "completed"
    assert completion["execution"]["arguments"][-1] == str(attested_output)


@pytest.mark.parametrize(
    "actual_suffix",
    [
        ("--ligand_mpnn_use_atom_context", "0"),
        ("--ligand_mpnn_use_atom_con", "0"),
        ("--unplanned_unique_override", "1"),
    ],
)
def test_pinned_runtime_rejects_actual_arguments_that_differ_from_complete_plan(
    tmp_path: Path,
    actual_suffix: tuple[str, ...],
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    planned_arguments = (
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        str(checkpoint),
        "--pdb_path",
        str(pdb),
        "--ligand_mpnn_use_atom_context",
        "1",
        "--output",
        str(tmp_path / "output.txt"),
    )

    with pytest.raises(ValueError, match="does not match the complete planned arguments"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            planned_arguments=planned_arguments,
            arguments=(*planned_arguments, *actual_suffix),
        )


def test_pinned_runtime_executes_pinned_tree_when_checkout_sources_are_dirty(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    (checkout / "data_utils.py").write_text("VALUE = 'modified'\n", encoding="utf-8")
    (checkout / "model_utils.py").write_text("HELPER = 'helper-modified'\n", encoding="utf-8")
    output = tmp_path / "output.txt"

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--output",
            str(output),
        ),
    )

    assert output.read_text(encoding="utf-8") == "attested:helper-attested:checkpoint-v1:input-v1:no-sidecar"


def test_generated_runtime_command_executes_and_records_complete_cli_arguments(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output = tmp_path / "output.txt"
    output_dir = tmp_path / "runtime"
    arguments = (
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        str(checkpoint),
        "--pdb_path",
        str(pdb),
        "--seed",
        "1",
        "--output",
        str(output),
    )
    context_inventory = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    command = build_pinned_runtime_command(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        request_id="test_request",
        context_inventory_path=context_inventory.path,
        context_inventory_sha256=context_inventory.sha256,
        execution_root=tmp_path,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        python_executable=sys.executable,
        output_dir=output_dir,
        arguments=arguments,
    )

    subprocess.run(command, cwd=tmp_path, check=True)

    completion = json.loads((output_dir / ".dnadesign-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert completion["execution"]["arguments"] == list(arguments)
    assert completion["execution"]["request_id"] == "test_request"
    assert completion["execution"]["context_inventory_path"] == context_inventory.path.as_posix()
    assert completion["execution"]["context_inventory_sha256"] == context_inventory.sha256
    assert completion["execution_sha256"].startswith("sha256:")


def test_public_design_builder_executes_only_with_bound_context_evidence(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    request = LigandMpnnRequest(
        request_id="direct_public_builder",
        pdb_path=pdb.relative_to(tmp_path),
        pdb_sha256=pdb_sha256,
        output_dir=Path("designs"),
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_path=checkpoint.relative_to(checkout),
        ),
        context_inventory=reference,
        seeds=(7,),
    )

    command = build_ligandmpnn_commands(
        request,
        checkout_root=checkout,
        execution_root=tmp_path,
        python_executable=sys.executable,
    )[0]
    subprocess.run(command.argv, cwd=tmp_path, check=True)

    output_root = tmp_path / command.output_dir
    completion = json.loads((output_root / ".dnadesign-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert (output_root / "design.txt").read_text(encoding="utf-8") == "input-v1"
    assert completion["execution"]["context_inventory_path"] == reference.path.as_posix()
    assert completion["execution"]["context_inventory_sha256"] == reference.sha256


def test_public_design_builder_runs_relative_checkout_against_execution_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    request = LigandMpnnRequest(
        request_id="relative_checkout_builder",
        pdb_path=pdb.relative_to(tmp_path),
        pdb_sha256=pdb_sha256,
        output_dir=Path("designs"),
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_path=checkpoint.relative_to(checkout),
        ),
        context_inventory=reference,
        seeds=(7,),
    )
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    command = build_ligandmpnn_commands(
        request,
        checkout_root=checkout.relative_to(tmp_path),
        execution_root=tmp_path,
        python_executable=sys.executable,
    )[0]
    subprocess.run(command.argv, cwd=tmp_path, check=True)

    assert command.argv[command.argv.index("--checkout-root") + 1] == str(checkout)
    assert (tmp_path / command.output_dir / "seqs/input.fa").is_file()


def test_public_score_builder_binds_and_revalidates_context_before_execution(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    request = LigandMpnnScoreRequest(
        request_id="bound_score_context",
        pdb_path=pdb.relative_to(tmp_path),
        pdb_sha256=pdb_sha256,
        output_dir=Path("scores"),
        upstream=LigandMpnnUpstreamPin(
            commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_path=checkpoint.relative_to(checkout),
        ),
        context_inventory=reference,
        seeds=(7,),
        number_of_batches=10,
        use_atom_context=False,
    )
    command = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout,
        execution_root=tmp_path,
        python_executable=sys.executable,
    )[0]
    inventory_path = tmp_path / reference.path
    inventory_bytes = inventory_path.read_bytes()
    inventory_path.write_bytes(b"tampered after planning")

    failed = subprocess.run(command.argv, cwd=tmp_path, text=True, capture_output=True)

    assert failed.returncode != 0
    assert "context inventory SHA256 mismatch" in failed.stderr
    assert not (tmp_path / command.output_dir / "input.pt").exists()
    assert not (tmp_path / command.output_dir / ".dnadesign-ligandmpnn-execution.json").exists()

    inventory_path.write_bytes(inventory_bytes)
    subprocess.run(command.argv, cwd=tmp_path, check=True)
    completion = json.loads(
        (tmp_path / command.output_dir / ".dnadesign-ligandmpnn-execution.json").read_text(encoding="utf-8")
    )
    assert completion["execution"]["context_inventory_path"] == reference.path.as_posix()
    assert completion["execution"]["context_inventory_sha256"] == reference.sha256
    assert completion["execution"]["execution_root"] == str(tmp_path)


def test_pinned_design_runtime_rejects_context_inventory_tampered_after_planning(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    (tmp_path / reference.path).write_bytes(b"tampered-after-planning")
    output_root = tmp_path / "designs/seed_7"

    with pytest.raises(ValueError, match="context inventory SHA256 mismatch"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            context_inventory_path=reference.path,
            context_inventory_sha256=reference.sha256,
            execution_root=tmp_path,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=output_root / ".dnadesign-ligandmpnn-execution.json",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not output_root.exists()


def test_pinned_design_runtime_rejects_symlinked_context_inventory_after_planning(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256((checkout / "data_utils.py").read_bytes()).hexdigest(),
    )
    inventory_path = tmp_path / reference.path
    outside = tmp_path / "outside-context.json"
    inventory_path.replace(outside)
    inventory_path.symlink_to(outside)
    output_root = tmp_path / "designs/seed_7"

    with pytest.raises(ValueError, match="context inventory could not be opened safely"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            context_inventory_path=reference.path,
            context_inventory_sha256=reference.sha256,
            execution_root=tmp_path,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=output_root / ".dnadesign-ligandmpnn-execution.json",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not output_root.exists()


def test_pinned_runtime_rejects_checkpoint_changed_after_planning(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    checkpoint.write_text("checkpoint-v2", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_ligand_mpnn SHA256 mismatch"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


def test_pinned_runtime_rejects_pdb_changed_after_planning(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    pdb.write_text("input-v2", encoding="utf-8")

    with pytest.raises(ValueError, match="pdb_path SHA256 mismatch"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


def test_pinned_runtime_rejects_sidecar_changed_after_planning(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    sidecar = tmp_path / "residue-alphabet.json"
    sidecar.write_text("sidecar-v1", encoding="utf-8")
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    sidecar.write_text("sidecar-v2", encoding="utf-8")

    with pytest.raises(ValueError, match="omit_AA_per_residue SHA256 mismatch"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=sidecar_sha256,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--omit_AA_per_residue",
                str(sidecar),
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--checkpoint_ligand_mpnn",
        "--checkpoint_path_sc",
        "--pdb_path",
        "--omit_AA_per_residue",
    ],
)
def test_runtime_staging_rejects_input_replaced_by_fifo_without_blocking(
    tmp_path: Path,
    flag: str,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"attested-input")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "staged"

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from contextlib import contextmanager\n"
                "from pathlib import Path\n"
                "import dnadesign.thread.adapters.ligandmpnn.pinned_runtime as module\n"
                "source, destination = Path(sys.argv[2]), Path(sys.argv[4])\n"
                "original = module.open_regular_file\n"
                "@contextmanager\n"
                "def replace_before_open(path):\n"
                "    if path == source:\n"
                "        path.unlink()\n"
                "        os.mkfifo(path)\n"
                "    with original(path) as handle:\n"
                "        yield handle\n"
                "module.open_regular_file = replace_before_open\n"
                "try:\n"
                "    module._replace_verified_file(\n"
                "        [sys.argv[1], str(source)],\n"
                "        flag=sys.argv[1],\n"
                "        expected_sha256=sys.argv[3],\n"
                "        destination=destination,\n"
                "    )\n"
                "except ValueError as error:\n"
                "    assert 'must reference a regular file' in str(error)\n"
                "else:\n"
                "    raise SystemExit('nonregular staged input was accepted')\n"
                "assert not destination.exists()\n"
            ),
            flag,
            str(source),
            expected,
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert probe.stderr == ""


def test_pinned_runtime_stages_verified_sidecar_before_execution(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    sidecar = tmp_path / "residue-alphabet.json"
    sidecar.write_text("sidecar-v1", encoding="utf-8")
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    output = tmp_path / "output.txt"

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=sidecar_sha256,
        entrypoint="run.py",
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--omit_AA_per_residue",
            str(sidecar),
            "--output",
            str(output),
        ),
    )

    assert output.read_text(encoding="utf-8") == "attested:helper-attested:checkpoint-v1:input-v1:sidecar-v1"


def test_pinned_runtime_rejects_simultaneous_singular_residue_selection_flags(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output = tmp_path / "output.txt"

    with pytest.raises(ValueError, match="mutually exclusive"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--fixed_residues",
                "A1 A2",
                "--redesigned_residues",
                "A3",
                "--output",
                str(output),
            ),
        )


@pytest.mark.parametrize("entrypoint", ["run.py", "score.py"])
@pytest.mark.parametrize("selection_flag", ["--fixed_residues", "--redesigned_residues"])
def test_pinned_runtime_rejects_selector_absent_from_attested_parser_input(
    tmp_path: Path,
    entrypoint: str,
    selection_flag: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / f"{entrypoint}-output"

    with pytest.raises(ValueError, match=rf"{selection_flag.removeprefix('--')}.*A13A.*not present"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint=entrypoint,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
                selection_flag,
                "A13A",
            ),
        )

    assert not output_root.exists()


def test_pinned_runtime_rejects_standalone_semantic_abbreviation() -> None:
    with pytest.raises(ValueError, match="unattested or ambiguous"):
        _validate_runtime_option_contract(
            (
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                "/tmp/checkpoint.pt",
                "--pdb_path",
                "/tmp/input.pdb",
                "--ligand_mpnn_use_atom_con",
                "0",
            )
        )


def test_pinned_runtime_preserves_pdb_basename_for_upstream_score_output(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    named_pdb = pdb.with_name("target-complex.pdb")
    pdb.rename(named_pdb)
    output_root = tmp_path / "scores"

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(named_pdb),
            "--out_folder",
            str(output_root),
        ),
    )

    published_score = output_root / "target-complex.pt"
    assert published_score.read_text(encoding="utf-8") == "input-v1"
    completion = json.loads((tmp_path / ".test-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert completion["score_output_sha256"] == f"sha256:{hashlib.sha256(published_score.read_bytes()).hexdigest()}"


def test_pinned_score_dot_output_keeps_private_attempt_inside_execution_root_with_unwritable_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = tmp_path / "container"
    execution_root = container / "workspace"
    execution_root.mkdir(parents=True)
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(execution_root)
    output_root = Path("seed_7")
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    monkeypatch.chdir(execution_root)
    container.chmod(0o500)
    try:
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )
    finally:
        container.chmod(0o700)

    assert (execution_root / output_root / "input.pt").read_text(encoding="utf-8") == "input-v1"
    assert (execution_root / completion_path).is_file()
    assert not tuple(execution_root.glob(".dnadesign-score-*"))
    assert not tuple(container.glob(".dnadesign-score-*"))


def test_pinned_score_runtime_rolls_back_publication_when_private_cleanup_fails_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_cleanup = pinned_runtime_module._cleanup_private_attempt_directory
    failed = False
    score_attempt_paths: list[Path] = []

    def _fail_first_score_cleanup(path: Path, identity: tuple[int, int], **kwargs: object) -> None:
        nonlocal failed
        if path.name.startswith(".dnadesign-score-"):
            score_attempt_paths.append(path)
            if not failed:
                failed = True
                raise OSError("simulated private score cleanup failure")
        original_cleanup(path, identity, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pinned_runtime_module, "_cleanup_private_attempt_directory", _fail_first_score_cleanup)
    arguments = (
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        str(checkpoint),
        "--pdb_path",
        str(pdb),
        "--out_folder",
        str(output_root),
    )

    with pytest.raises(ValueError, match="score attempt cleanup failed after publication"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=arguments,
        )

    assert failed
    assert not (output_root / "input.pt").exists()
    assert not completion_path.exists()
    assert score_attempt_paths
    assert all(path.parent == output_root.parent for path in score_attempt_paths)

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        arguments=arguments,
    )

    assert (output_root / "input.pt").read_text(encoding="utf-8") == "input-v1"
    assert completion_path.is_file()


def test_pinned_score_runtime_reports_uncertainty_when_cleanup_rollback_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    original_cleanup = pinned_runtime_module._cleanup_private_attempt_directory
    original_fsync = os.fsync
    cleanup_started = False

    def _fail_score_cleanup(path: Path, identity: tuple[int, int], **kwargs: object) -> None:
        nonlocal cleanup_started
        if path.name.startswith(".dnadesign-score-"):
            cleanup_started = True
            raise OSError("simulated private score cleanup failure")
        original_cleanup(path, identity, **kwargs)  # type: ignore[arg-type]

    def _fail_cleanup_rollback_sync(descriptor: int) -> None:
        if cleanup_started and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated cleanup rollback sync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(pinned_runtime_module, "_cleanup_private_attempt_directory", _fail_score_cleanup)
    monkeypatch.setattr(os, "fsync", _fail_cleanup_rollback_sync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="cleanup rollback durability is uncertain",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not (output_root / "input.pt").exists()


def test_pinned_score_cleanup_rollback_never_deletes_replacement_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    published_score = output_root / "input.pt"
    original_cleanup = pinned_runtime_module._cleanup_private_attempt_directory

    def _replace_score_then_fail_cleanup(path: Path, identity: tuple[int, int], **kwargs: object) -> None:
        if path.name.startswith(".dnadesign-score-"):
            published_score.unlink()
            published_score.write_text("unrelated replacement", encoding="utf-8")
            raise OSError("simulated cleanup failure after replacement")
        original_cleanup(path, identity, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pinned_runtime_module, "_cleanup_private_attempt_directory", _replace_score_then_fail_cleanup)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="cleanup rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert published_score.read_text(encoding="utf-8") == "unrelated replacement"


def test_pinned_score_cleanup_never_deletes_recreated_attempt_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    original_publish = pinned_runtime_module._publish_score_output
    displaced_attempt = tmp_path / "owned-score-attempt"
    recreated_attempt: Path | None = None

    def _publish_then_replace_attempt(source_path: Path, destination_path: Path) -> tuple[str, tuple[int, int]]:
        nonlocal recreated_attempt
        result = original_publish(source_path, destination_path)
        attempt_path = source_path.parent
        attempt_path.rename(displaced_attempt)
        attempt_path.mkdir()
        (attempt_path / "foreign.txt").write_text("foreign", encoding="utf-8")
        recreated_attempt = attempt_path
        return result

    monkeypatch.setattr(pinned_runtime_module, "_publish_score_output", _publish_then_replace_attempt)

    with pytest.raises(ValueError, match="score attempt cleanup failed after publication") as captured:
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert recreated_attempt is not None
    assert isinstance(captured.value.__cause__, pinned_runtime_module.LigandMpnnScorePublicationUncertainError)
    assert "score attempt cleanup target changed" in str(captured.value.__cause__)
    assert (recreated_attempt / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (displaced_attempt / "input.pt").read_text(encoding="utf-8") == "input-v1"
    assert not (output_root / "input.pt").exists()


def test_pinned_score_publication_binds_the_hashed_source_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    published_score = output_root / "input.pt"
    displaced_score = tmp_path / "hashed-score.pt"
    original_link = os.link

    def _replace_source_then_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        source_path = Path(source)
        if source_path.name == "input.pt" and source_path.parent.name.startswith(".dnadesign-score-"):
            source_path.rename(displaced_score)
            source_path.write_text("replacement-score", encoding="utf-8")
        original_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "link", _replace_source_then_link)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="score publication identity changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert displaced_score.read_text(encoding="utf-8") == "input-v1"
    assert published_score.read_text(encoding="utf-8") == "replacement-score"


def test_pinned_score_publication_rejects_source_replaced_by_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "score.pt"
    source.write_bytes(b"score")
    destination = tmp_path / "published" / "score.pt"
    destination.parent.mkdir()

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from contextlib import contextmanager\n"
                "from pathlib import Path\n"
                "import dnadesign.thread.adapters.ligandmpnn.pinned_runtime as module\n"
                "source, destination = Path(sys.argv[1]), Path(sys.argv[2])\n"
                "original = module.open_regular_file\n"
                "@contextmanager\n"
                "def replace_before_open(path):\n"
                "    if path == source:\n"
                "        path.unlink()\n"
                "        os.mkfifo(path)\n"
                "    with original(path) as handle:\n"
                "        yield handle\n"
                "module.open_regular_file = replace_before_open\n"
                "try:\n"
                "    module._publish_score_output(source, destination)\n"
                "except ValueError as error:\n"
                "    assert 'did not produce a regular output' in str(error)\n"
                "else:\n"
                "    raise SystemExit('nonregular score output was accepted')\n"
                "assert not destination.exists()\n"
            ),
            str(source),
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert probe.stderr == ""


def test_pinned_score_publication_rejects_fifo_linked_after_source_open_without_blocking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "score.pt"
    source.write_bytes(b"score")
    displaced_source = tmp_path / "opened-score.pt"
    destination = tmp_path / "published" / "score.pt"
    destination.parent.mkdir()

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, stat, sys\n"
                "from pathlib import Path\n"
                "import dnadesign.thread.adapters.ligandmpnn.pinned_runtime as module\n"
                "source, displaced, destination = map(Path, sys.argv[1:])\n"
                "original_link = os.link\n"
                "def replace_source_then_link(source_arg, destination_arg, *args, **kwargs):\n"
                "    source_path = Path(source_arg)\n"
                "    source_path.rename(displaced)\n"
                "    os.mkfifo(source_path)\n"
                "    original_link(source_arg, destination_arg, *args, **kwargs)\n"
                "os.link = replace_source_then_link\n"
                "try:\n"
                "    module._publish_score_output(source, destination)\n"
                "except module.LigandMpnnScorePublicationUncertainError as error:\n"
                "    assert 'could not be verified' in str(error)\n"
                "else:\n"
                "    raise SystemExit('replacement FIFO publication was accepted')\n"
                "assert displaced.read_bytes() == b'score'\n"
                "assert stat.S_ISFIFO(destination.lstat().st_mode)\n"
            ),
            str(source),
            str(displaced_source),
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert probe.stderr == ""


def test_pinned_score_publication_preserves_replacement_installed_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    published_score = output_root / "input.pt"
    displaced_score = tmp_path / "owned-linked-score.pt"
    original_link = os.link
    replacement_installed = False

    def _link_then_replace_destination(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replacement_installed
        original_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]
        source_path = Path(source)
        if source_path.name == "input.pt" and source_path.parent.name.startswith(".dnadesign-score-"):
            published_score.rename(displaced_score)
            published_score.write_text("post-link replacement", encoding="utf-8")
            replacement_installed = True

    monkeypatch.setattr(os, "link", _link_then_replace_destination)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="score publication identity changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert published_score.read_text(encoding="utf-8") == "post-link replacement"
    assert displaced_score.read_text(encoding="utf-8") == "input-v1"


def test_pinned_score_cleanup_failure_does_not_mask_original_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    original_cleanup = pinned_runtime_module._cleanup_private_attempt_directory
    original_run = subprocess.run

    def _fail_score_execution(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(command, list) and any(str(value).endswith("score.py") for value in command):
            raise subprocess.CalledProcessError(23, command)
        return original_run(command, *args, **kwargs)  # type: ignore[arg-type, return-value]

    def _fail_score_cleanup(path: Path, identity: tuple[int, int], **kwargs: object) -> None:
        if path.name.startswith(".dnadesign-score-"):
            raise OSError("simulated cleanup failure during execution failure")
        original_cleanup(path, identity, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", _fail_score_execution)
    monkeypatch.setattr(pinned_runtime_module, "_cleanup_private_attempt_directory", _fail_score_cleanup)

    with pytest.raises(subprocess.CalledProcessError) as captured:
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert any("private score attempt cleanup also failed" in note for note in captured.value.__notes__)
    assert not (output_root / "input.pt").exists()


@pytest.mark.parametrize(
    ("entrypoint", "output_root", "attempt_prefix"),
    [
        ("run.py", Path("designs/seed_7"), ".seed_7.attempt-"),
        ("score.py", Path("scores"), ".dnadesign-score-"),
    ],
)
def test_pinned_runtime_cleans_owned_attempt_after_upstream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    output_root: Path,
    attempt_prefix: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    absolute_output_root = tmp_path / output_root
    original_run = subprocess.run
    attempt_paths: list[Path] = []
    original_create = pinned_runtime_module._create_private_attempt_directory

    def _record_attempt(*, parent: Path, prefix: str) -> tuple[Path, tuple[int, int]]:
        result = original_create(parent=parent, prefix=prefix)
        attempt_paths.append(result[0])
        return result

    def _fail_upstream(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(command, list) and any(str(value).endswith(entrypoint) for value in command):
            raise subprocess.CalledProcessError(23, command)
        return original_run(command, *args, **kwargs)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(pinned_runtime_module, "_create_private_attempt_directory", _record_attempt)
    monkeypatch.setattr(subprocess, "run", _fail_upstream)

    with pytest.raises(subprocess.CalledProcessError):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint=entrypoint,
            completion_record_path=(
                absolute_output_root / ".dnadesign-ligandmpnn-execution.json"
                if entrypoint == "run.py"
                else tmp_path / ".test-ligandmpnn-execution.json"
            ),
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(absolute_output_root),
            ),
        )

    assert attempt_paths
    assert all(path.name.startswith(attempt_prefix) and not path.exists() for path in attempt_paths)


def test_pinned_design_cleanup_failure_does_not_mask_original_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    original_run = subprocess.run
    original_cleanup = pinned_runtime_module._cleanup_private_attempt_directory

    def _fail_design_execution(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(command, list) and any(str(value).endswith("run.py") for value in command):
            raise subprocess.CalledProcessError(23, command)
        return original_run(command, *args, **kwargs)  # type: ignore[arg-type, return-value]

    def _fail_design_cleanup(path: Path, identity: tuple[int, int], **kwargs: object) -> None:
        if path.name.startswith(".seed_7.attempt-"):
            raise OSError("simulated design cleanup failure during execution failure")
        original_cleanup(path, identity, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", _fail_design_execution)
    monkeypatch.setattr(pinned_runtime_module, "_cleanup_private_attempt_directory", _fail_design_cleanup)

    with pytest.raises(subprocess.CalledProcessError) as captured:
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=output_root / ".dnadesign-ligandmpnn-execution.json",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert any("private design attempt cleanup also failed" in note for note in captured.value.__notes__)


def test_pinned_score_runtime_retries_with_abandoned_legacy_private_attempt(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    abandoned = output_root / ".dnadesign-score-killed" / "partial.pt"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_text("killed-attempt", encoding="utf-8")

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="score.py",
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--out_folder",
            str(output_root),
        ),
    )

    assert (output_root / "input.pt").read_text(encoding="utf-8") == "input-v1"
    assert abandoned.read_text(encoding="utf-8") == "killed-attempt"


def test_pinned_design_runtime_rejects_preexisting_seed_output_lifecycle(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    output_root.mkdir(parents=True)
    stale_output = output_root / "stale-design.fa"
    stale_output.write_text("failed-attempt", encoding="utf-8")

    with pytest.raises(ValueError, match="design output directory already exists"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=output_root / ".dnadesign-ligandmpnn-execution.json",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert stale_output.read_text(encoding="utf-8") == "failed-attempt"
    assert not (output_root / "design.txt").exists()
    assert not (output_root / ".dnadesign-ligandmpnn-execution.json").exists()


def test_pinned_design_runtime_publishes_one_complete_attempt_owned_seed_directory(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    arguments = (
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        str(checkpoint),
        "--pdb_path",
        str(pdb),
        "--seed",
        "7",
        "--out_folder",
        str(output_root),
    )

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        completion_record_path=completion_path,
        arguments=arguments,
    )

    assert (output_root / "design.txt").read_text(encoding="utf-8") == "input-v1"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["execution"]["arguments"] == list(arguments)
    assert completion["score_output_sha256"] is None
    assert not tuple(output_root.parent.glob(".seed_7.attempt-*"))


def test_pinned_design_runtime_publishes_relative_generated_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_root = Path("designs/seed_7")
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        completion_record_path=completion_path,
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--out_folder",
            str(output_root),
        ),
    )

    assert (tmp_path / output_root / "design.txt").is_file()
    assert (tmp_path / completion_path).is_file()


def test_pinned_design_runtime_publishes_only_one_concurrent_seed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, first_pdb, _pdb_sha256 = _checkout(tmp_path)
    second_pdb = tmp_path / "second-input.pdb"
    first_pdb.write_text("first", encoding="utf-8")
    second_pdb.write_text("second", encoding="utf-8")
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    barrier = threading.Barrier(2)
    original_publish = pinned_runtime_module._publish_design_output_directory

    def _synchronize_publish(
        source_path: Path,
        destination_path: Path,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        barrier.wait(timeout=5)
        original_publish(source_path, destination_path, expected_identity=expected_identity)

    monkeypatch.setattr(pinned_runtime_module, "_publish_design_output_directory", _synchronize_publish)

    def _execute(label: str, input_path: Path) -> tuple[str, str]:
        try:
            execute_pinned_entrypoint(
                checkout_root=checkout,
                upstream_commit=commit,
                checkpoint_sha256=checkpoint_sha256,
                pdb_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                packing_checkpoint_sha256=None,
                residue_alphabet_sha256=None,
                entrypoint="run.py",
                completion_record_path=completion_path,
                arguments=(
                    "--model_type",
                    "ligand_mpnn",
                    "--checkpoint_ligand_mpnn",
                    str(checkpoint),
                    "--pdb_path",
                    str(input_path),
                    "--out_folder",
                    str(output_root),
                ),
            )
        except ValueError:
            return label, "rejected"
        return label, "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda item: _execute(*item),
                (("first", first_pdb), ("second", second_pdb)),
            )
        )

    completed = [label for label, status in outcomes if status == "completed"]
    assert len(completed) == 1
    assert (output_root / "design.txt").read_text(encoding="utf-8") == completed[0]
    assert completion_path.is_file()


def test_pinned_design_runtime_preserves_empty_directory_created_at_publication_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename_no_replace = pinned_runtime_module._rename_no_replace
    replacement_identity: tuple[int, int] | None = None

    def _install_empty_replacement_then_publish(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal replacement_identity
        if destination_name == output_root.name and replacement_identity is None:
            os.mkdir(destination_name, dir_fd=dst_dir_fd)
            status = os.stat(destination_name, dir_fd=dst_dir_fd, follow_symlinks=False)
            replacement_identity = (status.st_dev, status.st_ino)
        original_rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        pinned_runtime_module,
        "_rename_no_replace",
        _install_empty_replacement_then_publish,
    )

    with pytest.raises(ValueError, match="design output directory already exists"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_identity is not None
    observed = output_root.stat()
    assert (observed.st_dev, observed.st_ino) == replacement_identity
    assert not (output_root / "design.txt").exists()


def test_pinned_design_runtime_rejects_attempt_replaced_immediately_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    displaced_attempt = output_root.parent / "owned-attempt-recovery"
    original_rename_no_replace = pinned_runtime_module._rename_no_replace
    replacement_path: Path | None = None

    def _replace_checked_attempt_then_publish(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal replacement_path
        if destination_name == output_root.name and replacement_path is None:
            os.rename(
                source_name,
                displaced_attempt.name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            os.mkdir(source_name, dir_fd=src_dir_fd)
            replacement_path = output_root.parent / source_name
            replacement_fd = os.open(
                source_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=src_dir_fd,
            )
            try:
                foreign_fd = os.open(
                    "foreign.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=replacement_fd,
                )
                try:
                    os.write(foreign_fd, b"foreign replacement")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
                os.fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
        original_rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(pinned_runtime_module, "_rename_no_replace", _replace_checked_attempt_then_publish)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design attempt identity changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_path is not None
    assert (output_root / "foreign.txt").read_text(encoding="utf-8") == "foreign replacement"
    assert not replacement_path.exists()
    assert (displaced_attempt / "design.txt").read_text(encoding="utf-8") == "input-v1"


def test_pinned_design_runtime_preserves_replacement_installed_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    displaced_attempt = output_root.parent / "owned-published-recovery"
    original_rename_no_replace = pinned_runtime_module._rename_no_replace
    replacement_installed = False

    def _publish_then_replace_destination(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal replacement_installed
        original_rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_name == output_root.name and not replacement_installed:
            os.rename(
                destination_name,
                displaced_attempt.name,
                src_dir_fd=dst_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            os.mkdir(destination_name, dir_fd=dst_dir_fd)
            replacement_fd = os.open(
                destination_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=dst_dir_fd,
            )
            try:
                foreign_fd = os.open(
                    "foreign.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=replacement_fd,
                )
                try:
                    os.write(foreign_fd, b"post-rename replacement")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
                os.fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
            replacement_installed = True

    monkeypatch.setattr(pinned_runtime_module, "_rename_no_replace", _publish_then_replace_destination)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design attempt identity changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert (output_root / "foreign.txt").read_text(encoding="utf-8") == "post-rename replacement"
    assert (displaced_attempt / "design.txt").read_text(encoding="utf-8") == "input-v1"


def test_pinned_design_runtime_revalidates_child_manifest_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_manifest = pinned_runtime_module.build_design_output_manifest
    manifest_calls = 0

    def _replace_child_after_final_private_manifest(
        root: Path,
        *,
        expected_root_identity: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        nonlocal manifest_calls
        manifest = original_manifest(root, expected_root_identity=expected_root_identity)
        manifest_calls += 1
        if manifest_calls == 2:
            (root / "design.txt").write_text("post-manifest replacement", encoding="utf-8")
        return manifest

    monkeypatch.setattr(
        pinned_runtime_module,
        "build_design_output_manifest",
        _replace_child_after_final_private_manifest,
    )

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design output tree changed during atomic publication",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert manifest_calls >= 3
    assert (output_root / "design.txt").read_text(encoding="utf-8") == "post-manifest replacement"


def test_pinned_design_runtime_revalidates_completion_after_directory_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    displaced_completion = tmp_path / "owned-design-completion.json"
    original_publish = pinned_runtime_module._publish_design_output_directory
    replacement_installed = False

    def _replace_completion_then_publish(
        source_path: Path,
        destination_path: Path,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        nonlocal replacement_installed
        private_completion = source_path / completion_path.name
        private_completion.rename(displaced_completion)
        private_completion.write_text("foreign design completion", encoding="utf-8")
        replacement_installed = True
        original_publish(source_path, destination_path, expected_identity=expected_identity)

    monkeypatch.setattr(
        pinned_runtime_module,
        "_publish_design_output_directory",
        _replace_completion_then_publish,
    )

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design completion changed during atomic publication",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert completion_path.read_text(encoding="utf-8") == "foreign design completion"
    assert displaced_completion.is_file()


def test_pinned_design_runtime_preserves_recreated_attempt_after_successful_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_publish = pinned_runtime_module._publish_design_output_directory
    recreated_attempt: Path | None = None

    def _publish_then_recreate_attempt(
        source_path: Path,
        destination_path: Path,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        nonlocal recreated_attempt
        original_publish(source_path, destination_path, expected_identity=expected_identity)
        source_path.mkdir()
        (source_path / "foreign.txt").write_text("foreign", encoding="utf-8")
        recreated_attempt = source_path

    monkeypatch.setattr(
        pinned_runtime_module,
        "_publish_design_output_directory",
        _publish_then_recreate_attempt,
    )

    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=None,
        residue_alphabet_sha256=None,
        entrypoint="run.py",
        completion_record_path=completion_path,
        arguments=(
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(pdb),
            "--out_folder",
            str(output_root),
        ),
    )

    assert recreated_attempt is not None
    assert (recreated_attempt / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (output_root / "design.txt").read_text(encoding="utf-8") == "input-v1"


def test_pinned_design_publication_rejects_replaced_attempt_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    displaced_attempt = tmp_path / "original-design-attempt"
    replacement_attempt: Path | None = None
    original_run = subprocess.run

    def _replace_attempt_after_execution(
        command: object,
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal replacement_attempt
        result = original_run(command, *args, **kwargs)  # type: ignore[arg-type, return-value]
        if isinstance(command, list) and any(str(value).endswith("run.py") for value in command):
            attempt = Path(command[command.index("--out_folder") + 1])
            attempt.rename(displaced_attempt)
            attempt.mkdir()
            (attempt / "foreign.txt").write_text("foreign", encoding="utf-8")
            replacement_attempt = attempt
        return result

    monkeypatch.setattr(subprocess, "run", _replace_attempt_after_execution)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design attempt.*changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_attempt is not None
    assert (replacement_attempt / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (displaced_attempt / "design.txt").read_text(encoding="utf-8") == "input-v1"
    assert not output_root.exists()


def test_pinned_design_rollback_collision_never_deletes_foreign_attempt_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename = pinned_runtime_module._rename_no_replace
    original_fsync = os.fsync
    published = False
    publication_sync_failed = False
    foreign_attempt: Path | None = None

    def _rename_with_restore_collision(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal published, foreign_attempt
        if destination_name == output_root.name:
            original_rename(
                source_name,
                destination_name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            published = True
            return
        if source_name == "publication" and destination_name.startswith(".seed_7.attempt-"):
            os.mkdir(destination_name, dir_fd=dst_dir_fd)
            foreign_attempt = output_root.parent / destination_name
            (foreign_attempt / "foreign.txt").write_text("foreign", encoding="utf-8")
        original_rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def _fail_published_parent_sync(descriptor: int) -> None:
        nonlocal publication_sync_failed
        if published and not publication_sync_failed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            publication_sync_failed = True
            raise OSError("simulated publication durability failure")
        original_fsync(descriptor)

    monkeypatch.setattr(pinned_runtime_module, "_rename_no_replace", _rename_with_restore_collision)
    monkeypatch.setattr(os, "fsync", _fail_published_parent_sync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="displaced leaf retained",
    ) as captured:
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert foreign_attempt is not None
    assert (foreign_attempt / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert any("private design attempt cleanup also failed" in note for note in captured.value.__notes__)


def test_pinned_design_runtime_rolls_back_whole_directory_when_parent_fsync_fails_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename_no_replace = pinned_runtime_module._rename_no_replace
    original_fsync = os.fsync
    published = False
    failure_injected = False

    def _record_publication_rename(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal published
        original_rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_name == output_root.name:
            published = True

    def _fail_published_parent_fsync_once(file_descriptor: int) -> None:
        nonlocal failure_injected
        if published and not failure_injected and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            failure_injected = True
            assert (output_root / "design.txt").is_file()
            assert completion_path.is_file()
            raise OSError("simulated design parent fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(pinned_runtime_module, "_rename_no_replace", _record_publication_rename)
    monkeypatch.setattr(os, "fsync", _fail_published_parent_fsync_once)

    with pytest.raises(ValueError, match="design output publication could not be made durable"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert failure_injected
    assert not output_root.exists()
    assert not tuple(output_root.parent.glob(".seed_7.attempt-*"))


def test_pinned_design_runtime_preserves_concurrent_replacement_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    displaced_output = tmp_path / "displaced-published-design"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_published_tree_then_fail(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if not replacement_installed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode) and completion_path.is_file():
            output_root.rename(displaced_output)
            output_root.mkdir()
            replacement = output_root / "replacement-design.fa"
            replacement.write_text("concurrent replacement", encoding="utf-8")
            with replacement.open("rb") as handle:
                original_fsync(handle.fileno())
            replacement_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                original_fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
            original_fsync(file_descriptor)
            replacement_installed = True
            raise OSError("simulated design parent fsync failure after replacement")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_published_tree_then_fail)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert (output_root / "replacement-design.fa").read_text(encoding="utf-8") == "concurrent replacement"
    assert (displaced_output / "design.txt").read_text(encoding="utf-8") == "input-v1"
    assert not tuple(output_root.parent.glob(".seed_7.attempt-*"))


def test_pinned_design_runtime_reports_uncertainty_when_directory_rollback_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename_no_replace = pinned_runtime_module._rename_no_replace
    original_fsync = os.fsync
    published = False

    def _record_publication_rename(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal published
        original_rename_no_replace(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_name == output_root.name:
            published = True

    def _fail_published_and_rollback_parent_fsync(file_descriptor: int) -> None:
        if published and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError("simulated persistent design parent fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(pinned_runtime_module, "_rename_no_replace", _record_publication_rename)
    monkeypatch.setattr(os, "fsync", _fail_published_and_rollback_parent_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnDesignPublicationUncertainError,
        match="design publication rollback durability is uncertain",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            completion_record_path=completion_path,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not output_root.exists()
    assert not tuple(output_root.parent.glob(".seed_7.attempt-*"))


@pytest.mark.parametrize("leaf_kind", ["fifo", "socket"])
def test_design_output_tree_rejects_nonregular_leaf_without_blocking(leaf_kind: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ligandmpnn-nonregular-", dir="/tmp") as raw_root:
        root = Path(raw_root)
        leaf = root / "upstream-output"
        listener: socket.socket | None = None
        if leaf_kind == "fifo":
            os.mkfifo(leaf)
        else:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(leaf))
        try:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys\n"
                        "from pathlib import Path\n"
                        "from dnadesign.thread.adapters.ligandmpnn.pinned_runtime "
                        "import _sync_regular_directory_tree\n"
                        "try:\n"
                        "    _sync_regular_directory_tree(Path(sys.argv[1]))\n"
                        "except ValueError as error:\n"
                        "    print(error)\n"
                        "else:\n"
                        "    raise SystemExit('nonregular output was accepted')\n"
                    ),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        finally:
            if listener is not None:
                listener.close()

    assert "design output could not be synced safely" in probe.stdout


def test_pinned_score_runtime_rolls_back_link_when_score_directory_fsync_fails_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_link = os.link
    original_fsync = os.fsync
    score_link_created = False
    score_directory_failure_injected = False

    def _record_score_link(*args: object, **kwargs: object) -> None:
        nonlocal score_link_created
        original_link(*args, **kwargs)  # type: ignore[arg-type]
        score_link_created = True

    def _fail_score_directory_fsync_once(file_descriptor: int) -> None:
        nonlocal score_directory_failure_injected
        if (
            score_link_created
            and not score_directory_failure_injected
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
        ):
            score_directory_failure_injected = True
            assert (output_root / "input.pt").is_file()
            raise OSError("simulated score directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "link", _record_score_link)
    monkeypatch.setattr(os, "fsync", _fail_score_directory_fsync_once)

    with pytest.raises(ValueError, match="score output publication could not be made durable"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert score_directory_failure_injected
    assert not (output_root / "input.pt").exists()
    assert not completion_path.exists()


def test_pinned_score_publication_rollback_preserves_concurrent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_score_then_fail_directory_fsync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if not replacement_installed and stat.S_ISDIR(os.fstat(file_descriptor).st_mode) and published_score.is_file():
            published_score.unlink()
            published_score.write_text("concurrent replacement", encoding="utf-8")
            with published_score.open("rb") as handle:
                original_fsync(handle.fileno())
            replacement_installed = True
            raise OSError("simulated score directory fsync failure after replacement")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_score_then_fail_directory_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="score publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert published_score.read_text(encoding="utf-8") == "concurrent replacement"


def test_pinned_score_runtime_revalidates_score_after_publication_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_score_during_successful_publication_sync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if (
            not replacement_installed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and not completion_path.exists()
        ):
            published_score.unlink()
            published_score.write_text("concurrent replacement", encoding="utf-8")
            with published_score.open("rb") as handle:
                original_fsync(handle.fileno())
            replacement_installed = True
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_score_during_successful_publication_sync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="durable score publication identity changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert published_score.read_text(encoding="utf-8") == "concurrent replacement"
    assert not completion_path.exists()


def test_pinned_score_runtime_reports_uncertainty_when_score_rollback_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_link = os.link
    original_fsync = os.fsync
    score_link_created = False

    def _record_score_link(*args: object, **kwargs: object) -> None:
        nonlocal score_link_created
        original_link(*args, **kwargs)  # type: ignore[arg-type]
        score_link_created = True

    def _fail_score_and_rollback_directory_fsync(file_descriptor: int) -> None:
        if score_link_created and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError("simulated persistent score directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "link", _record_score_link)
    monkeypatch.setattr(os, "fsync", _fail_score_and_rollback_directory_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnScorePublicationUncertainError,
        match="score publication rollback durability is uncertain",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not (output_root / "input.pt").exists()
    assert not completion_path.exists()


def test_open_directory_path_syncs_each_parent_receiving_a_new_seed_directory_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "batch" / "seed-42"
    original_mkdir = os.mkdir
    original_fsync = os.fsync
    receiving_parent_ids: list[tuple[int, int]] = []
    synced_directory_ids: list[tuple[int, int]] = []

    def _record_mkdir(
        path: str | bytes,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        assert dir_fd is not None
        parent_stat = os.fstat(dir_fd)
        receiving_parent_ids.append((parent_stat.st_dev, parent_stat.st_ino))
        original_mkdir(path, mode=mode, dir_fd=dir_fd)

    def _record_fsync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            synced_directory_ids.append((descriptor_stat.st_dev, descriptor_stat.st_ino))
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "mkdir", _record_mkdir)
    monkeypatch.setattr(os, "fsync", _record_fsync)

    directory_fd = pinned_runtime_module._open_directory_path(output_root, create=True)
    os.close(directory_fd)

    assert len(receiving_parent_ids) == 2
    assert set(receiving_parent_ids) <= set(synced_directory_ids)


def test_pinned_score_runtime_rolls_back_output_when_completion_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    directory_fsync_count = 0

    def _fail_completion_directory_fsync(file_descriptor: int) -> None:
        nonlocal directory_fsync_count
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            directory_fsync_count += 1
            if (output_root / "input.pt").is_file() and completion_path.is_file():
                raise OSError("simulated completion directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _fail_completion_directory_fsync)

    with pytest.raises(ValueError, match="completion record publication could not be made durable"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert directory_fsync_count >= 2
    assert not (output_root / "input.pt").exists()
    assert not completion_path.exists()


def test_pinned_score_completion_rollback_preserves_concurrent_score_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_score_then_fail_completion_fsync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if (
            not replacement_installed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            published_score.unlink()
            published_score.write_text("concurrent replacement", encoding="utf-8")
            with published_score.open("rb") as handle:
                original_fsync(handle.fileno())
            replacement_installed = True
            raise OSError("simulated completion fsync failure after score replacement")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_score_then_fail_completion_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication output rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert published_score.read_text(encoding="utf-8") == "concurrent replacement"
    assert not completion_path.exists()


def test_pinned_score_runtime_revalidates_score_after_completion_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_score_during_successful_completion_sync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if (
            not replacement_installed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            published_score.unlink()
            published_score.write_text("concurrent replacement", encoding="utf-8")
            with published_score.open("rb") as handle:
                original_fsync(handle.fileno())
            replacement_installed = True
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_score_during_successful_completion_sync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication output rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert published_score.read_text(encoding="utf-8") == "concurrent replacement"
    assert not completion_path.exists()


def test_pinned_score_runtime_revalidates_completion_after_output_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    displaced_completion = tmp_path / "owned-completion.json"
    original_matches = pinned_runtime_module._owned_regular_leaf_matches_sha256
    replacement_installed = False

    def _replace_completion_after_output_validation(*args: object, **kwargs: object) -> bool:
        nonlocal replacement_installed
        owned = original_matches(*args, **kwargs)
        if owned and completion_path.is_file() and not replacement_installed:
            completion_path.rename(displaced_completion)
            completion_path.write_text("foreign completion", encoding="utf-8")
            replacement_installed = True
        return owned

    monkeypatch.setattr(
        pinned_runtime_module,
        "_owned_regular_leaf_matches_sha256",
        _replace_completion_after_output_validation,
    )

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert completion_path.read_text(encoding="utf-8") == "foreign completion"
    assert published_score.read_text(encoding="utf-8") == "input-v1"
    assert displaced_completion.is_file()


def test_pinned_score_completion_rollback_preserves_concurrent_completion_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    original_open = os.open
    completion_descriptor: int | None = None
    replacement_installed = False

    def _track_completion_descriptor(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o600,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal completion_descriptor
        file_descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == completion_path.name and flags & os.O_EXCL:
            completion_descriptor = file_descriptor
        return file_descriptor

    def _replace_completion_then_fail_fsync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if (
            not replacement_installed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            assert completion_descriptor is not None
            published_status = os.fstat(completion_descriptor)
            completion_path.unlink()
            completion_path.write_text("concurrent completion", encoding="utf-8")
            with completion_path.open("rb") as handle:
                original_fsync(handle.fileno())
                replacement_status = os.fstat(handle.fileno())
            assert (replacement_status.st_dev, replacement_status.st_ino) != (
                published_status.st_dev,
                published_status.st_ino,
            )
            replacement_installed = True
            raise OSError("simulated completion fsync failure after completion replacement")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "open", _track_completion_descriptor)
    monkeypatch.setattr(os, "fsync", _replace_completion_then_fail_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert completion_path.read_text(encoding="utf-8") == "concurrent completion"
    assert published_score.read_text(encoding="utf-8") == "input-v1"


def test_rollback_quarantine_restores_foreign_directory_replacement(
    tmp_path: Path,
) -> None:
    public_path = tmp_path / "publication"
    public_path.write_bytes(b"owned")
    public_status = public_path.stat()
    expected_identity = (public_status.st_dev, public_status.st_ino)
    public_path.unlink()
    public_path.mkdir()
    (public_path / "foreign.txt").write_text("foreign", encoding="utf-8")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
            match="rollback target changed",
        ):
            pinned_runtime_module._quarantine_and_remove_owned_leaf(
                directory_fd,
                public_path.name,
                expected_identity,
                expected_bytes=b"owned",
                expected_sha256=None,
                error_type=pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
                changed_message="rollback target changed",
                inspect_message="rollback target could not be inspected",
                durability_message="rollback durability is uncertain",
            )
    finally:
        os.close(directory_fd)

    assert (public_path / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not tuple(tmp_path.glob(".dnadesign-rollback-*"))


def test_directory_no_replace_rename_preserves_existing_empty_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "attempt"
    source.mkdir()
    (source / "design.txt").write_text("owned", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir()
    destination_status = destination.stat()
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FileExistsError):
            pinned_runtime_module._rename_no_replace(
                source.name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
    finally:
        os.close(parent_fd)

    observed_destination = destination.stat()
    assert (observed_destination.st_dev, observed_destination.st_ino) == (
        destination_status.st_dev,
        destination_status.st_ino,
    )
    assert (source / "design.txt").read_text(encoding="utf-8") == "owned"


@pytest.mark.parametrize("platform", ["freebsd13", "openbsd7"])
def test_pinned_runtime_rejects_unsupported_atomic_publish_before_upstream_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256(PINNED_CONTEXT_PARSER_PAYLOAD).hexdigest(),
    )
    upstream_called = False

    def _reject_upstream_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal upstream_called
        upstream_called = True
        raise AssertionError("upstream execution must not start")

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(subprocess, "run", _reject_upstream_run)

    with pytest.raises(ValueError, match="atomic no-replace publication is unavailable"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            context_inventory_path=reference.path,
            context_inventory_sha256=reference.sha256,
            execution_root=tmp_path,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not upstream_called
    assert not output_root.parent.exists()


def test_pinned_runtime_rejects_missing_atomic_publish_symbol_before_upstream_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    reference = write_context_inventory(
        tmp_path,
        input_path=pdb.relative_to(tmp_path),
        input_sha256=pdb_sha256,
        upstream_commit=commit,
        parse_all_atoms=False,
        parser_sha256=hashlib.sha256(PINNED_CONTEXT_PARSER_PAYLOAD).hexdigest(),
    )
    upstream_called = False

    def _missing_symbol(*args: object, **kwargs: object) -> object:
        return object()

    def _reject_upstream_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal upstream_called
        upstream_called = True
        raise AssertionError("upstream execution must not start")

    monkeypatch.setattr(ctypes, "CDLL", _missing_symbol)
    monkeypatch.setattr(subprocess, "run", _reject_upstream_run)

    with pytest.raises(ValueError, match="atomic no-replace publication is unavailable"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            context_inventory_path=reference.path,
            context_inventory_sha256=reference.sha256,
            execution_root=tmp_path,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert not upstream_called
    assert not output_root.parent.exists()


def test_pinned_score_completion_rollback_preserves_in_place_completion_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    replacement_installed = False

    def _replace_completion_then_fail_fsync(file_descriptor: int) -> None:
        nonlocal replacement_installed
        if (
            not replacement_installed
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            with completion_path.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"concurrent completion")
                handle.truncate()
                handle.flush()
                original_fsync(handle.fileno())
            replacement_installed = True
            raise OSError("simulated completion fsync failure after in-place replacement")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _replace_completion_then_fail_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert replacement_installed
    assert completion_path.read_text(encoding="utf-8") == "concurrent completion"
    assert published_score.read_text(encoding="utf-8") == "input-v1"


def test_pinned_score_completion_rollback_preserves_replacement_after_completion_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    original_matches = pinned_runtime_module._owned_regular_leaf_matches_bytes
    failure_injected = False
    replacement_installed = False

    def _fail_completion_directory_fsync(file_descriptor: int) -> None:
        nonlocal failure_injected
        if (
            not failure_injected
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            failure_injected = True
            raise OSError("simulated completion directory fsync failure")
        original_fsync(file_descriptor)

    def _replace_after_validation(*args: object, **kwargs: object) -> bool:
        nonlocal replacement_installed
        owned = original_matches(*args, **kwargs)
        if owned and not replacement_installed:
            completion_path.write_text("concurrent completion", encoding="utf-8")
            replacement_installed = True
        return owned

    monkeypatch.setattr(os, "fsync", _fail_completion_directory_fsync)
    monkeypatch.setattr(
        pinned_runtime_module,
        "_owned_regular_leaf_matches_bytes",
        _replace_after_validation,
    )

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert failure_injected
    assert replacement_installed
    assert completion_path.read_text(encoding="utf-8") == "concurrent completion"
    assert published_score.read_text(encoding="utf-8") == "input-v1"


def test_pinned_score_completion_rollback_preserves_replacement_after_output_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    published_score = output_root / "input.pt"
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    original_matches = pinned_runtime_module._owned_regular_leaf_matches_sha256
    failure_injected = False
    replacement_installed = False

    def _fail_completion_directory_fsync(file_descriptor: int) -> None:
        nonlocal failure_injected
        if (
            not failure_injected
            and stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            and published_score.is_file()
            and completion_path.is_file()
        ):
            failure_injected = True
            raise OSError("simulated completion directory fsync failure")
        original_fsync(file_descriptor)

    def _replace_after_validation(*args: object, **kwargs: object) -> bool:
        nonlocal replacement_installed
        owned = original_matches(*args, **kwargs)
        if owned and not replacement_installed:
            published_score.write_text("concurrent score", encoding="utf-8")
            replacement_installed = True
        return owned

    monkeypatch.setattr(os, "fsync", _fail_completion_directory_fsync)
    monkeypatch.setattr(
        pinned_runtime_module,
        "_owned_regular_leaf_matches_sha256",
        _replace_after_validation,
    )

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication output rollback target changed",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert failure_injected
    assert replacement_installed
    assert not completion_path.exists()
    assert published_score.read_text(encoding="utf-8") == "concurrent score"


def test_pinned_score_runtime_reports_uncertainty_when_completion_rollback_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    completion_path = tmp_path / ".test-ligandmpnn-execution.json"
    original_fsync = os.fsync
    directory_fsync_count = 0
    completion_failure_started = False

    def _fail_completion_and_rollback_directory_fsync(file_descriptor: int) -> None:
        nonlocal completion_failure_started, directory_fsync_count
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            directory_fsync_count += 1
            if completion_failure_started or completion_path.is_file():
                completion_failure_started = True
                raise OSError("simulated persistent completion directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", _fail_completion_and_rollback_directory_fsync)

    with pytest.raises(
        pinned_runtime_module.LigandMpnnCompletionPublicationUncertainError,
        match="completion publication rollback durability is uncertain",
    ):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert (output_root / "input.pt").read_text(encoding="utf-8") == "input-v1"
    assert not completion_path.exists()
    recoveries = list(tmp_path.glob(".dnadesign-rollback-*"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "publication").is_file()


def test_pinned_score_runtime_publishes_only_one_concurrent_same_name_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, _pdb_sha256 = _checkout(tmp_path)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_pdb = second_root / pdb.name
    pdb.write_text("first", encoding="utf-8")
    second_pdb.write_text("second", encoding="utf-8")
    output_root = tmp_path / "scores"
    barrier = threading.Barrier(2)
    original_reject = pinned_runtime_module._reject_existing_score_output

    def _synchronize(
        arguments: list[str],
        *,
        pdb_path: Path,
        execution_root: Path,
    ) -> tuple[int, Path, Path]:
        publication = original_reject(arguments, pdb_path=pdb_path, execution_root=execution_root)
        barrier.wait(timeout=5)
        return publication

    monkeypatch.setattr(pinned_runtime_module, "_reject_existing_score_output", _synchronize)

    def _execute(label: str, input_path: Path) -> tuple[str, str]:
        arguments = (
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint),
            "--pdb_path",
            str(input_path),
            "--out_folder",
            str(output_root),
        )
        try:
            execute_pinned_entrypoint(
                checkout_root=checkout,
                upstream_commit=commit,
                checkpoint_sha256=checkpoint_sha256,
                pdb_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                packing_checkpoint_sha256=None,
                residue_alphabet_sha256=None,
                entrypoint="score.py",
                arguments=arguments,
            )
        except ValueError:
            return label, "rejected"
        return label, "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda item: _execute(*item), (("first", pdb), ("second", second_pdb))))

    completed = [label for label, status in outcomes if status == "completed"]
    assert len(completed) == 1
    assert (output_root / "input.pt").read_text(encoding="utf-8") == completed[0]


def test_pinned_runtime_rejects_preexisting_score_output(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "scores"
    output_root.mkdir()
    stale_output = output_root / "input.pt"
    stale_output.write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="refuse stale or ambiguous result"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="score.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--out_folder",
                str(output_root),
            ),
        )

    assert stale_output.read_text(encoding="utf-8") == "stale"


@pytest.mark.parametrize(
    ("attached_argument", "message"),
    [
        ("--checkpoint_ligand_mpnn=/unattested.pt", "split form of --checkpoint_ligand_mpnn"),
        ("--pdb_path=/unattested.pdb", "split form of --pdb_path"),
    ],
)
def test_pinned_runtime_rejects_attached_duplicate_file_flags(
    tmp_path: Path,
    attached_argument: str,
    message: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)

    with pytest.raises(ValueError, match=message):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                attached_argument,
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


@pytest.mark.parametrize(
    "duplicate_arguments",
    [
        ("--ligand_mpnn_use_atom_context", "0"),
        ("--ligand_mpnn_use_atom_context=0",),
        ("--ligand_mpnn_use_atom_con", "0"),
    ],
)
def test_pinned_runtime_rejects_duplicate_semantic_flags(
    tmp_path: Path,
    duplicate_arguments: tuple[str, ...],
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)

    with pytest.raises(ValueError, match="duplicate LigandMPNN runtime option|unattested or ambiguous"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--ligand_mpnn_use_atom_context",
                "1",
                *duplicate_arguments,
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


@pytest.mark.parametrize("entrypoint", ["run.py", "score.py"])
def test_pinned_runtime_rejects_zero_seed_before_upstream_random_fallback(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)

    with pytest.raises(ValueError, match="--seed must be an integer from 1 through 4294967295"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint=entrypoint,
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                "--seed",
                "0",
                "--out_folder",
                str(tmp_path / "outputs"),
            ),
        )


@pytest.mark.parametrize(
    ("attached_argument", "message"),
    [
        ("--checkpoint_path_sc=/unattested.pt", "packing checkpoint was supplied without a pinned digest"),
        (
            "--omit_AA_per_residue=/unattested.json",
            "residue alphabet sidecar was supplied without a pinned digest",
        ),
    ],
)
def test_pinned_runtime_rejects_unpinned_attached_optional_file_flags(
    tmp_path: Path,
    attached_argument: str,
    message: str,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)

    with pytest.raises(ValueError, match=message):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                attached_argument,
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ("--checkpoint_ligand_m=/unattested.pt",),
        ("--checkpoint_protein_mpnn", "/unattested.pt"),
        ("--pdb_path_multi", "/unattested.json"),
        ("--fixed_residues_multi", "/unattested.json"),
        ("--fixed_residues_m", "/ambiguous.json"),
        ("--redesigned_residues_multi", "/unattested.json"),
        ("--redesigned_residues_m", "/ambiguous.json"),
        ("--bias_AA_per_residue", "/unattested.json"),
        ("--bias_AA_per_residue_multi", "/unattested.json"),
        ("--omit_AA_per_residue_multi", "/unattested.json"),
        ("--model_type", "protein_mpnn"),
        ("--model_type=protein_mpnn",),
    ],
)
def test_pinned_runtime_rejects_abbreviated_or_alternate_attestation_flags(
    tmp_path: Path,
    extra_arguments: tuple[str, ...],
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)

    with pytest.raises(ValueError, match="unattested or ambiguous LigandMPNN runtime option"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            checkpoint_sha256=checkpoint_sha256,
            pdb_sha256=pdb_sha256,
            packing_checkpoint_sha256=None,
            residue_alphabet_sha256=None,
            entrypoint="run.py",
            arguments=(
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                str(checkpoint),
                "--pdb_path",
                str(pdb),
                *extra_arguments,
                "--output",
                str(tmp_path / "output.txt"),
            ),
        )
