"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_pinned_runtime.py

Tests attested LigandMPNN entrypoint execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dnadesign.thread.adapters.ligandmpnn.pinned_runtime as pinned_runtime_module
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import (
    _validate_runtime_option_contract,
    build_pinned_runtime_command,
    pinned_execution_sha256,
)
from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import (
    execute_pinned_entrypoint as _execute_pinned_entrypoint,
)


def execute_pinned_entrypoint(**kwargs: object) -> None:
    """Exercise the direct runtime boundary with a canonical planned execution."""

    arguments = kwargs["arguments"]
    assert isinstance(arguments, tuple)
    checkout_root = kwargs["checkout_root"]
    assert isinstance(checkout_root, Path)
    completion_record_path = kwargs.pop(
        "completion_record_path",
        checkout_root.parent / ".test-ligandmpnn-execution.json",
    )
    assert isinstance(completion_record_path, Path)
    planned_arguments = kwargs.pop("planned_arguments", arguments)
    assert isinstance(planned_arguments, tuple)
    planned_execution_sha256 = pinned_execution_sha256(
        checkout_root=checkout_root,
        upstream_commit=str(kwargs["upstream_commit"]),
        checkpoint_sha256=str(kwargs["checkpoint_sha256"]),
        pdb_sha256=str(kwargs["pdb_sha256"]),
        packing_checkpoint_sha256=kwargs["packing_checkpoint_sha256"],  # type: ignore[arg-type]
        residue_alphabet_sha256=kwargs["residue_alphabet_sha256"],  # type: ignore[arg-type]
        entrypoint=str(kwargs["entrypoint"]),
        completion_record_path=completion_record_path,
        arguments=planned_arguments,
    )
    _execute_pinned_entrypoint(
        **kwargs,  # type: ignore[arg-type]
        planned_execution_sha256=planned_execution_sha256,
        completion_record_path=completion_record_path,
    )


def _checkout(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    root = tmp_path / "LigandMPNN"
    root.mkdir()
    (root / "data_utils.py").write_text("VALUE = 'attested'\n", encoding="utf-8")
    (root / "model_utils.py").write_text("HELPER = 'helper-attested'\n", encoding="utf-8")
    (root / "run.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "from data_utils import VALUE\n"
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
        "args = parser.parse_args()\n"
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
        "    (output_root / 'design.txt').write_text(pdb, encoding='utf-8')\n",
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
        "args = parser.parse_args()\n"
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


def test_pinned_runtime_ignores_timestamp_valid_poisoned_parser_bytecode(tmp_path: Path) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    parser_path = checkout / "data_utils.py"
    malicious_source = tmp_path / "data_utils.py"
    malicious_source.write_text("VALUE = 'poisoned'\n", encoding="utf-8")
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
        "--output",
        str(output),
    )
    command = build_pinned_runtime_command(
        checkout_root=checkout,
        upstream_commit=commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
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
    assert completion["execution_sha256"].startswith("sha256:")


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

    def _synchronize_publish(source_path: Path, destination_path: Path) -> None:
        barrier.wait(timeout=5)
        original_publish(source_path, destination_path)

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


def test_pinned_design_runtime_rolls_back_whole_directory_when_parent_fsync_fails_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename = os.rename
    original_fsync = os.fsync
    published = False
    failure_injected = False

    def _record_publication_rename(*args: object, **kwargs: object) -> None:
        nonlocal published
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        if args[1] == output_root.name:
            published = True

    def _fail_published_parent_fsync_once(file_descriptor: int) -> None:
        nonlocal failure_injected
        if published and not failure_injected and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            failure_injected = True
            assert (output_root / "design.txt").is_file()
            assert completion_path.is_file()
            raise OSError("simulated design parent fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "rename", _record_publication_rename)
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


def test_pinned_design_runtime_reports_uncertainty_when_directory_rollback_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, commit, checkpoint, checkpoint_sha256, pdb, pdb_sha256 = _checkout(tmp_path)
    output_root = tmp_path / "designs" / "seed_7"
    completion_path = output_root / ".dnadesign-ligandmpnn-execution.json"
    original_rename = os.rename
    original_fsync = os.fsync
    published = False

    def _record_publication_rename(*args: object, **kwargs: object) -> None:
        nonlocal published
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        if args[1] == output_root.name:
            published = True

    def _fail_published_and_rollback_parent_fsync(file_descriptor: int) -> None:
        if published and stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError("simulated persistent design parent fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "rename", _record_publication_rename)
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
            if directory_fsync_count == 2:
                assert (output_root / "input.pt").is_file()
                assert completion_path.is_file()
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

    def _fail_completion_and_rollback_directory_fsync(file_descriptor: int) -> None:
        nonlocal directory_fsync_count
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            directory_fsync_count += 1
            if directory_fsync_count >= 2:
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

    assert not (output_root / "input.pt").exists()
    assert not completion_path.exists()


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

    def _synchronize(arguments: list[str], *, pdb_path: Path) -> tuple[int, Path, Path]:
        publication = original_reject(arguments, pdb_path=pdb_path)
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
