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
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn import (
    LigandMpnnCommand,
    LigandMpnnRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_commands,
    parse_ligandmpnn_design_outputs,
)
from dnadesign.thread.tests.adapters.ligandmpnn._context_inventory import write_context_inventory
from dnadesign.thread.tests.adapters.ligandmpnn.test_pinned_runtime import _checkout


def _execute_design(tmp_path: Path) -> tuple[LigandMpnnRequest, tuple[LigandMpnnCommand, ...], Path]:
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
        request_id="admit_design",
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
    commands = build_ligandmpnn_commands(
        request,
        checkout_root=checkout,
        execution_root=tmp_path,
        python_executable=sys.executable,
    )
    subprocess.run(commands[0].argv, cwd=tmp_path, check=True)
    return request, commands, tmp_path / commands[0].output_dir


def test_design_admission_binds_exact_published_tree(tmp_path: Path) -> None:
    request, commands, output_root = _execute_design(tmp_path)

    result = parse_ligandmpnn_design_outputs(
        request,
        commands,
        execution_root=tmp_path,
    )

    completion = json.loads((output_root / ".dnadesign-ligandmpnn-execution.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == 3
    assert completion["design_output_manifest"] == result.outputs[0].manifest
    assert result.outputs[0].manifest["entries"] == [
        {
            "path": "design.txt",
            "type": "file",
            "size_bytes": len(b"input-v1"),
            "sha256": f"sha256:{hashlib.sha256(b'input-v1').hexdigest()}",
        }
    ]


@pytest.mark.parametrize("mutation", ["edit", "replace", "add", "delete"])
def test_design_admission_rejects_artifact_tree_mutation(tmp_path: Path, mutation: str) -> None:
    request, commands, output_root = _execute_design(tmp_path)
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
    request, commands, output_root = _execute_design(tmp_path)
    (output_root / "foreign-link").symlink_to(output_root / "design.txt")

    with pytest.raises(ValueError, match="design output entry must be regular"):
        parse_ligandmpnn_design_outputs(
            request,
            commands,
            execution_root=tmp_path,
        )
