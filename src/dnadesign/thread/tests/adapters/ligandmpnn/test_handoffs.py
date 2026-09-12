"""Public JSON handoffs preserve producer validation and full probabilities."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from dnadesign.thread.adapters.ligandmpnn.context_probe import materialize_ligandmpnn_context_inventory
from dnadesign.thread.adapters.ligandmpnn.handoff_requests import context_request_from_document
from dnadesign.thread.adapters.ligandmpnn.handoffs import (
    normalize_score_plan,
    plan_designs,
    plan_scores,
    score_request_document,
    score_request_from_document,
)
from dnadesign.thread.tests.adapters.ligandmpnn.test_score_results import (
    _parse,
    _prepare_request,
    _write_output,
)


def test_score_handoff_roundtrip_preserves_full_21_state_values(tmp_path):
    request = _prepare_request(tmp_path)
    document = score_request_document(request)
    assert score_request_from_document(document) == request
    plan = plan_scores(document, checkout_root=tmp_path / "LigandMPNN", execution_root=tmp_path)
    for seed in request.seeds:
        _write_output(tmp_path, request, seed)
    exported = normalize_score_plan(plan, execution_root=tmp_path, trust="pinned_local_execution")
    result = _parse(tmp_path, request)
    assert exported["result"] == result.to_dict()
    for artifact, output in zip(exported["probability_artifacts"], result.outputs, strict=True):
        values = np.asarray(artifact["values"], dtype=artifact["dtype"])
        assert values.shape == tuple(artifact["shape"])
        np.testing.assert_array_equal(values, output.raw_probabilities)
        assert values.shape[-1] == 21
        assert artifact["residue_names"] == list(output.residue_names)


def test_score_handoff_rejects_schema_drift(tmp_path):
    document = score_request_document(_prepare_request(tmp_path))
    document["silent_override"] = True
    with pytest.raises(ValueError, match="fields"):
        score_request_from_document(document)


def test_score_normalization_rejects_modified_plan_and_absent_trust(tmp_path):
    request = _prepare_request(tmp_path)
    plan = plan_scores(score_request_document(request), checkout_root=tmp_path / "LigandMPNN", execution_root=tmp_path)
    with pytest.raises(ValueError, match="trust"):
        normalize_score_plan(plan, execution_root=tmp_path, trust=None)
    plan["request"]["seeds"] = [31]
    with pytest.raises(ValueError, match="digest"):
        normalize_score_plan(plan, execution_root=tmp_path, trust="pinned_local_execution")


@pytest.mark.parametrize("seed", [True, 1.0])
def test_score_plan_rejects_noninteger_command_seed_even_when_numerically_equal(tmp_path, seed):
    request = _prepare_request(tmp_path, seeds=(1,))
    plan = plan_scores(score_request_document(request), checkout_root=tmp_path / "LigandMPNN", execution_root=tmp_path)
    _write_output(tmp_path, request, 1)
    plan["commands"][0]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        normalize_score_plan(plan, execution_root=tmp_path, trust="pinned_local_execution")


def test_cli_score_plan_is_json_and_failure_emits_no_partial_result(tmp_path):
    request = _prepare_request(tmp_path)
    source = tmp_path / "request.json"
    source.write_text(json.dumps(score_request_document(request)))
    argv = [
        sys.executable,
        "-m",
        "dnadesign.thread.adapters.ligandmpnn.cli",
        "score-plan",
        "--request",
        str(source),
        "--checkout-root",
        str(tmp_path / "LigandMPNN"),
        "--execution-root",
        str(tmp_path),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["request_id"] == request.request_id
    source.write_text('{"schema_version":999}')
    failed = subprocess.run(argv, capture_output=True, text=True)
    assert failed.returncode != 0
    assert failed.stdout == ""


def test_context_and_design_commands_keep_request_admission_producer_owned(tmp_path):
    request = _prepare_request(tmp_path)
    score = score_request_document(request)
    context = {
        "schema_id": "thread.ligandmpnn.context_command_request",
        "schema_version": 1,
        "request_id": "portable-context",
        "pdb_path": score["pdb_path"],
        "pdb_sha256": score["pdb_sha256"],
        "upstream": score["upstream"],
        "output_path": "evidence/portable-context.json",
        "minimum_nucleotide_atoms": 1,
        "required_polymer_types": ["dna"],
        "chains": [],
        "parse_all_atoms": False,
        "parse_atoms_with_zero_occupancy": False,
    }
    source = tmp_path / "context-request.json"
    source.write_text(json.dumps(context))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.thread.adapters.ligandmpnn.cli",
            "context",
            "--request",
            str(source),
            "--execution-root",
            str(tmp_path),
            "--checkout-root",
            str(tmp_path / "LigandMPNN"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    reference = json.loads(completed.stdout)
    replay = materialize_ligandmpnn_context_inventory(
        context_request_from_document(context),
        execution_root=tmp_path,
        checkout_root=tmp_path / "LigandMPNN",
    )
    assert reference == replay.to_dict()
    design = {key: value for key, value in score.items() if key not in {"mode", "use_sequence"}}
    design.update(
        schema_id="thread.ligandmpnn.design_command_request",
        context_inventory=reference,
        temperature=0.1,
        residue_alphabets=[],
        packing={
            "enabled": False,
            "number_of_packs_per_design": 4,
            "repack_everything": False,
            "use_ligand_context": True,
        },
    )
    planned = plan_designs(design, checkout_root=tmp_path / "LigandMPNN", execution_root=tmp_path)
    assert planned["status"] == "planned_not_run"
    assert len(planned["commands"]) == len(request.seeds)
    assert all("run.py" in row["argv"] for row in planned["commands"])
    assert planned["residue_alphabet_sidecar"] is None
    residue = {"chain_id": "A", "residue_number": 12, "insertion_code": ""}
    design["redesigned_residues"] = [residue]
    design["residue_alphabets"] = [{"residue": residue, "allowed_amino_acids": ["A", "C"]}]
    constrained = plan_designs(design, checkout_root=tmp_path / "LigandMPNN", execution_root=tmp_path)
    sidecar = constrained["residue_alphabet_sidecar"]
    content = (tmp_path / sidecar["path"]).read_bytes()
    assert sidecar["sha256"] == "sha256:" + hashlib.sha256(content).hexdigest()
    assert json.loads(content)["A12"].endswith("X")


@pytest.mark.parametrize("contents", ['{"schema_version":1,"schema_version":2}', '{"value":NaN}'])
def test_cli_rejects_ambiguous_json_before_emitting_any_plan(tmp_path, contents):
    source = tmp_path / "invalid.json"
    source.write_text(contents)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.thread.adapters.ligandmpnn.cli",
            "score-plan",
            "--request",
            str(source),
            "--execution-root",
            str(tmp_path),
            "--checkout-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
