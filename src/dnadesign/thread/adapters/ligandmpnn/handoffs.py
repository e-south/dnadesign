"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/handoffs.py

Domain commands for attested plans and portable, uncensored score results.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

from ._regular_files import open_regular_file
from .alphabets import materialize_residue_alphabet_sidecar
from .commands import build_ligandmpnn_commands
from .handoff_requests import (
    design_request_from_document,
    document_digest,
    exact_fields,
    score_request_document,
    score_request_from_document,
)
from .models import LigandMpnnCommand, validate_ligandmpnn_seeds
from .scoring import build_ligandmpnn_score_commands

__all__ = ["normalize_score_plan", "plan_scores", "score_request_document", "score_request_from_document"]


def producer_identity() -> dict:
    installed = distribution("dnadesign")
    source = json.loads(installed.read_text("direct_url.json") or "{}")
    source["kind"] = (
        "git"
        if source.get("vcs_info", {}).get("vcs") == "git"
        else "archive"
        if "archive_info" in source
        else "local"
        if "dir_info" in source
        else "unrecorded"
    )
    # Installed source identity is evidence; local absolute paths are not portable identity.
    if source.get("url", "").startswith("file:"):
        source = {key: value for key, value in source.items() if key != "url"}
    modules = {}
    for module in sorted(Path(__file__).parent.glob("*.py")):
        with open_regular_file(module) as stream:
            modules[module.name] = hashlib.sha256(stream.read()).hexdigest()
    return {
        "schema_id": "thread.ligandmpnn.command_identity",
        "schema_version": 1,
        "distribution": "dnadesign",
        "version": installed.version,
        "source": source,
        "adapter_source_sha256": document_digest(modules),
        "request_schema_version": 1,
        "score_result_schema_version": 3,
        "score_export_schema_version": 1,
    }


def plan_scores(
    document: dict,
    *,
    checkout_root: Path,
    execution_root: Path,
    input_root: Path | None = None,
    python_executable: str = "python",
) -> dict:
    from .score_results import score_request_sha256

    request = score_request_from_document(document)
    commands = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        execution_root=execution_root,
        input_root=input_root,
        python_executable=python_executable,
    )
    return {
        "schema_id": "thread.ligandmpnn.score_plan",
        "schema_version": 1,
        "status": "planned_not_run",
        "producer": producer_identity(),
        "request_id": request.request_id,
        "request": document,
        "request_document_sha256": document_digest(document),
        "request_sha256": score_request_sha256(request),
        "commands": [command.to_dict() for command in commands],
    }


def normalize_score_plan(plan: dict, *, execution_root: Path, trust: str | None) -> dict:
    from .score_results import LigandMpnnScoreOutputTrust, parse_ligandmpnn_score_outputs, score_request_sha256

    if trust != "pinned_local_execution":
        raise ValueError("score normalization requires explicit pinned-local-execution trust")
    exact_fields(
        plan,
        {
            "schema_id",
            "schema_version",
            "status",
            "producer",
            "request_id",
            "request",
            "request_document_sha256",
            "request_sha256",
            "commands",
        },
        "score plan",
    )
    if (
        plan["schema_id"] != "thread.ligandmpnn.score_plan"
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("unsupported score plan schema")
    if plan["status"] != "planned_not_run" or document_digest(plan["request"]) != plan["request_document_sha256"]:
        raise ValueError("score plan request status or digest mismatch")
    request = score_request_from_document(plan["request"])
    if request.request_id != plan["request_id"] or score_request_sha256(request) != plan["request_sha256"]:
        raise ValueError("score plan semantic request identity or digest mismatch")
    if not isinstance(plan["commands"], list):
        raise ValueError("score plan commands must be a list")
    commands = []
    for row in plan["commands"]:
        exact_fields(row, {"seed", "output_dir", "argv"}, "score command")
        if not isinstance(row["argv"], list) or any(not isinstance(arg, str) for arg in row["argv"]):
            raise ValueError("score command argv must be an array of strings")
        commands.append(LigandMpnnCommand(row["seed"], Path(row["output_dir"]), tuple(row["argv"])))
    validate_ligandmpnn_seeds(tuple(command.seed for command in commands))
    result = parse_ligandmpnn_score_outputs(
        request,
        tuple(commands),
        execution_root=execution_root,
        trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
    )
    artifacts = []
    for output in result.outputs:
        values = output.raw_probabilities
        artifact = {
            "seed": output.seed,
            "output_sha256": output.output_sha256,
            "residue_names": list(output.residue_names),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "values": values.tolist(),
        }
        artifacts.append({**artifact, "sha256": document_digest(artifact)})
    return {
        "schema_id": "thread.ligandmpnn.score_export",
        "schema_version": 1,
        "producer": producer_identity(),
        "plan_sha256": document_digest(plan),
        "result": result.to_dict(),
        "probability_artifacts": artifacts,
    }


def plan_designs(
    document: dict, *, checkout_root: Path, execution_root: Path, python_executable: str = "python"
) -> dict:
    request = design_request_from_document(document)
    producer = producer_identity()
    request_document_sha256 = document_digest(document)
    with TemporaryDirectory(prefix="ligandmpnn-design-") as staging:
        sidecar = None
        if request.residue_alphabets:
            path = Path("residue-alphabets") / f"{request.request_id}.json"
            sidecar = materialize_residue_alphabet_sidecar(
                request, path, write_path=Path(staging).resolve() / "alphabet.json"
            )
        commands = build_ligandmpnn_commands(
            request,
            checkout_root=checkout_root,
            execution_root=execution_root,
            python_executable=python_executable,
            residue_alphabet_sidecar=sidecar,
        )
        if sidecar is not None:
            sidecar = materialize_residue_alphabet_sidecar(request, path, write_path=execution_root / path)
    return {
        "schema_id": "thread.ligandmpnn.design_plan",
        "schema_version": 1,
        "status": "planned_not_run",
        "producer": producer,
        "request": document,
        "request_document_sha256": request_document_sha256,
        "residue_alphabet_sidecar": sidecar.to_dict() if sidecar is not None else None,
        "commands": [command.to_dict() for command in commands],
    }
