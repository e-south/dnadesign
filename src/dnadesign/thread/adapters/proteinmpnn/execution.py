"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/execution.py

Official ProteinMPNN request execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.thread.adapters.proteinmpnn.execution_preflight import (
    proteinmpnn_git_commit,
    resolve_proteinmpnn_root,
    validate_proteinmpnn_root,
)
from dnadesign.thread.adapters.proteinmpnn.samples import write_backend_run_manifest
from dnadesign.thread.adapters.proteinmpnn.sidecars import resolve_manifest_sidecar_paths


@dataclass(frozen=True)
class ProteinMpnnExecutionConfig:
    """Official ProteinMPNN run-scale settings for one named batch."""

    batch_id: str
    num_seq_per_target: int = 1
    batch_size: int = 1
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.batch_id or not self.batch_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("batch_id must contain only letters, numbers, underscores, or hyphens")
        if self.num_seq_per_target <= 0:
            raise ValueError("num_seq_per_target must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_seq_per_target % self.batch_size != 0:
            raise ValueError("num_seq_per_target must be divisible by batch_size")

    @property
    def run_dir_name(self) -> str:
        return self.batch_id

    def expected_sample_count(self, *, seed_count: int, temperature_count: int) -> int:
        return seed_count * temperature_count * self.num_seq_per_target


def run_official_proteinmpnn_request(
    *,
    request_manifest_path: Path,
    proteinmpnn_root: Path | None = None,
    output_dir: Path,
    python_executable: str | None = None,
    execution_config: ProteinMpnnExecutionConfig | None = None,
) -> dict[str, Any]:
    """Run official ProteinMPNN scripts for one validated request manifest."""

    root = resolve_proteinmpnn_root(proteinmpnn_root)
    issues = validate_proteinmpnn_root(root)
    if issues:
        messages = "; ".join(issue.message for issue in issues)
        raise FileNotFoundError(messages)

    manifest = _load_yaml(request_manifest_path)
    config = execution_config or _execution_config_from_manifest(manifest)
    request_hash = str(manifest["request_hash"])
    target_name = str(manifest["proteinmpnn_name"])
    chain_id = str(manifest["proteinmpnn_design_chain"])
    sidecar_paths = resolve_manifest_sidecar_paths(request_manifest_path, manifest["sidecar_paths"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir = output_dir / "batches" / config.run_dir_name
    if batch_output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"ProteinMPNN batch output already exists: {batch_output_dir}")
        shutil.rmtree(batch_output_dir)
    batch_output_dir.mkdir(parents=True)
    executable = python_executable or sys.executable

    helper_dir = batch_output_dir / "official_helper_parity"
    _run_helper_parity_check(
        manifest=manifest,
        request_manifest_path=request_manifest_path,
        proteinmpnn_root=root,
        helper_dir=helper_dir,
        python_executable=executable,
        chain_id=chain_id,
        sidecar_paths=sidecar_paths,
    )

    runs: list[dict[str, Any]] = []
    temp_text = " ".join(f"{float(temperature):g}" for temperature in manifest["temperature_schedule"])
    for seed in manifest["seed_set"]:
        seed_output_dir = batch_output_dir / f"seed_{int(seed)}"
        command = [
            executable,
            str(root / "protein_mpnn_run.py"),
            "--jsonl_path",
            str(sidecar_paths["parsed_pdbs_jsonl"]),
            "--chain_id_jsonl",
            str(sidecar_paths["assigned_chains_jsonl"]),
            "--fixed_positions_jsonl",
            str(sidecar_paths["fixed_positions_jsonl"]),
            "--out_folder",
            str(seed_output_dir),
            "--num_seq_per_target",
            str(config.num_seq_per_target),
            "--sampling_temp",
            temp_text,
            "--seed",
            str(int(seed)),
            "--batch_size",
            str(config.batch_size),
            "--omit_AAs",
            "C",
            "--save_score",
            "1",
            "--suppress_print",
            "1",
        ]
        if "omit_AA_jsonl" in sidecar_paths:
            command.extend(["--omit_AA_jsonl", str(sidecar_paths["omit_AA_jsonl"])])
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        run_record = {
            "seed": int(seed),
            "output_dir": str(seed_output_dir),
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        runs.append(run_record)
        if completed.returncode != 0:
            raise RuntimeError(f"ProteinMPNN run failed for seed {seed}: {completed.stderr}")

    backend_run_manifest_path = batch_output_dir / "backend_run_manifest.yaml"
    write_backend_run_manifest(
        backend_run_manifest_path,
        request_manifest_path=request_manifest_path,
        request_hash=request_hash,
        proteinmpnn_root=root,
        proteinmpnn_git_commit=proteinmpnn_git_commit(root),
        runs=runs,
        batch_id=config.batch_id,
        num_seq_per_target=config.num_seq_per_target,
        batch_size=config.batch_size,
        expected_sample_count=config.expected_sample_count(
            seed_count=len(manifest["seed_set"]),
            temperature_count=len(manifest["temperature_schedule"]),
        ),
    )
    return {
        "backend_run_manifest_path": backend_run_manifest_path,
        "backend_run_id": _backend_run_id(request_hash),
        "request_hash": request_hash,
        "target_name": target_name,
        "run_outputs": runs,
    }


def _run_helper_parity_check(
    *,
    manifest: Mapping[str, Any],
    request_manifest_path: Path,
    proteinmpnn_root: Path,
    helper_dir: Path,
    python_executable: str,
    chain_id: str,
    sidecar_paths: Mapping[str, Path],
) -> None:
    if helper_dir.exists():
        shutil.rmtree(helper_dir)
    helper_dir.mkdir(parents=True)
    request_dir = sidecar_paths["chain_a_backbone_pdb"].parent
    parsed_path = helper_dir / "parsed_pdbs.jsonl"
    assigned_path = helper_dir / "assigned_chains.jsonl"
    fixed_path = helper_dir / "fixed_positions.jsonl"

    _run_checked(
        [
            python_executable,
            str(proteinmpnn_root / "helper_scripts/parse_multiple_chains.py"),
            "--input_path",
            str(request_dir),
            "--output_path",
            str(parsed_path),
        ],
        cwd=proteinmpnn_root,
    )
    _run_checked(
        [
            python_executable,
            str(proteinmpnn_root / "helper_scripts/assign_fixed_chains.py"),
            "--input_path",
            str(parsed_path),
            "--output_path",
            str(assigned_path),
            "--chain_list",
            chain_id,
        ],
        cwd=proteinmpnn_root,
    )
    fixed_positions = manifest["fixed_positions_jsonl"][manifest["proteinmpnn_name"]][chain_id]
    _run_checked(
        [
            python_executable,
            str(proteinmpnn_root / "helper_scripts/make_fixed_positions_dict.py"),
            "--input_path",
            str(parsed_path),
            "--output_path",
            str(fixed_path),
            "--chain_list",
            chain_id,
            "--position_list",
            " ".join(str(position) for position in fixed_positions),
        ],
        cwd=proteinmpnn_root,
    )

    _assert_jsonl_payload_close(
        observed=_load_jsonl_record(parsed_path),
        expected=_load_jsonl_record(sidecar_paths["parsed_pdbs_jsonl"]),
        context=f"{request_manifest_path}:parsed_pdbs_jsonl",
    )
    _assert_jsonl_payload_close(
        observed=_load_jsonl_record(assigned_path),
        expected=_load_jsonl_record(sidecar_paths["assigned_chains_jsonl"]),
        context=f"{request_manifest_path}:assigned_chains_jsonl",
    )
    _assert_jsonl_payload_close(
        observed=_load_jsonl_record(fixed_path),
        expected=_load_jsonl_record(sidecar_paths["fixed_positions_jsonl"]),
        context=f"{request_manifest_path}:fixed_positions_jsonl",
    )


def _run_checked(argv: Sequence[str], *, cwd: Path) -> None:
    completed = subprocess.run(list(argv), cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ProteinMPNN helper failed: {' '.join(argv)}\n{completed.stderr}")


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _execution_config_from_manifest(manifest: Mapping[str, Any]) -> ProteinMpnnExecutionConfig:
    return ProteinMpnnExecutionConfig(
        batch_id=str(manifest["batch_id"]),
        num_seq_per_target=int(manifest["num_seq_per_target"]),
        batch_size=int(manifest["batch_size"]),
    )


def _load_jsonl_record(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return records[0]


def _assert_jsonl_payload_close(*, observed: Any, expected: Any, context: str) -> None:
    if not _payload_close(observed, expected):
        raise ValueError(f"Official ProteinMPNN helper output does not match generated sidecar at {context}")


def _payload_close(observed: Any, expected: Any) -> bool:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if set(observed) != set(expected):
            return False
        return all(_payload_close(observed[key], expected[key]) for key in observed)
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            return False
        return all(_payload_close(obs, exp) for obs, exp in zip(observed, expected, strict=True))
    if isinstance(observed, float | int) and isinstance(expected, float | int):
        return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-4)
    return observed == expected


def _backend_run_id(request_hash: str) -> str:
    return "proteinmpnn_" + request_hash.removeprefix("sha256:")[:12]
