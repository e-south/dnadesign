"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/proteinmpnn_request.py

ProteinMPNN request-sidecar validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _resolve_output_root
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.thread.adapters.proteinmpnn.validation import validate_request_manifest

_REQUEST_DIR_NAME = "proteinmpnn_request"
_PROTEINMPNN_NAME = "chain_a_backbone"
_CHAIN_ID = "A"
_THREAD_TO_ECO1_ISSUES = {
    "thread.proteinmpnn.request_metadata_mismatch": "eco1_rt.sampling.proteinmpnn_request_metadata_mismatch",
    "thread.proteinmpnn.invalid_omit_aas": "eco1_rt.sampling.proteinmpnn_invalid_omit_aas",
    "thread.proteinmpnn.invalid_omit_aa_jsonl": "eco1_rt.sampling.proteinmpnn_invalid_omit_aa_jsonl",
    "thread.proteinmpnn.missing_sidecar_hashes": "eco1_rt.sampling.proteinmpnn_missing_sidecar_hashes",
    "thread.proteinmpnn.sidecar_missing": "eco1_rt.sampling.proteinmpnn_sidecar_missing",
    "thread.proteinmpnn.sidecar_hash_mismatch": "eco1_rt.sampling.proteinmpnn_sidecar_hash_mismatch",
    "thread.proteinmpnn.sidecar_payload_mismatch": "eco1_rt.sampling.proteinmpnn_sidecar_payload_mismatch",
    "thread.proteinmpnn.sidecar_payload_invalid": "eco1_rt.sampling.proteinmpnn_sidecar_payload_invalid",
    "thread.proteinmpnn.request_hash_mismatch": "eco1_rt.sampling.proteinmpnn_request_hash_mismatch",
}


def validate_proteinmpnn_request_content(
    path: Path, *, repo_root: Path, output_root: Path | None = None
) -> list[ContractIssue]:
    """Validate ProteinMPNN sidecars against the current thread plan and residue map."""

    issues: list[ContractIssue] = []
    structure_root = _resolve_output_root(repo_root, output_root)
    manifest = _load_yaml(path)
    thread_plan_path = structure_root / "thread_plan.yaml"
    residue_map_path = structure_root / "residue_map.parquet"
    thread_plan = _load_yaml(thread_plan_path)
    canonical_to_mpnn = _canonical_to_proteinmpnn_position(residue_map_path)
    expected_fixed = _convert_positions(thread_plan.get("fixed_positions"), canonical_to_mpnn)
    expected_mutable = _convert_positions(thread_plan.get("mutable_positions"), canonical_to_mpnn)
    expected_excluded = list(thread_plan.get("excluded_non_fixed_missing_backbone_positions", []))

    issues.extend(_adapt_thread_issue(issue) for issue in validate_request_manifest(path))
    _validate_thread_plan(
        issues, manifest=manifest, thread_plan=thread_plan, thread_plan_path=thread_plan_path, path=path
    )
    _validate_positions(
        issues,
        manifest=manifest,
        expected_fixed=expected_fixed,
        expected_mutable=expected_mutable,
        expected_excluded=expected_excluded,
        canonical_to_mpnn=canonical_to_mpnn,
        path=path,
    )
    return issues


def _adapt_thread_issue(issue: Any) -> ContractIssue:
    return ContractIssue(
        check_id=_THREAD_TO_ECO1_ISSUES.get(issue.check_id, "eco1_rt.sampling.proteinmpnn_request_invalid"),
        message=issue.message,
        path=issue.path,
    )


def _validate_thread_plan(
    issues: list[ContractIssue],
    *,
    manifest: Mapping[str, Any],
    thread_plan: Mapping[str, Any],
    thread_plan_path: Path,
    path: Path,
) -> None:
    source = manifest.get("source_thread_plan")
    if not isinstance(source, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_missing_thread_plan_source",
                message="ProteinMPNN request must hash-link source_thread_plan",
                path=str(path),
            )
        )
        return
    expected = {
        "hash": "sha256:" + _sha256(thread_plan_path),
        "request_hash": thread_plan.get("request_hash"),
    }
    for field, value in expected.items():
        if source.get(field) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.proteinmpnn_thread_plan_source_mismatch",
                    message=f"ProteinMPNN request source_thread_plan field {field!r} must match current thread plan",
                    path=str(path),
                )
            )
    if not str(source.get("path", "")).strip():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_thread_plan_source_mismatch",
                message="ProteinMPNN request source_thread_plan field 'path' must be non-empty",
                path=str(path),
            )
        )
    if manifest.get("seed_set") != thread_plan.get("seed_set") or manifest.get(
        "temperature_schedule"
    ) != thread_plan.get("temperature_schedule"):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_sampling_schedule_mismatch",
                message="ProteinMPNN request seeds and temperatures must match thread_plan.yaml",
                path=str(path),
            )
        )
    for field in ("batch_id", "num_seq_per_target", "batch_size", "expected_sample_count"):
        if manifest.get(field) != thread_plan.get(field):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.proteinmpnn_sampling_scale_mismatch",
                    message=f"ProteinMPNN request field {field!r} must match thread_plan.yaml",
                    path=str(path),
                )
            )


def _validate_positions(
    issues: list[ContractIssue],
    *,
    manifest: Mapping[str, Any],
    expected_fixed: list[int],
    expected_mutable: list[int],
    expected_excluded: list[int],
    canonical_to_mpnn: Mapping[int, int],
    path: Path,
) -> None:
    fixed_payload = manifest.get("fixed_positions_jsonl")
    expected_payload = {_PROTEINMPNN_NAME: {_CHAIN_ID: expected_fixed}}
    if fixed_payload != expected_payload:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_fixed_positions_mismatch",
                message="ProteinMPNN fixed_positions_jsonl must use chain-local fixed positions from thread_plan.yaml",
                path=str(path),
            )
        )
    if manifest.get("mutable_positions_by_chain") != {_CHAIN_ID: expected_mutable}:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_mutable_positions_mismatch",
                message="ProteinMPNN mutable positions must use chain-local positions from thread_plan.yaml",
                path=str(path),
            )
        )
    if manifest.get("excluded_missing_backbone_positions") != expected_excluded:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_excluded_positions_mismatch",
                message="ProteinMPNN request must carry terminal missing-backbone positions as exclusions",
                path=str(path),
            )
        )
    if manifest.get("canonical_position_count") != len(canonical_to_mpnn):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_position_count_mismatch",
                message="ProteinMPNN request canonical_position_count must match mapped fixed-backbone residues",
                path=str(path),
            )
        )


def _canonical_to_proteinmpnn_position(residue_map_path: Path) -> dict[int, int]:
    rows = pq.read_table(residue_map_path).to_pylist()
    mapped = [
        row for row in rows if row.get("mapping_status") == "mapped" and row.get("structure_chain_id") == _CHAIN_ID
    ]
    mapped.sort(key=lambda row: int(row["canonical_position"]))
    return {int(row["canonical_position"]): index for index, row in enumerate(mapped, start=1)}


def _convert_positions(value: Any, mapping: Mapping[int, int]) -> list[int]:
    if not isinstance(value, list):
        return []
    converted = [mapping[int(position)] for position in value if int(position) in mapping]
    return sorted(converted)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
