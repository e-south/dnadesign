"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/contracts.py

Public ops contracts for producer destination resolution and resume-readiness
policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.construct.contracts import resolve_construct_usr_output_contract
from dnadesign.densegen.contracts import resolve_densegen_usr_producer_contract
from dnadesign.infer.contracts import resolve_infer_usr_output_contract


@dataclass(frozen=True)
class ResumeReadinessPolicy:
    tool: str
    required_record_columns: tuple[str, ...]
    orphan_artifact_markers: tuple[str, ...]


@dataclass(frozen=True)
class USRProducerContract:
    tool: str
    config_path: Path
    run_root: Path | None
    usr_root: Path
    usr_dataset: str
    supports_overlay_parts: bool
    supports_records_parts: bool
    usr_chunk_size: int | None
    records_path: Path | None
    parquet_chunk_size: int | None
    round_robin: bool | None
    max_accepted_per_library: int | None
    generation_total_quota: int | None


_DENSEGEN_RESUME_POLICY = ResumeReadinessPolicy(
    tool="densegen",
    required_record_columns=(
        "densegen__run_id",
        "densegen__input_name",
        "densegen__plan",
        "densegen__used_tfbs_detail",
    ),
    orphan_artifact_markers=(
        "outputs/pools/pool_manifest.json",
        "outputs/tables/attempts.parquet",
        "outputs/tables/solutions.parquet",
        "outputs/tables/run_metrics.parquet",
        "outputs/meta/effective_config.json",
    ),
)

_RESUME_READINESS_POLICIES: dict[str, ResumeReadinessPolicy] = {
    "densegen": _DENSEGEN_RESUME_POLICY,
}

_TOOLS_WITHOUT_RESUME_POLICY = frozenset({"infer"})


def resolve_resume_readiness_policy(tool: str) -> ResumeReadinessPolicy | None:
    tool_name = str(tool or "").strip().lower()
    if not tool_name:
        raise ValueError("resume readiness policy tool must be non-empty")
    policy = _RESUME_READINESS_POLICIES.get(tool_name)
    if policy is not None:
        return policy
    if tool_name in _TOOLS_WITHOUT_RESUME_POLICY:
        return None
    supported = ", ".join(sorted(set(_RESUME_READINESS_POLICIES) | set(_TOOLS_WITHOUT_RESUME_POLICY)))
    raise ValueError(f"unsupported resume readiness policy tool: {tool_name} (supported: {supported})")


def _resolve_densegen_usr_contract(config_path: Path) -> USRProducerContract:
    densegen_contract = resolve_densegen_usr_producer_contract(config_path)
    return USRProducerContract(
        tool="densegen",
        config_path=densegen_contract.config_path,
        run_root=densegen_contract.run_root,
        usr_root=densegen_contract.usr_root,
        usr_dataset=densegen_contract.usr_dataset,
        supports_overlay_parts=densegen_contract.supports_overlay_parts,
        supports_records_parts=densegen_contract.supports_records_parts,
        usr_chunk_size=densegen_contract.usr_chunk_size,
        records_path=densegen_contract.records_path,
        parquet_chunk_size=densegen_contract.parquet_chunk_size,
        round_robin=densegen_contract.round_robin,
        max_accepted_per_library=densegen_contract.max_accepted_per_library,
        generation_total_quota=densegen_contract.generation_total_quota,
    )


def _resolve_infer_usr_contract(config_path: Path) -> USRProducerContract:
    destination = resolve_infer_usr_output_contract(config_path)
    return USRProducerContract(
        tool="infer",
        config_path=destination.config_path,
        run_root=None,
        usr_root=destination.usr_root,
        usr_dataset=destination.usr_dataset,
        supports_overlay_parts=False,
        supports_records_parts=False,
        usr_chunk_size=None,
        records_path=None,
        parquet_chunk_size=None,
        round_robin=None,
        max_accepted_per_library=None,
        generation_total_quota=None,
    )


def _resolve_construct_usr_contract(config_path: Path) -> USRProducerContract:
    destination = resolve_construct_usr_output_contract(config_path)
    return USRProducerContract(
        tool="construct",
        config_path=destination.config_path,
        run_root=None,
        usr_root=destination.usr_root,
        usr_dataset=destination.usr_dataset,
        supports_overlay_parts=False,
        supports_records_parts=False,
        usr_chunk_size=None,
        records_path=None,
        parquet_chunk_size=None,
        round_robin=None,
        max_accepted_per_library=None,
        generation_total_quota=None,
    )


_USR_PRODUCER_ADAPTERS = {
    "construct": _resolve_construct_usr_contract,
    "densegen": _resolve_densegen_usr_contract,
    "infer": _resolve_infer_usr_contract,
}


def resolve_usr_producer_contract(*, tool: str, config_path: Path) -> USRProducerContract:
    tool_name = str(tool or "").strip().lower()
    adapter = _USR_PRODUCER_ADAPTERS.get(tool_name)
    if adapter is None:
        supported = ", ".join(sorted(_USR_PRODUCER_ADAPTERS))
        raise ValueError(f"unsupported usr producer tool: {tool_name} (supported: {supported})")
    return adapter(config_path)


__all__ = [
    "ResumeReadinessPolicy",
    "USRProducerContract",
    "resolve_resume_readiness_policy",
    "resolve_usr_producer_contract",
]
