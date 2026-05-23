"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/contracts/usr.py

USR producer destination contract adapters for Ops runbook planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .models import USRProducerContract


def _resolve_densegen_usr_contract(config_path: Path) -> USRProducerContract:
    from dnadesign.densegen.contracts import resolve_densegen_usr_producer_contract

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
    from dnadesign.infer.contracts import resolve_infer_usr_output_contract

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
    from dnadesign.construct import resolve_construct_usr_output_contract

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
