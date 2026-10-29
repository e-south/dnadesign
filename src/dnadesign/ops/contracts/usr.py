"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/contracts/usr.py

SeqLedger producer destination contract adapters for Ops runbook planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .models import USRDestination, USRProducerContract


def _infer_destination(config_path: Path) -> USRDestination:
    from dnadesign.infer.contracts import resolve_infer_usr_output_contract

    destination = resolve_infer_usr_output_contract(config_path)
    return USRDestination("infer", destination.config_path, destination.usr_root, destination.usr_dataset)


def _construct_destination(config_path: Path) -> USRDestination:
    from dnadesign.construct import resolve_construct_usr_output_contract

    destination = resolve_construct_usr_output_contract(config_path)
    return USRDestination("construct", destination.config_path, destination.usr_root, destination.usr_dataset)


def _resolve_infer_usr_contract(config_path: Path) -> USRProducerContract:
    destination = _infer_destination(config_path)
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
    destination = _construct_destination(config_path)
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


@dataclass(frozen=True)
class _ProducerAdapter:
    destination: Callable[[Path], USRDestination]
    producer: Callable[[Path], USRProducerContract]


_USR_PRODUCER_ADAPTERS = {
    "construct": _ProducerAdapter(_construct_destination, _resolve_construct_usr_contract),
    "infer": _ProducerAdapter(_infer_destination, _resolve_infer_usr_contract),
}


def _resolve_adapter(tool: str) -> _ProducerAdapter:
    tool_name = str(tool or "").strip().lower()
    adapter = _USR_PRODUCER_ADAPTERS.get(tool_name)
    if adapter is None:
        supported = ", ".join(sorted(_USR_PRODUCER_ADAPTERS))
        raise ValueError(f"unsupported usr producer tool: {tool_name} (supported: {supported})")
    return adapter


def resolve_usr_destination(*, tool: str, config_path: Path) -> USRDestination:
    """Resolve storage location without requiring scheduling or chunking settings."""
    return _resolve_adapter(tool).destination(config_path)


def resolve_usr_producer_contract(*, tool: str, config_path: Path) -> USRProducerContract:
    return _resolve_adapter(tool).producer(config_path)
