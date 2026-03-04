"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/usr_overlay_inputs.py

Tool-adapter contract for parsing USR overlay guard inputs from producer configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign._contracts import (
    resolve_densegen_usr_output_contract,
    resolve_infer_evo2_usr_output_contract,
    resolve_usr_producer_contract,
)


@dataclass(frozen=True)
class UsrOverlayGuardInputs:
    run_root: Path | None
    usr_root: Path
    usr_dataset: str
    usr_chunk_size: int | None
    records_path: Path | None
    parquet_chunk_size: int | None
    round_robin: bool | None
    max_accepted_per_library: int | None
    generation_total_quota: int | None
    supports_overlay_parts: bool
    supports_records_parts: bool


def parse_usr_overlay_guard_inputs(*, tool: str, config_path: Path) -> UsrOverlayGuardInputs:
    tool_name = str(tool or "").strip().lower()
    if tool_name == "densegen":
        # Keep densegen events-source contract explicit and shared with notify.
        resolve_densegen_usr_output_contract(config_path)
    if tool_name in {"infer", "infer-evo2", "infer_evo2"}:
        # Keep infer events-source contract explicit and shared with notify.
        resolve_infer_evo2_usr_output_contract(config_path)
    try:
        contract = resolve_usr_producer_contract(tool=tool_name, config_path=config_path)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("unsupported usr producer tool:"):
            supported = message.split("(supported:", maxsplit=1)[-1].rstrip(")")
            raise ValueError(f"unsupported usr-overlay-guard tool: {tool_name} (supported: {supported})") from exc
        raise
    return UsrOverlayGuardInputs(
        run_root=contract.run_root,
        usr_root=contract.usr_root,
        usr_dataset=contract.usr_dataset,
        usr_chunk_size=contract.usr_chunk_size,
        records_path=contract.records_path,
        parquet_chunk_size=contract.parquet_chunk_size,
        round_robin=contract.round_robin,
        max_accepted_per_library=contract.max_accepted_per_library,
        generation_total_quota=contract.generation_total_quota,
        supports_overlay_parts=contract.supports_overlay_parts,
        supports_records_parts=contract.supports_records_parts,
    )
