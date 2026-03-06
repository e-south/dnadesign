"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/extract_execution.py

Executes chunked extract calls with explicit fail-fast output contracts and OOM derating.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .._logging import get_logger
from .adapter_dispatch import invoke_extract_callable
from ..errors import RuntimeOOMError

_LOG = get_logger(__name__)


def execute_extract_output(
    *,
    seqs: List[str],
    need_idx: List[int],
    existing: List[object],
    method_name: str,
    fn,
    params: Dict[str, Any],
    output_format: str,
    micro_batch_size: int,
    default_batch_size: int,
    auto_derate: bool,
    is_oom: Callable[[BaseException], bool],
    on_progress: Callable[[int], None],
    on_chunk: Optional[Callable[[List[int], List[object]], None]],
) -> List[object]:
    if len(need_idx) == 0:
        return list(existing)

    all_vals = list(existing)
    start_bs = micro_batch_size if micro_batch_size > 0 else min(len(need_idx), default_batch_size)
    bs = start_bs
    start = 0

    while start < len(need_idx):
        take = min(bs, len(need_idx) - start)
        idx_chunk = need_idx[start : start + take]
        chunk = [seqs[row_index] for row_index in idx_chunk]

        try:
            vals = invoke_extract_callable(
                fn=fn,
                method_name=method_name,
                chunk=chunk,
                params=params,
                output_format=output_format,
            )
        except RuntimeError as exc:
            if is_oom(exc) and auto_derate and bs > 1:
                new_bs = max(1, bs // 2)
                _LOG.warning("OOM at batch=%d -> retry at batch=%d", bs, new_bs)
                bs = new_bs
                continue
            raise RuntimeOOMError(str(exc))

        if len(vals) != len(idx_chunk):
            raise RuntimeError("Adapter returned wrong number of outputs for chunk")

        for k, j in enumerate(idx_chunk):
            all_vals[j] = vals[k]

        if on_chunk is not None:
            on_chunk(idx_chunk, vals)
        on_progress(len(idx_chunk))
        start += take

    return all_vals
