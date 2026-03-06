"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/runtime/generate_execution.py

Executes chunked generation calls with explicit payload contracts and OOM derating.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .._logging import get_logger
from ..errors import RuntimeOOMError

_LOG = get_logger(__name__)


def validate_generate_payload(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("Generate function must return a mapping with 'gen_seqs'")
    if "gen_seqs" not in payload:
        raise RuntimeError("Generate output must include 'gen_seqs'")
    gen_seqs = payload["gen_seqs"]
    if not isinstance(gen_seqs, list):
        raise RuntimeError("Generate output field 'gen_seqs' must be a list")
    return payload


def execute_generate_batches(
    *,
    prompts: List[str],
    fn,
    params: Dict[str, object],
    micro_batch_size: int,
    auto_derate: bool,
    is_oom: Callable[[BaseException], bool],
    on_progress: Callable[[int], None],
) -> Dict[str, List[object]]:
    gen_all: Dict[str, List[object]] = {"gen_seqs": []}
    start = 0
    bs = micro_batch_size

    while start < len(prompts):
        take = min(bs, len(prompts) - start)
        chunk = prompts[start : start + take]

        try:
            out = fn(chunk, **params)
        except RuntimeError as exc:
            if is_oom(exc) and auto_derate and bs > 1:
                new_bs = max(1, bs // 2)
                _LOG.warning("OOM at batch=%d -> retry at batch=%d", bs, new_bs)
                bs = new_bs
                continue
            raise RuntimeOOMError(str(exc))

        payload = validate_generate_payload(out)
        generated = payload["gen_seqs"]
        if len(generated) != len(chunk):
            raise RuntimeError("Adapter returned wrong number of generated sequences for chunk")

        gen_all["gen_seqs"].extend(generated)
        on_progress(len(chunk))
        start += take

    return gen_all
