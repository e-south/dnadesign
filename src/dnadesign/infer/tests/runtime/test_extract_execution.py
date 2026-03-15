"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_extract_execution.py

Chunk-execution contract tests for infer extract execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.errors import RuntimeOOMError
from dnadesign.infer.src.runtime.extract_execution import execute_extract_output


def test_execute_extract_output_populates_values_and_calls_hooks() -> None:
    progress_updates: list[int] = []
    chunk_calls: list[tuple[list[int], list[object]]] = []

    def _fn(chunk, **_kwargs):
        return [f"v:{seq}" for seq in chunk]

    values = execute_extract_output(
        seqs=["A", "B", "C"],
        need_idx=[0, 2],
        existing=[None, "keep", None],
        method_name="log_likelihood",
        fn=_fn,
        params={},
        output_format="float",
        micro_batch_size=2,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=progress_updates.append,
        on_chunk=lambda idx, vals: chunk_calls.append((list(idx), list(vals))),
    )

    assert values == ["v:A", "keep", "v:C"]
    assert progress_updates == [2]
    assert chunk_calls == [([0, 2], ["v:A", "v:C"])]


def test_execute_extract_output_derates_after_oom_and_retries() -> None:
    calls: list[int] = []

    def _fn(chunk, **_kwargs):
        calls.append(len(chunk))
        if len(chunk) > 1:
            raise RuntimeError("out of memory")
        return [f"ok:{chunk[0]}"]

    values = execute_extract_output(
        seqs=["A", "B", "C"],
        need_idx=[0, 1, 2],
        existing=[None, None, None],
        method_name="log_likelihood",
        fn=_fn,
        params={},
        output_format="float",
        micro_batch_size=3,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda exc: "out of memory" in str(exc).lower(),
        on_progress=lambda _n: None,
        on_chunk=None,
    )

    assert values == ["ok:A", "ok:B", "ok:C"]
    assert calls[0] == 3
    assert calls[1:] == [1, 1, 1]


def test_execute_extract_output_raises_runtime_oom_when_not_derating() -> None:
    def _oom(*_args, **_kwargs):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeOOMError, match="out of memory"):
        execute_extract_output(
            seqs=["A", "B"],
            need_idx=[0, 1],
            existing=[None, None],
            method_name="log_likelihood",
            fn=_oom,
            params={},
            output_format="float",
            micro_batch_size=2,
            default_batch_size=64,
            auto_derate=False,
            is_oom=lambda exc: "out of memory" in str(exc).lower(),
            on_progress=lambda _n: None,
            on_chunk=None,
        )


def test_execute_extract_output_fails_on_wrong_chunk_output_length() -> None:
    def _fn(_chunk, **_kwargs):
        return [1.0]

    with pytest.raises(RuntimeError, match="wrong number of outputs for chunk"):
        execute_extract_output(
            seqs=["A", "B"],
            need_idx=[0, 1],
            existing=[None, None],
            method_name="log_likelihood",
            fn=_fn,
            params={},
            output_format="float",
            micro_batch_size=2,
            default_batch_size=64,
            auto_derate=True,
            is_oom=lambda _exc: False,
            on_progress=lambda _n: None,
            on_chunk=None,
        )
