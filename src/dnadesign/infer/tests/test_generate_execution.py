"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_generate_execution.py

Chunk-execution and payload contract tests for infer generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.errors import RuntimeOOMError
from dnadesign.infer.generate_execution import execute_generate_batches, validate_generate_payload


def test_validate_generate_payload_accepts_mapping_with_gen_seqs() -> None:
    out = validate_generate_payload({"gen_seqs": ["A"], "meta": [1]})
    assert out["gen_seqs"] == ["A"]
    assert out["meta"] == [1]


def test_validate_generate_payload_rejects_missing_gen_seqs() -> None:
    with pytest.raises(RuntimeError, match="must include 'gen_seqs'"):
        validate_generate_payload({"other": ["A"]})


def test_execute_generate_batches_aggregates_outputs_and_progress() -> None:
    progress: list[int] = []

    def _fn(chunk, **_kwargs):
        return {"gen_seqs": [f"{p}-gen" for p in chunk]}

    out = execute_generate_batches(
        prompts=["A", "B", "C"],
        fn=_fn,
        params={},
        micro_batch_size=2,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=progress.append,
    )

    assert out == {"gen_seqs": ["A-gen", "B-gen", "C-gen"]}
    assert progress == [2, 1]


def test_execute_generate_batches_derates_after_oom() -> None:
    calls: list[int] = []

    def _fn(chunk, **_kwargs):
        calls.append(len(chunk))
        if len(chunk) > 1:
            raise RuntimeError("out of memory")
        return {"gen_seqs": [f"{chunk[0]}-gen"]}

    out = execute_generate_batches(
        prompts=["A", "B", "C"],
        fn=_fn,
        params={},
        micro_batch_size=3,
        auto_derate=True,
        is_oom=lambda exc: "out of memory" in str(exc).lower(),
        on_progress=lambda _n: None,
    )

    assert out == {"gen_seqs": ["A-gen", "B-gen", "C-gen"]}
    assert calls[0] == 3
    assert calls[1:] == [1, 1, 1]


def test_execute_generate_batches_raises_runtime_oom_without_derating() -> None:
    def _oom(*_args, **_kwargs):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeOOMError, match="out of memory"):
        execute_generate_batches(
            prompts=["A", "B"],
            fn=_oom,
            params={},
            micro_batch_size=2,
            auto_derate=False,
            is_oom=lambda exc: "out of memory" in str(exc).lower(),
            on_progress=lambda _n: None,
        )


def test_execute_generate_batches_fails_on_prompt_output_count_mismatch() -> None:
    def _fn(chunk, **_kwargs):
        return {"gen_seqs": [chunk[0] + "-gen"]}

    with pytest.raises(RuntimeError, match="wrong number of generated sequences"):
        execute_generate_batches(
            prompts=["A", "B"],
            fn=_fn,
            params={},
            micro_batch_size=2,
            auto_derate=True,
            is_oom=lambda _exc: False,
            on_progress=lambda _n: None,
        )
