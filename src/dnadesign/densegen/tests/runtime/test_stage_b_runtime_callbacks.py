"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_stage_b_runtime_callbacks.py

Tests for Stage-B runtime callback state objects and library-build progression.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.densegen.src.core.pipeline.stage_b_runtime_callbacks import (
    StageBLibraryRuntimeState,
    build_next_library_callback,
)


@dataclass(frozen=True)
class _FakeLibraryContext:
    sampling_info: dict


class _FakeBuilder:
    def __init__(self, library_context: _FakeLibraryContext) -> None:
        self._library_context = library_context
        self.starts: list[int] = []

    def build_next(self, *, library_index_start: int):
        self.starts.append(int(library_index_start))
        return self._library_context


def test_build_next_library_callback_updates_build_mode_indexes() -> None:
    state = StageBLibraryRuntimeState(libraries_built=2, libraries_built_start=2)
    builder = _FakeBuilder(_FakeLibraryContext(sampling_info={"library_index": 9}))

    library_context = build_next_library_callback(
        builder=builder,
        state=state,
        library_source_label="build",
    )

    assert library_context.sampling_info["library_index"] == 9
    assert builder.starts == [2]
    assert state.libraries_used == 1
    assert state.libraries_built == 9


def test_build_next_library_callback_updates_artifact_mode_indexes() -> None:
    state = StageBLibraryRuntimeState(libraries_built=6, libraries_built_start=6)
    builder = _FakeBuilder(_FakeLibraryContext(sampling_info={}))

    build_next_library_callback(
        builder=builder,
        state=state,
        library_source_label="artifact",
    )

    assert builder.starts == [6]
    assert state.libraries_used == 1
    assert state.libraries_built == 1
