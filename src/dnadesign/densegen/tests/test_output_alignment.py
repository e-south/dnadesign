from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.outputs.base import AlignmentDigest
from dnadesign.densegen.src.core.pipeline import _assert_sink_alignment


class DummySinkA:
    def __init__(self, ids: set[str]) -> None:
        self._ids = set(ids)

    def existing_ids(self) -> set[str]:
        return set(self._ids)

    def alignment_digest(self) -> AlignmentDigest:
        return AlignmentDigest(len(self._ids), "a" * 32)


class DummySinkB:
    def __init__(self, ids: set[str]) -> None:
        self._ids = set(ids)

    def existing_ids(self) -> set[str]:
        return set(self._ids)

    def alignment_digest(self) -> AlignmentDigest:
        return AlignmentDigest(len(self._ids), "b" * 32)


def test_sink_alignment_mismatch_raises() -> None:
    sinks = [DummySinkA({"a", "b"}), DummySinkB({"a"})]
    with pytest.raises(RuntimeError, match="out of sync"):
        _assert_sink_alignment(sinks)
