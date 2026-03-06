"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_progress_handles.py

Tests for infer progress handle creation and lifecycle contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

from dnadesign.infer.progress import create_progress_handle


class _CaptureProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, label: str, total: int):
        self.calls.append((label, total))
        return SimpleNamespace(update=lambda _n: None, close=lambda: None)


def test_create_progress_handle_prefers_custom_factory() -> None:
    factory = _CaptureProgress()
    handle = create_progress_handle(
        progress_factory=factory,
        label="job/out",
        total=3,
        unit="seq",
    )

    handle.update(1)
    handle.close()
    assert factory.calls == [("job/out", 3)]


def test_create_progress_handle_uses_no_progress_mode(monkeypatch) -> None:
    monkeypatch.setenv("DNADESIGN_PROGRESS", "0")

    handle = create_progress_handle(
        progress_factory=None,
        label="job/out",
        total=3,
        unit="seq",
    )

    handle.update(2)
    handle.close()


def test_create_progress_handle_uses_tqdm_factory(monkeypatch) -> None:
    observed = {}

    class _FakeTQDM:
        def __init__(self, total=None, **kwargs):
            observed["total"] = total
            observed["kwargs"] = kwargs

        def update(self, _n):
            return None

        def close(self):
            return None

    monkeypatch.setattr("dnadesign.infer.progress.resolve_tqdm_factory", lambda: (_FakeTQDM, True))

    handle = create_progress_handle(
        progress_factory=None,
        label="job/out",
        total=5,
        unit="seq",
    )

    handle.update(1)
    handle.close()
    assert observed["total"] == 5
    assert observed["kwargs"]["unit"] == "seq"
    assert observed["kwargs"]["desc"] == "job/out"
