"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_progress_output.py

Validate non-interactive progress behavior for tqdm-backed adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import dnadesign.cruncher.app.progress as progress_mod


def test_progress_adapter_disables_tqdm_when_noninteractive(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_tqdm(iterable, **kwargs):
        calls.append(dict(kwargs))
        return iterable

    monkeypatch.setattr(progress_mod, "tqdm", _fake_tqdm)
    monkeypatch.setenv("CRUNCHER_NONINTERACTIVE", "1")
    adapter = progress_mod.progress_adapter(True)

    assert list(adapter(range(3), desc="sample")) == [0, 1, 2]
    assert calls
    assert calls[-1]["disable"] is True


def test_progress_adapter_keeps_tqdm_enabled_for_interactive_terminals(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_tqdm(iterable, **kwargs):
        calls.append(dict(kwargs))
        return iterable

    monkeypatch.setattr(progress_mod, "tqdm", _fake_tqdm)
    monkeypatch.delenv("CRUNCHER_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(progress_mod.sys, "stderr", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(progress_mod.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    adapter = progress_mod.progress_adapter(True)

    assert list(adapter(range(1), desc="sample")) == [0]
    assert calls
    assert calls[-1]["disable"] is False
    assert calls[-1]["dynamic_ncols"] is True


def test_progress_adapter_preserves_explicit_disable(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_tqdm(iterable, **kwargs):
        calls.append(dict(kwargs))
        return iterable

    monkeypatch.setattr(progress_mod, "tqdm", _fake_tqdm)
    monkeypatch.delenv("CRUNCHER_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(progress_mod.sys, "stderr", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(progress_mod.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    adapter = progress_mod.progress_adapter(True)

    assert list(adapter(range(1), disable=True)) == [0]
    assert calls
    assert calls[-1]["disable"] is True
