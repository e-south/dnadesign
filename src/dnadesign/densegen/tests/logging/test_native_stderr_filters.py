"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/logging/test_native_stderr_filters.py

Native stderr filter installation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from dnadesign.densegen.src.utils import logging_utils


def test_densegen_import_installs_native_stderr_filters() -> None:
    code = """
import sys
from dnadesign.densegen.src.utils import logging_utils
import dnadesign.densegen.src  # triggers stderr filter installation
flag = getattr(logging_utils._install_native_stderr_deduper, "_installed", False)
sys.stdout.write("1" if flag else "0")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "1"


def test_install_native_stderr_filters_warns_once_on_install_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(logging_utils._install_native_stderr_deduper, "_installed", False, raising=False)
    monkeypatch.setattr(logging_utils._install_native_stderr_deduper, "_warned_install_failure", False, raising=False)
    monkeypatch.setattr(logging_utils.os, "dup", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no-fd")))
    with caplog.at_level(logging.WARNING):
        logging_utils.install_native_stderr_filters(suppress_solver_messages=False)
        logging_utils.install_native_stderr_filters(suppress_solver_messages=False)
    assert caplog.text.count("Native stderr deduper installation failed") == 1


def test_install_native_stderr_filters_strict_raises_on_install_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(logging_utils._install_native_stderr_deduper, "_installed", False, raising=False)
    monkeypatch.setattr(logging_utils._install_native_stderr_deduper, "_warned_install_failure", False, raising=False)
    monkeypatch.setattr(logging_utils.os, "dup", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no-fd")))
    with pytest.raises(RuntimeError, match="Native stderr deduper installation failed"):
        logging_utils.install_native_stderr_filters(suppress_solver_messages=False, strict=True)
