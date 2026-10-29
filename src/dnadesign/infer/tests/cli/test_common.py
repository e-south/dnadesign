"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_common.py

Contracts for shared infer CLI error mapping and rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
import typer

from dnadesign.infer.src.cli.common import raise_cli_error
from dnadesign.infer.src.errors import ConfigError


def test_raise_cli_error_renders_error_type_and_message(monkeypatch) -> None:
    rendered: list[str] = []

    monkeypatch.setattr("dnadesign.infer.src.cli.common.console.print", lambda message: rendered.append(str(message)))

    with pytest.raises(typer.Exit) as exc:
        raise_cli_error(ConfigError("missing config"))

    assert exc.value.exit_code == 2
    assert rendered == ["[red]ConfigError: missing config[/red]"]
