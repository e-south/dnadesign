"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_formatting_markup.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.cli import formatting as fmt


def test_formatting_markup_respects_env(monkeypatch):
    monkeypatch.setenv("OPAL_CLI_TUI", "0")

    monkeypatch.setenv("OPAL_CLI_MARKUP", "0")
    txt = fmt.kv_block("Title", {"alpha": 1})
    assert "[bold" not in txt

    monkeypatch.setenv("OPAL_CLI_MARKUP", "1")
    txt2 = fmt.kv_block("Title", {"alpha": 1})
    assert "[bold" in txt2
