"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_presets.py

Public API:
  - run_extract
  - run_generate
  - run_job (YAML-driven)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.infer.cli import app
from dnadesign.infer.presets import list_presets, load_preset

runner = CliRunner()


def test_presets_registry_has_evo2_extract():
    items = list_presets()
    ids = [it["id"] for it in items]
    assert "evo2/extract_logits_ll" in ids
    p = load_preset("evo2/extract_logits_ll")
    assert p["kind"] == "extract"
    outs = [o["id"] for o in p["outputs"]]
    assert {"logits_mean", "ll_mean", "ll_sum"}.issubset(set(outs))


def test_cli_presets_list():
    res = runner.invoke(app, ["presets", "list"])
    assert res.exit_code == 0
    assert "Available Presets" in res.stdout


def test_cli_extract_preset_dry_run_ok():
    # No adapter load in dry-run; should validate and print tables.
    res = runner.invoke(
        app,
        ["extract", "--preset", "evo2/extract_logits_ll", "--seq", "ACGT", "--dry-run"],
    )
    assert res.exit_code == 0
    assert "Output Specs" in res.stdout
