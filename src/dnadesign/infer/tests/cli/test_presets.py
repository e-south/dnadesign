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

import pytest
from typer.testing import CliRunner

from dnadesign.infer.cli import app
from dnadesign.infer.src.presets import list_presets, load_preset
from dnadesign.infer.src.presets import registry as preset_registry

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


def test_preset_registry_reuses_cached_scan(monkeypatch):
    calls = {"count": 0}

    def _fake_scan():
        calls["count"] += 1
        return [
            (
                "evo2/extract_logits_ll",
                """
id: evo2/extract_logits_ll
kind: extract
description: demo
model:
  id: evo2_7b
outputs: []
params: {}
""".strip()
                + "\n",
            )
        ]

    preset_registry.clear_preset_cache()
    monkeypatch.setattr(preset_registry, "_scan", _fake_scan)

    assert preset_registry.list_presets()[0]["id"] == "evo2/extract_logits_ll"
    assert preset_registry.list_presets()[0]["id"] == "evo2/extract_logits_ll"
    assert preset_registry.load_preset("evo2/extract_logits_ll")["id"] == "evo2/extract_logits_ll"
    assert calls["count"] == 1


def test_load_preset_fails_fast_on_ambiguous_stem(monkeypatch):
    def _fake_scan():
        return [
            (
                "evo2/extract_logits_ll",
                """
id: evo2/extract_logits_ll
kind: extract
model:
  id: evo2_7b
outputs: []
params: {}
""".strip()
                + "\n",
            ),
            (
                "other/extract_logits_ll",
                """
id: other/extract_logits_ll
kind: extract
model:
  id: evo2_7b
outputs: []
params: {}
""".strip()
                + "\n",
            ),
        ]

    preset_registry.clear_preset_cache()
    monkeypatch.setattr(preset_registry, "_scan", _fake_scan)

    with pytest.raises(KeyError, match="Ambiguous preset stem"):
        preset_registry.load_preset("extract_logits_ll")
