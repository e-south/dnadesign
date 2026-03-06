"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_cli_builders.py

Characterization tests for shared infer CLI helper functions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

from dnadesign.infer.cli_builders import build_model_config, run_with_progress


def test_build_model_config_prefers_cli_overrides() -> None:
    model = build_model_config(
        model_id="override_model",
        device=None,
        precision=None,
        alphabet="protein",
        batch_size=16,
        preset_model={"id": "preset_model", "precision": "bf16", "alphabet": "dna"},
    )

    assert model.id == "override_model"
    assert model.device == "cpu"
    assert model.precision == "bf16"
    assert model.alphabet == "protein"
    assert model.batch_size == 16


def test_run_with_progress_disables_progress_when_requested(monkeypatch) -> None:
    monkeypatch.delenv("DNADESIGN_PROGRESS", raising=False)

    observed = {"progress_factory_is_none": False}

    def _runner(progress_factory):
        observed["progress_factory_is_none"] = progress_factory is None
        return {"ok": True}

    result = run_with_progress(progress=False, runner=_runner)

    assert result == {"ok": True}
    assert observed["progress_factory_is_none"] is True
    assert os.environ.get("DNADESIGN_PROGRESS") == "0"
