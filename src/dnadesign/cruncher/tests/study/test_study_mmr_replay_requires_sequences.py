"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_mmr_replay_requires_sequences.py

Ensure MMR replay enforces required sequence artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.study_workflow import run_study
from dnadesign.cruncher.tests.study._helpers import write_study_spec, write_workspace_config


def test_study_mmr_replay_allows_minimal_profile_when_sequences_enabled(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="minimal",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    assert run_dir.exists()


def test_study_mmr_replay_requires_save_sequences_enabled(tmp_path: Path) -> None:
    config_path = write_workspace_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    assert isinstance(payload, dict)
    cruncher = payload.get("cruncher")
    assert isinstance(cruncher, dict)
    sample = cruncher.get("sample")
    assert isinstance(sample, dict)
    output = sample.get("output")
    assert isinstance(output, dict)
    output["save_sequences"] = False
    config_path.write_text(yaml.safe_dump(payload))

    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="minimal",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    with pytest.raises(ValueError, match="requires sample\\.output\\.save_sequences=true"):
        run_study(spec_path)
