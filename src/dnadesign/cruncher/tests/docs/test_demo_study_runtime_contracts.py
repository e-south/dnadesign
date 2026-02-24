"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_demo_study_runtime_contracts.py

Enforce bounded runtime knobs in workspace length-sweep study grids.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_study(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    study = payload.get("study")
    assert isinstance(study, dict)
    return study


def _load_base_sequence_length(path: Path) -> int:
    config_path = path.parents[1] / "config.yaml"
    payload = yaml.safe_load(config_path.read_text())
    assert isinstance(payload, dict)
    cruncher = payload.get("cruncher")
    assert isinstance(cruncher, dict), f"{config_path}: missing cruncher mapping"
    sample = cruncher.get("sample")
    assert isinstance(sample, dict), f"{config_path}: missing cruncher.sample mapping"
    value = sample.get("sequence_length")
    assert isinstance(value, int), f"{config_path}: missing integer sample.sequence_length"
    return int(value)


def test_workspace_length_sweep_grids_define_sequence_axis_only() -> None:
    paths = sorted((ROOT / "workspaces").glob("*/configs/studies/length_vs_score.study.yaml"))
    assert paths, "expected workspace length_vs_score study specs"
    for path in paths:
        study = _load_study(path)
        grids = study.get("trial_grids")
        assert isinstance(grids, list) and grids
        first = grids[0]
        assert isinstance(first, dict)
        factors = first.get("factors")
        assert isinstance(factors, dict)
        assert "sample.budget.tune" not in factors, f"{path}: study must inherit tune budget from config.yaml"
        assert "sample.budget.draws" not in factors, f"{path}: study must inherit draw budget from config.yaml"
        assert "sample.output.save_sequences" not in factors, (
            f"{path}: study must inherit save_sequences from config.yaml"
        )
        length_axis = factors.get("sample.sequence_length")
        assert isinstance(length_axis, list) and len(length_axis) >= 2, f"{path}: length axis must remain a sweep"


def test_workspace_length_sweeps_use_step_two_and_include_base_length() -> None:
    paths = sorted((ROOT / "workspaces").glob("*/configs/studies/length_vs_score.study.yaml"))
    assert paths, "expected workspace length_vs_score study specs"
    for path in paths:
        study = _load_study(path)
        grids = study.get("trial_grids")
        assert isinstance(grids, list) and grids, f"{path}: missing trial_grids"
        first = grids[0]
        assert isinstance(first, dict), f"{path}: trial_grids[0] must be a mapping"
        factors = first.get("factors")
        assert isinstance(factors, dict), f"{path}: trial_grids[0].factors must be a mapping"
        raw_axis = factors.get("sample.sequence_length")
        assert isinstance(raw_axis, list) and raw_axis, f"{path}: missing sample.sequence_length axis"
        length_axis = [int(value) for value in raw_axis]
        base_length = _load_base_sequence_length(path)
        assert base_length in length_axis, (
            f"{path}: length sweep must include base config sample.sequence_length={base_length}"
        )
        non_base_axis = sorted({value for value in length_axis if int(value) != base_length})
        if len(non_base_axis) <= 1:
            continue
        deltas = [int(curr) - int(prev) for prev, curr in zip(non_base_axis, non_base_axis[1:], strict=False)]
        assert all(delta == 2 for delta in deltas), (
            f"{path}: sample.sequence_length sweep must use step-2 increments outside base-length anchor"
        )


def test_workspace_study_specs_default_to_parallelism_six() -> None:
    paths = sorted((ROOT / "workspaces").glob("*/configs/studies/*.study.yaml"))
    assert paths, "expected workspace study specs"
    for path in paths:
        study = _load_study(path)
        execution = study.get("execution")
        assert isinstance(execution, dict), f"{path}: missing study.execution"
        assert execution.get("parallelism") == 6, f"{path}: parallelism must default to 6"
