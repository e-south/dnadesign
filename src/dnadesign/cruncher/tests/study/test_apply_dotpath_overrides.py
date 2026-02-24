"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_apply_dotpath_overrides.py

Validate Study dot-path override behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import pytest

from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.study.overrides import apply_dotpath_overrides, extract_factor_columns


def _base_cfg() -> CruncherConfig:
    return CruncherConfig.model_validate(
        {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": ".cruncher", "pwm_source": "matrix"},
            "sample": {
                "seed": 1,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
            },
        }
    )


def test_apply_dotpath_overrides_updates_and_revalidates() -> None:
    cfg = _base_cfg()
    updated = apply_dotpath_overrides(
        cfg,
        {
            "sample.sequence_length": 20,
            "sample.elites.select.diversity": 0.25,
        },
    )
    assert updated.sample is not None
    assert updated.sample.sequence_length == 20
    assert updated.sample.elites.select.diversity == pytest.approx(0.25)
    assert cfg.sample is not None
    assert cfg.sample.sequence_length == 12


def test_apply_dotpath_overrides_rejects_unknown_path() -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="Unknown override path"):
        apply_dotpath_overrides(cfg, {"sample.not_a_field": 1})


def test_apply_dotpath_overrides_rejects_invalid_type() -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="sequence_length"):
        apply_dotpath_overrides(cfg, {"sample.sequence_length": "long"})


def test_extract_factor_columns_serializes_non_scalar_values() -> None:
    values = extract_factor_columns(
        {
            "sample.sequence_length": 16,
            "sample.moves.overrides.move_probs": {"S": 0.2, "B": 0.7, "M": 0.1},
            "sample.elites.select.pool_candidates": [100, 250],
        }
    )
    assert values["param__sample__sequence_length"] == 16
    assert isinstance(values["param__sample__moves__overrides__move_probs"], str)
    assert isinstance(values["param__sample__elites__select__pool_candidates"], str)
    assert json.loads(values["param__sample__moves__overrides__move_probs"]) == {"B": 0.7, "M": 0.1, "S": 0.2}
    assert json.loads(values["param__sample__elites__select__pool_candidates"]) == [100, 250]
