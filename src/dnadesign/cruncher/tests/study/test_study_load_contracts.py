"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_load_contracts.py

Validate Study spec load-time error formatting and path contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.study.load import load_study_spec


def test_load_formats_schema_validation_errors_without_pydantic_noise(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.study.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 1,
                    "name": "bad_study",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "replicates": {"seed_path": "sample.seed", "seeds": []},
                    "trials": [],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Study schema validation failed") as exc_info:
        load_study_spec(spec_path)
    message = str(exc_info.value)
    assert "- study.schema_version:" in message
    assert "- study.replicates.seeds:" in message
    assert "https://errors.pydantic.dev" not in message
