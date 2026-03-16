"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_config.py

Direct configuration-normalization tests for construct.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from dnadesign.construct.src.config import RealizeConfig


def test_realize_config_normalizes_legacy_window_fields() -> None:
    realize = RealizeConfig.model_validate(
        {
            "mode": "window",
            "focal_part": "anchor",
            "focal_point": "end",
            "anchor_offset_bp": -2,
            "window_bp": 11,
        }
    )

    assert realize.window is not None
    assert realize.window.semantics == "fixed_total"
    assert realize.window.reference == "end"
    assert realize.window.direction == "symmetric"
    assert realize.window.size_bp == 11
    assert realize.window.offset_bp == -2


def test_realize_config_rejects_mixed_window_block_and_legacy_fields() -> None:
    with pytest.raises(PydanticValidationError, match="Use either realize.window or the legacy"):
        RealizeConfig.model_validate(
            {
                "mode": "window",
                "focal_part": "anchor",
                "window": {
                    "semantics": "fixed_total",
                    "reference": "center",
                    "direction": "symmetric",
                    "size_bp": 8,
                },
                "window_bp": 8,
            }
        )


def test_realize_config_requires_window_mapping_when_explicitly_null() -> None:
    with pytest.raises(PydanticValidationError, match="realize.window must be a mapping"):
        RealizeConfig.model_validate(
            {
                "mode": "window",
                "focal_part": "anchor",
                "window": None,
            }
        )


def test_realize_config_rejects_window_fields_for_full_construct_mode() -> None:
    with pytest.raises(PydanticValidationError, match="only allowed when realize.mode='window'"):
        RealizeConfig.model_validate(
            {
                "mode": "full_construct",
                "window_bp": 8,
            }
        )
