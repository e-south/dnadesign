"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_config.py

Direct configuration-normalization tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from dnadesign.construct.src.config import (
    InputConfig,
    JobConfig,
    OutputConfig,
    PlacementConfig,
    RealizeConfig,
    TemplateConfig,
)


def test_realize_config_rejects_legacy_window_fields() -> None:
    with pytest.raises(
        PydanticValidationError,
        match="realize.focal_point, realize.anchor_offset_bp, realize.window_bp",
    ):
        RealizeConfig.model_validate(
            {
                "mode": "window",
                "focal_part": "anchor",
                "focal_point": "end",
                "anchor_offset_bp": -2,
                "window_bp": 11,
            }
        )


def test_realize_config_rejects_window_block_when_legacy_fields_are_present() -> None:
    with pytest.raises(PydanticValidationError, match="realize.window_bp is no longer supported"):
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
    with pytest.raises(PydanticValidationError, match="realize.window is only allowed when realize.mode='window'"):
        RealizeConfig.model_validate(
            {
                "mode": "full_construct",
                "window": {
                    "semantics": "fixed_total",
                    "reference": "center",
                    "direction": "symmetric",
                    "size_bp": 8,
                },
            }
        )


def test_placement_config_accepts_coordinate_locator_and_guards() -> None:
    placement = PlacementConfig.model_validate(
        {
            "kind": "replace",
            "orientation": "forward",
            "locator": {"kind": "coordinates", "start": 4, "end": 8},
            "guards": {
                "replaced_sequence": "TTTT",
                "upstream_sequence": "AAAA",
                "downstream_sequence": "CCCC",
                "require_unique_forward_matches": True,
            },
        }
    )

    assert placement.locator.kind == "coordinates"
    assert placement.guards is not None
    assert placement.guards.replaced_sequence == "TTTT"


def test_placement_config_accepts_flank_locator_without_coordinate_fields() -> None:
    placement = PlacementConfig.model_validate(
        {
            "kind": "replace",
            "orientation": "forward",
            "locator": {
                "kind": "flanks",
                "upstream_sequence": "AAAA",
                "downstream_sequence": "CCCC",
            },
            "guards": {
                "replaced_span_bp": 4,
            },
        }
    )

    assert placement.locator.kind == "flanks"
    assert placement.guards is not None
    assert placement.guards.replaced_span_bp == 4


def test_placement_config_rejects_coordinate_fields_outside_locator() -> None:
    with pytest.raises(PydanticValidationError, match="locator"):
        PlacementConfig.model_validate(
            {
                "kind": "replace",
                "orientation": "forward",
                "start": 4,
                "end": 8,
            }
        )


def test_placement_config_rejects_flank_guards_when_locator_is_flanks() -> None:
    with pytest.raises(PydanticValidationError, match="placement.guards.upstream_sequence/downstream_sequence"):
        PlacementConfig.model_validate(
            {
                "kind": "replace",
                "orientation": "forward",
                "locator": {
                    "kind": "flanks",
                    "upstream_sequence": "AAAA",
                    "downstream_sequence": "CCCC",
                },
                "guards": {
                    "upstream_sequence": "AAAA",
                },
            }
        )


def test_input_config_rejects_legacy_flat_shape() -> None:
    with pytest.raises(
        PydanticValidationError,
        match="input.dataset, input.root, input.source is no longer supported",
    ):
        InputConfig.model_validate(
            {
                "source": "usr",
                "dataset": "anchors_demo",
                "root": "outputs/usr_datasets",
            }
        )


def test_template_config_rejects_legacy_flat_shape() -> None:
    with pytest.raises(
        PydanticValidationError,
        match="template.kind, template.sequence is no longer supported",
    ):
        TemplateConfig.model_validate(
            {
                "id": "template_demo",
                "kind": "literal",
                "sequence": "AAAATTTTCCCCGGGG",
            }
        )


def test_output_config_rejects_legacy_flat_shape() -> None:
    with pytest.raises(
        PydanticValidationError,
        match="output.dataset, output.root, output.source is no longer supported",
    ):
        OutputConfig.model_validate(
            {
                "dataset": "anchors_constructed",
                "root": "outputs/usr_datasets",
                "source": "construct run demo",
            }
        )


def test_job_config_normalize_anchor_requires_normalize_block() -> None:
    with pytest.raises(PydanticValidationError, match="job.normalize_anchor is required"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "normalize_demo",
                    "mode": "normalize_anchor",
                    "input": {
                        "source": {"kind": "usr", "dataset": "anchors", "root": "/tmp/usr"},
                        "field": "sequence",
                    },
                    "output": {
                        "target": {"kind": "usr", "dataset": "anchors_norm", "root": "/tmp/usr"},
                    },
                }
            }
        )


def test_job_config_normalize_anchor_rejects_template_realization_fields() -> None:
    with pytest.raises(PydanticValidationError, match="job.template is only allowed"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "normalize_demo",
                    "mode": "normalize_anchor",
                    "input": {
                        "source": {"kind": "usr", "dataset": "anchors", "root": "/tmp/usr"},
                        "field": "sequence",
                    },
                    "template": {
                        "id": "template",
                        "source": {"kind": "literal", "sequence": "AAAATTTT"},
                    },
                    "normalize_anchor": {
                        "product_kind": "analysis_core60",
                        "target_length": 60,
                        "focal_selector": {
                            "kind": "chain",
                            "selectors": [{"kind": "sequence_midpoint", "allowed": True}],
                        },
                        "over_length_policy": {"kind": "trim", "target_length": 60},
                    },
                    "output": {
                        "target": {"kind": "usr", "dataset": "anchors_norm", "root": "/tmp/usr"},
                    },
                }
            }
        )
