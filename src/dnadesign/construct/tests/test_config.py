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

from dnadesign.construct.src.contracts.config import (
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
                        "product_kind": "analysis_window",
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


def test_job_config_rejects_unknown_full_construct_focal_part() -> None:
    with pytest.raises(PydanticValidationError, match="realize.focal_part 'missing_anchor' is not defined"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "multi_anchor_demo",
                    "input": {
                        "source": {"kind": "usr", "dataset": "anchors", "root": "/tmp/usr"},
                        "field": "sequence",
                    },
                    "template": {
                        "id": "template",
                        "source": {"kind": "literal", "sequence": "AAAATTTTCCCCGGGG"},
                    },
                    "parts": [
                        {
                            "name": "anchor",
                            "role": "anchor",
                            "sequence": {"source": "input_field", "field": "sequence"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 4, "end": 8},
                            },
                        }
                    ],
                    "realize": {"mode": "full_construct", "focal_part": "missing_anchor"},
                    "output": {
                        "target": {"kind": "usr", "dataset": "constructs", "root": "/tmp/usr"},
                    },
                }
            }
        )


def test_job_config_rejects_implicit_focal_part_when_multiple_anchor_parts_exist() -> None:
    with pytest.raises(PydanticValidationError, match="multiple anchor parts"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "multi_anchor_demo",
                    "input": {
                        "source": {"kind": "usr", "dataset": "anchors", "root": "/tmp/usr"},
                        "field": "sequence",
                    },
                    "template": {
                        "id": "template",
                        "source": {"kind": "literal", "sequence": "AAAATTTTCCCCGGGG"},
                    },
                    "parts": [
                        {
                            "name": "anchor_a",
                            "role": "anchor",
                            "sequence": {"source": "input_field", "field": "sequence"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 4, "end": 8},
                            },
                        },
                        {
                            "name": "anchor_b",
                            "role": "anchor",
                            "sequence": {"source": "literal", "literal": "GG"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 12, "end": 16},
                            },
                        },
                    ],
                    "realize": {"mode": "full_construct"},
                    "output": {
                        "target": {"kind": "usr", "dataset": "constructs", "root": "/tmp/usr"},
                    },
                }
            }
        )


def test_job_config_rejects_output_variants_without_anchor_handoff_part() -> None:
    with pytest.raises(PydanticValidationError, match="job.output_variants requires realize.focal_part"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "multi_slot_missing_handoff",
                    "input": {
                        "source": {"kind": "usr", "dataset": "rt_lnrna_candidates", "root": "/tmp/usr"},
                        "field": None,
                    },
                    "template": {
                        "id": "dual_cassette_template",
                        "source": {"kind": "literal", "sequence": "AAAATTTTCCCCGGGG"},
                    },
                    "parts": [
                        {
                            "name": "lnrna",
                            "role": "lnrna_cassette",
                            "sequence": {"source": "input_field", "field": "candidate__lnrna_sequence"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 4, "end": 8},
                            },
                        },
                        {
                            "name": "rt_cds",
                            "role": "rt_cds",
                            "sequence": {"source": "input_field", "field": "candidate__rt_cds_sequence"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 12, "end": 16},
                            },
                        },
                    ],
                    "realize": {"mode": "full_construct", "required_slots": ["lnrna", "rt_cds"]},
                    "output_variants": [
                        {
                            "product_kind": "realized_context",
                            "orientation": "forward",
                            "recommended_pooling": "anchor_mean",
                        }
                    ],
                    "output": {
                        "target": {"kind": "usr", "dataset": "rt_lnrna_constructs", "root": "/tmp/usr"},
                    },
                }
            }
        )


def test_job_config_rejects_output_variant_anchor_part_missing_from_parts() -> None:
    with pytest.raises(PydanticValidationError, match="anchor_part 'missing_slot' is not defined"):
        JobConfig.model_validate(
            {
                "job": {
                    "id": "multi_slot_missing_variant_anchor",
                    "input": {
                        "source": {"kind": "usr", "dataset": "rt_lnrna_candidates", "root": "/tmp/usr"},
                        "field": None,
                    },
                    "template": {
                        "id": "dual_cassette_template",
                        "source": {"kind": "literal", "sequence": "AAAATTTTCCCCGGGG"},
                    },
                    "parts": [
                        {
                            "name": "lnrna",
                            "role": "lnrna_cassette",
                            "sequence": {"source": "input_field", "field": "candidate__lnrna_sequence"},
                            "placement": {
                                "kind": "replace",
                                "locator": {"kind": "coordinates", "start": 4, "end": 8},
                            },
                        }
                    ],
                    "realize": {"mode": "full_construct", "required_slots": ["lnrna"]},
                    "output_variants": [
                        {
                            "product_kind": "realized_context",
                            "orientation": "forward",
                            "recommended_pooling": "anchor_mean",
                            "anchor_part": "missing_slot",
                        }
                    ],
                    "output": {
                        "target": {"kind": "usr", "dataset": "rt_lnrna_constructs", "root": "/tmp/usr"},
                    },
                }
            }
        )
