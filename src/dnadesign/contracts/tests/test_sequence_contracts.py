"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_sequence_contracts.py

Shared sequence-contract validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

import dnadesign.contracts as contracts
from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1, SecondaryStructurePredictionV1
from dnadesign.contracts.sequence import LinearSsdnaCompositionV1, MsdDesignCatalogV1, MsdDesignReferenceV1
from dnadesign.contracts.visual import CompositionReviewSvgV1


def test_root_contract_exports_include_composition_review_manifest() -> None:
    assert contracts.CompositionReviewSvgV1 is CompositionReviewSvgV1


def test_msd_design_reference_contract_accepts_scar_nick_reference() -> None:
    reference = MsdDesignReferenceV1.model_validate(
        {
            "contract": "msd_design_reference_v1",
            "schema_version": 1,
            "construct_id": "pES-retron-177",
            "construct_label": "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "msd_design_id": "msd-tetr-c172-lcggt-racag-mxmm",
            "payload_or_target": {"id": "TetR"},
            "cap": {
                "id": "C172",
                "source_construct": "retron-172",
                "snapback_topology": {
                    "kind": "snapback_foldback_geometry_v1",
                    "retained_stem_span": {"start": 0, "end": 3},
                    "cap_span": {"start": 3, "end": 6},
                    "foldback_return_span": {"start": 6, "end": 9},
                },
            },
            "scar_nick": {
                "left_base": "CGGT",
                "right_base": "ACAG",
                "profile_s3s2s1s0": "MXMM",
                "route_status": "note_only",
                "route_note": "26-derived base / 172-cap crossover",
            },
            "source_notes": "tests 172-cap permissiveness",
        }
    )

    assert reference.scar_nick.profile_s3s2s1s0 == "MXMM"
    assert reference.scar_nick.left_base == "CGGT"
    assert reference.cap.snapback_topology is not None
    assert reference.cap.snapback_topology.cap_span.model_dump(mode="json") == {"start": 3, "end": 6}


def test_msd_design_reference_contract_rejects_profile_drift() -> None:
    with pytest.raises(PydanticValidationError, match="profile_s3s2s1s0 does not match"):
        MsdDesignReferenceV1.model_validate(
            {
                "contract": "msd_design_reference_v1",
                "schema_version": 1,
                "construct_id": "pES-retron-177",
                "construct_label": "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
                "msd_design_id": "msd-tetr-c172-lcggt-racag-mxmm",
                "payload_or_target": {"id": "TetR"},
                "cap": {"id": "C172"},
                "scar_nick": {
                    "left_base": "CGGT",
                    "right_base": "ACAG",
                    "profile_s3s2s1s0": "MMMM",
                },
            }
        )


def test_msd_design_catalog_contract_rejects_duplicate_design_ids() -> None:
    row = {
        "contract": "msd_design_reference_v1",
        "schema_version": 1,
        "construct_id": "pES-retron-177",
        "construct_label": "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
        "msd_design_id": "msd-tetr-c172-lcggt-racag-mxmm",
        "payload_or_target": {"id": "TetR"},
        "cap": {"id": "C172"},
        "scar_nick": {"left_base": "CGGT", "right_base": "ACAG", "profile_s3s2s1s0": "MXMM"},
    }

    with pytest.raises(PydanticValidationError, match="Duplicate msd_design_id"):
        MsdDesignCatalogV1.model_validate(
            {
                "contract": "msd_design_catalog_v1",
                "schema_version": 1,
                "records": [row, {**row, "construct_id": "pES-retron-177b"}],
            }
        )


def test_linear_ssdna_composition_contract_accepts_retron43_literal() -> None:
    config = LinearSsdnaCompositionV1.model_validate(
        {
            "contract": "linear_ssdna_composition_v1",
            "schema_version": 1,
            "composition_id": "retron43_teto_manual_x8",
            "alphabet": "dna",
            "topology": "linear_ssdna",
            "coordinate_system": "zero_based_half_open",
            "units": [
                {
                    "unit_id": "retron43_teto_unit",
                    "repeat_count": 8,
                    "segments": [
                        {
                            "segment_id": "flank_5p",
                            "role": "flank_5p",
                            "sequence": "gtcagaaaaaaCAAG",
                            "source": {"kind": "literal", "label": "manual_retron43_example"},
                        },
                        {
                            "segment_id": "payload_primary",
                            "role": "payload_primary",
                            "sequence": "tccctatcagtgatagaga",
                            "source": {"kind": "literal", "label": "manual_teto_payload"},
                        },
                        {
                            "segment_id": "snapback_foldback_geometry",
                            "role": "snapback_foldback_geometry",
                            "sequence": "tCCTCAGcccGCTGAGGa",
                            "source": {"kind": "literal", "label": "manual_snapback_43_foldback"},
                        },
                        {
                            "segment_id": "payload_complement",
                            "role": "payload_complement",
                            "sequence": "tctctatcactgataggga",
                            "transform": {
                                "kind": "reverse_complement",
                                "source_segment_id": "payload_primary",
                                "assert_expected_sequence": True,
                            },
                            "source": {"kind": "derived", "from_segment_id": "payload_primary"},
                        },
                        {
                            "segment_id": "flank_3p",
                            "role": "flank_3p",
                            "sequence": "CTCGacagtaactcaga",
                            "source": {"kind": "literal", "label": "manual_retron43_example"},
                        },
                    ],
                    "annotations": [
                        {
                            "annotation_id": "stem_base_left",
                            "role": "stem_base_left",
                            "location": {
                                "basis": "segment",
                                "segment_id": "flank_5p",
                                "start": 11,
                                "end": 15,
                            },
                        }
                    ],
                    "assertions": [
                        {
                            "assertion_id": "payload_rc",
                            "kind": "reverse_complement",
                            "left_segment_id": "payload_primary",
                            "right_segment_id": "payload_complement",
                            "severity": "error",
                        }
                    ],
                }
            ],
            "visual": {
                "display_profile": {
                    "title": "Retron 43 TetO x8",
                    "component_labels": {"payload_primary": "TetO primary"},
                    "annotation_labels": {"stem_base_left": "Left stem base"},
                    "component_hues": {"payload_primary": "#F58518"},
                    "component_styles": {
                        "payload_primary": {
                            "fill": "#34D399",
                            "alpha": 0.58,
                            "edge_color": "#059669",
                        }
                    },
                    "base_highlight_color": "#111827",
                }
            },
        }
    )

    assert config.composition_id == "retron43_teto_manual_x8"
    assert config.units[0].repeat_count == 8
    assert config.units[0].segments[3].transform is not None
    assert config.visual.display_profile.title == "Retron 43 TetO x8"
    assert config.visual.display_profile.component_labels["payload_primary"] == "TetO primary"


def test_linear_ssdna_composition_contract_rejects_invalid_visual_style_alpha() -> None:
    with pytest.raises(PydanticValidationError, match="less than or equal to 1"):
        LinearSsdnaCompositionV1.model_validate(
            {
                "contract": "linear_ssdna_composition_v1",
                "schema_version": 1,
                "composition_id": "bad_visual_profile",
                "units": [{"unit_id": "unit", "segments": [{"segment_id": "payload", "sequence": "ACGT"}]}],
                "visual": {
                    "display_profile": {
                        "component_styles": {
                            "payload": {
                                "fill": "#34D399",
                                "alpha": 1.2,
                            }
                        }
                    }
                },
            }
        )


def test_linear_ssdna_composition_contract_rejects_duplicate_segment_ids() -> None:
    with pytest.raises(PydanticValidationError, match="Duplicate segment_id 'payload'"):
        LinearSsdnaCompositionV1.model_validate(
            {
                "contract": "linear_ssdna_composition_v1",
                "schema_version": 1,
                "composition_id": "bad_duplicate",
                "units": [
                    {
                        "unit_id": "unit",
                        "segments": [
                            {"segment_id": "payload", "sequence": "ACGT"},
                            {"segment_id": "payload", "sequence": "TGCA"},
                        ],
                    }
                ],
            }
        )


def test_secondary_structure_prediction_contract_rejects_length_mismatch() -> None:
    with pytest.raises(PydanticValidationError, match="dot_bracket length must equal input length"):
        SecondaryStructurePredictionV1.model_validate(
            {
                "contract": "secondary_structure_prediction_v1",
                "schema_version": 1,
                "prediction_id": "demo.rnafold.canonical_component_unit",
                "status": "ok",
                "input": {
                    "sequence_id": "demo",
                    "sequence_sha256": "abc",
                    "alphabet": "dna",
                    "topology": "linear_ssdna",
                    "length": 4,
                },
                "backend": {
                    "name": "ViennaRNA",
                    "version": "2.7.0",
                    "command": ["RNAfold", "--noPS"],
                    "parameters": {"temperature_c": 37.0},
                },
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "submitted_alphabet": "rna_surrogate",
                    "coordinates_mapped_to": "original_dna_sequence",
                },
                "result": {
                    "dot_bracket": "...",
                    "mfe_kcal_mol": -1.2,
                    "pair_map": [],
                },
            }
        )


def test_secondary_structure_prediction_contract_accepts_valid_pair_map() -> None:
    prediction = SecondaryStructurePredictionV1.model_validate(
        {
            "contract": "secondary_structure_prediction_v1",
            "schema_version": 1,
            "prediction_id": "demo.rnafold.canonical_component_unit",
            "status": "ok",
            "input": {
                "sequence_id": "demo",
                "sequence_sha256": "abc",
                "alphabet": "dna",
                "topology": "linear_ssdna",
                "length": 4,
            },
            "backend": {
                "name": "ViennaRNA",
                "version": "2.7.0",
                "command": ["RNAfold", "--noPS"],
                "parameters": {"temperature_c": 37.0},
            },
            "dna_policy": {
                "mode": "convert_t_to_u_for_rna_backend",
                "submitted_alphabet": "rna_surrogate",
                "coordinates_mapped_to": "original_dna_sequence",
            },
            "result": {
                "dot_bracket": "(())",
                "mfe_kcal_mol": -1.2,
                "pair_map": [
                    {"left": 0, "right": 3, "pair": "AU"},
                    {"left": 1, "right": 2, "pair": "CG"},
                ],
            },
        }
    )

    assert prediction.result.pair_map[0].left == 0


def test_secondary_structure_prediction_request_requires_explicit_dna_policy() -> None:
    request = SecondaryStructurePredictionRequestV1.model_validate(
        {
            "contract": "secondary_structure_prediction_request_v1",
            "schema_version": 1,
            "request_id": "demo.rnafold.canonical_component_unit",
            "input": {
                "sequence_artifact": "../assembled_sequence.json",
                "sequence_id": "demo",
                "sequence_sha256": "abc",
                "alphabet": "dna",
                "topology": "linear_ssdna",
                "length": 4,
            },
            "scope": {"mode": "canonical_component_unit"},
            "backend": {
                "name": "ViennaRNA",
                "executable": "RNAfold",
                "parameters": {"temperature_c": 37.0},
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "output_coordinates": "original_dna_sequence",
                },
            },
            "policy": {
                "required": False,
                "fail_on_malformed_output": True,
                "fail_on_length_mismatch": True,
            },
        }
    )

    assert request.backend.dna_policy.mode == "convert_t_to_u_for_rna_backend"

    payload = request.model_dump(mode="json")
    del payload["backend"]["dna_policy"]
    with pytest.raises(PydanticValidationError, match="Field required"):
        SecondaryStructurePredictionRequestV1.model_validate(payload)


def test_secondary_structure_prediction_request_accepts_viennarna_python_api_backend() -> None:
    request = SecondaryStructurePredictionRequestV1.model_validate(
        {
            "contract": "secondary_structure_prediction_request_v1",
            "schema_version": 1,
            "request_id": "demo.viennarna.canonical_component_unit",
            "input": {
                "sequence_artifact": "../assembled_sequence.json",
                "sequence_id": "demo",
                "sequence_sha256": "abc",
                "alphabet": "dna",
                "topology": "linear_ssdna",
                "length": 4,
            },
            "scope": {"mode": "canonical_component_unit"},
            "backend": {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {"temperature_c": 37.0},
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "output_coordinates": "original_dna_sequence",
                },
            },
            "policy": {"required": False},
        }
    )

    assert request.backend.interface == "python_api"
    assert request.backend.python_module == "RNA"
