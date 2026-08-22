"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_sequence_contracts.py

Shared sequence-contract validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError as PydanticValidationError

import dnadesign.contracts as contracts
from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1, SecondaryStructurePredictionV2
from dnadesign.contracts.folding.secondary_structure_prediction_v2 import (
    SecondaryStructurePredictionRequestBackendV1,
)
from dnadesign.contracts.sequence import (
    LinearSsdnaCompositionV1,
    MsdDesignCatalogV1,
    MsdDesignReferenceV1,
    RtPartPublicationV1,
)
from dnadesign.contracts.sequence.linear_ssdna_composition_v1 import (
    LinearSsdnaFoldingBackendConfigV1,
)
from dnadesign.contracts.visual import CompositionReviewSvgV1


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SecondaryStructurePredictionRequestBackendV1,
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {"unsupported_parameter": 1},
                "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
            },
        ),
        (
            LinearSsdnaFoldingBackendConfigV1,
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {"unsupported_parameter": 1},
            },
        ),
    ],
)
def test_viennarna_request_surfaces_reject_unsupported_parameters(
    model: type[object],
    payload: dict[str, object],
) -> None:
    with pytest.raises(PydanticValidationError, match="Unsupported ViennaRNA parameters"):
        model.model_validate(payload)  # type: ignore[attr-defined]


def test_viennarna_request_rejects_non_string_parameter_keys_as_validation_error() -> None:
    with pytest.raises(PydanticValidationError, match="parameter names must be strings"):
        SecondaryStructurePredictionRequestBackendV1.model_validate(
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {1: "ignored"},
                "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
            }
        )


@pytest.mark.parametrize("temperature", [True, 0, -1.0, float("nan"), "37"])
def test_viennarna_request_rejects_ineffective_temperature_values(temperature: object) -> None:
    with pytest.raises(PydanticValidationError, match="finite positive number"):
        SecondaryStructurePredictionRequestBackendV1.model_validate(
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {"temperature_c": temperature},
                "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
            }
        )


def test_folding_request_rejects_an_unimplemented_backend() -> None:
    with pytest.raises(PydanticValidationError, match="ViennaRNA"):
        SecondaryStructurePredictionRequestBackendV1.model_validate(
            {
                "name": "ImaginaryFold",
                "interface": "python_api",
                "python_module": "imaginary",
                "parameters": {},
                "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
            }
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SecondaryStructurePredictionRequestBackendV1,
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "backend_contract": "secondary_structure_prediction_v1",
                "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
            },
        ),
        (
            LinearSsdnaFoldingBackendConfigV1,
            {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "backend_contract": "secondary_structure_prediction_v1",
            },
        ),
    ],
)
def test_folding_request_surfaces_reject_unsupported_result_contracts(
    model: type[object],
    payload: dict[str, object],
) -> None:
    with pytest.raises(PydanticValidationError, match="secondary_structure_prediction_v2"):
        model.model_validate(payload)  # type: ignore[attr-defined]


def _rt_part_publication_payload() -> dict[str, object]:
    return {
        "contract": "rt_part_publication_v1",
        "schema_version": 1,
        "owner_study_id": "literature_rt_parts",
        "publication_id": "literature_rt_parts_v1",
        "provenance": {
            "source_ref": "/path/to/study/record/source.yaml",
            "source_contract": "curated_rt_source_v1",
            "source_sha256": _sha256("curated-source"),
        },
        "parts": [
            {
                "part_id": "LiteratureRT-Short",
                "provider_ref": "provider:literature_rt_parts/rt-short",
                "cds_sha256": _sha256("private-cds"),
                "cds_length_nt": 9,
                "terminal_stop_codon": "included",
                "protein_sha256": _sha256("MK"),
                "protein_length_aa": 2,
            }
        ],
    }


def test_root_contract_exports_include_composition_review_manifest() -> None:
    assert contracts.CompositionReviewSvgV1 is CompositionReviewSvgV1


def test_rt_part_publication_accepts_provider_neutral_short_rt() -> None:
    publication = RtPartPublicationV1.model_validate(_rt_part_publication_payload())

    assert publication.owner_study_id == "literature_rt_parts"
    assert publication.parts[0].provider_ref == "provider:literature_rt_parts/rt-short"
    assert publication.parts[0].cds_length_nt == 9
    assert contracts.RtPartPublicationV1 is RtPartPublicationV1


def test_rt_part_publication_accepts_cds_without_terminal_stop_when_declared() -> None:
    payload = _rt_part_publication_payload()
    payload["parts"][0]["cds_length_nt"] = 6
    payload["parts"][0]["terminal_stop_codon"] = "omitted"

    publication = RtPartPublicationV1.model_validate(payload)

    assert publication.parts[0].terminal_stop_codon == "omitted"


def test_rt_part_publication_rejects_private_sequence_bytes() -> None:
    payload = _rt_part_publication_payload()
    part = payload["parts"][0]
    part["cds_sequence_5to3"] = "ATGAAATAA"

    with pytest.raises(PydanticValidationError, match=r"(?s)cds_sequence_5to3.*Extra inputs are not permitted"):
        RtPartPublicationV1.model_validate(payload)


def test_rt_part_publication_rejects_internal_provider_record_ids() -> None:
    payload = _rt_part_publication_payload()
    part = payload["parts"][0]
    part["provider_record_id"] = "internal-generated-candidate-id"

    with pytest.raises(PydanticValidationError, match=r"(?s)provider_record_id.*Extra inputs are not permitted"):
        RtPartPublicationV1.model_validate(payload)


def test_rt_part_publication_rejects_declared_protein_length_drift() -> None:
    payload = _rt_part_publication_payload()
    payload["parts"][0]["protein_length_aa"] = 3

    with pytest.raises(PydanticValidationError, match="declared CDS length 9 does not match protein length 3"):
        RtPartPublicationV1.model_validate(payload)


def test_rt_part_publication_rejects_malformed_sequence_digest() -> None:
    payload = _rt_part_publication_payload()
    payload["parts"][0]["cds_sha256"] = "sha256:not-a-digest"

    with pytest.raises(PydanticValidationError, match="lowercase sha256"):
        RtPartPublicationV1.model_validate(payload)


def test_rt_part_publication_rejects_duplicate_part_ids() -> None:
    payload = _rt_part_publication_payload()
    payload["parts"].append(dict(payload["parts"][0]))

    with pytest.raises(PydanticValidationError, match="Duplicate part_id"):
        RtPartPublicationV1.model_validate(payload)


def test_rt_part_publication_rejects_duplicate_provider_refs() -> None:
    payload = _rt_part_publication_payload()
    duplicate = dict(payload["parts"][0])
    duplicate["part_id"] = "LiteratureRT-Other"
    payload["parts"].append(duplicate)

    with pytest.raises(PydanticValidationError, match="Duplicate provider_ref"):
        RtPartPublicationV1.model_validate(payload)


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


def test_linear_ssdna_composition_contract_accepts_neutral_sources_and_display_policy() -> None:
    config = LinearSsdnaCompositionV1.model_validate(
        {
            "composition_id": "provider_neutral_example",
            "units": [
                {
                    "unit_id": "unit",
                    "segments": [
                        {
                            "segment_id": "record_segment",
                            "sequence": "ACGT",
                            "source": {
                                "kind": "record",
                                "authority": "catalog.example",
                                "record_id": "part-1",
                            },
                        },
                        {
                            "segment_id": "artifact_segment",
                            "sequence": "TGCA",
                            "source": {
                                "kind": "artifact",
                                "contract": "sequence_part_v1",
                                "uri": "artifact://parts/2",
                            },
                        },
                    ],
                }
            ],
            "visual": {
                "display_profile": {
                    "facts": [{"fact_id": "source", "label": "Source", "value": "review fixture"}],
                    "overview_hidden_components": ["artifact_segment"],
                }
            },
        }
    )

    assert config.units[0].segments[0].source.kind == "record"
    assert config.units[0].segments[1].source.kind == "artifact"
    assert config.visual.display_profile.facts[0].fact_id == "source"
    assert config.visual.display_profile.overview_hidden_components == ["artifact_segment"]


@pytest.mark.parametrize("source_kind", ["study_record", "cruncher_artifact"])
def test_linear_ssdna_composition_contract_rejects_retired_provider_source_kinds(source_kind: str) -> None:
    with pytest.raises(PydanticValidationError):
        LinearSsdnaCompositionV1.model_validate(
            {
                "composition_id": "retired_source_kind",
                "units": [
                    {
                        "unit_id": "unit",
                        "segments": [
                            {
                                "segment_id": "payload",
                                "sequence": "ACGT",
                                "source": {"kind": source_kind},
                            }
                        ],
                    }
                ],
            }
        )


def test_linear_ssdna_composition_contract_rejects_study_specific_display_shape() -> None:
    with pytest.raises(PydanticValidationError, match=r"(?s)scar_nick.*Extra inputs are not permitted"):
        LinearSsdnaCompositionV1.model_validate(
            {
                "composition_id": "study_display_shape",
                "units": [{"unit_id": "unit", "segments": [{"segment_id": "payload", "sequence": "ACGT"}]}],
                "visual": {"display_profile": {"scar_nick": {"payload": "TetR"}}},
            }
        )


def test_linear_ssdna_composition_contract_bounds_display_fact_count() -> None:
    with pytest.raises(PydanticValidationError, match="at most 32 items"):
        LinearSsdnaCompositionV1.model_validate(
            {
                "composition_id": "too_many_display_facts",
                "units": [{"unit_id": "unit", "segments": [{"segment_id": "payload", "sequence": "ACGT"}]}],
                "visual": {
                    "display_profile": {
                        "facts": [
                            {"fact_id": f"fact-{index}", "label": "Fact", "value": str(index)} for index in range(33)
                        ]
                    }
                },
            }
        )


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
        SecondaryStructurePredictionV2.model_validate(
            {
                "contract": "secondary_structure_prediction_v2",
                "schema_version": 2,
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
    prediction = SecondaryStructurePredictionV2.model_validate(
        {
            "contract": "secondary_structure_prediction_v2",
            "schema_version": 2,
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
