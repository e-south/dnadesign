"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/tests/test_visual_contracts.py

Shared cassette visual-contract validation tests.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.contracts.visual import (
    CassetteViewsManifestV1,
    HairpinTopologyViewV1,
    LinearDuplexViewV1,
    SequenceEvidenceMapV1,
    YiuHairpinTopologyV1,
    YiuLinearStateV1,
    YiuTopologyCartoonV1,
)


def test_linear_duplex_view_contract_validates_example_payload() -> None:
    payload = {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": "hit_001.linear_duplex",
        "solution_id": "abc123def456",
        "title": "Hit 1 - Linear duplex",
        "coordinate_semantics": "boundary_inclusive_v2",
        "primary_sequence_5to3": "TTTACCTCAGCAAAGCTGAGGTAAA",
        "sequence_span": {"start": 0, "end": 25},
        "cassette_span": {"start": 0, "end": 25},
        "row_labels": {
            "primary": "5' -> 3' primary",
            "complement": "3' -> 5' complement",
        },
        "target_strand": "complement",
        "segments": [
            {"id": "stem5p_arm", "start": 0, "end": 10, "semantic": "stem5p_arm", "label": "Stem 5' arm"},
            {"id": "loop", "start": 10, "end": 15, "semantic": "loop", "label": "Loop"},
            {"id": "stem3p_arm", "start": 15, "end": 25, "semantic": "stem3p_arm", "label": "Stem 3' arm"},
        ],
        "site_instances": [
            {
                "id": "left_site",
                "variant_id": "Nb.BbvCI",
                "specificity_id": "BbvCI",
                "start": 2,
                "end": 9,
                "orientation": "forward",
                "intent": "intended_left",
                "label": "Nb.BbvCI",
                "site_target_strand": "complement",
            },
            {
                "id": "right_site",
                "variant_id": "Nt.BbvCI",
                "specificity_id": "BbvCI",
                "start": 16,
                "end": 23,
                "orientation": "reverse",
                "intent": "intended_right",
                "label": "Nt.BbvCI",
                "site_target_strand": "complement",
            },
        ],
        "nick_events": [
            {
                "id": "left_nick",
                "boundary": 7,
                "target_strand": "complement",
                "source_site_id": "left_site",
                "intent": "intended_left",
                "label": "Nick",
            },
            {
                "id": "right_nick",
                "boundary": 20,
                "target_strand": "complement",
                "source_site_id": "right_site",
                "intent": "intended_right",
                "label": "Nick",
            },
        ],
        "bounded_segment": {
            "start_boundary": 7,
            "end_boundary": 20,
            "target_strand": "complement",
            "label": "Bounded nicked segment",
        },
        "labels": [{"text": "Target strand: complement", "placement": "header"}],
        "meta": {
            "rank": 1,
            "left_variant_id": "Nb.BbvCI",
            "right_variant_id": "Nt.BbvCI",
            "left_boundary": 7,
            "right_boundary": 20,
            "bounded_length_nt": 13,
        },
    }

    view = LinearDuplexViewV1.model_validate(payload)

    assert view.kind == "linear_duplex_v1"
    assert view.target_strand == "complement"
    assert view.bounded_segment.end_boundary - view.bounded_segment.start_boundary == 13


def test_hairpin_topology_view_contract_validates_example_payload() -> None:
    payload = {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": "ACCTCAGCAAAGCTGAGGT",
        "topology": {
            "stem5p_span": {"start": 0, "end": 7},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [
            {"left_index": 0, "right_index": 18},
            {"left_index": 1, "right_index": 17},
            {"left_index": 2, "right_index": 16},
        ],
        "feature_spans": [
            {
                "id": "left_site_projection",
                "start": 1,
                "end": 7,
                "semantic": "motif_projection",
                "label": "Nb.BbvCI motif",
            },
            {
                "id": "right_site_projection",
                "start": 12,
                "end": 18,
                "semantic": "motif_projection",
                "label": "Nt.BbvCI motif",
            },
        ],
        "duplex_derived_annotations": [
            {
                "kind": "informational_note",
                "text": "Nicking is defined in the linear duplex interpretation.",
            }
        ],
        "meta": {"rank": 1, "left_variant_id": "Nb.BbvCI", "right_variant_id": "Nt.BbvCI"},
    }

    view = HairpinTopologyViewV1.model_validate(payload)

    assert view.kind == "ssdna_hairpin_v1"
    assert len(view.pair_map) == 3
    assert view.topology.loop_span.end - view.topology.loop_span.start == 5


def test_views_manifest_validates_relative_view_and_job_paths() -> None:
    payload = {
        "version": 1,
        "kind": "cassette_views_manifest_v1",
        "solution_id": "abc123def456",
        "rank": 1,
        "views": [
            {"view_kind": "linear_duplex_v1", "path": "linear_duplex.v1.json"},
            {"view_kind": "ssdna_hairpin_v1", "path": "ssdna_hairpin.v1.json"},
        ],
        "recommended_jobs": [
            {"name": "linear_duplex", "path": "../baserender_jobs/linear_duplex.job.yaml"},
            {"name": "ssdna_hairpin", "path": "../baserender_jobs/ssdna_hairpin.job.yaml"},
        ],
    }

    manifest = CassetteViewsManifestV1.model_validate(payload)

    assert manifest.kind == "cassette_views_manifest_v1"
    assert manifest.views[0].path == "linear_duplex.v1.json"
    assert manifest.recommended_jobs[1].name == "ssdna_hairpin"


def test_hairpin_topology_view_rejects_empty_pair_map() -> None:
    payload = {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": "ACCTCAGCAAAGCTGAGGT",
        "topology": {
            "stem5p_span": {"start": 0, "end": 7},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [],
    }

    with pytest.raises(ValueError, match="pair_map must be non-empty"):
        HairpinTopologyViewV1.model_validate(payload)


def test_hairpin_topology_view_rejects_overlapping_spans() -> None:
    payload = {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": "ACCTCAGCAAAGCTGAGGT",
        "topology": {
            "stem5p_span": {"start": 0, "end": 9},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [{"left_index": 0, "right_index": 18}],
    }

    with pytest.raises(ValueError, match="stem5p_span must end at or before loop_span.start"):
        HairpinTopologyViewV1.model_validate(payload)


def test_linear_duplex_view_rejects_overlapping_segments() -> None:
    payload = {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": "hit_001.linear_duplex",
        "solution_id": "abc123def456",
        "title": "Hit 1 - Linear duplex",
        "coordinate_semantics": "boundary_inclusive_v2",
        "primary_sequence_5to3": "TTTACCTCAGCAAAGCTGAGGTAAA",
        "sequence_span": {"start": 0, "end": 25},
        "cassette_span": {"start": 0, "end": 25},
        "row_labels": {
            "primary": "5' -> 3' primary",
            "complement": "3' -> 5' complement",
        },
        "target_strand": "complement",
        "segments": [
            {"id": "stem5p_arm", "start": 0, "end": 10, "semantic": "stem5p_arm", "label": "Stem 5' arm"},
            {"id": "loop", "start": 9, "end": 15, "semantic": "loop", "label": "Loop"},
        ],
    }

    with pytest.raises(ValueError, match="segments must be ordered and non-overlapping"):
        LinearDuplexViewV1.model_validate(payload)


def test_yiu_linear_state_contract_validates_minimal_payload() -> None:
    payload = {
        "contract_kind": "yiu_linear_state_v1",
        "state_id": "hairpin_pcr_linear_insert",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": "CCTCAGCCCGCTGATCCCTATCAGTGATAGAR",
        "complement_sequence": "YTCTATCACTGATAGGGATCAGCGGGCTGAGG",
        "segments": [],
        "annotations": [],
        "cuts": [],
        "junctions": [],
        "fragments": [],
        "display": {"title": "Split-payload insert"},
        "meta": {"evidence_mode": "pattern_compatibility"},
    }

    contract = YiuLinearStateV1.model_validate(payload)

    assert contract.contract_kind == "yiu_linear_state_v1"
    assert contract.alphabet == "iupac_dna"


def test_yiu_hairpin_topology_contract_validates_minimal_payload() -> None:
    payload = {
        "contract_kind": "yiu_hairpin_topology_v1",
        "state_id": "ligated_ssdna_hairpin",
        "topology_kind": "ssdna_hairpin",
        "sequence": "CCTCAGCCCGCTGATCAGCGGGCTGAGG",
        "stem_left_span": {"start": 0, "end": 8},
        "stem_right_span": {"start": 20, "end": 28},
        "loop_span": {"start": 8, "end": 20},
        "pair_map": [{"left_index": 0, "right_index": 27}],
        "adapter_branches": [],
        "annotations": [],
        "display": {"title": "Ligation hairpin"},
        "meta": {"evidence_mode": "concrete_realization"},
    }

    contract = YiuHairpinTopologyV1.model_validate(payload)

    assert contract.contract_kind == "yiu_hairpin_topology_v1"
    assert len(contract.pair_map) == 1


def test_yiu_topology_cartoon_contract_validates_minimal_payload() -> None:
    payload = {
        "contract_kind": "yiu_topology_cartoon_v1",
        "state_id": "circularized_payload_candidate",
        "topology_kind": "circular_duplex",
        "sequence": "CCGATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
        "segments": [],
        "annotations": [],
        "cuts": [],
        "junctions": [{"id": "junction", "join_index": 15}],
        "fragments": [],
        "display": {"title": "Circularized payload"},
        "meta": {"evidence_mode": "concrete_realization"},
    }

    contract = YiuTopologyCartoonV1.model_validate(payload)

    assert contract.contract_kind == "yiu_topology_cartoon_v1"
    assert contract.topology_kind == "circular_duplex"


def test_sequence_evidence_map_contract_validates_minimal_payload() -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "hairpin_pcr_linear_insert",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": "ACGTACGT",
        "complement_sequence": "TGCATGCA",
        "owners": [
            {
                "owner_id": "payload_left_half",
                "row_id": "primary",
                "start": 0,
                "end": 4,
                "display_label": "Payload",
                "short_label": "PAY",
            }
        ],
        "effect_tags": [
            {
                "tag_id": "overhang",
                "tag_kind": "payload_overhang_left",
                "row_id": "primary",
                "start": 0,
                "end": 2,
                "display_label": "Overhang",
                "short_label": "OVL",
            }
        ],
        "boundaries": [
            {
                "boundary_id": "nick-1",
                "row_id": "primary",
                "boundary": 4,
                "boundary_kind": "nick",
                "display_label": "Nick",
                "short_label": "NCK",
            }
        ],
        "pairings": [
            {
                "pairing_id": "pair-1",
                "primary_start": 0,
                "primary_end": 2,
                "complement_start": 6,
                "complement_end": 8,
                "display_label": "Pairing",
                "short_label": "PR",
            }
        ],
        "display": {"title": "Example"},
        "meta": {"source": "test"},
    }

    contract = SequenceEvidenceMapV1.model_validate(payload)

    assert contract.contract_kind == "sequence_evidence_map_v1"
    assert contract.boundaries[0].boundary_kind == "nick"


def test_sequence_evidence_map_contract_rejects_invalid_owner_bounds() -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-owner",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "owners": [
            {
                "owner_id": "payload_left_half",
                "row_id": "primary",
                "start": 1,
                "end": 1,
                "display_label": "Payload",
                "short_label": "PAY",
            }
        ],
    }

    with pytest.raises(ValueError, match="owner span end must be > start"):
        SequenceEvidenceMapV1.model_validate(payload)


def test_sequence_evidence_map_contract_rejects_invalid_effect_bounds() -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-effect",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "effect_tags": [
            {
                "tag_id": "effect-1",
                "tag_kind": "payload_overhang_left",
                "row_id": "primary",
                "start": 2,
                "end": 2,
                "display_label": "Effect",
                "short_label": "EFF",
            }
        ],
    }

    with pytest.raises(ValueError, match="effect span end must be > start"):
        SequenceEvidenceMapV1.model_validate(payload)


def test_sequence_evidence_map_contract_rejects_boundary_length_overflow() -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-pairing",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCA",
        "boundaries": [
            {
                "boundary_id": "boundary-1",
                "row_id": "primary",
                "boundary": 5,
                "boundary_kind": "cut",
                "display_label": "Boundary",
                "short_label": "BND",
            }
        ],
    }

    with pytest.raises(ValueError, match="boundary exceeds row sequence length"):
        SequenceEvidenceMapV1.model_validate(payload)


@pytest.mark.parametrize(
    ("pairing_updates", "message"),
    [
        (
            {"primary_start": 2, "primary_end": 2},
            "pairing primary span end must be > start",
        ),
        (
            {"complement_start": 2, "complement_end": 2},
            "pairing complement span end must be > start",
        ),
    ],
)
def test_sequence_evidence_map_contract_rejects_invalid_pairing_bounds(
    pairing_updates: dict[str, int],
    message: str,
) -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-pairing",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCA",
        "pairings": [
            {
                "pairing_id": "pair-1",
                "primary_start": 0,
                "primary_end": 2,
                "complement_start": 0,
                "complement_end": 2,
                "display_label": "Pairing",
                "short_label": "PR",
                **pairing_updates,
            }
        ],
    }

    with pytest.raises(ValueError, match=message):
        SequenceEvidenceMapV1.model_validate(payload)


@pytest.mark.parametrize(
    ("payload_key", "row_id", "message"),
    [
        ("owners", "complement", "owner span exceeds row sequence length"),
        ("effect_tags", "complement", "effect span exceeds row sequence length"),
    ],
)
def test_sequence_evidence_map_contract_rejects_complement_span_overflow(
    payload_key: str,
    row_id: str,
    message: str,
) -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-span",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCA",
        payload_key: [
            {
                "owner_id" if payload_key == "owners" else "tag_id": "overflow",
                "owner_id": "payload_left_half" if payload_key == "owners" else None,
                "tag_kind": "payload_overhang_left" if payload_key == "effect_tags" else None,
                "row_id": row_id,
                "start": 0,
                "end": 5,
                "display_label": "Overflow",
                "short_label": "OVR",
            }
        ],
    }
    payload[payload_key][0] = {k: v for k, v in payload[payload_key][0].items() if v is not None}

    with pytest.raises(ValueError, match=message):
        SequenceEvidenceMapV1.model_validate(payload)


@pytest.mark.parametrize(
    ("pairing_updates", "message"),
    [
        (
            {"primary_start": 0, "primary_end": 5, "complement_start": 0, "complement_end": 2},
            "pairing primary span exceeds primary sequence length",
        ),
        (
            {"primary_start": 0, "primary_end": 2, "complement_start": 0, "complement_end": 5},
            "pairing complement span exceeds complement sequence length",
        ),
    ],
)
def test_sequence_evidence_map_contract_rejects_pairing_length_overflow(
    pairing_updates: dict[str, int],
    message: str,
) -> None:
    payload = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "bad-pairing-length",
        "topology_kind": "linear_dsdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCA",
        "pairings": [
            {
                "pairing_id": "pair-1",
                "display_label": "Pairing",
                "short_label": "PR",
                **pairing_updates,
            }
        ],
    }

    with pytest.raises(ValueError, match=message):
        SequenceEvidenceMapV1.model_validate(payload)
