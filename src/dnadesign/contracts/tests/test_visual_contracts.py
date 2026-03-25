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

from dnadesign.contracts.visual import CassetteViewsManifestV1, HairpinTopologyViewV1, LinearDuplexViewV1


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
