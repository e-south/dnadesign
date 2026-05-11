"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/tests/test_visual_contracts.py

Shared cassette visual-contract validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.contracts.visual import (
    CassetteViewsManifestV1,
    HairpinTopologyViewV1,
    LinearDuplexViewV1,
    ScarNickVisualV1,
    SequenceEvidenceMapV1,
    SnapbackVisualV1,
    YiuHairpinTopologyV1,
    YiuLinearStateV1,
    YiuTopologyCartoonV1,
)


def _scar_nick_pair_classes(profile: str = "MXMX") -> list[dict[str, object]]:
    base_rows = [
        {
            "position": 0,
            "site": "S3",
            "source_offset": 0,
            "left_base": "G",
            "right_base": "C",
            "aligned_right_base": "G",
        },
        {
            "position": 1,
            "site": "S2",
            "source_offset": 1,
            "left_base": "C",
            "right_base": "T",
            "aligned_right_base": "A",
        },
        {
            "position": 2,
            "site": "S1",
            "source_offset": 2,
            "left_base": "C",
            "right_base": "G",
            "aligned_right_base": "C",
        },
        {
            "position": 3,
            "site": "S0",
            "source_offset": 3,
            "left_base": "C",
            "right_base": "T",
            "aligned_right_base": "A",
        },
    ]
    return [{**row, "class_label": label} for row, label in zip(base_rows, profile, strict=True)]


def _scar_nick_fill(panel: dict[str, object], semantic: str, fill_id: str, fill: str) -> dict[str, object]:
    span = panel[f"{semantic}_span"] if semantic != "nickase_site" else panel["nickase_site_span"]
    return {
        "fill_id": fill_id,
        "semantic": {
            "release_site": "type_iis_release_site",
            "type_iis_offset": "type_iis_offset_spacer",
            "retained_scar": "retained_type_iis_scar",
            "nickase_site": "nickase_footprint",
        }[semantic],
        "start": span["start"],
        "end": span["end"],
        "cover_rows": "both",
        "fill": fill,
        "alpha": {
            "release_site": 0.34,
            "type_iis_offset": 0.28,
            "retained_scar": 0.36,
            "nickase_site": 0.24,
        }[semantic],
        "corner_radius": 0.0,
    }


def _scar_nick_visual_payload() -> dict[str, object]:
    pre_sequence = "GGTCTCGGCCC"
    pre_complement = "CCAGAGCCGGG"
    post_complement = "CCAGAGCCTGT"
    spacer = "NNNN"
    post_offset = len(pre_sequence) + len(spacer)
    pre_panel = {
        "panel_id": "pre_release",
        "title": "before terminal nick",
        "state_kind": "pre_terminal_nick",
        "nick_state": "intact",
        "start": 0,
        "end": len(pre_sequence),
        "terminal_boundary": 11,
        "nick_boundary": 11,
        "retained_product_span": {"start": 7, "end": 11},
        "release_site_span": {"start": 0, "end": 6},
        "type_iis_offset_span": {"start": 6, "end": 7},
        "retained_scar_span": {"start": 7, "end": 11},
        "nickase_site_span": {"start": 0, "end": 11},
        "fragment_spans": [],
    }
    post_panel = {
        "panel_id": "post_release",
        "title": "after terminal nick",
        "state_kind": "post_terminal_nick",
        "nick_state": "nicked",
        "start": post_offset,
        "end": post_offset + len(pre_sequence),
        "terminal_boundary": post_offset + 11,
        "nick_boundary": post_offset + 11,
        "retained_product_span": {"start": post_offset + 7, "end": post_offset + 11},
        "release_site_span": {"start": post_offset, "end": post_offset + 6},
        "type_iis_offset_span": {"start": post_offset + 6, "end": post_offset + 7},
        "retained_scar_span": {"start": post_offset + 7, "end": post_offset + 11},
        "nickase_site_span": {"start": post_offset, "end": post_offset + 11},
        "fragment_spans": [{"row": "complement", "start": post_offset, "end": post_offset + 11}],
    }
    rectangular_fills = []
    for panel in (pre_panel, post_panel):
        prefix = panel["panel_id"]
        rectangular_fills.extend(
            [
                _scar_nick_fill(panel, "release_site", f"{prefix}_type_iis_release_site", "#F0E442"),
                _scar_nick_fill(panel, "type_iis_offset", f"{prefix}_type_iis_offset_spacer", "#FFF6B3"),
                {
                    **_scar_nick_fill(panel, "retained_scar", f"{prefix}_retained_type_iis_scar", "#009E73"),
                    "cover_rows": "primary" if panel["panel_id"] == "post_release" else "both",
                },
            ]
        )
        if panel["panel_id"] == "pre_release":
            rectangular_fills.append(_scar_nick_fill(panel, "nickase_site", f"{prefix}_nickase_footprint", "#56B4E9"))
        if panel["panel_id"] == "post_release":
            fragment_span = panel["fragment_spans"][0]
            rectangular_fills.append(
                {
                    "fill_id": f"{prefix}_annealed_adapter_fragment_0",
                    "semantic": "annealed_adapter_fragment",
                    "start": fragment_span["start"],
                    "end": fragment_span["end"],
                    "cover_rows": fragment_span["row"],
                    "fill": "#CBD5E1",
                    "alpha": 0.48,
                    "corner_radius": 4.0,
                    "edge_color": "#94A3B8",
                    "edge_alpha": 0.64,
                    "edge_linewidth": 0.45,
                }
            )
    return {
        "contract_kind": "scar_nick_visual_v1",
        "state_id": "candidate_01.pre_post_terminal_nick",
        "state_kind": "pre_post_terminal_nick",
        "event_scope": "terminal_nick",
        "alphabet": "iupac_dna",
        "primary_sequence": pre_sequence + spacer + pre_sequence,
        "complement_sequence": pre_complement + spacer + post_complement,
        "primary_row_label": "Top",
        "complement_row_label": "Bottom",
        "terminal_boundary": post_offset + 11,
        "nick_boundary": post_offset + 11,
        "retained_product_span": {"start": post_offset + 7, "end": post_offset + 11},
        "release_site_span": {"start": post_offset, "end": post_offset + 6},
        "type_iis_offset_span": {"start": post_offset + 6, "end": post_offset + 7},
        "retained_scar_span": {"start": post_offset + 7, "end": post_offset + 11},
        "junction_partner_span": None,
        "nickase_site_span": {"start": post_offset, "end": post_offset + 11},
        "nickase_site_source_span": {"start": -7, "end": 4},
        "nick_state": "pre_post",
        "retained_scar": "GCCC",
        "left_base": "GCCC",
        "right_base": "TGTC",
        "nicked_strand": "bottom",
        "surviving_strand": "top",
        "profile_s3s2s1s0": "MXMX",
        "profile_payload_outward": "XMXM",
        "pair_classes": _scar_nick_pair_classes(),
        "panels": [pre_panel, post_panel],
        "rectangular_fills": rectangular_fills,
        "release_placement": {
            "variant_id": "BsaI-HFv2",
            "orientation": "forward",
            "recognition_sequence": "GGTCTC",
            "recognition_site_excised": True,
            "source_catalog_id": "type_iis_release_v1",
            "source_url": "https://www.neb.com/en-us/products/r3733-bsai-hf-v2",
            "commercial_confidence": "primary_vendor_current",
            "warning_codes": [],
            "recognition_site_start": -7,
            "recognition_site_end": -1,
            "top_cut_boundary": 0,
            "bottom_cut_boundary": 4,
            "retained_scar_start": 0,
            "retained_scar_end": 4,
            "retained_scar_nt": 4,
        },
        "nickase": {
            "variant_id": "Test.TerminalBottomNickase",
            "specificity_id": "TerminalBottomNickase",
            "orientation": "forward",
            "canonical_read_row": "primary",
            "motif_top_5to3": "GGTCTCGNNNN",
            "canonical_motif_top_5to3": "GGTCTCGNNNN",
            "recognition_nt": 7,
            "vendor": "dnadesign test fixture",
            "source_url": "https://example.invalid/dnadesign/scar-nick-terminal-fixture",
            "source_family": "nicking_endonuclease",
            "commercial_confidence": "primary_vendor_current",
            "warning_codes": [],
            "source_site_start": -7,
            "source_site_end": 4,
            "strand": "bottom",
            "boundary": 4,
            "terminal_boundary": 4,
            "display_boundary": post_offset + 11,
            "display_site_span": {"start": post_offset, "end": post_offset + 11},
            "exact_terminal": True,
            "site": "Test.TerminalBottomNickase:forward[-7,4)",
        },
        "meta": {
            "panel_spacer_indices": list(range(len(pre_sequence), post_offset)),
            "mismatch_indices": [post_offset + 8, post_offset + 10],
        },
    }


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


def test_scar_nick_visual_contract_validates_nick_state_payload() -> None:
    payload = _scar_nick_visual_payload()

    contract = ScarNickVisualV1.model_validate(payload)

    assert contract.contract_kind == "scar_nick_visual_v1"
    assert contract.state_kind == "pre_post_terminal_nick"
    assert [panel.panel_id for panel in contract.panels] == ["pre_release", "post_release"]
    assert contract.rectangular_fills[2].corner_radius == 0.0
    nickase_fills = [fill for fill in contract.rectangular_fills if fill.semantic == "nickase_footprint"]
    assert len(nickase_fills) == 1
    assert nickase_fills[0].start == contract.panels[0].nickase_site_span.start
    fragment_span = contract.panels[1].fragment_spans[0]
    assert fragment_span.end == contract.panels[1].nick_boundary
    fragment_fill = next(fill for fill in contract.rectangular_fills if fill.semantic == "annealed_adapter_fragment")
    assert fragment_fill.edge_color == "#94A3B8"
    assert fragment_fill.edge_alpha == 0.64
    assert fragment_fill.edge_linewidth == 0.45
    post_scar_fill = next(
        fill for fill in contract.rectangular_fills if fill.fill_id == "post_release_retained_type_iis_scar"
    )
    assert post_scar_fill.cover_rows == "primary"


def test_scar_nick_visual_contract_requires_nickase_strand_and_fragment_row_consistency() -> None:
    mismatched_strand = _scar_nick_visual_payload()
    mismatched_strand["nickase"]["strand"] = "top"
    with pytest.raises(ValueError, match="nickase strand must match nicked_strand"):
        ScarNickVisualV1.model_validate(mismatched_strand)

    wrong_fragment_row = _scar_nick_visual_payload()
    wrong_fragment_row["panels"][1]["fragment_spans"][0]["row"] = "primary"
    with pytest.raises(ValueError, match="fragment spans must be on the nicked strand"):
        ScarNickVisualV1.model_validate(wrong_fragment_row)

    fragment_stops_before_nick = _scar_nick_visual_payload()
    fragment_stops_before_nick["panels"][1]["fragment_spans"][0]["end"] = fragment_stops_before_nick["panels"][1][
        "retained_scar_span"
    ]["start"]
    with pytest.raises(ValueError, match="fragment spans must terminate at the nick boundary"):
        ScarNickVisualV1.model_validate(fragment_stops_before_nick)

    post_scar_covers_nicked_strand = _scar_nick_visual_payload()
    for fill in post_scar_covers_nicked_strand["rectangular_fills"]:
        if fill["fill_id"] == "post_release_retained_type_iis_scar":
            fill["cover_rows"] = "both"
    with pytest.raises(ValueError, match="post-release retained Type IIS scar fill must cover the surviving strand"):
        ScarNickVisualV1.model_validate(post_scar_covers_nicked_strand)


def test_scar_nick_visual_contract_accepts_catalog_type_iis_release_without_bsa_i_pin() -> None:
    payload = _scar_nick_visual_payload()
    payload["release_placement"]["variant_id"] = "BbsI-HF"
    payload["release_placement"]["source_url"] = "https://www.neb.com/en-us/products/r3539-bbsi-hf"

    contract = ScarNickVisualV1.model_validate(payload)

    assert contract.release_placement.variant_id == "BbsI-HF"
    assert contract.release_placement.retained_scar_nt == 4


def test_scar_nick_visual_contract_rejects_non_rectangular_scar_fill() -> None:
    payload = _scar_nick_visual_payload()
    payload["rectangular_fills"][2]["corner_radius"] = 6.0

    with pytest.raises(ValueError, match="retained Type IIS scar fill must be rectangular"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_complement_drift() -> None:
    payload = _scar_nick_visual_payload()
    complement = list(str(payload["complement_sequence"]))
    complement[15] = "A"
    payload["complement_sequence"] = "".join(complement)

    with pytest.raises(ValueError, match="complement only inside post_release retained scar"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_pre_release_adapter_mismatch() -> None:
    payload = _scar_nick_visual_payload()
    complement = list(str(payload["complement_sequence"]))
    complement[10] = "T"
    payload["complement_sequence"] = "".join(complement)

    with pytest.raises(ValueError, match="pre_release panel must be Watson-Crick paired"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_requires_release_sequence_identity() -> None:
    payload = _scar_nick_visual_payload()
    del payload["release_placement"]["recognition_sequence"]

    with pytest.raises(ValueError, match="recognition_sequence"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_requires_nickase_catalog_identity_and_motif() -> None:
    missing_motif = _scar_nick_visual_payload()
    del missing_motif["nickase"]["motif_top_5to3"]
    with pytest.raises(ValueError, match="motif_top_5to3"):
        ScarNickVisualV1.model_validate(missing_motif)

    missing_variant = _scar_nick_visual_payload()
    del missing_variant["nickase"]["variant_id"]
    with pytest.raises(ValueError, match="variant_id"):
        ScarNickVisualV1.model_validate(missing_variant)


def test_scar_nick_visual_contract_rejects_downstream_partner_span() -> None:
    payload = _scar_nick_visual_payload()
    payload["junction_partner_span"] = {"start": 8, "end": 10}

    with pytest.raises(ValueError, match="partner sequence downstream"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_requires_release_and_nickase_fills() -> None:
    no_release_fill = _scar_nick_visual_payload()
    no_release_fill["rectangular_fills"] = [
        fill for fill in no_release_fill["rectangular_fills"] if fill["semantic"] != "type_iis_release_site"
    ]
    with pytest.raises(ValueError, match="type_iis_release_site"):
        ScarNickVisualV1.model_validate(no_release_fill)

    no_nickase_fill = _scar_nick_visual_payload()
    no_nickase_fill["rectangular_fills"] = [
        fill for fill in no_nickase_fill["rectangular_fills"] if fill["semantic"] != "nickase_footprint"
    ]
    with pytest.raises(ValueError, match="nickase_footprint"):
        ScarNickVisualV1.model_validate(no_nickase_fill)


def test_scar_nick_visual_contract_requires_post_release_fragment_fill() -> None:
    payload = _scar_nick_visual_payload()
    payload["rectangular_fills"] = [
        fill for fill in payload["rectangular_fills"] if fill["semantic"] != "annealed_adapter_fragment"
    ]

    with pytest.raises(ValueError, match="annealed_adapter_fragment"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_requires_canonical_nickase_read_row() -> None:
    payload = _scar_nick_visual_payload()
    payload["nickase"]["orientation"] = "reverse"
    payload["nickase"]["canonical_read_row"] = "complement"
    payload["nickase"]["canonical_motif_top_5to3"] = "NNNNCGAGACC"

    ScarNickVisualV1.model_validate(payload)

    payload["nickase"]["canonical_motif_top_5to3"] = "NNNNCGAGACA"

    with pytest.raises(ValueError, match="motif_top_5to3 must match canonical_motif_top_5to3"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_post_release_nickase_fill() -> None:
    payload = _scar_nick_visual_payload()
    post_panel = payload["panels"][1]
    payload["rectangular_fills"].append(
        _scar_nick_fill(post_panel, "nickase_site", "post_release_nickase_footprint", "#56B4E9")
    )

    with pytest.raises(ValueError, match="pre_release panel only"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_short_nickase_recognition_site() -> None:
    payload = _scar_nick_visual_payload()
    payload["nickase"]["variant_id"] = "Nt.CviPII"
    payload["nickase"]["specificity_id"] = "CviPII"
    payload["nickase"]["motif_top_5to3"] = "NNNNHGGNNNN"
    payload["nickase"]["canonical_motif_top_5to3"] = "NNNNHGGNNNN"
    payload["nickase"]["recognition_nt"] = 3

    with pytest.raises(ValueError, match="at least 4 nt"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_pair_class_drift() -> None:
    payload = _scar_nick_visual_payload()
    payload["pair_classes"] = _scar_nick_pair_classes("MMXM")

    with pytest.raises(ValueError, match="pair_classes class labels"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_wobble_label_without_gt_pair() -> None:
    payload = _scar_nick_visual_payload()
    payload["profile_s3s2s1s0"] = "MWMX"
    payload["profile_payload_outward"] = "XMWM"
    payload["pair_classes"] = _scar_nick_pair_classes("MWMX")

    with pytest.raises(ValueError, match="W pair_classes must be G:T or T:G physical pairs"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_terminal_boundary_off_scar_end() -> None:
    payload = _scar_nick_visual_payload()
    payload["terminal_boundary"] = 10
    payload["nick_boundary"] = 10
    payload["nickase"]["display_boundary"] = 10

    with pytest.raises(ValueError, match="terminal_boundary must equal retained_scar_span.end"):
        ScarNickVisualV1.model_validate(payload)


def test_scar_nick_visual_contract_rejects_non_nucleotide_symbols() -> None:
    payload = _scar_nick_visual_payload()
    sequence_length = len(str(payload["primary_sequence"]))
    payload["primary_sequence"] = "Z" * sequence_length
    payload["complement_sequence"] = "Z" * sequence_length

    with pytest.raises(ValueError, match="primary_sequence contains symbols outside iupac_dna"):
        ScarNickVisualV1.model_validate(payload)


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


def test_snapback_visual_contract_validates_foldback_payload() -> None:
    payload = {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.post_nick_foldback",
        "state_kind": "post_nick_foldback",
        "alphabet": "dna",
        "title": "Foldback",
        "primary_sequence": "TCAGCATCTGA",
        "complement_sequence": "GACTTGCAACT",
        "primary_row_label": "Primary",
        "complement_row_label": "Partner",
        "ligation_junction_boundary": 0,
        "protected_region_span": {"start": 0, "end": 4},
        "retained_stem_span": {"start": 0, "end": 4},
        "cap_span": {"start": 4, "end": 7},
        "foldback_revcomp_span": {"start": 7, "end": 11},
        "loop_geometry": {
            "kind": "hairpin_corner_triloop_v1",
            "source_cap_span": {"start": 4, "end": 6},
            "cap_extension_span": {"start": 6, "end": 7},
            "display_primary_span": {"start": 0, "end": 4},
            "display_complement_span": {"start": 7, "end": 11},
        },
        "pairings": [
            {"left_index": 0, "right_index": 10},
            {"left_index": 1, "right_index": 9},
        ],
        "primary_mismatch_positions": [],
        "complement_mismatch_positions": [],
        "meta": {"source": "test"},
    }

    contract = SnapbackVisualV1.model_validate(payload)

    assert contract.contract_kind == "snapback_visual_v1"
    assert contract.state_kind == "post_nick_foldback"
    assert contract.loop_geometry is not None
    assert contract.pairings[0].left_index == 0


def test_snapback_visual_contract_rejects_pairings_outside_foldback_spans() -> None:
    payload = {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.post_nick_foldback",
        "state_kind": "post_nick_foldback",
        "alphabet": "dna",
        "primary_sequence": "TCAGCAGTCTTGACT",
        "complement_sequence": "AGTCGTCAGAACTGA",
        "primary_row_label": "Primary",
        "complement_row_label": "Partner",
        "ligation_junction_boundary": 5,
        "released_prefix_span": {"start": 0, "end": 5},
        "retained_stem_span": {"start": 5, "end": 9},
        "cap_span": {"start": 9, "end": 11},
        "foldback_revcomp_span": {"start": 11, "end": 15},
        "pairings": [
            {"left_index": 4, "right_index": 14},
        ],
        "primary_mismatch_positions": [],
        "complement_mismatch_positions": [],
    }

    with pytest.raises(ValueError, match="pairings left_index must remain inside retained_stem_span"):
        SnapbackVisualV1.model_validate(payload)


def test_snapback_visual_contract_rejects_pre_nick_origin_not_at_nick() -> None:
    payload = {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.pre_nick_duplex",
        "state_kind": "pre_nick_duplex",
        "alphabet": "dna",
        "primary_sequence": "ACGTACGTAA",
        "complement_sequence": "TGCATGCATT",
        "primary_row_label": "Top",
        "complement_row_label": "Partner",
        "nick_boundary": 2,
        "ligation_junction_boundary": 3,
        "retained_stem_span": {"start": 3, "end": 6},
        "cap_span": {"start": 6, "end": 8},
        "foldback_revcomp_span": {"start": 8, "end": 10},
        "pairings": [],
    }

    with pytest.raises(ValueError, match="pre/exposed states must use the nick boundary as the snapback origin"):
        SnapbackVisualV1.model_validate(payload)


def test_snapback_visual_contract_rejects_foldback_loop_geometry_with_noncontiguous_segments() -> None:
    payload = {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.post_nick_foldback",
        "state_kind": "post_nick_foldback",
        "alphabet": "dna",
        "primary_sequence": "TCAGACATCTGA",
        "complement_sequence": "GACTTGCAACTA",
        "primary_row_label": "Primary",
        "complement_row_label": "Partner",
        "ligation_junction_boundary": 0,
        "retained_stem_span": {"start": 0, "end": 4},
        "cap_span": {"start": 5, "end": 8},
        "foldback_revcomp_span": {"start": 8, "end": 12},
        "loop_geometry": {
            "kind": "hairpin_corner_triloop_v1",
            "source_cap_span": {"start": 5, "end": 7},
            "cap_extension_span": {"start": 7, "end": 8},
            "display_primary_span": {"start": 0, "end": 4},
            "display_complement_span": {"start": 8, "end": 12},
        },
        "pairings": [
            {"left_index": 0, "right_index": 11},
            {"left_index": 1, "right_index": 10},
        ],
    }

    with pytest.raises(ValueError, match="loop_geometry requires retained_stem_span.end == cap_span.start"):
        SnapbackVisualV1.model_validate(payload)


def test_snapback_visual_contract_rejects_foldback_loop_geometry_with_unequal_display_spans() -> None:
    payload = {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.post_nick_foldback",
        "state_kind": "post_nick_foldback",
        "alphabet": "dna",
        "primary_sequence": "TCAGCATCTG",
        "complement_sequence": "GACTTGCAAC",
        "primary_row_label": "Primary",
        "complement_row_label": "Partner",
        "ligation_junction_boundary": 0,
        "retained_stem_span": {"start": 0, "end": 4},
        "cap_span": {"start": 4, "end": 7},
        "foldback_revcomp_span": {"start": 7, "end": 10},
        "loop_geometry": {
            "kind": "hairpin_corner_triloop_v1",
            "source_cap_span": {"start": 4, "end": 6},
            "cap_extension_span": {"start": 6, "end": 7},
            "display_primary_span": {"start": 0, "end": 4},
            "display_complement_span": {"start": 7, "end": 10},
        },
        "pairings": [
            {"left_index": 0, "right_index": 9},
            {"left_index": 1, "right_index": 8},
        ],
    }

    with pytest.raises(ValueError, match="loop_geometry display spans must have equal length"):
        SnapbackVisualV1.model_validate(payload)
