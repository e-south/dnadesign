"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/scar_nick/test_planner.py

Planning, rejection, and ranking tests for scar-nick candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dnadesign.cruncher.app.scar_nick_workflow import validate_scar_nick_spec
from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.scar_nick.candidates import evaluate_pair_candidate
from dnadesign.cruncher.scar_nick.geometry import placements_for_entry
from dnadesign.cruncher.scar_nick.models import CandidateRankingContext, NickasePlacement, ReleasePlacement
from dnadesign.cruncher.scar_nick.ranking import rank_pair_candidates, unique_sequence_candidates
from dnadesign.cruncher.scar_nick.view_contracts import build_terminal_nick_visual_contract
from dnadesign.cruncher.scar_nick.view_models import ScarNickTerminalNickViewV1


def _base_spec(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scar_nick": {
            "schema_version": 1,
            "contract": "terminal_type_iis_scar_nick_v1",
            "name": "teto_upstream_processing",
        },
        "junction": {
            "left_base": "CGGG",
            "right_base": "ACAG",
            "profile_order": "S3_S2_S1_S0",
            "s0_match_required": True,
            "overhang_length": 4,
        },
        "processing": {
            "release": {
                "variant_id": "BsaI-HFv2",
                "catalog": {"preset": "type_iis_release_v1"},
                "required_terminal_scar_nt": 4,
                "recognition_site_must_be_excised": True,
            },
            "nick": {
                "target_strand": "either",
                "terminal_nick_required": True,
                "downstream_protected_nt_allowed": 0,
                "downstream_must_be_degenerate": True,
                "catalog": {
                    "preset": "neb_nicking_v1",
                    "additional_presets": ["thermo_nicking_v1"],
                },
            },
        },
        "ranking_context": {
            "anchor_mode": "profile_analog",
            "optional_reference_profiles": {
                "working_control": {
                    "id": "retron_26",
                    "left_base": "CGGG",
                    "right_base": "ACAG",
                    "profile_s3s2s1s0": "MXMX",
                },
                "failed_control": {
                    "id": "retron_43",
                    "left_base": "CAAG",
                    "right_base": "CTCG",
                    "profile_s3s2s1s0": "MXMM",
                },
            },
            "target_profile_buckets": [
                "MXMM",
                "XWMM",
                "MWXM",
                "MXWM",
                "XMWM",
                "MWMM",
                "MMWM",
                "MWWM",
                "XXMM",
                "XMXM",
            ],
            "reject_profiles": ["MMMM"],
            "allow_gt_wobble": True,
            "active_max_hard_mismatches": 2,
            "active_max_non_watson_crick_pairs": 2,
            "forbid_active_middle_middle_double_hard": True,
            "min_ligation_support": 2.0,
            "max_effective_disruption": 2.5,
            "prefer_lower_middle_hard_mismatch_tier": True,
            "prefer_lower_hard_mismatch_tier": True,
            "reduce_gc_when_tied": True,
        },
        "search": {
            "mode": "curated_panel",
            "max_hits": 16,
            "materialize_top_k": 8,
        },
        "output": {"run_dir": "outputs/scar_nick/teto_upstream_processing"},
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


def _write_spec(tmp_path: Path, payload: dict[str, Any]) -> Path:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    spec_path = workspace / "configs" / "scar_nick" / "teto_upstream_processing.scar_nick.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return spec_path


def _write_terminal_nickase_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    nick_catalog = workspace / "inputs" / "nickases" / "terminal.nickases.yaml"
    nick_catalog.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Test.TerminalBottomNickase",
                            "specificity_id": "TerminalBottomNickase",
                            "motif_top_5to3": "GGTCTCGNNNN",
                            "vendor_diagram_top_5to3": "GGTCTCGNNNN",
                            "bottom_cut_offset": 11,
                            "vendor": "dnadesign test fixture",
                            "source_url": "https://example.invalid/dnadesign/scar-nick-terminal-fixture",
                            "source_family": "nicking_endonuclease",
                            "commercial_confidence": "primary_vendor_current",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _with_terminal_nickase_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    payload["processing"]["nick"]["catalog"] = {"additional_paths": ["inputs/nickases/terminal.nickases.yaml"]}
    return payload


def _release_placement(
    *,
    variant_id: str = "BsaI-HFv2",
    recognition_sequence: str = "GGTCTC",
    recognition_site_start: int = -7,
    recognition_site_end: int = -1,
) -> ReleasePlacement:
    return ReleasePlacement(
        variant_id=variant_id,
        orientation="forward",
        recognition_sequence=recognition_sequence,
        source_catalog_id="type_iis_release_v1",
        source_url=f"https://example.invalid/dnadesign/{variant_id}",
        commercial_confidence="primary_vendor_current",
        warning_codes=[],
        recognition_site_start=recognition_site_start,
        recognition_site_end=recognition_site_end,
        top_cut_boundary=0,
        bottom_cut_boundary=4,
        retained_scar_start=0,
        retained_scar_end=4,
        retained_scar_nt=4,
        recognition_site_excised=True,
    )


def _nickase_placement(
    *,
    variant_id: str = "Test.TerminalBottomNickase",
    strand: str = "bottom",
) -> NickasePlacement:
    return NickasePlacement(
        variant_id=variant_id,
        specificity_id="TerminalBottomNickase",
        orientation="forward",
        motif_top_5to3="GGTCTCGNNNN",
        canonical_motif_top_5to3="GGTCTCGNNNN",
        vendor="dnadesign test fixture",
        source_url="https://example.invalid/dnadesign/scar-nick-terminal-fixture",
        source_family="nicking_endonuclease",
        commercial_confidence="primary_vendor_current",
        warning_codes=[],
        source_site_start=-7,
        source_site_end=4,
        strand=strand,
        boundary=4,
        terminal_boundary=4,
        boundary_distance=0,
        exact_terminal=True,
    )


def test_validate_scar_nick_spec_returns_ranked_hits(tmp_path: Path) -> None:
    _write_terminal_nickase_catalog(tmp_path)
    spec_path = _write_spec(tmp_path, _with_terminal_nickase_catalog(_base_spec()))

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "satisfied"
    assert report.spec_name == "teto_upstream_processing"
    assert report.candidates
    assert len(report.candidates) <= 16
    assert report.candidates[0].rank == 1
    assert report.candidates[0].profile_s3s2s1s0 in {
        "MXMM",
        "WXMM",
        "XWMM",
        "MWXM",
        "MXWM",
        "XMWM",
        "WMMM",
        "MWMM",
        "MMWM",
        "WWMM",
        "WMWM",
        "MWWM",
    }
    assert report.candidates[0].retained_scar == report.candidates[0].left_base
    assert report.candidates[0].retained_product_sequence == report.candidates[0].left_base
    assert report.candidates[0].nick_boundary == 4
    assert report.candidates[0].release_placement is not None
    assert report.candidates[0].release_placement.variant_id == "BsaI-HFv2"
    assert report.candidates[0].nickase_placement is not None
    assert report.candidates[0].nickase_placement.variant_id == "Test.TerminalBottomNickase"
    assert report.candidates[0].nickase_placement.orientation == "forward"
    assert report.candidates[0].nickase_placement.strand == "bottom"
    assert report.candidates[0].rejection_reasons == []
    assert report.metadata.compatible_nickase_placement_count == 1
    assert report.metadata.enzyme_compatible_scar_count == 256
    terminal_fixture = next(
        entry
        for entry in report.nickase_geometry_audit
        if entry.variant_id == "Test.TerminalBottomNickase" and entry.orientation == "forward"
    )
    assert terminal_fixture.compatible is True
    assert terminal_fixture.type_iis_offset_sequence == "G"
    assert [domain.bases for domain in terminal_fixture.retained_scar_domains] == [
        ["A", "C", "G", "T"],
        ["A", "C", "G", "T"],
        ["A", "C", "G", "T"],
        ["A", "C", "G", "T"],
    ]


def test_public_catalog_geometry_audit_explains_no_current_solution(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, _base_spec())

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.metadata.compatible_nickase_placement_count == 0
    assert report.metadata.enzyme_compatible_scar_count == 0
    assert len(report.nickase_geometry_audit) >= 14

    bsrdi = next(
        entry
        for entry in report.nickase_geometry_audit
        if entry.variant_id == "Nb.BsrDI" and entry.orientation == "forward"
    )
    assert "NICKASE_RELEASE_SITE_OVERLAP_CONFLICT" in bsrdi.rejection_reasons
    assert [
        (conflict.raw_coordinate, conflict.nickase_symbol, conflict.release_symbol)
        for conflict in bsrdi.release_overlap_conflicts
    ] == [(-2, "G", "C")]
    assert [domain.bases for domain in bsrdi.retained_scar_domains] == [["A"], ["A"], ["T"], ["G"]]

    bspqi = next(
        entry
        for entry in report.nickase_geometry_audit
        if entry.variant_id == "Nt.BspQI" and entry.orientation == "reverse"
    )
    assert "NON_DEGENERATE_DOWNSTREAM_OF_TERMINAL_NICK" in bspqi.rejection_reasons
    assert any(not symbol.fully_degenerate for symbol in bspqi.downstream_symbols)

    cvi = next(
        entry
        for entry in report.nickase_geometry_audit
        if entry.variant_id == "Nt.CviPII" and entry.orientation == "reverse"
    )
    assert "NICKASE_POLICY_REJECTED" in cvi.rejection_reasons
    assert set(cvi.policy_rejection_reasons) == {
        "NICKASE_RECOGNITION_SITE_TOO_SHORT",
        "NICKASE_WARNING_CODE_DISALLOWED:FREQUENT_CUTTER",
    }


def test_bbs_i_hf_release_expansion_finds_public_catalog_hits(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["processing"]["release"]["variant_id"] = "BbsI-HF"
    spec_path = _write_spec(tmp_path, payload)

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "satisfied"
    assert report.release_placement is not None
    assert report.release_placement.variant_id == "BbsI-HF"
    assert report.release_placement.retained_scar_nt == 4
    assert report.metadata.compatible_nickase_placement_count == 3
    assert report.metadata.enzyme_compatible_scar_count == 6
    assert report.candidates == unique_sequence_candidates(report.candidates)
    assert report.metadata.accepted_candidate_count == len(report.candidates)
    assert report.metadata.materialized_candidate_count == min(8, len(report.candidates))
    assert {candidate.profile_s3s2s1s0 for candidate in report.candidates} >= {
        "MXMM",
        "XWMM",
        "MWXM",
        "MXWM",
        "XMWM",
        "MWMM",
        "MMWM",
        "MWWM",
        "XXMM",
        "XMXM",
    }
    assert "MXXM" not in {candidate.profile_s3s2s1s0 for candidate in report.candidates}
    assert all(candidate.profile_policy_status == "active" for candidate in report.candidates)
    assert {candidate.profile_s3s2s1s0 for candidate in report.candidates} >= {"XXMM", "XMXM"}
    assert "MXXM" in {candidate.profile_s3s2s1s0 for candidate in report.reserve_candidates}
    assert all(candidate.x_count <= 2 for candidate in report.candidates)
    assert all(
        not (candidate.pair_classes[1].class_label == "X" and candidate.pair_classes[2].class_label == "X")
        for candidate in report.candidates
    )
    assert all(candidate.non_watson_crick_count <= 2 for candidate in report.candidates)
    assert {candidate.left_base for candidate in report.candidates} <= {
        "AATG",
        "AGTG",
        "CTCA",
        "CTCC",
        "CTCG",
        "CTCT",
    }
    assert {entry.strand for entry in report.nickase_geometry_audit if entry.compatible} == {"top", "bottom"}
    bsssi_entries = [entry for entry in report.nickase_geometry_audit if entry.variant_id == "Nb.BssSI"]
    assert bsssi_entries
    assert all(not entry.compatible for entry in bsssi_entries)
    assert all("NON_DEGENERATE_DOWNSTREAM_OF_TERMINAL_NICK" in entry.rejection_reasons for entry in bsssi_entries)
    observed_nickases = {
        candidate.nickase_placement.variant_id for candidate in report.candidates if candidate.nickase_placement
    }
    assert observed_nickases <= {
        "Nb.BsrDI",
        "Nb.BtsI",
        "Nt.BsmAI",
    }


def test_target_profile_bucket_must_fit_declared_hard_gates(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["processing"]["release"]["variant_id"] = "BbsI-HF"
    payload["ranking_context"]["target_profile_buckets"] = ["MXXX"]
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="target_profile_buckets conflict"):
        validate_scar_nick_spec(spec_path)


def test_terminal_nick_required_false_is_rejected_at_spec_boundary(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["processing"]["nick"]["terminal_nick_required"] = False
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="terminal_nick_required=true"):
        validate_scar_nick_spec(spec_path)


def test_reference_profile_payload_must_match_declared_bases(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["ranking_context"]["optional_reference_profiles"]["working_control"]["profile_s3s2s1s0"] = "MMMM"
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="reference profile mismatch"):
        validate_scar_nick_spec(spec_path)


def test_ranking_profiles_cannot_be_rejected_and_preferred_or_allowed() -> None:
    with pytest.raises(ValueError, match="both rejected"):
        CandidateRankingContext(
            target_profile_buckets=["MMMM"],
            reject_profiles=["MMMM"],
            allow_gt_wobble=True,
            active_max_hard_mismatches=4,
            active_max_non_watson_crick_pairs=4,
        )


def test_target_profile_buckets_must_be_unique() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        CandidateRankingContext(
            target_profile_buckets=["MXXM", "MXXM"],
            reject_profiles=[],
            allow_gt_wobble=True,
            active_max_hard_mismatches=2,
            active_max_non_watson_crick_pairs=2,
        )


def test_max_hits_must_cover_declared_target_bucket_count(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["search"]["max_hits"] = 2
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="max_hits"):
        validate_scar_nick_spec(spec_path)


def test_output_run_dir_must_stay_under_scar_nick_outputs(tmp_path: Path) -> None:
    payload = _base_spec()
    payload["output"]["run_dir"] = "src/generated/scar_nick/leak"
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="outputs/scar_nick"):
        validate_scar_nick_spec(spec_path)


def test_non_terminal_bsai_scar_is_a_hard_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    release_catalog = workspace / "inputs" / "release" / "bad.release.yaml"
    release_catalog.parent.mkdir(parents=True, exist_ok=True)
    release_catalog.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "BadI",
                            "display_name": "BadI",
                            "recognition_sequence": "GGTCTC",
                            "top_cut_offset": 7,
                            "bottom_cut_offset": 10,
                            "class_label": "type_iis",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "test",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = _base_spec()
    payload["processing"]["release"]["variant_id"] = "BadI"
    payload["processing"]["release"]["catalog"] = {"additional_paths": ["inputs/release/bad.release.yaml"]}
    spec_path = _write_spec(tmp_path, payload)

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["NON_TERMINAL_RELEASE_SCAR"]


def test_no_exact_terminal_nick_is_a_hard_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    nick_catalog = workspace / "inputs" / "nickases" / "top_only.nickases.yaml"
    nick_catalog.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.only",
                            "specificity_id": "Only",
                            "motif_top_5to3": "AACGA",
                            "top_cut_offset": 2,
                            "vendor": "dnadesign test fixture",
                            "source_url": "https://example.invalid/dnadesign/top-only-fixture",
                            "source_family": "nicking_endonuclease",
                            "commercial_confidence": "primary_vendor_current",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = _base_spec()
    payload["processing"]["nick"]["catalog"] = {"additional_paths": ["inputs/nickases/top_only.nickases.yaml"]}
    spec_path = _write_spec(tmp_path, payload)

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["NO_EXACT_TERMINAL_NICK"]


def test_nickase_catalog_entries_must_satisfy_scar_nick_catalog_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    nick_catalog = workspace / "inputs" / "nickases" / "generic.nickases.yaml"
    nick_catalog.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Generic.Nickase",
                            "specificity_id": "GenericNickase",
                            "motif_top_5to3": "GGTCTCGNNNN",
                            "vendor_diagram_top_5to3": "GGTCTCGNNNN",
                            "bottom_cut_offset": 11,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = _base_spec()
    payload["processing"]["nick"]["catalog"] = {"additional_paths": ["inputs/nickases/generic.nickases.yaml"]}
    spec_path = _write_spec(tmp_path, payload)

    report = validate_scar_nick_spec(spec_path)

    assert report.status == "unsatisfied"
    assert [issue.code for issue in report.issues] == ["NO_ELIGIBLE_NICKASE_CATALOG_ENTRIES"]
    assert report.issues[0].details["entry_rejection_counts"] == {
        "NICKASE_COMMERCIAL_CONFIDENCE_MISSING": 1,
        "NICKASE_SOURCE_FAMILY_NOT_NICKING_ENDONUCLEASE": 1,
        "NICKASE_SOURCE_URL_MISSING": 1,
        "NICKASE_VENDOR_METADATA_MISSING": 1,
    }


def test_terminal_nick_placements_allow_top_and_bottom_routes_and_reject_invalid_terminal_tails() -> None:
    short_frequent_cutter = NickaseCatalogEntry(
        id="Nt.CviPII",
        specificity_id="CviPII",
        motif_top_5to3="CCD",
        top_cut_offset=-3,
        vendor="NEB",
        source_url="https://www.neb.com/en-us/products/r0626-ntcvipii",
        source_family="nicking_endonuclease",
        selection={"commercial_confidence": "primary_vendor_current", "warning_codes": ["FREQUENT_CUTTER"]},
    )
    nt_reverse = NickaseCatalogEntry(
        id="Nt.LongReverse",
        specificity_id="LongReverse",
        motif_top_5to3="CCCC",
        top_cut_offset=0,
        vendor="dnadesign test fixture",
        source_url="https://example.invalid/dnadesign/long-reverse-fixture",
        source_family="nicking_endonuclease",
        selection={"commercial_confidence": "primary_vendor_current"},
    )
    top_valid = NickaseCatalogEntry(
        id="Nt.TopValid",
        specificity_id="TopValid",
        motif_top_5to3="AAAA",
        top_cut_offset=4,
        vendor="dnadesign test fixture",
        source_url="https://example.invalid/dnadesign/top-valid-fixture",
        source_family="nicking_endonuclease",
        selection={"commercial_confidence": "primary_vendor_current"},
    )
    bad_downstream_tail = NickaseCatalogEntry(
        id="Nb.BadTail",
        specificity_id="BadTail",
        motif_top_5to3="AAAAA",
        bottom_cut_offset=4,
        vendor="dnadesign test fixture",
        source_url="https://example.invalid/dnadesign/bad-tail-fixture",
        source_family="nicking_endonuclease",
        selection={"commercial_confidence": "primary_vendor_current"},
    )
    degenerate_downstream_tail = NickaseCatalogEntry(
        id="Nb.DegenerateTail",
        specificity_id="DegenerateTail",
        motif_top_5to3="AAAAN",
        bottom_cut_offset=4,
        vendor="dnadesign test fixture",
        source_url="https://example.invalid/dnadesign/degenerate-tail-fixture",
        source_family="nicking_endonuclease",
        selection={"commercial_confidence": "primary_vendor_current"},
    )

    short_placements = placements_for_entry(short_frequent_cutter, terminal_boundary=4, boundary=4)
    nt_placements = placements_for_entry(nt_reverse, terminal_boundary=4, boundary=4, target_strand="either")
    top_placements = placements_for_entry(top_valid, terminal_boundary=4, boundary=4, target_strand="either")
    bad_tail_placements = placements_for_entry(bad_downstream_tail, terminal_boundary=4, boundary=4)
    degenerate_tail_placements = placements_for_entry(degenerate_downstream_tail, terminal_boundary=4, boundary=4)

    assert short_placements == []
    assert [(placement.variant_id, placement.orientation, placement.strand) for placement in nt_placements] == [
        ("Nt.LongReverse", "reverse", "bottom"),
    ]
    assert [(placement.variant_id, placement.orientation, placement.strand) for placement in top_placements] == [
        ("Nt.TopValid", "forward", "top"),
    ]
    assert bad_tail_placements == []
    assert len(degenerate_tail_placements) == 1


def test_candidate_rejections_cover_terminal_pair_profile_and_mismatch_limits() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=["MMMM"],
        allow_gt_wobble=True,
        active_max_hard_mismatches=2,
        active_max_non_watson_crick_pairs=2,
    )

    non_terminal = evaluate_pair_candidate(
        left_base="AAAA",
        right_base="AAAA",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    perfect = evaluate_pair_candidate(
        left_base="AAAA",
        right_base="TTTT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    three_mismatch = evaluate_pair_candidate(
        left_base="AAAA",
        right_base="GGGT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    three_non_wc = evaluate_pair_candidate(
        left_base="GTGA",
        right_base="TTGA",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )

    assert "S0_PAIR_NOT_WATSON_CRICK" in non_terminal.rejection_reasons
    assert "REJECTED_PROFILE_BUCKET" in perfect.rejection_reasons
    assert perfect.profile_s3s2s1s0 == "MMMM"
    assert "S0_PAIR_NOT_WATSON_CRICK" in three_mismatch.rejection_reasons
    assert three_mismatch.profile_s3s2s1s0 == "MXXX"
    assert (
        "PROFILE_POLICY_RESERVE:MIDDLE_MIDDLE_DOUBLE_HARD"
        in evaluate_pair_candidate(
            left_base="AAAA",
            right_base="TGGT",
            context=context,
            s0_match_required=True,
            forbidden_release_sites=[],
        ).rejection_reasons
    )
    assert three_non_wc.profile_s3s2s1s0 == "XWWM"
    assert three_non_wc.x_count == 1
    assert three_non_wc.w_count == 2
    assert three_non_wc.non_watson_crick_count == 3
    assert "PROFILE_POLICY_RESERVE:MORE_THAN_TWO_NON_WATSON_CRICK" in three_non_wc.rejection_reasons


def test_pair_identity_uses_physical_right_base_for_profile_calls() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )

    candidate = evaluate_pair_candidate(
        left_base="AGTG",
        right_base="CATT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )

    assert candidate.profile_s3s2s1s0 == "MWMM"
    assert candidate.s2_pair_identity == "G:T"
    assert candidate.pair_classes[1].right_base == "T"
    assert candidate.pair_classes[1].aligned_right_base == "A"
    assert candidate.pair_classes[1].class_label == "W"


def test_retained_release_recognition_site_is_rejected() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )

    candidate = evaluate_pair_candidate(
        left_base="GGTC",
        right_base="TCAA",
        context=context,
        s0_match_required=False,
        forbidden_release_sites=["GGTC"],
    )

    assert "RETAINED_RELEASE_RECOGNITION_SITE" in candidate.rejection_reasons


def test_ranking_prefers_exact_nick_profile_order_low_gc_and_reference_similarity() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=["MWXM", "MXWM"],
        reject_profiles=["MMMM"],
        allow_gt_wobble=True,
        active_max_hard_mismatches=2,
        active_max_non_watson_crick_pairs=2,
        reduce_gc_when_tied=True,
    )
    candidates = [
        evaluate_pair_candidate(
            left_base="CGGT",
            right_base="AATG",
            context=context,
            s0_match_required=True,
            forbidden_release_sites=[],
            nick_distance=1,
        ),
        evaluate_pair_candidate(
            left_base="CGGT",
            right_base="AATG",
            context=context,
            s0_match_required=True,
            forbidden_release_sites=[],
            nick_distance=0,
        ),
        evaluate_pair_candidate(
            left_base="CGGT",
            right_base="ATAG",
            context=context,
            s0_match_required=True,
            forbidden_release_sites=[],
            nick_distance=0,
        ),
    ]

    ranked = rank_pair_candidates(candidates, context=context)

    assert [candidate.nick_distance for candidate in ranked[:2]] == [0, 0]
    assert ranked[0].profile_s3s2s1s0 == "MWXM"
    assert ranked[0].x_count == 1
    assert ranked[0].w_count == 1

    low_gc = evaluate_pair_candidate(
        left_base="GTGA",
        right_base="TTCC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    high_gc = evaluate_pair_candidate(
        left_base="GTGC",
        right_base="GTCC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    assert [candidate.gc_fraction for candidate in rank_pair_candidates([high_gc, low_gc], context=context)] == [
        low_gc.gc_fraction,
        high_gc.gc_fraction,
    ]


def test_ranking_prefers_lower_hard_mismatch_tier_before_gc_tie_break() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=["MXMM"],
        reject_profiles=["MMMM"],
        allow_gt_wobble=True,
        active_max_hard_mismatches=2,
        active_max_non_watson_crick_pairs=2,
        reduce_gc_when_tied=True,
    )
    lower_gc_tier3 = evaluate_pair_candidate(
        left_base="AAAA",
        right_base="TTAT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )
    higher_gc_tier2 = evaluate_pair_candidate(
        left_base="AAAA",
        right_base="TTGT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
    )

    ranked = rank_pair_candidates([lower_gc_tier3, higher_gc_tier2], context=context)

    assert ranked[0].hard_mismatch_tier_sum == 2
    assert ranked[0].gc_fraction > ranked[1].gc_fraction
    assert ranked[0].left_base == "AAAA"
    assert ranked[0].right_base == "TTGT"


def test_candidate_id_includes_enzyme_route() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )
    bsa_candidate = evaluate_pair_candidate(
        left_base="GCCC",
        right_base="TGTC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
        release_placement=_release_placement(),
        nickase_placement=_nickase_placement(),
    )
    paq_candidate = evaluate_pair_candidate(
        left_base="GCCC",
        right_base="TGTC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
        release_placement=_release_placement(
            variant_id="PaqCI",
            recognition_sequence="CACCTGC",
            recognition_site_start=-8,
        ),
        nickase_placement=_nickase_placement(),
    )

    assert bsa_candidate.left_base == paq_candidate.left_base
    assert bsa_candidate.right_base == paq_candidate.right_base
    assert bsa_candidate.profile_s3s2s1s0 == paq_candidate.profile_s3s2s1s0
    assert bsa_candidate.candidate_id != paq_candidate.candidate_id


def test_terminal_nick_visual_includes_release_site_scar_and_full_nickase_span() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )
    candidate = evaluate_pair_candidate(
        left_base="GCCC",
        right_base="TGTC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
        release_placement=_release_placement(),
        nickase_placement=_nickase_placement(),
    )

    visual = build_terminal_nick_visual_contract(
        candidate=candidate,
        solution_id="demo.candidate_01",
        state_kind="pre_post_terminal_nick",
    )

    assert visual["state_kind"] == "pre_post_terminal_nick"
    assert visual["primary_sequence"] == "GGTCTCGGCCCNNNNGGTCTCGGCCC"
    assert visual["complement_sequence"] == "CCAGAGCCGGGNNNNCCAGAGCCTGT"
    assert visual["primary_sequence"][0:6] == "GGTCTC"
    assert visual["release_site_span"] == {"start": 15, "end": 21}
    assert visual["retained_scar_span"] == {"start": 22, "end": 26}
    assert visual["terminal_boundary"] == 26
    assert visual["nick_boundary"] == 26
    assert visual["retained_product_span"] == {"start": 22, "end": 26}
    assert visual["junction_partner_span"] is None
    assert visual["nickase_site_span"] == {"start": 15, "end": 26}
    assert visual["nickase_site_source_span"] == {"start": -7, "end": 4}
    assert visual["nickase_site_span_clipped"] is False
    assert [panel["panel_id"] for panel in visual["panels"]] == ["pre_release", "post_release"]
    post_panel = visual["panels"][1]
    assert post_panel["fragment_spans"] == [{"row": "complement", "start": 15, "end": 22}]
    assert post_panel["fragment_spans"][0]["end"] == post_panel["retained_scar_span"]["start"]
    fragment_fill = next(
        fill for fill in visual["rectangular_fills"] if fill["semantic"] == "annealed_adapter_fragment"
    )
    assert fragment_fill["start"] == post_panel["fragment_spans"][0]["start"]
    assert fragment_fill["end"] == post_panel["fragment_spans"][0]["end"]
    assert fragment_fill["cover_rows"] == "complement"
    assert fragment_fill["corner_radius"] > 0
    assert visual["nickase"]["canonical_read_row"] == "primary"
    assert visual["nickase"]["recognition_nt"] == 7
    assert visual["meta"]["profile_order"] == "S3_S2_S1_S0"
    assert visual["meta"]["type_iis_label"] == "BsaI-HFv2 GGTCTC"
    assert visual["meta"]["nickase_label"] == "Test.TerminalBottomNickase GGTCTCGNNNN"
    assert visual["meta"]["junction_label"] == ""
    assert {fill["semantic"] for fill in visual["rectangular_fills"]} >= {
        "type_iis_release_site",
        "retained_type_iis_scar",
        "nickase_footprint",
    }


def test_terminal_nick_visual_keeps_pre_release_duplex_watson_crick_until_adapter_anneals() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )
    candidate = evaluate_pair_candidate(
        left_base="GCCC",
        right_base="TGTC",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
        release_placement=_release_placement(),
        nickase_placement=_nickase_placement(),
    )

    visual = build_terminal_nick_visual_contract(
        candidate=candidate,
        solution_id="demo.candidate_01",
        state_kind="pre_post_terminal_nick",
    )

    pre_panel, post_panel = visual["panels"]
    pre_slice = slice(pre_panel["start"], pre_panel["end"])
    post_scar = slice(post_panel["retained_scar_span"]["start"], post_panel["retained_scar_span"]["end"])
    pre_scar = slice(pre_panel["retained_scar_span"]["start"], pre_panel["retained_scar_span"]["end"])

    assert visual["complement_sequence"][pre_slice] == "CCAGAGCCGGG"
    assert visual["complement_sequence"][pre_scar] == "CGGG"
    assert visual["complement_sequence"][post_scar] == candidate.right_base[::-1]
    assert visual["meta"]["right_base_raw_display_order"] == candidate.right_base[::-1]
    assert visual["meta"]["aligned_right_base_display_order"] == "".join(
        pair.aligned_right_base for pair in candidate.pair_classes
    )
    assert visual["meta"]["mismatch_indices"] == [
        post_panel["retained_scar_span"]["start"] + 1,
        post_panel["retained_scar_span"]["start"] + 3,
    ]


def test_terminal_nick_visual_renders_wobble_labels_as_gt_aligned_pairs() -> None:
    context = CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=[],
        allow_gt_wobble=True,
        active_max_hard_mismatches=4,
        active_max_non_watson_crick_pairs=4,
    )
    candidate = evaluate_pair_candidate(
        left_base="GTGA",
        right_base="TCTT",
        context=context,
        s0_match_required=True,
        forbidden_release_sites=[],
        release_placement=_release_placement(),
        nickase_placement=_nickase_placement(),
    )

    visual = build_terminal_nick_visual_contract(
        candidate=candidate,
        solution_id="demo.candidate_01",
        state_kind="pre_post_terminal_nick",
    )

    post_panel = visual["panels"][1]
    post_scar = slice(post_panel["retained_scar_span"]["start"], post_panel["retained_scar_span"]["end"])

    assert candidate.profile_s3s2s1s0 == "WXMM"
    assert candidate.pair_classes[0].class_label == "W"
    assert candidate.pair_classes[0].left_base == "G"
    assert candidate.pair_classes[0].right_base == "T"
    assert visual["primary_sequence"][post_scar] == "GTGA"
    assert visual["complement_sequence"][post_scar] == "TTCT"
    assert visual["meta"]["right_base_raw_display_order"] == "TTCT"


def test_terminal_nick_view_rejects_boundary_off_retained_scar_end() -> None:
    payload = {
        "view_id": "demo.candidate_01.post_terminal_nick",
        "solution_id": "demo.candidate_01",
        "candidate_id": "candidate_01",
        "title": "candidate 01",
        "state_kind": "post_terminal_nick",
        "event_scope": "terminal_nick",
        "primary_sequence_5to3": "GCCNNNNN",
        "complement_sequence_3to5": "CGGNNNNN",
        "terminal_boundary": 3,
        "nick_boundary": 3,
        "retained_product_span": {"start": 0, "end": 4},
        "release_site_span": {"start": 0, "end": 1},
        "retained_scar_span": {"start": 0, "end": 4},
        "junction_partner_span": None,
        "nickase_site_span": {"start": 0, "end": 1},
        "nickase_site_source_span": {"start": 0, "end": 1},
        "nick_state": "nicked",
        "profile_s3s2s1s0": "MXMX",
        "profile_payload_outward": "XMXM",
    }

    with pytest.raises(ValueError, match="terminal_boundary must equal retained_scar_span.end"):
        ScarNickTerminalNickViewV1.model_validate(payload)
