"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/workbench/test_teto_retained_span_trim_tetr_pwm_elite_design_set.py

Tests for the tetO PWM trim study-owned design set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import load_msd_compiler_spec

from ..support.paths import repo_root_from


def test_teto_retained_span_trim_tetr_pwm_elite_design_set_is_study_owned_and_compiler_ready() -> None:
    root = repo_root_from(__file__)
    study_dir = root / "docs" / "studies" / "retron_hairpin_design"
    design_set_path = study_dir / "workbench" / "design_sets" / "teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
    spec_path = study_dir / "compiler" / "inputs" / "teto_retained_span_trim_tetr_pwm_elite_v1.spec.yaml"
    directions = yaml.safe_load((study_dir / "workbench" / "ontology" / "directions.yaml").read_text(encoding="utf-8"))
    design_set = yaml.safe_load(design_set_path.read_text(encoding="utf-8"))
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    resolved = load_msd_compiler_spec(spec_path, study_dir=study_dir)

    assert design_set["contract"] == "retron_msd_design_set_v1"
    assert design_set["design_set_id"] == "teto_retained_span_trim_tetr_pwm_elite_v1"
    assert design_set["authority"] == "persistent study workbench design cohort"
    assert design_set["expected_variant_count"] == 9
    assert design_set["compiler_spec_ref"] == (
        "docs/studies/retron_hairpin_design/compiler/inputs/teto_retained_span_trim_tetr_pwm_elite_v1.spec.yaml"
    )
    assert design_set["non_goals"]["rt_fusions"] == "outside_msd_design_scope"
    assert "decision_logic" not in design_set
    assert "outcome_bins" not in design_set
    assert design_set["assay_handoff"] == {
        "contract": "retron_msd_assay_handoff_ref_v1",
        "owner_study_id": "rt_lnrna_sponging_construct_triage",
        "route_ref": ("docs/studies/rt_lnrna_sponging_construct_triage/routes/reporter-response-evidence.md"),
        "subject_set_id": "teto_retained_span_trim_tetr_pwm_elite_v1",
        "subject_identity_field": "construct_id",
        "handoff_role": "subject_identity_only",
    }
    assert set(design_set["payload_trims"]) == {"TetR_w00_19", "TetR_w02_17", "TetR_w03_16"}
    assert {trim["payload_trim_id"] for trim in design_set["payload_trims"].values()} == set(spec["payload_sequences"])
    assert {trim["exact_sequence_5to3"] for trim in design_set["payload_trims"].values()} == {
        entry["sequence"] for entry in spec["payload_sequences"].values()
    }
    assert design_set["source_refs"]["tetr_monotypic_yiu"].endswith("configs/yiu/tetr_monotypic_hit.yiu.yaml")
    assert design_set["parent_payload"]["parent_payload_id"] == "TetR_w00_19"
    assert design_set["parent_payload"]["source_sequence_5to3"] == "CTCTATATCTGATATAGAG"
    assert design_set["parent_payload"]["motif_occurrences"] == [
        {"motif_instance_id": "tetR:0:17:+:1", "start": 0, "end": 17, "strand": "+", "occurrence_rank": 1},
        {"motif_instance_id": "tetR:2:19:-:2", "start": 2, "end": 19, "strand": "-", "occurrence_rank": 2},
    ]
    assert design_set["retron180_context_selection"] == {
        "selected_precedent_construct_id": "pES-retron-180",
        "selected_precedent_label": "pES-retron-180-msd[TetR]; C172-LAGTG-RCATG-XWMM",
        "source_design_set_ref": (
            "docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
        ),
        "selection_basis": "Retain the existing pES-retron-180 MSD precedent as one bounded scaffold context.",
        "cap_id": "C172",
        "left_base": "AGTG",
        "right_base": "CATG",
        "profile_s3s2s1s0": "XWMM",
        "nick_orientation": "bottom",
        "nickase": "Nb.BtsI",
    }
    assert design_set["payload_trims"]["TetR_w00_19"]["exact_sequence_5to3"] == "CTCTATATCTGATATAGAG"
    assert design_set["payload_trims"]["TetR_w02_17"]["exact_sequence_5to3"] == "CTATATCTGATATAG"
    assert design_set["payload_trims"]["TetR_w02_17"]["sequence_length_nt"] == 15
    assert design_set["payload_trims"]["TetR_w02_17"]["retained_parent_span_0"] == {"start": 2, "end": 17}
    assert design_set["payload_trims"]["TetR_w02_17"]["retained_information_fraction"] == 0.964248
    assert design_set["payload_trims"]["TetR_w02_17"]["selection_rule"] == {
        "algorithm": "dual_site_sliding_window_max_ic",
        "requested_length_nt": 15,
        "tie_breaker": "closest_to_parent_center",
    }
    assert design_set["payload_trims"]["TetR_w03_16"]["exact_sequence_5to3"] == "TATATCTGATATA"
    assert design_set["payload_trims"]["TetR_w03_16"]["sequence_length_nt"] == 13
    assert design_set["payload_trims"]["TetR_w03_16"]["retained_parent_span_0"] == {"start": 3, "end": 16}
    assert design_set["payload_trims"]["TetR_w03_16"]["retained_information_fraction"] == 0.915756

    direction_ids = {direction["id"] for direction in directions["directions"]}
    effect_tags = set(directions["effect_tags"])
    known_construct_ids = {design["construct_id"] for design in design_set["designs"]}
    assert known_construct_ids == {
        "pES-tetr-r26-w00-19",
        "pES-tetr-r26-w02-17",
        "pES-tetr-r26-w03-16",
        "pES-tetr-r43-w00-19",
        "pES-tetr-r43-w02-17",
        "pES-tetr-r43-w03-16",
        "pES-tetr-r180-w00-19",
        "pES-tetr-r180-w02-17",
        "pES-tetr-r180-w03-16",
    }
    assert design_set["label_count"] == len(design_set["designs"]) == len(spec["designs"])
    assert design_set["label_count"] == len(resolved.catalog.records)
    assert {design["construct_id"] for design in spec["designs"]} == known_construct_ids
    assert [record.variant_metadata.rt_mode for record in resolved.catalog.records] == ["wt_eco1"] * 9

    for design in design_set["designs"]:
        assert design["direction_ids"]
        assert design["effect_tags"]
        assert set(design["direction_ids"]) <= direction_ids
        assert set(design["effect_tags"]) <= effect_tags
        assert design["payload_trim_id"] in design_set["payload_trims"]
        assert design["rt_mode"] == "wt_eco1"
        assert design["variant_role"] in {"control", "scaffold_target", "trim_candidate"}

    assert {design["scaffold_context"] for design in design_set["designs"]} == {"retron26", "retron43", "retron180"}

    assert spec["allow_non_ligatable_s0"] is True
    assert all("source" not in payload for payload in spec["payload_sequences"].values())
    assert set(spec["cap_sequences"]) == {"C26", "C43", "C172"}
    r180_designs = [design for design in design_set["designs"] if design["construct_id"].startswith("pES-tetr-r180")]
    assert len(r180_designs) == 3
    assert {design["scaffold_context"] for design in r180_designs} == {"retron180"}
    assert {tag for design in r180_designs for tag in design["effect_tags"] if tag == "retron180_target"} == {
        "retron180_target"
    }
    assert {design["right_base"] for design in r180_designs} == {"CATG"}
    assert {design["profile_s3s2s1s0"] for design in r180_designs} == {"XWMM"}
    assert {design["source_precedent_id"] for design in r180_designs} == {"pES-retron-180"}
