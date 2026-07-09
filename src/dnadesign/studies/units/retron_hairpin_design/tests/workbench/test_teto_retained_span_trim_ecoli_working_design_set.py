"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/workbench/test_teto_retained_span_trim_ecoli_working_design_set.py

Tests for the Eco1 tetO retained-span trim workbench design set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import load_msd_compiler_spec

from ..support.paths import repo_root_from


def test_teto_retained_span_trim_ecoli_working_design_set_is_payload_family_based_and_compiler_ready() -> None:
    root = repo_root_from(__file__)
    study_dir = root / "docs" / "studies" / "retron_hairpin_design"
    design_set_path = study_dir / "workbench" / "design_sets" / "teto_retained_span_trim_ecoli_working_v1.yaml"
    spec_path = study_dir / "compiler" / "inputs" / "teto_retained_span_trim_ecoli_working_v1.spec.yaml"

    design_set = yaml.safe_load(design_set_path.read_text(encoding="utf-8"))
    directions = yaml.safe_load((study_dir / "workbench" / "ontology" / "directions.yaml").read_text(encoding="utf-8"))
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    resolved = load_msd_compiler_spec(spec_path, study_dir=study_dir)

    assert design_set["contract"] == "retron_msd_design_set_v1"
    assert design_set["design_set_id"] == "teto_retained_span_trim_ecoli_working_v1"
    assert design_set["payload_source_policy"] == "payload_family_catalog_terms_not_variant_specific_semantics"
    assert design_set["source_refs"]["payload_binding_catalog"].endswith(
        "workbench/ontology/payload_binding_sites.yaml"
    )
    assert design_set["non_goals"]["final_plasmid_number_assignment"] == "deliverable_plan_owned_not_design_set_owned"
    assert design_set["parent_payload"]["payload_family_id"] == "tetO_ecoli_working"
    assert design_set["parent_payload"]["parent_payload_id"] == "tetO_ecoli_working_w00_19"
    assert design_set["parent_payload"]["motif_occurrences"] == [
        {"motif_instance_id": "tetR:1:18:+:1", "start": 1, "end": 18, "strand": "+", "occurrence_rank": 1},
        {"motif_instance_id": "tetR:1:18:-:2", "start": 1, "end": 18, "strand": "-", "occurrence_rank": 2},
    ]
    assert set(design_set["payload_trims"]) == {
        "tetO_ecoli_working_w00_19",
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    }
    assert {trim["exact_sequence_5to3"] for trim in design_set["payload_trims"].values()} == {
        "TCCCTATCAGTGATAGAGA",
        "CCTATCAGTGATAGA",
        "CTATCAGTGATAG",
    }
    assert {design["payload_family_id"] for design in design_set["designs"]} == {"tetO_ecoli_working"}
    assert {design["source_precedent_id"] for design in design_set["designs"]} == {
        "pES-retron-26",
        "pES-retron-43",
        "pES-retron-180",
    }
    assert {design["scaffold_context"] for design in design_set["designs"]} == {"retron26", "retron43", "retron180"}
    ecoli_direction = {direction["id"]: direction for direction in directions["directions"]}[
        "teto_retained_span_trim_ecoli_working"
    ]
    assert {"retron26_control", "retron43_target", "retron180_target"} <= set(ecoli_direction["effect_tags"])
    assert {design["construct_id"] for design in design_set["designs"]} == {
        "pES-teto-r26-w02-17",
        "pES-teto-r26-w03-16",
        "pES-teto-r43-w02-17",
        "pES-teto-r43-w03-16",
        "pES-teto-r180-w02-17",
        "pES-teto-r180-w03-16",
    }
    assert spec["allow_non_ligatable_s0"] is True
    assert set(spec["payload_sequences"]) == {
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    }
    assert set(spec["payload_sequences"]) < set(design_set["payload_trims"])
    assert set(spec["cap_sequences"]) == {"C26", "C43", "C172"}
    assert len(resolved.catalog.records) == 6
    assert [record.variant_metadata.rt_mode for record in resolved.catalog.records] == ["wt_eco1"] * 6
