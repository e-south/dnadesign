"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/infer/test_representation_contract.py

Representation handoff checks for the RT-lnRNA sponging construct triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.permuter import read_infer_feature_request_manifest
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.representation_contract import (
    REQUIRED_SOURCE_VIEW_NAMES,
    validate_infer_feature_bundle_payload,
    validate_registered_representation_table_contract,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _infer_fixture_payload() -> dict[str, object]:
    path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/fixtures/infer/"
        "evo2-7b-six-view-feature-bundle.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _permuter_plan_path() -> Path:
    return (
        _repo_root()
        / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/fixtures/permuter/rt-cds-dms-plan.yaml"
    )


def _permuter_infer_handoff_path() -> Path:
    return (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/fixtures/permuter/"
        "rt-cds-dms-infer-handoff.yaml"
    )


def _pipeline_path() -> Path:
    return (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/operations/runtime/command-groups/pipeline.yaml"
    )


def _permuter_plan_payload() -> dict[str, object]:
    payload = yaml.safe_load(_permuter_plan_path().read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_representation_table_contract_declares_fixed_size_gallery_and_overlay_inputs() -> None:
    audit = validate_registered_representation_table_contract(repo_root=_repo_root())

    assert audit.ok, "\n".join(audit.errors)
    assert audit.source_view_names == REQUIRED_SOURCE_VIEW_NAMES
    assert audit.overlay_reference_ids == (
        "khan_cross_retron_rt_dna_abundance_v1",
        "crawford_eco1_lnrna_msd_abundance_v1",
        "crawford_eco1_lnrna_msd_design_reference_v1",
    )
    assert audit.fixed_size_vectors["intermediate_embedding_7b_lnrna_span_in_construct_anchor_mean_bidir_concat"] == (
        "float32",
        8192,
    )
    assert audit.fixed_size_vectors["intermediate_embedding_7b_rt_cds_span_in_construct_anchor_mean_bidir_concat"] == (
        "float32",
        8192,
    )
    assert audit.fixed_size_vectors["intermediate_embedding_7b_lnrna_rt_slot_pair_anchor_mean_concat"] == (
        "float32",
        16384,
    )
    assert audit.row_key_source == {
        "dataset": "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1",
        "namespace": "construct_subject",
        "column": "construct_subject__id",
        "materialized_by": "construct_output_subject_bridge",
        "output_join_field": "construct__input_id",
        "input_dataset": "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1",
        "input_construct_subject_field": "construct_subject__id",
    }
    assert audit.construct_subject_promotion == {
        "source_dataset": "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1",
        "consolidated_construct_dataset": "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1",
        "required_sequence_fields": ("construct_subject__lnrna_sequence", "construct_subject__rt_cds_sequence"),
        "required_construct_views": REQUIRED_SOURCE_VIEW_NAMES,
    }


def test_rt_lnrna_infer_feature_bundle_fixture_selects_every_view_by_explicit_view_name() -> None:
    audit = validate_infer_feature_bundle_payload(_infer_fixture_payload())

    assert audit.ok, "\n".join(audit.errors)
    assert audit.selected_view_names == REQUIRED_SOURCE_VIEW_NAMES


def test_rt_lnrna_permuter_plan_fixture_links_construct_subject_envelope_to_six_view_infer_handoff() -> None:
    payload = _permuter_plan_payload()
    construct_subject_envelope = payload["construct_subject_envelope"]
    infer_handoff = payload["infer_handoff"]
    feature_bundle_path = (_permuter_plan_path().parent / str(infer_handoff["feature_bundle_ref"])).resolve()
    feature_bundle_payload = yaml.safe_load(feature_bundle_path.read_text(encoding="utf-8"))
    assert isinstance(feature_bundle_payload, dict)

    assert payload["kind"] == "permuter_rt_cds_dms_plan_v1"
    assert payload["owner_boundary"] == "study"
    assert payload["variant_owner"] == "permuter"
    assert payload["construct_owner"] == "construct"
    assert payload["infer_owner"] == "infer"
    assert construct_subject_envelope == {
        "record_kind": "construct_subject_envelope",
        "sequence_authority": "overlay_only",
        "biological_sequence_fields": ["construct_subject__lnrna_sequence", "construct_subject__rt_cds_sequence"],
        "semantic_identity": "construct_subject__id",
        "usr_row_identity": "canonical_sequence_id",
    }
    assert infer_handoff["source_owner"] == "construct"
    assert tuple(infer_handoff["required_view_names"]) == REQUIRED_SOURCE_VIEW_NAMES
    audit = validate_infer_feature_bundle_payload(feature_bundle_payload)
    assert audit.ok, "\n".join(audit.errors)
    assert audit.selected_view_names == REQUIRED_SOURCE_VIEW_NAMES


def test_rt_lnrna_permuter_infer_handoff_fixture_uses_public_manifest_contract() -> None:
    request = read_infer_feature_request_manifest(_permuter_infer_handoff_path())

    assert request.source_owner == "construct"
    assert request.source_dataset.dataset_id == "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1"
    assert request.execution_owner == "infer"
    assert request.writeback_owner == "infer"
    assert tuple(selector.view_name for selector in request.sequence_view_selectors) == REQUIRED_SOURCE_VIEW_NAMES
    assert request.requested_outputs == ("log_likelihood", "output_layer_mean", "intermediate_embedding")


def test_rt_lnrna_infer_feature_bundle_rejects_product_kind_orientation_only_selectors() -> None:
    payload = {
        "kind": "evo2_sequence_feature_v1",
        "intermediate_block": 26,
        "collect_log_likelihood": True,
        "collect_output_layer_mean": True,
        "collect_intermediate_embedding": True,
        "sequence_view_inputs": [
            {
                "dataset": "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1",
                "view_selector": {
                    "product_kind": "realized_context",
                    "orientation": "forward",
                },
                "pooling": {
                    "operation": "seq_mean",
                },
            }
        ],
    }

    audit = validate_infer_feature_bundle_payload(payload)

    assert not audit.ok
    assert any("must select by explicit view_name" in error for error in audit.errors)


def test_rt_lnrna_infer_inventory_command_is_a_hard_missing_sidecar_gate() -> None:
    payload = yaml.safe_load(_pipeline_path().read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    groups = {group["id"]: group for group in payload["command_groups"]}
    commands = "\n".join(groups["infer_handoff"]["commands"])

    assert "--max-missing-products 0" in commands
    assert "--max-missing-vectors 0" in commands
    assert "--max-missing-scalars 0" in commands
    assert "--max-stale-vectors 0" in commands
    assert "--max-stale-scalars 0" in commands
