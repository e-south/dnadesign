"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_rt_lnrna_sponging_construct_triage_representation_contract.py

Representation handoff checks for the RT-lnRNA sponging construct triage study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.representation_contract import (
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


def test_rt_lnrna_infer_feature_bundle_fixture_selects_every_view_by_explicit_view_name() -> None:
    audit = validate_infer_feature_bundle_payload(_infer_fixture_payload())

    assert audit.ok, "\n".join(audit.errors)
    assert audit.selected_view_names == REQUIRED_SOURCE_VIEW_NAMES


def test_rt_lnrna_infer_feature_bundle_rejects_product_kind_orientation_only_selectors() -> None:
    payload = {
        "kind": "evo2_sequence_feature_v1",
        "intermediate_block": 26,
        "collect_log_likelihood": True,
        "collect_output_layer_mean": True,
        "collect_intermediate_embedding": True,
        "sequence_view_inputs": [
            {
                "dataset": "rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1",
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
