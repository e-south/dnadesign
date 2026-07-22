"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_projection.py

Construct projection contract checks for the RT-lnRNA sponging construct triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_projection import (
    validate_projection_manifest_payload,
    validate_registered_projection_manifest,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _manifest_payload() -> dict[str, object]:
    path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/fixtures/construct/"
        "construct-projection-manifest.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_construct_projection_manifest_uses_public_multi_slot_construct_strategy() -> None:
    audit = validate_registered_projection_manifest(repo_root=_repo_root())

    assert audit.ok, "\n".join(audit.errors)
    assert audit.strategy_id == "construct_multi_slot_assembly_v1"
    assert audit.construct_template_id == "dual_cassette_rt_lnrna_expression_template_v1"
    assert audit.required_view_names == (
        "dual_cassette_2000bp_seq_mean",
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "lnrna_fixed_384bp_window_in_construct_anchor_mean",
        "lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean",
    )
    assert audit.construct_subject_count == 2
    assert audit.construct_subject_spans["rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"] == {
        "lnrna": (130, 303),
        "rt_cds": (468, 1431),
    }
    assert audit.construct_subject_spans["rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO"] == {
        "lnrna": (123, 310),
        "rt_cds": (475, 1438),
    }


def test_construct_projection_manifest_rejects_precomposed_single_anchor_strategy() -> None:
    audit = validate_projection_manifest_payload(
        {
            "manifest_id": "broken",
            "strategy": {"strategy_id": "precomposed_single_anchor_v1"},
            "construct_contract": "dual_cassette_rt_lnrna_expression_v1",
            "representation_contract": "dual_cassette_construct_context_embedding_v1",
            "construct_template": {},
            "slots": [],
            "representation_views": [],
            "candidates": [],
        }
    )

    assert not audit.ok
    assert "strategy.strategy_id must be construct_multi_slot_assembly_v1" in audit.errors


def test_construct_projection_manifest_rejects_missing_plasmid_context_source() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["construct_template"]["plasmid_context_source_id"] = ""

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert "construct_template.plasmid_context_source_id is required" in audit.errors


def test_construct_projection_manifest_rejects_oversized_required_slot() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["candidates"][0]["slot_bindings"]["rt_cds"]["sequence_length_nt"] = 2000
    payload["candidates"][0]["slot_bindings"]["rt_cds"]["source_sequence_span_0"] = [0, 2000]

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert any("required slot rt_cds resolves to" in error for error in audit.errors)


def test_construct_projection_manifest_rejects_missing_reverse_complement_view() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["representation_views"][1]["orientation"] = "forward"

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert "dual_cassette_2000bp_reverse_complement_seq_mean: orientation must be reverse_complement" in audit.errors


def test_construct_projection_manifest_rejects_pre_infer_concat_transform() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["representation_views"][1]["downstream_transform"] = "block_normalized_concatenate"

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert (
        "dual_cassette_2000bp_reverse_complement_seq_mean: downstream_transform must be empty; concat is post-inference"
    ) in audit.errors


def test_construct_projection_manifest_rejects_missing_lnrna_anchor_part_mapping() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["representation_views"][2]["construct_output_anchor_part"] = ""

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert (
        "lnrna_fixed_384bp_window_in_construct_anchor_mean: construct_output_anchor_part must be lnrna" in audit.errors
    )


def test_construct_projection_manifest_rejects_missing_rt_cds_anchor_part_mapping() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["representation_views"][4]["construct_output_anchor_part"] = ""

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert (
        "rt_cds_fixed_1600bp_window_in_construct_anchor_mean: construct_output_anchor_part must be rt_cds"
        in audit.errors
    )


def test_construct_projection_manifest_reports_missing_lnrna_slot_without_crashing() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["slots"] = [slot for slot in payload["slots"] if slot["slot_id"] != "lnrna"]

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert "slots must declare required lnrna and rt_cds slots in construct order" in audit.errors


def test_construct_projection_manifest_reports_missing_lnrna_binding_without_crashing() -> None:
    payload = copy.deepcopy(_manifest_payload())
    del payload["candidates"][0]["slot_bindings"]["lnrna"]

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert any(".slot_bindings.lnrna must be a mapping" in error for error in audit.errors)


def test_construct_projection_manifest_rejects_swapped_slot_sequence_fields() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["slots"][0]["sequence_field"] = "construct_subject__rt_cds_sequence"
    payload["slots"][1]["sequence_field"] = "construct_subject__lnrna_sequence"

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert (
        "slot lnrna must use role=lnrna_cassette and sequence_field=construct_subject__lnrna_sequence" in audit.errors
    )
    assert "slot rt_cds must use role=rt_cds and sequence_field=construct_subject__rt_cds_sequence" in audit.errors
