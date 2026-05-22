"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_rt_lnrna_sponging_construct_triage_construct_projection.py

Construct projection contract checks for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.construct_projection import (
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
        "dual_cassette_1600bp_seq_mean",
        "dual_cassette_1600bp_fwd_rc_concat",
        "lnrna_span_in_construct_anchor_mean",
    )
    assert audit.candidate_count == 2
    assert audit.candidate_spans["rt_lnrna_anchor__eco1_wt_rt__retron26_lnrna__tetO"] == {
        "lnrna": (186, 359),
        "rt_cds": (524, 1487),
    }
    assert audit.candidate_spans["rt_lnrna_anchor__eco1_wt_rt__retron43_lnrna__tetO"] == {
        "lnrna": (186, 373),
        "rt_cds": (538, 1501),
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
    assert any("required slot rt_cds ends at" in error for error in audit.errors)


def test_construct_projection_manifest_rejects_missing_reverse_complement_view() -> None:
    payload = copy.deepcopy(_manifest_payload())
    payload["representation_views"][1]["required_orientations"] = ["forward"]

    audit = validate_projection_manifest_payload(payload)

    assert not audit.ok
    assert (
        "dual_cassette_1600bp_fwd_rc_concat: required_orientations must be forward, reverse_complement"
    ) in audit.errors
