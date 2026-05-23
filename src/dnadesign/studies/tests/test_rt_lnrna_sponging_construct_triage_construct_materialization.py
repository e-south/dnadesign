"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_rt_lnrna_sponging_construct_triage_construct_materialization.py

Construct materialization checks for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from Bio.Seq import Seq

from dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.construct_materialization import (
    MaterializationContractError,
    materialize_control_construct_contexts,
)
from dnadesign.usr import Dataset, load_sequence_views


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_rt_lnrna_controls_materialize_real_1600bp_construct_context_views(tmp_path: Path) -> None:
    report = materialize_control_construct_contexts(repo_root=_repo_root(), work_root=tmp_path)

    assert report.input_dataset == "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
    assert report.output_dataset == "rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1"
    output = Dataset(report.usr_root, report.output_dataset).head(n=10)
    assert {result.records_total for result in report.run_results} == {4, 8}
    assert output.shape[0] == 4

    expected_spans = {
        "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO": {
            "window_start": 56,
            "lnrna": (130, 303),
            "rt_cds": (468, 1431),
        },
        "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO": {
            "window_start": 63,
            "lnrna": (123, 310),
            "rt_cds": (475, 1438),
        },
    }
    for candidate_id, expected in expected_spans.items():
        input_id = report.input_ids_by_candidate_id[candidate_id]
        forward = output[
            (output["construct__input_id"] == input_id) & (output["construct__orientation"] == "forward")
        ].iloc[0]
        reverse = output[
            (output["construct__input_id"] == input_id) & (output["construct__orientation"] == "reverse_complement")
        ].iloc[0]
        assert forward["usr_label__primary"] == f"{candidate_id}_realized_context_forward"
        assert reverse["usr_label__primary"] == f"{candidate_id}_realized_context_reverse_complement"
        assert len(forward["sequence"]) == 1600
        assert reverse["sequence"] == str(Seq(forward["sequence"]).reverse_complement())
        assert forward["sequence"] == report.expected_sequences[candidate_id]
        assert forward["construct__window_start"] == expected["window_start"]
        assert {
            slot["slot_id"]: (slot["start"], slot["end"])
            for slot in forward["construct__slots"]
            if slot["slot_id"] in {"lnrna", "rt_cds"}
        } == {slot_id: span for slot_id, span in expected.items() if slot_id in {"lnrna", "rt_cds"}}

        lnrna_start, lnrna_end = expected["lnrna"]
        rt_start, rt_end = expected["rt_cds"]
        window_start = expected["window_start"]
        assert forward["sequence"][:lnrna_start] == report.template_sequence[window_start:186]
        assert forward["sequence"][lnrna_end:rt_start] == report.template_sequence[359:524]
        suffix_length = 1600 - rt_end
        assert forward["sequence"][rt_end:] == report.template_sequence[1487 : 1487 + suffix_length]

    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 12
    assert {view.context_kind for view in views} == {"template_custom"}
    views_by_name = {}
    for view in views:
        views_by_name.setdefault(view.view_name, []).append(view)
    assert sorted(views_by_name) == [
        "dual_cassette_1600bp_fwd_rc_concat",
        "dual_cassette_1600bp_seq_mean",
        "lnrna_span_in_construct_anchor_mean",
        "lnrna_span_in_construct_reverse_complement_anchor_mean",
        "rt_cds_span_in_construct_anchor_mean",
        "rt_cds_span_in_construct_reverse_complement_anchor_mean",
    ]
    assert all(len(view_rows) == 2 for view_rows in views_by_name.values())
    assert {view.orientation for view in views_by_name["dual_cassette_1600bp_seq_mean"]} == {"forward"}
    assert {view.orientation for view in views_by_name["dual_cassette_1600bp_fwd_rc_concat"]} == {"reverse_complement"}
    assert {
        (view.anchor_start_0, view.anchor_end_0) for view in views_by_name["lnrna_span_in_construct_anchor_mean"]
    } == {(130, 303), (123, 310)}
    assert {view.recommended_pooling for view in views_by_name["lnrna_span_in_construct_anchor_mean"]} == {
        "anchor_mean"
    }
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["lnrna_span_in_construct_reverse_complement_anchor_mean"]
    } == {(1290, 1477), (1297, 1470)}
    assert {
        (view.anchor_start_0, view.anchor_end_0) for view in views_by_name["rt_cds_span_in_construct_anchor_mean"]
    } == {(468, 1431), (475, 1438)}
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["rt_cds_span_in_construct_reverse_complement_anchor_mean"]
    } == {(162, 1125), (169, 1132)}


def test_rt_lnrna_materialization_rejects_swapped_candidate_slot_sequences(tmp_path: Path) -> None:
    with pytest.raises(MaterializationContractError, match="candidate__lnrna_sequence length"):
        materialize_control_construct_contexts(
            repo_root=_repo_root(),
            work_root=tmp_path,
            candidate_sequence_overrides={
                "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO": {
                    "candidate__lnrna_sequence": "A" * 963,
                    "candidate__rt_cds_sequence": "C" * 173,
                }
            },
        )


def test_rt_lnrna_materialization_fails_fast_when_rt_cds_field_is_missing(tmp_path: Path) -> None:
    with pytest.raises(MaterializationContractError, match="candidate__rt_cds_sequence"):
        materialize_control_construct_contexts(
            repo_root=_repo_root(),
            work_root=tmp_path,
            omitted_candidate_fields=("candidate__rt_cds_sequence",),
        )
