"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_materialization.py

Construct materialization checks for the RT-lnRNA sponging construct triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from Bio.Seq import Seq

from dnadesign.permuter import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    default_codon_table_path,
    generate_variants,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    MaterializationContractError,
    materialize_control_construct_contexts,
    materialize_rt_cds_dms_construct_contexts,
    materialize_unified_construct_subject_contexts,
    materialize_variant_construct_contexts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.infer_readiness import (
    REQUIRED_INFER_READY_VIEW_NAMES,
    validate_construct_infer_readiness,
)
from dnadesign.usr import Dataset, load_sequence_views, sequence_views_path

from .helpers import (
    _assert_construct_output_subject_bridge,
    _assert_construct_subject_envelope_inputs,
    _assert_usr_contracts_strictly_validate,
    _repo_root,
    _write_msd_compiler_pool_spec,
)
from .source_fixtures import _fixture_source_record_resolver, _write_source_promotion_fixture

GENBANK_CONSTRUCT_SUBJECT_COUNT = 46
RT_CDS_DMS_CONSTRUCT_SUBJECT_COUNT = 19
CRAWFORD_FIXTURE_CONSTRUCT_SUBJECT_COUNT = 2
KHAN_FIXTURE_CONSTRUCT_SUBJECT_COUNT = 0
MSD_COMPILER_CONSTRUCT_SUBJECT_COUNT = 80
CONTEXT_ROWS_PER_SUBJECT, SEQUENCE_VIEWS_PER_SUBJECT = 2, 6


def test_rt_lnrna_materialization_source_is_split_by_contract_domain() -> None:
    unit_root = _repo_root() / "src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage"
    max_lines_by_module = {
        "construct_materialization.py": 360,
        "materialization/subjects.py": 650,
        "materialization/usr_io.py": 350,
        "materialization/views.py": 350,
        "materialization/unified.py": 420,
        "materialization/execution.py": 220,
        "materialization/contracts.py": 150,
        "materialization/manifest.py": 180,
        "materialization/common.py": 80,
        "tests/construct/helpers.py": 340,
        "tests/construct/source_fixtures.py": 220,
        "tests/construct/test_materialization.py": 420,
        "tests/construct/test_msd_compiler.py": 260,
        "tests/construct/test_msd_compiler_primitives.py": 120,
        "tests/construct/test_projection.py": 220,
        "tests/construct/test_source_promotions.py": 340,
    }

    for module_name, max_lines in max_lines_by_module.items():
        line_count = len((unit_root / module_name).read_text(encoding="utf-8").splitlines())
        assert line_count <= max_lines, f"{module_name} has {line_count} lines; expected <= {max_lines}"


def test_rt_lnrna_controls_materialize_real_2000bp_construct_context_views(tmp_path: Path) -> None:
    report = materialize_control_construct_contexts(repo_root=_repo_root(), work_root=tmp_path)

    assert report.input_dataset == "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
    assert report.output_dataset == "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1"
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)
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
        input_id = report.input_ids_by_subject_id[candidate_id]
        forward = output[
            (output["construct__input_id"] == input_id) & (output["construct__orientation"] == "forward")
        ].iloc[0]
        reverse = output[
            (output["construct__input_id"] == input_id) & (output["construct__orientation"] == "reverse_complement")
        ].iloc[0]
        assert forward["usr_label__primary"] == f"{candidate_id}_realized_context_forward"
        assert reverse["usr_label__primary"] == f"{candidate_id}_realized_context_reverse_complement"
        assert len(forward["sequence"]) == 2000
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
        suffix_length = 2000 - rt_end
        assert forward["sequence"][rt_end:] == report.template_sequence[1487 : 1487 + suffix_length]

    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 12
    assert {view.context_kind for view in views} == {"template_custom"}
    views_by_name = {}
    for view in views:
        views_by_name.setdefault(view.view_name, []).append(view)
    assert sorted(views_by_name) == [
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "dual_cassette_2000bp_seq_mean",
        "lnrna_fixed_384bp_window_in_construct_anchor_mean",
        "lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean",
    ]
    assert all(len(view_rows) == 2 for view_rows in views_by_name.values())
    assert {view.orientation for view in views_by_name["dual_cassette_2000bp_seq_mean"]} == {"forward"}
    assert {view.orientation for view in views_by_name["dual_cassette_2000bp_reverse_complement_seq_mean"]} == {
        "reverse_complement"
    }
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["lnrna_fixed_384bp_window_in_construct_anchor_mean"]
    } == {(24, 408)}
    assert {
        view.recommended_pooling for view in views_by_name["lnrna_fixed_384bp_window_in_construct_anchor_mean"]
    } == {"anchor_mean"}
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean"]
    } == {(1591, 1975)}
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["rt_cds_fixed_1600bp_window_in_construct_anchor_mean"]
    } == {(149, 1749), (156, 1756)}
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean"]
    } == {(243, 1843), (250, 1850)}


def test_rt_lnrna_catalog_variants_materialize_consolidated_construct_views(tmp_path: Path) -> None:
    report = materialize_variant_construct_contexts(repo_root=_repo_root(), work_root=tmp_path)

    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)
    assert len(report.input_ids_by_subject_id) == GENBANK_CONSTRUCT_SUBJECT_COUNT
    output = Dataset(report.usr_root, report.output_dataset).head(n=100)
    assert output.shape[0] == GENBANK_CONSTRUCT_SUBJECT_COUNT * CONTEXT_ROWS_PER_SUBJECT

    candidate_id = "rt_lnrna_pair__retron47_rt_fusion__retron47_lnrna__tetO"
    input_id = report.input_ids_by_subject_id[candidate_id]
    forward = output[
        (output["construct__input_id"] == input_id) & (output["construct__orientation"] == "forward")
    ].iloc[0]
    assert len(forward["sequence"]) == 2000
    assert forward["construct__window_start"] == 159
    assert {
        slot["slot_id"]: (slot["start"], slot["end"])
        for slot in forward["construct__slots"]
        if slot["slot_id"] in {"lnrna", "rt_cds"}
    } == {
        "lnrna": (27, 200),
        "rt_cds": (365, 1535),
    }

    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == GENBANK_CONSTRUCT_SUBJECT_COUNT * SEQUENCE_VIEWS_PER_SUBJECT
    assert {view.view_name for view in views if view.parent_sequence_id == input_id and view.view_name is not None} == {
        "dual_cassette_2000bp_seq_mean",
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "lnrna_fixed_384bp_window_in_construct_anchor_mean",
        "lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean",
    }


def test_rt_lnrna_rt_cds_dms_variants_materialize_through_permuter_public_api(tmp_path: Path) -> None:
    report = materialize_rt_cds_dms_construct_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path,
        base_construct_subject_id="rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
        rt_cds_positions=(1,),
    )

    assert report.input_dataset == "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
    assert report.output_dataset == "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1"
    assert report.permuter_request_id
    assert len(report.input_ids_by_subject_id) == 19
    assert all("__rt_cds_dms__" in candidate_id for candidate_id in report.input_ids_by_subject_id)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=25)
    assert set(inputs["construct_subject__dms_slot"]) == {"rt_cds"}
    assert set(inputs["construct_subject__study_id"]) == {"rt_lnrna_sponging_construct_triage"}
    assert set(inputs["construct_subject__construct_contract"]) == {"dual_cassette_rt_lnrna_expression_v1"}
    assert set(inputs["construct_subject__representation_contract"]) == {"dual_cassette_construct_context_embedding_v1"}
    assert set(inputs["construct_subject__payload_program_id"]) == {"tetO_sponging_v1"}
    assert set(inputs["construct_subject__source_basis"]) == {"in_silico_rt_cds_dms"}
    _assert_construct_subject_envelope_inputs(report)
    assert set(inputs["construct_subject__variant_derivation"]) == {"rt_cds_dms_top_codon_policy_v1"}
    assert set(inputs["construct_subject__construct_projection_status"]) == {"representable"}
    assert set(inputs["construct_subject__role"]) == {"in_silico_rt_cds_dms_variant"}
    assert set(inputs["construct_subject__parent_id"]) == {"rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"}
    assert set(inputs["construct_subject__permuter_request_id"]) == {report.permuter_request_id}
    assert set(inputs["construct_subject__rt_cds_dms_aa_pos"]) == {1}
    assert inputs["construct_subject__lnrna_sequence"].nunique() == 1
    assert inputs["construct_subject__rt_cds_sequence"].nunique() == 19
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    output = Dataset(report.usr_root, report.output_dataset).head(n=100)
    assert output.shape[0] == 38
    assert {
        slot["slot_id"]
        for slots in output["construct__slots"]
        for slot in slots
        if slot["slot_id"] in {"lnrna", "rt_cds"}
    } == {"lnrna", "rt_cds"}


def test_rt_lnrna_rt_cds_dms_public_api_is_exhaustive_without_stop_intrusions(tmp_path: Path) -> None:
    control_report = materialize_control_construct_contexts(repo_root=_repo_root(), work_root=tmp_path / "controls")
    inputs = Dataset(control_report.usr_root, control_report.input_dataset).head(n=10)
    parent = inputs[inputs["construct_subject__id"] == "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"].iloc[0]
    rt_cds_sequence = str(parent["construct_subject__rt_cds_sequence"])

    result = generate_variants(
        CodingDnaDmsRequest(
            ref_name="rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO__rt_cds",
            sequence=rt_cds_sequence,
            codon_table=default_codon_table_path("ecoli"),
        )
    )

    stop_codons = {"TAA", "TAG", "TGA"}
    codons = [rt_cds_sequence[index : index + 3].upper() for index in range(0, len(rt_cds_sequence), 3)]
    assert codons[-1] in stop_codons
    assert not any(codon in stop_codons for codon in codons[:-1])
    assert len(result.records) == (len(codons) - 1) * 19
    assert result.metadata["permuter"]["positions"] == tuple(range(1, len(codons)))
    assert result.metadata["permuter"]["excluded_codon_positions"] == (len(codons),)

    variants_by_position: dict[int, set[str]] = {}
    for record in result.records:
        meta = CodingDnaDmsVariantMetadata.from_record(record)
        variants_by_position.setdefault(meta.aa_pos, set()).add(meta.aa_alt)
        assert len(record.sequence) == len(rt_cds_sequence)
        assert meta.aa_alt != "*"
        assert meta.codon_new not in stop_codons
        variant_codons = [record.sequence[index : index + 3].upper() for index in range(0, len(record.sequence), 3)]
        assert not any(codon in stop_codons for codon in variant_codons[:-1])
        assert variant_codons[-1] == codons[-1]

    assert set(variants_by_position) == set(range(1, len(codons)))
    assert {len(alternates) for alternates in variants_by_position.values()} == {19}


def test_rt_lnrna_unified_construct_subjects_materialize_genbank_and_rt_dms(tmp_path: Path) -> None:
    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path,
        include_source_promotions=False,
        include_msd_compiler_promotions=False,
        rt_cds_positions=(1,),
    )

    assert report.genbank_construct_subject_count == GENBANK_CONSTRUCT_SUBJECT_COUNT
    assert report.rt_cds_dms_construct_subject_count == RT_CDS_DMS_CONSTRUCT_SUBJECT_COUNT
    expected_subjects = GENBANK_CONSTRUCT_SUBJECT_COUNT + RT_CDS_DMS_CONSTRUCT_SUBJECT_COUNT
    assert len(report.input_ids_by_subject_id) == expected_subjects
    assert report.permuter_request_id
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=70)
    assert set(inputs["construct_subject__source_basis"]) == {"genbank_variant_catalog", "in_silico_rt_cds_dms"}
    assert set(
        inputs.loc[inputs["construct_subject__source_basis"] == "genbank_variant_catalog"][
            "construct_subject__source_collection_id"
        ]
    ) == {"rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1"}
    assert set(
        inputs.loc[inputs["construct_subject__source_basis"] == "in_silico_rt_cds_dms"]["construct_subject__role"]
    ) == {"in_silico_rt_cds_dms_variant"}

    output = Dataset(report.usr_root, report.output_dataset).head(n=200)
    assert output.shape[0] == expected_subjects * CONTEXT_ROWS_PER_SUBJECT
    assert set(output["construct_subject__source_basis"]) == {"genbank_variant_catalog", "in_silico_rt_cds_dms"}
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == expected_subjects * SEQUENCE_VIEWS_PER_SUBJECT
    assert {view.view_name for view in views if view.view_name is not None} == {
        "dual_cassette_2000bp_seq_mean",
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "lnrna_fixed_384bp_window_in_construct_anchor_mean",
        "lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_anchor_mean",
        "rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean",
    }


def test_rt_lnrna_unified_construct_subjects_are_infer_ready_across_source_families(tmp_path: Path) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)
    msd_pool = _write_msd_compiler_pool_spec(tmp_path / "msd-pool.yaml")

    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path / "work",
        include_genbank_catalog=True,
        include_source_promotions=True,
        include_msd_compiler_promotions=True,
        include_rt_cds_dms=True,
        dnadesign_data_root=data_root,
        source_record_resolver=_fixture_source_record_resolver,
        msd_variant_pool_spec_paths=(msd_pool,),
        rt_cds_positions=(1,),
    )

    audit = validate_construct_infer_readiness(
        usr_root=report.usr_root,
        input_dataset=report.input_dataset,
        output_dataset=report.output_dataset,
        expected_construct_subject_ids=tuple(report.input_ids_by_subject_id),
    )

    assert audit.ok, "\n".join(audit.errors)
    assert report.genbank_construct_subject_count == GENBANK_CONSTRUCT_SUBJECT_COUNT
    assert report.crawford_construct_subject_count == CRAWFORD_FIXTURE_CONSTRUCT_SUBJECT_COUNT
    assert report.khan_construct_subject_count == KHAN_FIXTURE_CONSTRUCT_SUBJECT_COUNT
    assert report.msd_compiler_construct_subject_count == MSD_COMPILER_CONSTRUCT_SUBJECT_COUNT
    assert report.rt_cds_dms_construct_subject_count == RT_CDS_DMS_CONSTRUCT_SUBJECT_COUNT
    expected_subjects = (
        GENBANK_CONSTRUCT_SUBJECT_COUNT
        + CRAWFORD_FIXTURE_CONSTRUCT_SUBJECT_COUNT
        + KHAN_FIXTURE_CONSTRUCT_SUBJECT_COUNT
        + MSD_COMPILER_CONSTRUCT_SUBJECT_COUNT
        + RT_CDS_DMS_CONSTRUCT_SUBJECT_COUNT
    )
    assert audit.input_count == expected_subjects
    assert audit.output_count == expected_subjects * CONTEXT_ROWS_PER_SUBJECT
    assert audit.sequence_view_count == expected_subjects * SEQUENCE_VIEWS_PER_SUBJECT
    assert audit.view_names == REQUIRED_INFER_READY_VIEW_NAMES
    assert audit.construct_subject_count == expected_subjects


def test_rt_lnrna_infer_readiness_rejects_missing_required_sequence_view(tmp_path: Path) -> None:
    report = materialize_control_construct_contexts(repo_root=_repo_root(), work_root=tmp_path)
    output_dataset = Dataset(report.usr_root, report.output_dataset)
    view_path = sequence_views_path(output_dataset)
    table = pq.read_table(view_path)
    pq.write_table(table.slice(1), view_path)

    audit = validate_construct_infer_readiness(
        usr_root=report.usr_root,
        input_dataset=report.input_dataset,
        output_dataset=report.output_dataset,
        expected_construct_subject_ids=tuple(report.input_ids_by_subject_id),
    )

    assert not audit.ok
    assert any("Sequence view row count 11 must equal 6 per construct subject" in error for error in audit.errors)
    assert any("missing Infer source view" in error for error in audit.errors)


def test_rt_lnrna_materialization_rejects_swapped_candidate_slot_sequences(tmp_path: Path) -> None:
    with pytest.raises(MaterializationContractError, match="construct_subject__lnrna_sequence length"):
        materialize_control_construct_contexts(
            repo_root=_repo_root(),
            work_root=tmp_path,
            construct_subject_sequence_overrides={
                "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO": {
                    "construct_subject__lnrna_sequence": "A" * 963,
                    "construct_subject__rt_cds_sequence": "C" * 173,
                }
            },
        )


def test_rt_lnrna_materialization_fails_fast_when_rt_cds_field_is_missing(tmp_path: Path) -> None:
    with pytest.raises(MaterializationContractError, match="construct_subject__rt_cds_sequence"):
        materialize_control_construct_contexts(
            repo_root=_repo_root(),
            work_root=tmp_path,
            omitted_construct_subject_fields=("construct_subject__rt_cds_sequence",),
        )
