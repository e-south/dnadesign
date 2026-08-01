"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_source_promotions.py

Source-promotion tests for RT-lnRNA Construct materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    materialize_unified_construct_subject_contexts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import (
    resolve_source_construct_subject_promotions,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions.common import (
    construct_window_fit_issue,
)
from dnadesign.usr import Dataset, load_sequence_views

from .helpers import (
    _WT_RT_CDS_SEQUENCE,
    _assert_construct_output_subject_bridge,
    _assert_construct_subject_envelope_inputs,
    _assert_usr_contracts_strictly_validate,
    _repo_root,
    _reverse_complement,
    _source_window_policy,
)
from .source_fixtures import _fixture_source_record_resolver, _write_source_promotion_fixture


def test_rt_lnrna_source_promotion_window_policy_matches_construct_geometry() -> None:
    policy = _source_window_policy()

    assert not construct_window_fit_issue(
        lnrna_sequence="A" * 184,
        rt_cds_sequence="ATG" * 320 + "TAA",
        window_policy=policy,
    )
    assert "dual_cassette_2000bp_context_v1" in construct_window_fit_issue(
        lnrna_sequence="A" * 370,
        rt_cds_sequence="ATG" * 709 + "TAA",
        window_policy=policy,
    )


def test_rt_lnrna_unified_construct_subjects_promote_crawford_and_block_khan_without_rt_cds(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)

    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path / "work",
        allow_partial_byte_resolution=True,
        include_rt_cds_dms=False,
        include_source_promotions=True,
        include_msd_compiler_promotions=False,
        dnadesign_data_root=data_root,
        source_record_resolver=_fixture_source_record_resolver,
    )

    assert report.subject_binding_resolved_subject_count == 46
    assert len(report.blocked_subject_bindings) == 3
    assert report.crawford_construct_subject_count == 2
    assert report.khan_construct_subject_count == 0
    assert report.msd_compiler_construct_subject_count == 0
    assert report.rt_cds_dms_construct_subject_count == 0
    assert report.source_promotion_report is not None
    assert report.source_promotion_report.issues_by_reason == {
        "missing_affiliated_abundance_observation": 1,
        "missing_source_rt_cds_sequence": 1,
    }
    assert len(report.input_ids_by_subject_id) == 48
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=60)
    crawford_inputs = inputs[inputs["construct_subject__source_basis"] == "crawford_eco1_lnrna_fixed_wt_rt"]
    assert crawford_inputs.shape[0] == 2
    assert set(crawford_inputs["construct_subject__source_collection_id"]) == {
        "crawford_eco1_lnrna_abundance_affiliated_sequence_v1"
    }
    assert set(crawford_inputs["construct_subject__source_reference_record_count"]) == {0, 1}
    assert set(crawford_inputs["construct_subject__source_abundance_record_count"]) == {1}
    assert set(crawford_inputs["construct_subject__rt_cds_authority_kind"]) == {"fixed_eco1_wt_rt"}
    assert set(crawford_inputs["construct_subject__source_orientation"]) == {"forward"}
    assert set(crawford_inputs["construct_subject__crawford_sequence_qc_policy"]) == {"eco1_forward_kmer_similarity_v1"}
    assert set(crawford_inputs["construct_subject__crawford_msd_anchor_status"]) == {
        "exact_declared_msd_substring",
        "not_declared_for_abundance_only_sequence",
    }
    assert set(crawford_inputs["construct_subject__crawford_source_context_note"]) == {
        "source_lnrna_sequence_projected_into_dnadesign_dual_cassette_not_native_expression_context"
    }

    output = Dataset(report.usr_root, report.output_dataset).head(n=110)
    assert output.shape[0] == 96
    crawford_output = output[output["construct_subject__source_basis"] == "crawford_eco1_lnrna_fixed_wt_rt"]
    assert crawford_output.shape[0] == 4
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 288


def test_rt_lnrna_source_promotions_resolve_tables_through_public_source_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)
    resolved_source_ids: list[str] = []

    def resolver(source_id: str, root: Path) -> dict[str, object]:
        resolved_source_ids.append(source_id)
        return _fixture_source_record_resolver(source_id, root)

    report = resolve_source_construct_subject_promotions(
        dnadesign_data_root=data_root,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
        source_record_resolver=resolver,
    )

    assert resolved_source_ids == [
        "crawford_2025_retron_ncrna_ml_eco1_lnrna_msd_designs_tsv",
        "crawford_2025_retron_ncrna_ml_eco1_ncrna_abundance_observations_tsv",
        "khan_2024_retron_census_rt_lnrna_sequence_authority_tsv",
        "khan_2024_retron_census_abundance_prior_overlay_tsv",
    ]
    assert report.candidates_by_basis == {"crawford_eco1_lnrna_fixed_wt_rt": 2}


def test_rt_lnrna_source_promotions_include_validated_khan_rt_lnrna_rows(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)
    khan_reference = (
        data_root
        / "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    )
    lnrna_sequence = "ACGT" * 44
    rt_cds_sequence = _WT_RT_CDS_SEQUENCE
    khan_reference.write_text(
        "\t".join(
            [
                "sequence_authority_id",
                "terminal_node",
                "ncrna_sequence_dna",
                "ncrna_sequence_status",
                "rt_accession",
                "rt_cds_sequence",
                "rt_cds_sequence_status",
                "rt_cds_sequence_authority",
                "rt_cds_validation_status",
                "rt_cds_locus_authority_id",
                "rtdna_product_sequence",
                "construct_projection_status",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "khan_terminal_56_sequence_authority",
                "56",
                lnrna_sequence,
                "resolved",
                "WP_000111473.1",
                rt_cds_sequence,
                "resolved",
                "source_genomic_cds_from_mestre_coordinates",
                "translation_exact_match",
                "mestre_node_56_rt_locus",
                "ACGT",
                "representable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = resolve_source_construct_subject_promotions(
        dnadesign_data_root=data_root,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
        source_record_resolver=_fixture_source_record_resolver,
    )

    assert report.issues_by_reason == {"missing_affiliated_abundance_observation": 1}
    assert report.candidates_by_basis == {
        "crawford_eco1_lnrna_fixed_wt_rt": 2,
        "khan_abundance_affiliated_rt_lnrna_reference": 1,
    }
    khan_candidate = [
        candidate
        for candidate in report.candidates
        if candidate.source_basis == "khan_abundance_affiliated_rt_lnrna_reference"
    ][0]
    assert khan_candidate.lnrna_sequence == lnrna_sequence
    assert khan_candidate.rt_cds_sequence == rt_cds_sequence
    assert khan_candidate.rt_cds_authority_kind == "source_genomic_cds_from_mestre_coordinates"
    assert khan_candidate.overlay_fields["construct_subject__rt_cds_validation_status"] == "translation_exact_match"
    assert khan_candidate.overlay_fields["construct_subject__rt_cds_locus_authority_id"] == "mestre_node_56_rt_locus"
    assert khan_candidate.overlay_fields["construct_subject__source_abundance_record_count"] == 1
    assert khan_candidate.overlay_fields["construct_subject__khan_abundance_ordinal_bins"] == "low"


def test_rt_lnrna_source_promotions_reject_khan_rt_cds_without_translation_exact_validation(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)
    khan_reference = (
        data_root
        / "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    )
    khan_reference.write_text(
        "\t".join(
            [
                "sequence_authority_id",
                "terminal_node",
                "ncrna_sequence_dna",
                "ncrna_sequence_status",
                "rt_accession",
                "rt_cds_sequence",
                "rt_cds_sequence_status",
                "rt_cds_sequence_authority",
                "rt_cds_validation_status",
                "construct_projection_status",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "khan_terminal_56_sequence_authority",
                "56",
                "ACGT" * 44,
                "resolved",
                "WP_000111473.1",
                _WT_RT_CDS_SEQUENCE,
                "resolved",
                "source_genomic_cds_from_mestre_coordinates",
                "basic_cds_sanity",
                "representable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = resolve_source_construct_subject_promotions(
        dnadesign_data_root=data_root,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
        source_record_resolver=_fixture_source_record_resolver,
    )

    assert report.candidates_by_basis == {"crawford_eco1_lnrna_fixed_wt_rt": 2}
    assert report.issues_by_reason == {
        "invalid_source_rt_cds_sequence": 1,
        "missing_affiliated_abundance_observation": 1,
    }


def test_rt_lnrna_source_promotion_rejects_reverse_complemented_crawford_lnrna(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)
    reference_path = (
        data_root
        / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv"
    )
    rows = reference_path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split("\t")
    values = rows[1].split("\t")
    lnrna_index = header.index("lnrna_sequence")
    values[lnrna_index] = _reverse_complement(values[lnrna_index])
    reference_path.write_text(rows[0] + "\n" + "\t".join(values) + "\n", encoding="utf-8")
    abundance_path = (
        data_root / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/"
        "eco1_ncrna_abundance_observations.tsv"
    )
    abundance_rows = abundance_path.read_text(encoding="utf-8").splitlines()
    abundance_header = abundance_rows[0].split("\t")
    abundance_values = abundance_rows[1].split("\t")
    abundance_lnrna_index = abundance_header.index("lnrna_sequence")
    abundance_values[abundance_lnrna_index] = _reverse_complement(abundance_values[abundance_lnrna_index])
    abundance_path.write_text(
        abundance_rows[0] + "\n" + "\t".join(abundance_values) + "\n" + abundance_rows[2] + "\n",
        encoding="utf-8",
    )

    report = resolve_source_construct_subject_promotions(
        dnadesign_data_root=data_root,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
        source_record_resolver=_fixture_source_record_resolver,
    )

    assert report.candidates_by_basis == {"crawford_eco1_lnrna_fixed_wt_rt": 1}
    assert report.issues_by_reason == {
        "missing_source_rt_cds_sequence": 1,
        "reverse_complement_lnrna_orientation": 1,
    }
