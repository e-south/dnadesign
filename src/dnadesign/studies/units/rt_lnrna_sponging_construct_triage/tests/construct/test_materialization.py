"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_materialization.py

Construct materialization checks for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from dnadesign.permuter import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    default_codon_table_path,
    generate_variants,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    ControlConstructMaterializationReport,
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
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import (
    ConstructWindowPolicy,
    SourcePromotionContractError,
    resolve_msd_compiler_promotions,
    resolve_source_construct_subject_promotions,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions.common import (
    construct_window_fit_issue,
)
from dnadesign.usr import Dataset, load_sequence_views, sequence_views_path

_CONSTRUCT_SUBJECT_SEQUENCE_FIELDS = ("construct_subject__lnrna_sequence", "construct_subject__rt_cds_sequence")
_WT_RT_CDS_SEQUENCE = "ATG" * 320 + "TAA"
_TETO_PAYLOAD = "tccctatcagtgatagaga"
_SNAPBACK_FOLDBACK = "GAGAGACTC"
_SOURCE_ID_TO_FIXTURE_PATH = {
    "crawford_2025_retron_ncrna_ml_eco1_lnrna_msd_designs_tsv": (
        "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv"
    ),
    "crawford_2025_retron_ncrna_ml_eco1_ncrna_abundance_observations_tsv": (
        "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/"
        "eco1_ncrna_abundance_observations.tsv"
    ),
    "khan_2024_retron_census_rt_lnrna_sequence_authority_tsv": (
        "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    ),
}


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _source_window_policy() -> ConstructWindowPolicy:
    return ConstructWindowPolicy(
        context_id="dual_cassette_2000bp_context_v1",
        target_start_0=56,
        target_length_nt=2000,
        template_length_nt=4956,
        lnrna_template_span_0=(186, 359),
        rt_cds_template_span_0=(524, 1487),
    )


def _assert_construct_subject_envelope_inputs(report: ControlConstructMaterializationReport) -> None:
    inputs = Dataset(report.usr_root, report.input_dataset).head(n=len(report.input_ids_by_subject_id) + 5)

    assert set(inputs["construct_subject__record_kind"]) == {"construct_subject_envelope"}
    assert set(inputs["construct_subject__sequence_authority"]) == {"overlay_only"}
    assert set(inputs["construct_subject__envelope_carrier_policy"]) == {"synthetic_unique_dna4_v1"}
    assert inputs["id"].is_unique
    assert {tuple(fields) for fields in inputs["construct_subject__biological_sequence_fields"]} == {
        _CONSTRUCT_SUBJECT_SEQUENCE_FIELDS
    }


def _assert_construct_output_subject_bridge(report: ControlConstructMaterializationReport) -> None:
    output = Dataset(report.usr_root, report.output_dataset).head(n=len(report.input_ids_by_subject_id) * 4 + 20)

    assert set(output["construct_subject__record_kind"]) == {"construct_output"}
    assert set(output["construct_subject__sequence_authority"]) == {"realized_construct_sequence"}
    assert {tuple(fields) for fields in output["construct_subject__biological_sequence_fields"]} == {
        _CONSTRUCT_SUBJECT_SEQUENCE_FIELDS
    }
    for construct_subject_id, input_id in report.input_ids_by_subject_id.items():
        subject_output = output[output["construct__input_id"] == input_id]
        assert subject_output.shape[0] == 2
        assert set(subject_output["construct_subject__id"]) == {construct_subject_id}
        assert subject_output["construct_subject__lnrna_sequence"].nunique() == 1
        assert subject_output["construct_subject__rt_cds_sequence"].nunique() == 1


def _assert_usr_contracts_strictly_validate(report: ControlConstructMaterializationReport) -> None:
    Dataset(report.usr_root, report.input_dataset).validate(strict=True)
    Dataset(report.usr_root, report.output_dataset).validate(strict=True)


def _write_source_promotion_fixture(data_root: Path) -> None:
    crawford_reference = (
        data_root
        / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv"
    )
    crawford_abundance = (
        data_root / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/"
        "eco1_ncrna_abundance_observations.tsv"
    )
    khan_reference = (
        data_root
        / "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    )
    for path in (crawford_reference, crawford_abundance, khan_reference):
        path.parent.mkdir(parents=True, exist_ok=True)

    lnrna_sequence = (
        "TGCGCACCCTTAGCGAGAGGTTTATCATTAAGGTCAACCTCTGGATGTTGTTTCGGCATCCTGCATTGAAT"
        "CTGAGTTACTGTCTGTTTTCCTTGTTGGAACGGAGAGCATCGCCTGATGCTCTCCGAGCCAACCAGGAAAC"
        "CCGTTTTTTCTGACGTAAGGGTGCGCA"
    )
    msd_sequence = "CTGAGTTACTGTCTGTTTTCCTTGTTGGAACGGAGAGCATCGCCTGATGCTCTCCGAGCCAACCAGGAAACCCGTTTTTTCTGAC"
    crawford_reference.write_text(
        "\t".join(
            [
                "record_id",
                "reference_overlay_id",
                "regime",
                "label_kind",
                "lnrna_design_id",
                "lnrna_sequence",
                "msd_sequence",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "crawford_ref_1",
                "crawford_eco1_lnrna_msd_designs_v1",
                "eco1_local_variant_library",
                "sequence_design_reference",
                "86_r2_L1_wt",
                lnrna_sequence,
                msd_sequence,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    crawford_abundance.write_text(
        "\t".join(
            ["observation_id", "reference_overlay_id", "regime", "label_kind", "lnrna_design_id", "lnrna_sequence"]
        )
        + "\n"
        + "\t".join(
            [
                "crawford_obs_1",
                "crawford_eco1_lnrna_msd_abundance_v1",
                "eco1_local_variant_library",
                "msdna_abundance_score_relative_to_mean_wt",
                "crawford_score_fasta_1",
                lnrna_sequence,
            ]
        )
        + "\n"
        + "\t".join(
            [
                "crawford_obs_abundance_only",
                "crawford_eco1_lnrna_msd_abundance_v1",
                "eco1_local_variant_library",
                "msdna_abundance_score_relative_to_mean_wt",
                "crawford_score_fasta_abundance_only",
                lnrna_sequence[:-1] + "T",
            ]
        )
        + "\n",
        encoding="utf-8",
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
                "fig|source.peg.1",
                "",
                "unresolved",
                "ACGT",
                "blocked_missing_rt_cds",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _fixture_source_record_resolver(source_id: str, root: Path) -> dict[str, object]:
    relative_path = _SOURCE_ID_TO_FIXTURE_PATH[source_id]
    path = root / relative_path
    return {
        "source_id": source_id,
        "available": path.is_file(),
        "absolute_path": str(path),
    }


def _write_msd_compiler_pool_spec(
    path: Path,
    *,
    pool_id: str = "test_compiler_msd_pool_v1",
    construct_id_prefix: str = "rt-lnrna-compiler",
    cap_ids: tuple[str, ...] = ("C172",),
    expected_5p_flank: str = "TCCTGCATTGAA",
    expected_3p_flank: str = "GTAAGGGTGCGC",
    max_variant_count: int = 4,
    expected_variant_count: int | None = 1,
    extra_cap_sequences: dict[str, str] | None = None,
    extra_stem_base_fields: str = "",
) -> Path:
    cap_sequences = {
        "C26": "AGGC",
        "C172": _SNAPBACK_FOLDBACK,
        **(extra_cap_sequences or {}),
    }
    expected_line = "" if expected_variant_count is None else f"expected_variant_count: {expected_variant_count}\n"
    cap_sequence_lines = "\n".join(
        f"    {cap_id}:\n      sequence: {sequence}" for cap_id, sequence in cap_sequences.items()
    )
    cap_id_lines = "\n".join(f"    - {cap_id}" for cap_id in cap_ids)
    source_ref = (
        "docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml#pES-retron-179"
    )
    path.write_text(
        f"""contract: rt_lnrna_msd_variant_pool_spec_v1
schema_version: 1
pool_id: {pool_id}
study_id: rt_lnrna_sponging_construct_triage
payload_program_id: tetO_sponging_v1
max_variant_count: {max_variant_count}
{expected_line}dedupe_policy: fail
template_lnrna:
  sequence_ref: genbank:pes-retron-26-a1-a2.gb#a1-a2
  genbank_path: docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/pes-retron-26-a1-a2.gb
  sequence_span_0: [0, 173]
template_msd_design:
  construct_id: rt-lnrna-template-retron26
  payload_id: TetR
  cap_id: C26
  left_base: CGGG
  right_base: ACAG
  profile_s3s2s1s0: MXMX
placement:
  expected_5p_flank: {expected_5p_flank}
  expected_3p_flank: {expected_3p_flank}
compiler_inputs:
  payload_sequences:
    TetR:
      sequence: {_TETO_PAYLOAD.upper()}
  cap_sequences:
{cap_sequence_lines}
design_space:
  construct_id_prefix: {construct_id_prefix}
  payload_ids:
    - TetR
  cap_ids:
{cap_id_lines}
  stem_bases:
    - stem_base_id: lagtg_rcaat
      left_base: AGTG
      right_base: CAAT
      profile_s3s2s1s0: MXMM
      source_ref: {source_ref}
      nick_orientation: bottom
      nickase: Nb.BtsI
{extra_stem_base_fields}source_refs:
  - {source_ref}
""",
        encoding="utf-8",
    )
    return path


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


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
        "lnrna_span_in_construct_anchor_mean",
        "lnrna_span_in_construct_reverse_complement_anchor_mean",
        "rt_cds_span_in_construct_anchor_mean",
        "rt_cds_span_in_construct_reverse_complement_anchor_mean",
    ]
    assert all(len(view_rows) == 2 for view_rows in views_by_name.values())
    assert {view.orientation for view in views_by_name["dual_cassette_2000bp_seq_mean"]} == {"forward"}
    assert {view.orientation for view in views_by_name["dual_cassette_2000bp_reverse_complement_seq_mean"]} == {
        "reverse_complement"
    }
    assert {
        (view.anchor_start_0, view.anchor_end_0) for view in views_by_name["lnrna_span_in_construct_anchor_mean"]
    } == {(130, 303), (123, 310)}
    assert {view.recommended_pooling for view in views_by_name["lnrna_span_in_construct_anchor_mean"]} == {
        "anchor_mean"
    }
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["lnrna_span_in_construct_reverse_complement_anchor_mean"]
    } == {(1690, 1877), (1697, 1870)}
    assert {
        (view.anchor_start_0, view.anchor_end_0) for view in views_by_name["rt_cds_span_in_construct_anchor_mean"]
    } == {(468, 1431), (475, 1438)}
    assert {
        (view.anchor_start_0, view.anchor_end_0)
        for view in views_by_name["rt_cds_span_in_construct_reverse_complement_anchor_mean"]
    } == {(562, 1525), (569, 1532)}


def test_rt_lnrna_catalog_variants_materialize_consolidated_construct_views(tmp_path: Path) -> None:
    report = materialize_variant_construct_contexts(repo_root=_repo_root(), work_root=tmp_path)

    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)
    assert len(report.input_ids_by_subject_id) == 36
    output = Dataset(report.usr_root, report.output_dataset).head(n=100)
    assert output.shape[0] == 72

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
    assert len(views) == 216
    assert {view.view_name for view in views if view.parent_sequence_id == input_id and view.view_name is not None} == {
        "dual_cassette_2000bp_seq_mean",
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "lnrna_span_in_construct_anchor_mean",
        "lnrna_span_in_construct_reverse_complement_anchor_mean",
        "rt_cds_span_in_construct_anchor_mean",
        "rt_cds_span_in_construct_reverse_complement_anchor_mean",
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

    assert report.genbank_construct_subject_count == 36
    assert report.rt_cds_dms_construct_subject_count == 19
    assert len(report.input_ids_by_subject_id) == 55
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
    assert output.shape[0] == 110
    assert set(output["construct_subject__source_basis"]) == {"genbank_variant_catalog", "in_silico_rt_cds_dms"}
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 330
    assert {view.view_name for view in views if view.view_name is not None} == {
        "dual_cassette_2000bp_seq_mean",
        "dual_cassette_2000bp_reverse_complement_seq_mean",
        "lnrna_span_in_construct_anchor_mean",
        "lnrna_span_in_construct_reverse_complement_anchor_mean",
        "rt_cds_span_in_construct_anchor_mean",
        "rt_cds_span_in_construct_reverse_complement_anchor_mean",
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
    assert report.genbank_construct_subject_count == 36
    assert report.crawford_construct_subject_count == 2
    assert report.khan_construct_subject_count == 0
    assert report.msd_compiler_construct_subject_count == 1
    assert report.rt_cds_dms_construct_subject_count == 19
    assert audit.input_count == 58
    assert audit.output_count == 116
    assert audit.sequence_view_count == 348
    assert audit.view_names == REQUIRED_INFER_READY_VIEW_NAMES
    assert audit.construct_subject_count == 58


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


def test_rt_lnrna_unified_construct_subjects_promote_crawford_and_block_khan_without_rt_cds(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dnadesign-data"
    _write_source_promotion_fixture(data_root)

    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path / "work",
        include_genbank_catalog=False,
        include_rt_cds_dms=False,
        include_source_promotions=True,
        include_msd_compiler_promotions=False,
        dnadesign_data_root=data_root,
        source_record_resolver=_fixture_source_record_resolver,
    )

    assert report.genbank_construct_subject_count == 0
    assert report.crawford_construct_subject_count == 2
    assert report.khan_construct_subject_count == 0
    assert report.msd_compiler_construct_subject_count == 0
    assert report.rt_cds_dms_construct_subject_count == 0
    assert report.source_promotion_report is not None
    assert report.source_promotion_report.issues_by_reason == {"missing_source_rt_cds_sequence": 1}
    assert len(report.input_ids_by_subject_id) == 2
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=5)
    assert set(inputs["construct_subject__source_basis"]) == {"crawford_eco1_lnrna_fixed_wt_rt"}
    assert set(inputs["construct_subject__source_collection_id"]) == {"crawford_eco1_lnrna_sequence_union_v1"}
    assert set(inputs["construct_subject__source_reference_record_count"]) == {0, 1}
    assert set(inputs["construct_subject__source_abundance_record_count"]) == {1}
    assert set(inputs["construct_subject__rt_cds_authority_kind"]) == {"fixed_eco1_wt_rt"}
    assert set(inputs["construct_subject__source_orientation"]) == {"forward"}
    assert set(inputs["construct_subject__crawford_sequence_qc_policy"]) == {"eco1_forward_kmer_similarity_v1"}
    assert set(inputs["construct_subject__crawford_msd_anchor_status"]) == {
        "exact_declared_msd_substring",
        "not_declared_for_abundance_only_sequence",
    }
    assert set(inputs["construct_subject__crawford_source_context_note"]) == {
        "source_lnrna_sequence_projected_into_dnadesign_dual_cassette_not_native_expression_context"
    }

    output = Dataset(report.usr_root, report.output_dataset).head(n=10)
    assert output.shape[0] == 4
    assert set(output["construct_subject__source_basis"]) == {"crawford_eco1_lnrna_fixed_wt_rt"}
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 12


def test_rt_lnrna_msd_compiler_promotes_reverse_complement_inserted_lnrna(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(tmp_path / "msd-pool.yaml")

    promotions = resolve_msd_compiler_promotions(
        repo_root=_repo_root(),
        pool_spec_path=spec_path,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
    )

    assert len(promotions) == 1
    promotion = promotions[0]
    variant_product_5to3 = (
        "GTCAGAAAAAA"
        + "AGTG"
        + _TETO_PAYLOAD.upper()
        + _SNAPBACK_FOLDBACK.upper()
        + _reverse_complement(_TETO_PAYLOAD)
        + "CAAT"
        + "ACAGTAACTCAGA"
    )
    template_product_5to3 = (
        "GTCAGAAAAAA"
        + "CGGG"
        + _TETO_PAYLOAD.upper()
        + "AGGC"
        + _reverse_complement(_TETO_PAYLOAD)
        + "ACAG"
        + "ACAGTAACTCAGA"
    )
    template_sequence = str(
        SeqIO.read(
            _repo_root()
            / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/pes-retron-26-a1-a2.gb",
            "genbank",
        ).seq
    ).upper()
    template_insert = _reverse_complement(template_product_5to3)
    insert_start = template_sequence.index(template_insert)
    expected_lnrna = (
        template_sequence[:insert_start]
        + _reverse_complement(variant_product_5to3)
        + template_sequence[insert_start + len(template_insert) :]
    )

    assert promotion.lnrna_sequence == expected_lnrna
    assert promotion.lnrna_sequence != template_sequence
    assert variant_product_5to3 not in promotion.lnrna_sequence
    assert _reverse_complement(variant_product_5to3) in promotion.lnrna_sequence
    assert promotion.rt_cds_sequence == _WT_RT_CDS_SEQUENCE
    assert promotion.source_basis == "compiler_generated_msd_lnrna_variant"
    assert promotion.lnrna_authority_kind == "compiler_generated_lnrna_sequence"
    assert promotion.rt_cds_authority_kind == "fixed_eco1_wt_rt"
    assert promotion.overlay_fields["construct_subject__role"] == "compiler_lnrna_variant"
    assert promotion.overlay_fields["construct_subject__msd_cap_id"] == "C172"
    assert promotion.overlay_fields["construct_subject__msd_stem_base_left"] == "AGTG"
    assert promotion.overlay_fields["construct_subject__msd_stem_base_right"] == "CAAT"
    assert promotion.overlay_fields["construct_subject__msd_insert_orientation"] == "reverse_complement"
    assert promotion.overlay_fields["construct_subject__msd_scar_nick_route_status"] == "note_only"
    assert promotion.overlay_fields["construct_subject__msd_nick_orientation"] == "bottom"
    assert promotion.overlay_fields["construct_subject__msd_nickase"] == "Nb.BtsI"
    assert "pES-retron-179" in str(promotion.overlay_fields["construct_subject__msd_scar_nick_route_note"])


def test_rt_lnrna_msd_compiler_rejects_template_flank_mismatch(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(tmp_path / "msd-pool.yaml", expected_5p_flank="AAAAAAAAAAAA")

    with pytest.raises(SourcePromotionContractError, match="5' flank"):
        resolve_msd_compiler_promotions(
            repo_root=_repo_root(),
            pool_spec_path=spec_path,
            wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
            window_policy=_source_window_policy(),
        )


def test_rt_lnrna_msd_compiler_rejects_over_budget_combinatorics(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool.yaml",
        cap_ids=("C172", "C173"),
        max_variant_count=1,
        expected_variant_count=None,
        extra_cap_sequences={"C173": "GGAAGATCC"},
    )

    with pytest.raises(SourcePromotionContractError, match="exceeds max_variant_count"):
        resolve_msd_compiler_promotions(
            repo_root=_repo_root(),
            pool_spec_path=spec_path,
            wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
            window_policy=_source_window_policy(),
        )


def test_rt_lnrna_msd_compiler_rejects_duplicate_generated_lnrna(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool.yaml",
        cap_ids=("C172", "C172dup"),
        expected_variant_count=2,
        extra_cap_sequences={"C172dup": _SNAPBACK_FOLDBACK},
    )

    with pytest.raises(SourcePromotionContractError, match="Duplicate compiler-generated lnRNA sequence"):
        resolve_msd_compiler_promotions(
            repo_root=_repo_root(),
            pool_spec_path=spec_path,
            wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
            window_policy=_source_window_policy(),
        )


def test_rt_lnrna_msd_compiler_rejects_unknown_pool_fields(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool.yaml",
        extra_stem_base_fields="      unsupported_geometry: silent_drop_risk\n",
    )

    with pytest.raises(SourcePromotionContractError, match="design_space.stem_bases.0.unsupported_geometry"):
        resolve_msd_compiler_promotions(
            repo_root=_repo_root(),
            pool_spec_path=spec_path,
            wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
            window_policy=_source_window_policy(),
        )


def test_rt_lnrna_unified_construct_subjects_reject_cross_pool_duplicate_msd_lnrna(tmp_path: Path) -> None:
    pool_a = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool-a.yaml",
        pool_id="test_compiler_msd_pool_a_v1",
        construct_id_prefix="rt-lnrna-compiler-a",
    )
    pool_b = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool-b.yaml",
        pool_id="test_compiler_msd_pool_b_v1",
        construct_id_prefix="rt-lnrna-compiler-b",
        cap_ids=("C172dup",),
        extra_cap_sequences={"C172dup": _SNAPBACK_FOLDBACK},
    )

    with pytest.raises(SourcePromotionContractError, match="Duplicate compiler-generated lnRNA sequence across pools"):
        materialize_unified_construct_subject_contexts(
            repo_root=_repo_root(),
            work_root=tmp_path,
            include_genbank_catalog=False,
            include_source_promotions=False,
            include_msd_compiler_promotions=True,
            include_rt_cds_dms=False,
            msd_variant_pool_spec_paths=(pool_a, pool_b),
        )


def test_rt_lnrna_unified_construct_subjects_include_msd_compiler_pool(tmp_path: Path) -> None:
    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path,
        include_genbank_catalog=False,
        include_source_promotions=False,
        include_msd_compiler_promotions=True,
        include_rt_cds_dms=False,
        msd_variant_pool_spec_paths=(_write_msd_compiler_pool_spec(tmp_path / "msd-pool.yaml"),),
    )

    assert report.genbank_construct_subject_count == 0
    assert report.crawford_construct_subject_count == 0
    assert report.khan_construct_subject_count == 0
    assert report.msd_compiler_construct_subject_count == 1
    assert report.rt_cds_dms_construct_subject_count == 0
    assert len(report.input_ids_by_subject_id) == 1
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=5)
    assert set(inputs["construct_subject__source_basis"]) == {"compiler_generated_msd_lnrna_variant"}
    assert set(inputs["construct_subject__role"]) == {"compiler_lnrna_variant"}
    assert set(inputs["construct_subject__msd_insert_orientation"]) == {"reverse_complement"}

    output = Dataset(report.usr_root, report.output_dataset).head(n=10)
    assert output.shape[0] == 2
    assert set(output["construct_subject__source_basis"]) == {"compiler_generated_msd_lnrna_variant"}
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 6


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

    assert report.issues_by_reason == {}
    assert report.candidates_by_basis == {
        "crawford_eco1_lnrna_fixed_wt_rt": 2,
        "khan_source_rt_lnrna_reference": 1,
    }
    khan_candidate = [
        candidate for candidate in report.candidates if candidate.source_basis == "khan_source_rt_lnrna_reference"
    ][0]
    assert khan_candidate.lnrna_sequence == lnrna_sequence
    assert khan_candidate.rt_cds_sequence == rt_cds_sequence
    assert khan_candidate.rt_cds_authority_kind == "source_genomic_cds_from_mestre_coordinates"
    assert khan_candidate.overlay_fields["construct_subject__rt_cds_validation_status"] == "translation_exact_match"
    assert khan_candidate.overlay_fields["construct_subject__rt_cds_locus_authority_id"] == "mestre_node_56_rt_locus"


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
    assert report.issues_by_reason == {"invalid_source_rt_cds_sequence": 1}


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
