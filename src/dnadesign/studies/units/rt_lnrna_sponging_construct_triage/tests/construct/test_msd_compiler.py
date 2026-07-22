"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_msd_compiler.py

MSD compiler promotion tests for RT-lnRNA Construct materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from Bio import SeqIO

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    materialize_unified_construct_subject_contexts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import (
    SourcePromotionContractError,
    resolve_msd_compiler_promotions,
)
from dnadesign.usr import Dataset, load_sequence_views

from .helpers import (
    _SNAPBACK_FOLDBACK,
    _TETO_PAYLOAD,
    _WT_RT_CDS_SEQUENCE,
    _assert_construct_output_subject_bridge,
    _assert_construct_subject_envelope_inputs,
    _assert_usr_contracts_strictly_validate,
    _repo_root,
    _reverse_complement,
    _source_window_policy,
    _write_msd_compiler_pool_spec,
)


def test_rt_lnrna_msd_compiler_promotes_reverse_complement_inserted_lnrna(tmp_path: Path) -> None:
    spec_path = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool.yaml",
        cap_ranks=(1,),
        stem_base_ranks=(1,),
        max_variant_count=1,
        expected_variant_count=1,
    )

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
    assert promotion.overlay_fields["construct_subject__msd_cap_id"] == "CDE033R01"
    assert promotion.overlay_fields["construct_subject__msd_cloning_method"] == "YIU"
    assert promotion.overlay_fields["construct_subject__msd_cloning_compatibility"] == "YIU_compatible_cloning_method"
    assert (
        promotion.overlay_fields["construct_subject__msd_primitive_composition"]
        == "snapback_cap_plus_scar_nick_stem_base"
    )
    assert promotion.overlay_fields["construct_subject__msd_stem_base_left"] == "AGTG"
    assert promotion.overlay_fields["construct_subject__msd_stem_base_right"] == "CAAT"
    assert promotion.overlay_fields["construct_subject__msd_insert_orientation"] == "reverse_complement"
    assert promotion.overlay_fields["construct_subject__msd_scar_nick_route_status"] == "resolved"
    assert promotion.overlay_fields["construct_subject__msd_nick_orientation"] == "bottom"
    assert promotion.overlay_fields["construct_subject__msd_nickase"] == "Nb.BtsI"
    assert "scar_nick primitive" in str(promotion.overlay_fields["construct_subject__msd_scar_nick_route_note"])
    assert "YIU-compatible cloning method" in str(promotion.overlay_fields["construct_subject__msd_source_notes"])


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
        cap_ranks=(1, 2),
        stem_base_ranks=(1,),
        max_variant_count=1,
        expected_variant_count=None,
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
        use_primitive_sources=False,
        cap_ids=("C172", "C172dup"),
        max_variant_count=2,
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
        use_primitive_sources=False,
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
        use_primitive_sources=False,
        max_variant_count=1,
        expected_variant_count=1,
    )
    pool_b = _write_msd_compiler_pool_spec(
        tmp_path / "msd-pool-b.yaml",
        pool_id="test_compiler_msd_pool_b_v1",
        construct_id_prefix="rt-lnrna-compiler-b",
        use_primitive_sources=False,
        cap_ids=("C172dup",),
        max_variant_count=1,
        expected_variant_count=1,
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
    assert report.msd_compiler_construct_subject_count == 80
    assert report.rt_cds_dms_construct_subject_count == 0
    assert len(report.input_ids_by_subject_id) == 80
    _assert_construct_subject_envelope_inputs(report)
    _assert_construct_output_subject_bridge(report)
    _assert_usr_contracts_strictly_validate(report)

    inputs = Dataset(report.usr_root, report.input_dataset).head(n=5)
    assert set(inputs["construct_subject__source_basis"]) == {"compiler_generated_msd_lnrna_variant"}
    assert set(inputs["construct_subject__role"]) == {"compiler_lnrna_variant"}
    assert set(inputs["construct_subject__msd_insert_orientation"]) == {"reverse_complement"}

    output = Dataset(report.usr_root, report.output_dataset).head(n=170)
    assert output.shape[0] == 160
    assert set(output["construct_subject__source_basis"]) == {"compiler_generated_msd_lnrna_variant"}
    views = load_sequence_views(Dataset(report.usr_root, report.output_dataset))
    assert len(views) == 480
