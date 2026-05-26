"""Source-promotion fixtures for RT-lnRNA Construct materialization tests."""

from __future__ import annotations

from pathlib import Path

_SOURCE_ID_TO_FIXTURE_PATH = {
    "crawford_2025_retron_ncrna_ml_eco1_lnrna_msd_designs_tsv": (
        "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv"
    ),
    "crawford_2025_retron_ncrna_ml_eco1_ncrna_abundance_observations_tsv": (
        "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/"
        "eco1_ncrna_abundance_observations.tsv"
    ),
    "khan_2024_retron_census_abundance_prior_overlay_tsv": (
        "sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv"
    ),
    "khan_2024_retron_census_rt_lnrna_sequence_authority_tsv": (
        "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    ),
}


def _write_source_promotion_fixture(data_root: Path) -> None:
    crawford_reference = (
        data_root
        / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv"
    )
    crawford_abundance = (
        data_root
        / "sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays"
        / "eco1_ncrna_abundance_observations.tsv"
    )
    khan_reference = (
        data_root
        / "sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv"
    )
    khan_abundance = (
        data_root / "sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv"
    )
    for path in (crawford_reference, crawford_abundance, khan_reference, khan_abundance):
        path.parent.mkdir(parents=True, exist_ok=True)

    lnrna_sequence = (
        "TGCGCACCCTTAGCGAGAGGTTTATCATTAAGGTCAACCTCTGGATGTTGTTTCGGCATCCTGCATTGAAT"
        "CTGAGTTACTGTCTGTTTTCCTTGTTGGAACGGAGAGCATCGCCTGATGCTCTCCGAGCCAACCAGGAAAC"
        "CCGTTTTTTCTGACGTAAGGGTGCGCA"
    )
    msd_sequence = "".join(
        [
            "CTGAGTTACTGT",
            "CTGTTTTCCTTG",
            "TTGGAACGGAGA",
            "GCATCGCCTGAT",
            "GCTCTCCGAGCC",
            "AACCAGGAAACC",
            "CGTTTTTTCTGAC",
        ]
    )
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
        + "\n"
        + "\t".join(
            [
                "crawford_ref_reference_only",
                "crawford_eco1_lnrna_msd_designs_v1",
                "eco1_local_variant_library",
                "sequence_design_reference",
                "86_r2_L1_reference_only",
                lnrna_sequence[:-2] + "AT",
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
    khan_abundance.write_text(
        "\t".join(
            [
                "abundance_prior_id",
                "reference_overlay_id",
                "maps_to_reference_record_id",
                "raw_value",
                "normalized_value",
                "ordinal_bin",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "khan_terminal_56_rt_dna_abundance",
                "khan_cross_retron_rt_dna_abundance_v1",
                "khan_terminal_56",
                "0.0794138",
                "0.0794138",
                "low",
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
