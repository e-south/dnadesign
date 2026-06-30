"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_visual_content.py

Eco1 review-deliverable visual content tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_chimerax_context_scripts,
    assert_review_notebook_contract,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def test_review_deliverable_visual_content_is_plain_and_linked(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    _assert_mask_and_msa_content(result.manifest_path, deliverables)
    _assert_proteinmpnn_content(result.manifest_path, deliverables)
    _assert_linked_fold_and_esmc_content(result.manifest_path, deliverables)
    assert_chimerax_context_scripts(
        manifest_path=result.manifest_path,
        deliverables=deliverables,
        forbidden_path_text=str(tmp_path),
    )
    assert_review_notebook_contract(result.notebook_path.read_text(encoding="utf-8"))


def _assert_mask_and_msa_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    msa_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_mask_panel")
    assert "4-record clade 9 MSA: 25% plurality mask" in msa_text
    assert "all accepted clade 9 alignment rows" in str(deliverables["msa_plurality_mask_panel"]["description"])
    assert "current conservation mask uses this clade 9 denominator" in str(
        deliverables["msa_plurality_mask_panel"]["description"]
    )
    assert deliverables["msa_plurality_mask_panel"]["evidence_summary"]["current_mask_denominator"] is True
    assert "ec86_clade9_conservation_v1__" not in msa_text
    assert "WT plurality &gt;=25% (clade 9)" in msa_text
    assert "C9 001 fig|fixture.1.peg.1" in msa_text
    assert "Mask-protected" in msa_text

    subtype_text = _read_deliverable(manifest_path, deliverables, "msa_subtype_plurality_panel")
    assert "3-record II-A3/42_1 Eco1 subtype MSA" in subtype_text
    assert "all accepted II-A3/42_1 subtype alignment rows" in str(
        deliverables["msa_subtype_plurality_panel"]["description"]
    )
    assert "does not replace the clade 9 denominator" in str(deliverables["msa_subtype_plurality_panel"]["description"])
    assert deliverables["msa_subtype_plurality_panel"]["evidence_summary"]["current_mask_denominator"] is False
    assert "II-A3 002 fig|fixture.2.peg.1" in subtype_text
    assert "WT plurality &gt;=25% (Eco1 subtype II-A3/42_1)" in subtype_text

    mask_text = _read_deliverable(manifest_path, deliverables, "linear_mask_tracks")
    assert "Protected evidence defines fixed residues and the design canvas" in mask_text
    assert "WT residue" not in mask_text
    assert "Ec86 positions 1-6" in mask_text
    assert "Mask evidence track" in mask_text
    assert "M" in mask_text
    assert "K" in mask_text


def _assert_proteinmpnn_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    diversity_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_score_mutation_burden")
    assert "ProteinMPNN proposes sequence diversity inside the mutable canvas" in diversity_text
    assert "Sequence identity to Ec86 WT (%)" in diversity_text
    assert "Accepted designs retain a minority of WT residues." not in diversity_text
    assert "Sampling temperature" in diversity_text
    assert "Reported ProteinMPNN score" in diversity_text

    mutation_density_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_mutation_density")
    assert "RT1" in mutation_density_text
    assert "NAxxH" in mutation_density_text

    tao_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_tao_style_fold_validation")
    assert "ProteinMPNN designs cluster by ColabFold RMSD and pLDDT" in tao_text
    assert "WT-runtime C-alpha RMSD" in tao_text
    assert "Mean pLDDT" in tao_text
    assert "Tao-style" in str(deliverables["proteinmpnn_tao_style_fold_validation"]["description"])
    assert "single active mask policy" in str(
        deliverables["proteinmpnn_tao_style_fold_validation"]["interpretation_limit"]
    )


def _assert_linked_fold_and_esmc_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    linked_fold_plot = resolve_manifest_path(
        manifest_path,
        str(deliverables["foldcheck_review_fold_metric_scatter"]["path"]),
    )
    assert linked_fold_plot.exists()
    assert linked_fold_plot.parent.name == "plots"
    linked_structure_overlay = resolve_manifest_path(
        manifest_path,
        str(deliverables["foldcheck_review_structure_overlay_panel"]["path"]),
    )
    assert linked_structure_overlay.exists()

    linked_esmc_plot = resolve_manifest_path(
        manifest_path,
        str(deliverables["wt_esmc_substitution_llr_heatmap"]["path"]),
    )
    assert linked_esmc_plot.exists()
    assert linked_esmc_plot.parent.name == "plots"
    linked_esmc_text = linked_esmc_plot.read_text(encoding="utf-8")
    assert "<title" in linked_esmc_text
    assert "<desc" in linked_esmc_text
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["title"] == (
        "ESMC masked-marginal scores form a WT substitution matrix"
    )
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["render_mode"] == "wide_visual"
    assert "LLR = log P(alternate) - log P(WT)" in str(
        deliverables["wt_esmc_substitution_llr_heatmap"]["method_summary"]
    )
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["evidence_summary"]["substitution_llr_rows"] == 114

    esmc_scatter_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_vs_esmc_entropy")
    assert "High clade 9 plurality usually corresponds to low ESMC entropy" in esmc_scatter_text
    assert "Pearson r =" in esmc_scatter_text
    assert "R2 =" in esmc_scatter_text
    assert "25% plurality threshold" in esmc_scatter_text
    assert "model-derived audit" in str(deliverables["msa_plurality_vs_esmc_entropy"]["interpretation_limit"])


def _read_deliverable(manifest_path: Path, deliverables: dict[str, dict[str, object]], deliverable_id: str) -> str:
    path = resolve_manifest_path(manifest_path, str(deliverables[deliverable_id]["path"]))
    return path.read_text(encoding="utf-8")


def test_msa_plurality_panel_renders_all_source_rows_without_arbitrary_cutoff(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    alignment_path = tmp_path / "conservation_alignments" / "ec86_clade9_conservation_v1.aligned.fasta"
    source_manifest_path = tmp_path / "conservation_sources" / "ec86_clade9_conservation_v1.source_manifest.yaml"
    alignment_records = [">eco1_rt_ec86kit_reference", "MKSAYL"]
    source_records = []
    for index in range(1, 52):
        record_id = f"clade9_neighbor_{index:03d}"
        sequence = "MKSAFL" if index % 2 else "MRSAYI"
        alignment_records.extend([f">{record_id}", sequence])
        source_records.append(
            {
                "record_id": record_id,
                "provider_id": "fixture_provider",
                "accession": f"fig|fixture.{index}.peg.1",
            }
        )
    alignment_path.write_text("\n".join(alignment_records) + "\n", encoding="utf-8")
    source_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["included_record_count"] = len(source_records)
    source_manifest["included_records"] = source_records
    source_manifest_path.write_text(yaml.safe_dump(source_manifest, sort_keys=False), encoding="utf-8")

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    msa_text = _read_deliverable(result.manifest_path, deliverables, "msa_plurality_mask_panel")
    assert "52-record clade 9 MSA: 25% plurality mask" in msa_text
    assert "all 52 accepted alignment rows" in str(deliverables["msa_plurality_mask_panel"]["alt_text"])
    assert "C9 051 fig|fixture.51.peg.1" in msa_text
    assert "display subset" not in str(deliverables["msa_plurality_mask_panel"]["description"])


def test_msa_subtype_panel_requires_clade_source_superset(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    subtype_manifest_path = (
        tmp_path / "conservation_sources" / "ec86_iia3_cluster42_1_conservation_v1.source_manifest.yaml"
    )
    subtype_manifest = yaml.safe_load(subtype_manifest_path.read_text(encoding="utf-8"))
    subtype_manifest["included_records"][0]["accession"] = "WP_000000000.1"
    subtype_manifest_path.write_text(yaml.safe_dump(subtype_manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="subtype MSA source accessions must be a subset"):
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)
