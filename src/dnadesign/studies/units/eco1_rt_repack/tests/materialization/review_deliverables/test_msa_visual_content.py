"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_msa_visual_content.py

Eco1 review-deliverable MSA visual content tests.

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
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


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
    assert "The 52-record clade 9 MSA shows the active 25% WT-plurality mask denominator" in msa_text
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


def _read_deliverable(manifest_path: Path, deliverables: dict[str, dict[str, object]], deliverable_id: str) -> str:
    path = resolve_manifest_path(manifest_path, str(deliverables[deliverable_id]["path"]))
    return path.read_text(encoding="utf-8")
