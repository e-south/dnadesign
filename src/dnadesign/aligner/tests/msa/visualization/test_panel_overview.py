"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/visualization/test_panel_overview.py

Tests for all-record MSA overview panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa import write_fasta_records
from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    materialize_msa_visualizations,
)
from dnadesign.aligner.tests.msa.visualization._fixtures import (
    TARGET,
    target_hash,
    write_panel_spec,
)


def test_visualization_overview_can_render_all_records_with_manifest_labels(tmp_path: Path) -> None:
    alignment_root = tmp_path / "alignments"
    alignment_root.mkdir()
    source_manifest = _write_source_manifest(tmp_path)
    profile_id = "profile_a"
    target_aligned = TARGET[:10] + "-" + TARGET[10:]
    write_fasta_records(
        alignment_root / f"{profile_id}.aligned.fasta",
        {
            "target": target_aligned,
            "profile_a__1552__1552": target_aligned[:4] + "A" + target_aligned[5:],
            "profile_a__1553__1553": target_aligned[:5] + "A" + target_aligned[6:],
            "profile_a__1540__1540": target_aligned[:6] + "A" + target_aligned[7:],
            "profile_a__1542__1542": target_aligned[:7] + "A" + target_aligned[8:],
        },
    )
    panel_spec = write_panel_spec(
        tmp_path,
        row_source="all_records",
        max_display_rows="all",
        profiles={
            profile_id: {
                "group": "Mestre clade 9",
                "target_label": "Eco1/Ec86 reference",
                "label_template": "c9 node {node} {accession}",
                "label_max_chars": 40,
                "source_manifest_path": str(source_manifest),
            }
        },
    )

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=tmp_path / "visualizations",
            profile_ids=(profile_id,),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            panel_spec_yaml=panel_spec,
        )
    )

    overview_svg = result.profile_alignment_overview_svg_paths[profile_id].read_text(encoding="utf-8")
    assert "all-record overview" in overview_svg
    assert "this panel shows all 5 aligned rows" in overview_svg
    assert "Eco1/Ec86 reference" in overview_svg
    assert "c9 node 1552 fig|158822.8.peg.1905" in overview_svg
    assert "c9 node 1542 fig|1423.436.peg.4365" in overview_svg
    assert "Mestre clade 9" in overview_svg
    html_report = result.html_report_path.read_text(encoding="utf-8")
    assert "all-record whole-alignment overview" in html_report


def test_visualization_overview_rejects_missing_manifest_metadata_for_all_records(tmp_path: Path) -> None:
    alignment_root = tmp_path / "alignments"
    alignment_root.mkdir()
    source_manifest = _write_source_manifest(tmp_path, included_record_ids=("profile_a__1552__1552",))
    profile_id = "profile_a"
    target_aligned = TARGET[:10] + "-" + TARGET[10:]
    write_fasta_records(
        alignment_root / f"{profile_id}.aligned.fasta",
        {
            "target": target_aligned,
            "profile_a__1552__1552": target_aligned,
            "profile_a__1553__1553": target_aligned,
        },
    )
    panel_spec = write_panel_spec(
        tmp_path,
        row_source="all_records",
        max_display_rows="all",
        profiles={
            profile_id: {
                "label_template": "c9 node {node} {accession}",
                "source_manifest_path": str(source_manifest),
            }
        },
    )

    with pytest.raises(ValueError, match="missing source-manifest metadata"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=(profile_id,),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                panel_spec_yaml=panel_spec,
            )
        )


def _write_source_manifest(
    tmp_path: Path,
    *,
    included_record_ids: tuple[str, ...] = (
        "profile_a__1552__1552",
        "profile_a__1553__1553",
        "profile_a__1540__1540",
        "profile_a__1542__1542",
    ),
) -> Path:
    path = tmp_path / "profile_a.source_manifest.yaml"
    accession_by_record_id = {
        "profile_a__1552__1552": "fig|158822.8.peg.1905",
        "profile_a__1553__1553": "fig|1444060.3.peg.4830",
        "profile_a__1540__1540": "fig|1396.1518.peg.4860",
        "profile_a__1542__1542": "fig|1423.436.peg.4365",
    }
    path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "test.source_manifest",
                "included_records": [
                    {
                        "record_id": record_id,
                        "provider_id": "bv_brc_feature_protein_fasta",
                        "accession": accession_by_record_id[record_id],
                    }
                    for record_id in included_record_ids
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
