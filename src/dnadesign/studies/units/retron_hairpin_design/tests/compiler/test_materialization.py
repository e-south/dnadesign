"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_materialization.py

Materialization command and artifact contract tests for the Retron MSD compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.compiler.materialization import materialize_msd_design_artifacts
from dnadesign.studies.units.retron_hairpin_design.compiler.references import compile_msd_design_catalog
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ..support.cli import RUNNER
from ..support.compiler_fixtures import SNAPBACK_FOLDBACK, TETO_PAYLOAD
from ..support.registry import write_minimal_retron_msd_registry
from ..support.viennarna import install_fake_viennarna_python_api


def test_retron_msd_materialize_requires_concrete_sequences(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "requires concrete sequence subcomponents" in payload["error"]
    assert "payload(s): TetR" in payload["error"]
    assert "cap(s): C172" in payload["error"]
    assert "--payload-sequence ID=ACGT" in payload["next_step"]
    assert "explicit 5'->3' cap sequences" in payload["next_step"]


def test_retron_msd_materialize_api_validates_direct_sequence_maps(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    catalog = compile_msd_design_catalog(
        ["pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"],
        study_dir=study_dir,
    )

    with pytest.raises(RetronMsdCompilerError, match="payload_sequences.TetR contains non-DNA bases: N"):
        materialize_msd_design_artifacts(
            catalog,
            out_dir=tmp_path / "sequence_bundle",
            payload_sequences={"TetR": "ACNT"},
            cap_sequences={"C172": SNAPBACK_FOLDBACK},
        )


@pytest.mark.parametrize(
    ("sequence_flag", "first_value", "second_value", "error_text"),
    [
        ("--payload-sequence", "TetR=AAAA", "TetR=CCCC", "Duplicate payload sequence override ID: TetR"),
        ("--cap-sequence", "C172=AAAA", "C172=CCCC", "Duplicate cap sequence override ID: C172"),
    ],
)
def test_retron_msd_materialize_rejects_duplicate_cli_sequence_override_ids(
    tmp_path: Path,
    sequence_flag: str,
    first_value: str,
    second_value: str,
    error_text: str,
) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            sequence_flag,
            first_value,
            sequence_flag,
            second_value,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert error_text in payload["error"]


@pytest.mark.parametrize(
    ("sequence_flag", "override_value", "error_text"),
    [
        ("--payload-sequence", "TetR=ACNT", "payload sequence override TetR contains non-DNA bases: N"),
        ("--cap-sequence", "C172=ACNT", "cap sequence override C172 contains non-DNA bases: N"),
    ],
)
def test_retron_msd_materialize_rejects_invalid_cli_sequence_override_bases(
    tmp_path: Path,
    sequence_flag: str,
    override_value: str,
    error_text: str,
) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            sequence_flag,
            override_value,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert error_text in payload["error"]


def test_retron_msd_materialize_flag_admits_non_ligatable_s0_before_sequence_checks(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGG-RACAG-MXMX",
            "--allow-non-ligatable-s0",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "requires concrete sequence subcomponents" in payload["error"]
    assert "must be scar-compatible" not in payload["error"]


def test_retron_msd_materialize_preserves_construct_label_in_variant_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_viennarna_python_api(tmp_path, monkeypatch)
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    requested_label = "pES-retron-177-msd[TetR]; C172-LCGGG-RACAG-MXMX"
    expected_design_id = "msd-tetr-C172-LCGGG-RACAG-MXMX"
    expected_variant_dirname = f"pES-retron-177__{expected_design_id}"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            requested_label,
            "--allow-non-ligatable-s0",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            "C172=GAGAGACTC",
            "--render-format",
            "png",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    variant = payload["variants"][0]
    assert variant["construct_id"] == "pES-retron-177"
    assert variant["construct_label"] == requested_label
    assert variant["msd_design_id"] == expected_design_id
    assert variant["artifact_bundle"] == f"variants/{expected_variant_dirname}"

    rows = list(
        csv.DictReader(
            (out_dir / "manifest" / "indexes" / "sequence_index.tsv").read_text().splitlines(), delimiter="\t"
        )
    )
    assert rows[0]["construct_label"] == requested_label
    assert rows[0]["msd_design_id"] == expected_design_id

    catalog = json.loads((out_dir / "manifest" / "catalog" / "msd_design_catalog_v1.json").read_text())
    record = catalog["records"][0]
    assert record["construct_label"] == requested_label
    assert record["scar_nick"]["left_base"] == "CGGG"
    assert record["scar_nick"]["profile_s3s2s1s0"] == "MXMX"
    assert record["scar_nick"]["s0_match_required"] is False


def test_retron_msd_materialize_accepts_literal_cap_segment_without_snapback_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_viennarna_python_api(tmp_path, monkeypatch)
    monkeypatch.setenv("DNADESIGN_INKSCAPE", "__must_not_be_used__")
    study_dir = tmp_path / "study"
    compiler_dir = study_dir / "compiler" / "catalog"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "msd_design_registry.yaml").write_text(
        """
contract: retron_msd_design_registry_v1
schema_version: 1
payloads:
  TetR:
    display_name: msd[teto]
caps:
  C26:
    source_construct: retron-26
    display_name: C26 cap source
constructs:
  pES-retron-178:
    scar_nick:
      route_status: note_only
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-178-msd[TetR]; C26-LCAAG-RCTCG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            "C26=AGGC",
            "--render-format",
            "png",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["record_count"] == 1
    variant_dir = out_dir / "variants" / "pES-retron-178__msd-tetr-C26-LCAAG-RCTCG-MXMM"
    assert (variant_dir / "sequences" / "forward.gb").is_file()
    assert (variant_dir / "plots" / "secondary_structure.native.png").is_file()

    catalog = json.loads((out_dir / "manifest" / "catalog" / "msd_design_catalog_v1.json").read_text())
    record = catalog["records"][0]
    flank_5p_len = len("gtcagaaaaaa") + 4
    flank_3p_len = 4 + len("acagtaactcaga")
    unit_len = flank_5p_len + len(TETO_PAYLOAD) + len("AGGC") + len(TETO_PAYLOAD) + flank_3p_len
    assert record["sequence"]["length"] == unit_len

    feature_rows = list(csv.DictReader((variant_dir / "sequences" / "features.csv").read_text().splitlines()))
    annotation_ids = {row["feature_id"] for row in feature_rows if row["feature_kind"] == "annotation"}
    assert annotation_ids == {"stem_base_left", "stem_base_right"}
    assert "Foldback" in {row["display_label"] for row in feature_rows}
    assert not {"snapback_retained_stem", "snapback_cap", "snapback_foldback_return"} & {
        row["feature_id"] for row in feature_rows
    }
    visual_contract = json.loads((variant_dir / "manifest" / "visual" / "sequence_evidence_map_v1.json").read_text())
    labels_by_text = {label["text"]: label for label in visual_contract["meta"]["segment_labels"]}
    assert labels_by_text["Foldback"]["start"] == flank_5p_len + len(TETO_PAYLOAD)
    assert labels_by_text["Foldback"]["end"] == labels_by_text["Foldback"]["start"] + len("AGGC")
    assert not {"Cap", "Foldback stem", "Foldback return"} & set(labels_by_text)
    assert any(
        backdrop["semantic"] == "snapback_foldback_geometry" for backdrop in visual_contract["meta"]["span_backdrops"]
    )

    annotation_manifest = json.loads(
        (
            variant_dir
            / "runtime"
            / "construct"
            / "visual"
            / "viennarna_secondary_structure"
            / "secondary_structure.annotation_manifest.json"
        ).read_text()
    )
    layout_normalization = annotation_manifest["layout_normalization"]
    assert layout_normalization["applied"] is True
    assert layout_normalization["requested_orientation"] == "cap_right"
    sections_by_id = {section["section_id"]: section for section in annotation_manifest["section_annotations"]}
    anchor_section = sections_by_id[layout_normalization["anchor"]]
    assert anchor_section["label"] == "Foldback"
    assert "snapback_foldback_geometry" in anchor_section["semantic_tokens"]
    assert anchor_section["section_kind"] == "cap_foldback"

    annotated_svg = (
        variant_dir
        / "runtime"
        / "construct"
        / "visual"
        / "viennarna_secondary_structure"
        / "secondary_structure.annotated.svg"
    ).read_text()
    composition_overview_svg = (variant_dir / "plots" / "composition_overview.svg").read_text()
    assert 'data-dnadesign-orientation="cap_right"' in annotated_svg
    assert 'data-dnadesign-section-label="Foldback"' in annotated_svg
    assert 'data-dnadesign-source-orientation="cap_right"' in composition_overview_svg
    assert 'data-dnadesign-section-label="Foldback"' in composition_overview_svg


def test_retron_msd_materialize_requires_viennarna_for_deliverable_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    monkeypatch.setitem(sys.modules, "RNA", None)

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={SNAPBACK_FOLDBACK}",
            "--render-format",
            "png",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Folding backend Python module 'RNA' is not available" in payload["error"]
    assert "Retron MSD GenBank/structure/review deliverables require folding status ok" in payload["next_step"]


def test_retron_msd_materialize_writes_single_unit_genbank_png_and_reverse_complement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_viennarna_python_api(tmp_path, monkeypatch)
    monkeypatch.setenv("DNADESIGN_INKSCAPE", "__must_not_be_used__")
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={SNAPBACK_FOLDBACK}",
            "--render-format",
            "png",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["record_count"] == 1
    assert sorted(item.name for item in out_dir.iterdir()) == ["README.md", "manifest", "variants"]
    assert sorted(item.name for item in (out_dir / "manifest").iterdir()) == ["bundle", "catalog", "configs", "indexes"]
    assert Path(payload["sequence_manifest_path"]) == out_dir / "manifest" / "bundle" / "sequence_manifest.json"
    assert Path(payload["sequence_index_path"]) == out_dir / "manifest" / "indexes" / "sequence_index.tsv"
    assert Path(payload["variants_dir"]) == out_dir / "variants"
    assert Path(payload["composition_configs_dir"]) == out_dir / "manifest" / "configs" / "composition"
    assert (out_dir / "manifest" / "bundle" / "manifest.json").is_file()
    assert (out_dir / "manifest" / "catalog" / "references").is_dir()
    expected_variant_dirname = "pES-retron-177__msd-tetr-C172-LCGGT-RACAG-MXMM"
    assert (
        out_dir / "manifest" / "configs" / "composition" / f"{expected_variant_dirname}.linear_ssdna_composition.yaml"
    ).is_file()
    assert payload["finder_open"] == f"open {out_dir}"
    assert "Single-unit MSD sequence bundle emitted" in payload["next_step"]

    variant = payload["variants"][0]
    variant_dir = out_dir / "variants" / expected_variant_dirname
    expected_variant = Path("variants") / expected_variant_dirname
    genbank_path = variant_dir / "sequences" / "forward.gb"
    revcom_genbank_path = variant_dir / "sequences" / "reverse_complement.gb"
    features_path = variant_dir / "sequences" / "features.csv"
    composition_overview_svg_path = variant_dir / "plots" / "composition_overview.svg"
    composition_overview_png_path = variant_dir / "plots" / "composition_overview.png"
    secondary_structure_native_png_path = variant_dir / "plots" / "secondary_structure.native.png"
    construct_bundle = variant_dir / "runtime" / "construct"
    assert sorted(item.name for item in variant_dir.iterdir()) == ["manifest", "plots", "runtime", "sequences"]
    assert sorted(item.name for item in (variant_dir / "manifest").iterdir()) == [
        "composition",
        "construct",
        "folding",
        "provenance",
        "reviews",
        "visual",
    ]
    assert sorted(item.name for item in (variant_dir / "plots").iterdir()) == [
        "composition_overview.png",
        "composition_overview.svg",
        "secondary_structure.native.png",
    ]
    assert variant["unit_count"] == 1
    assert Path(variant["genbank"]) == expected_variant / "sequences" / "forward.gb"
    assert Path(variant["reverse_complement_genbank"]) == expected_variant / "sequences" / "reverse_complement.gb"
    assert Path(variant["composition_overview_svg"]) == expected_variant / "plots" / "composition_overview.svg"
    assert Path(variant["composition_overview_png"]) == expected_variant / "plots" / "composition_overview.png"
    assert Path(variant["secondary_structure_native_png"]) == (
        expected_variant / "plots" / "secondary_structure.native.png"
    )
    assert "component_span_png" not in variant
    assert "folding_png" not in variant
    assert "combined_plot_png" not in variant
    assert genbank_path.is_file()
    assert revcom_genbank_path.is_file()
    assert features_path.is_file()
    assert composition_overview_svg_path.is_file()
    assert composition_overview_png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert secondary_structure_native_png_path.is_file()
    assert (construct_bundle / "manifest.json").is_file()
    assert (construct_bundle / "manifest" / "composition" / "assembled_sequence.json").is_file()
    assert (construct_bundle / "manifest" / "provenance" / "provenance.json").is_file()
    assert (construct_bundle / "manifest" / "reviews" / "composition_review_svg_v1.json").is_file()
    assert (construct_bundle / "manifest" / "visual" / "sequence_evidence_map_v1.json").is_file()
    assert (construct_bundle / "folding" / "secondary_structure_prediction_v1.json").is_file()
    assert (
        construct_bundle / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg"
    ).is_file()
    assert (construct_bundle / "visual" / "viennarna_secondary_structure" / "secondary_structure.native.svg").is_file()
    assert (construct_bundle / "visual" / "reviews" / "composition_overview.svg").is_file()
    assert (construct_bundle / "visual" / "reviews" / "composition_overview.png").is_file()
    assert (construct_bundle / "visual" / "reviews" / "composition_review_svg_v1.json").is_file()
    assert (variant_dir / "manifest" / "folding" / "secondary_structure_prediction_v1.json").is_file()
    assert (variant_dir / "manifest" / "reviews" / "composition_review_svg_v1.json").is_file()
    assert (variant_dir / "manifest" / "visual" / "secondary_structure" / "native.svg").is_file()
    assert composition_overview_svg_path.stat().st_size > 0
    assert composition_overview_png_path.stat().st_size > 0
    assert secondary_structure_native_png_path.stat().st_size > 0
    visual_contract = json.loads((variant_dir / "manifest" / "visual" / "sequence_evidence_map_v1.json").read_text())
    assert visual_contract["meta"]["scar_nick"] == {
        "left_base": "CGGT",
        "right_base": "ACAG",
        "profile_s3s2s1s0": "MXMM",
    }
    annotated_structure_svg = (
        construct_bundle / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg"
    ).read_text(encoding="utf-8")
    composition_overview_svg = composition_overview_svg_path.read_text(encoding="utf-8")
    assert "mismatch profile MXMM" in annotated_structure_svg
    assert "mismatch profile MXMM" in composition_overview_svg
    assert "Cap AGA (3 nt)" in annotated_structure_svg
    assert "Cap AGA (3 nt)" in composition_overview_svg
    assert "Cap Geometry" not in annotated_structure_svg
    assert "Cap Geometry" not in composition_overview_svg
    assert "Foldback stem" not in composition_overview_svg
    assert "Foldback return" not in composition_overview_svg
    assert 'data-dnadesign-section-label="Foldback"' not in composition_overview_svg
    assert 'data-dnadesign-source-svg="secondary_structure.annotated.svg"' in composition_overview_svg
    assert 'data-dnadesign-source-orientation="cap_right"' in composition_overview_svg
    fills_by_semantic = {
        item["semantic"]: item["fill"]
        for item in visual_contract["meta"]["span_backdrops"]
        if item["semantic"]
        in {
            "flank_5p",
            "payload_primary",
            "snapback_cap",
            "payload_complement",
            "flank_3p",
        }
    }
    assert set(fills_by_semantic) == {
        "flank_5p",
        "payload_primary",
        "snapback_cap",
        "payload_complement",
        "flank_3p",
    }
    assert len(set(fills_by_semantic.values())) == len(fills_by_semantic)
    edge_colors_by_semantic = {
        item["semantic"]: item["edge_color"]
        for item in visual_contract["meta"]["span_backdrops"]
        if item["semantic"]
        in {
            "flank_5p",
            "payload_primary",
            "snapback_cap",
            "payload_complement",
            "flank_3p",
        }
    }
    assert set(edge_colors_by_semantic) == set(fills_by_semantic)
    assert len(set(edge_colors_by_semantic.values())) == len(edge_colors_by_semantic)
    assert 'id="dnadesign-secondary-structure-semantic-edges"' in composition_overview_svg
    for semantic, edge_color in edge_colors_by_semantic.items():
        assert f'data-dnadesign-edge-semantic="{semantic}"' in composition_overview_svg
        assert f"stroke: {edge_color}" in composition_overview_svg
    for semantic in ["stem_base_left", "stem_base_right"]:
        assert f'data-dnadesign-edge-semantic="{semantic}"' in composition_overview_svg
        assert f"stroke: {visual_contract['meta']['component_palette'][semantic]}" in composition_overview_svg

    rows = list(
        csv.DictReader(
            (out_dir / "manifest" / "indexes" / "sequence_index.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    assert rows[0]["unit_count"] == "1"
    assert rows[0]["genbank"] == (expected_variant / "sequences" / "forward.gb").as_posix()
    assert (
        rows[0]["reverse_complement_genbank"] == (expected_variant / "sequences" / "reverse_complement.gb").as_posix()
    )
    assert rows[0]["composition_overview_svg"] == (expected_variant / "plots" / "composition_overview.svg").as_posix()
    assert rows[0]["composition_overview_png"] == (expected_variant / "plots" / "composition_overview.png").as_posix()
    assert (
        rows[0]["secondary_structure_native_png"]
        == (expected_variant / "plots" / "secondary_structure.native.png").as_posix()
    )
    assert "component_span_png" not in rows[0]
    assert "folding_png" not in rows[0]
    assert "combined_plot_png" not in rows[0]
    assert rows[0]["folding_status"] == "ok"
    assert rows[0]["finder_reveal"].startswith("open -R ")

    catalog = json.loads((out_dir / "manifest" / "catalog" / "msd_design_catalog_v1.json").read_text(encoding="utf-8"))
    record = catalog["records"][0]
    flank_5p_len = len("gtcagaaaaaa") + 4
    flank_3p_len = 4 + len("acagtaactcaga")
    unit_len = flank_5p_len + len(TETO_PAYLOAD) + len(SNAPBACK_FOLDBACK) + len(TETO_PAYLOAD) + flank_3p_len
    assert record["sequence"]["length"] == unit_len
    assert record["source"]["dnadesign_bundle"] == expected_variant.as_posix()
    assert record["artifacts"]["genbank"] == (expected_variant / "sequences" / "forward.gb").as_posix()
    assert (
        record["artifacts"]["reverse_complement_genbank"]
        == (expected_variant / "sequences" / "reverse_complement.gb").as_posix()
    )
    assert (
        record["artifacts"]["composition_overview_svg"]
        == (expected_variant / "plots" / "composition_overview.svg").as_posix()
    )
    assert (
        record["artifacts"]["composition_overview_png"]
        == (expected_variant / "plots" / "composition_overview.png").as_posix()
    )
    assert (
        record["artifacts"]["secondary_structure_native_png"]
        == (expected_variant / "plots" / "secondary_structure.native.png").as_posix()
    )
    assert "component_span_png" not in record["artifacts"]
    assert "folding_png" not in record["artifacts"]
    assert "combined_plot_png" not in record["artifacts"]

    genbank_record = next(SeqIO.parse(genbank_path, "genbank"))
    revcom_record = next(SeqIO.parse(revcom_genbank_path, "genbank"))
    assert str(revcom_record.seq).upper() == str(genbank_record.seq.reverse_complement()).upper()
    features_by_label = {
        feature.qualifiers["label"][0]: feature for feature in genbank_record.features if "label" in feature.qualifiers
    }
    assert "payload_complement" not in features_by_label
    assert "snapback_cap" not in features_by_label
    assert not any("copy 0" in label for label in features_by_label)
    assert {"5' Flanking", "msd[teto]", "Foldback", "msd[teto] complement"} <= set(features_by_label)
    features_by_id = {
        feature.qualifiers["dnadesign_feature_id"][0]: feature
        for feature in genbank_record.features
        if "dnadesign_feature_id" in feature.qualifiers
    }
    payload_complement = features_by_id["payload_complement"]
    assert int(payload_complement.location.start) == flank_5p_len + len(TETO_PAYLOAD) + len(SNAPBACK_FOLDBACK)
    assert int(payload_complement.location.end) == (
        flank_5p_len + len(TETO_PAYLOAD) + len(SNAPBACK_FOLDBACK) + len(TETO_PAYLOAD)
    )
    assert payload_complement.location.strand == -1
    assert payload_complement.qualifiers["label"] == ["msd[teto] complement"]
    assert payload_complement.qualifiers["dnadesign_copy_index"] == ["0"]
    assert payload_complement.qualifiers["dnadesign_transform"] == ["reverse_complement"]
    assert str(payload_complement.extract(genbank_record.seq)).upper() == TETO_PAYLOAD.upper()

    feature_rows = list(csv.DictReader(features_path.read_text(encoding="utf-8").splitlines()))
    annotation_ids = {row["feature_id"] for row in feature_rows if row["feature_kind"] == "annotation"}
    assert annotation_ids == {
        "stem_base_left",
        "stem_base_right",
        "snapback_retained_stem",
        "snapback_cap",
        "snapback_foldback_return",
    }
    assert {
        "5' Flanking",
        "Left Base",
        "msd[teto]",
        "Foldback",
        "Foldback stem",
        "Cap",
        "Foldback return",
        "msd[teto] complement",
        "Right Base",
        "3' Flanking",
    } <= {row["display_label"] for row in feature_rows}
    assert not {"flank_5p", "payload_primary", "snapback_foldback_geometry", "payload_complement"} & {
        row["display_label"] for row in feature_rows
    }
    duplicate_display_spans = [
        key
        for key, count in Counter(
            (row["display_label"], row["start_0"], row["end_0"], row["strand"]) for row in feature_rows
        ).items()
        if count > 1
    ]
    assert duplicate_display_spans == []
    payload_complement_rows = [
        row for row in feature_rows if row["feature_kind"] == "segment" and row["feature_id"] == "payload_complement"
    ]
    assert len(payload_complement_rows) == 1
    assert payload_complement_rows[0]["display_label"] == "msd[teto] complement"
    assert payload_complement_rows[0]["strand"] == "-1"
    assert payload_complement_rows[0]["source_segment_id"] == "payload_primary"
    assert payload_complement_rows[0]["transform_kind"] == "reverse_complement"
    assert payload_complement_rows[0]["genbank_location"].startswith("complement(")
    assert payload_complement_rows[0]["sequence"].upper() == str(Seq(TETO_PAYLOAD).reverse_complement()).upper()

    forward_by_key = {
        (
            feature.qualifiers["dnadesign_feature_kind"][0],
            feature.qualifiers["dnadesign_feature_id"][0],
        ): feature
        for feature in genbank_record.features
        if "dnadesign_feature_kind" in feature.qualifiers and "dnadesign_feature_id" in feature.qualifiers
    }
    revcom_by_key = {
        (
            feature.qualifiers["dnadesign_feature_kind"][0],
            feature.qualifiers["dnadesign_feature_id"][0],
        ): feature
        for feature in revcom_record.features
        if "dnadesign_feature_kind" in feature.qualifiers and "dnadesign_feature_id" in feature.qualifiers
    }
    assert set(forward_by_key) == set(revcom_by_key)
    for key, feature in forward_by_key.items():
        revcom_feature = revcom_by_key[key]
        assert int(revcom_feature.location.start) == unit_len - int(feature.location.end)
        assert int(revcom_feature.location.end) == unit_len - int(feature.location.start)
        assert revcom_feature.location.strand == -feature.location.strand
        assert revcom_feature.qualifiers["label"] == feature.qualifiers["label"]
        assert revcom_feature.qualifiers["dnadesign_orientation"] == ["reverse_complement"]


def test_retron_msd_materialize_rejects_repeat_count_flag(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    help_result = RUNNER.invoke(app, ["materialize", "--help"])

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            (tmp_path / "sequence_bundle").as_posix(),
            "--repeat-count",
            "8",
        ],
    )

    assert help_result.exit_code == 0
    assert "--repeat-count" not in help_result.output
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_retron_msd_materialize_refuses_flat_legacy_sequence_layout(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    out_dir.mkdir()
    (out_dir / "sequence_manifest.json").write_text("{}\n", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={SNAPBACK_FOLDBACK}",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Unexpected MSD materialize output entries" in payload["error"]
    assert "fresh --out-dir" in payload["next_step"]


def test_retron_msd_materialize_refuses_stale_legacy_plot_deliverables(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    stale_plots_dir = out_dir / "variants" / "pES-retron-177__msd-tetr-C172-LCGGT-RACAG-MXMM" / "plots"
    stale_plots_dir.mkdir(parents=True)
    (stale_plots_dir / "component_span_and_folding.png").write_text("stale\n", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={SNAPBACK_FOLDBACK}",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Stale MSD plot output" in payload["error"]
    assert "component_span_and_folding.png" in payload["error"]
    assert "archive/remove stale plot artifacts" in payload["next_step"]


def test_retron_msd_materialize_refuses_stale_variant_sequence_outputs(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    stale_sequences_dir = out_dir / "variants" / "pES-retron-177__msd-tetr-C172-LCGGT-RACAG-MXMM" / "sequences"
    stale_sequences_dir.mkdir(parents=True)
    (stale_sequences_dir / "legacy_sequence.gb").write_text("stale\n", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "materialize",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--payload-sequence",
            f"TetR={TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={SNAPBACK_FOLDBACK}",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Stale MSD sequence artifact output" in payload["error"]
    assert "legacy_sequence.gb" in payload["error"]
    assert "archive/remove stale sequence artifacts" in payload["next_step"]
