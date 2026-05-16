"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py

Runtime tests for generic linear ssDNA composition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import io
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from Bio import SeqIO
from Bio.GenBank import BiopythonParserWarning

import dnadesign.baserender as baserender
from dnadesign.construct.src.composition import run_linear_ssdna_composition
from dnadesign.construct.src.composition_review import publish_composition_review_svg
from dnadesign.construct.src.errors import ValidationError

_DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
RETRON43_UNIT = "gtcagaaaaaaCAAGtccctatcagtgatagagatCCTCAGcccGCTGAGGatctctatcactgatagggaCTCGacagtaactcaga"
RETRON43_UNIT_COMPLEMENT = RETRON43_UNIT.translate(_DNA_COMPLEMENT)


def _png_delta_without_secondary_structure_nucleotides(review_svg_path: Path, review_png_path: Path) -> tuple[int, int]:
    import vl_convert as vlc
    from PIL import Image, ImageChops

    root = ET.parse(review_svg_path).getroot()
    parent_by_child = {child: parent for parent in root.iter() for child in list(parent)}
    removed_count = 0
    for node in list(root.iter()):
        if not node.tag.endswith("text"):
            continue
        if "nucleotide" not in str(node.attrib.get("class", "")).split():
            continue
        current = parent_by_child.get(node)
        while current is not None and current.attrib.get("data-dnadesign-panel") != "secondary_structure":
            current = parent_by_child.get(current)
        if current is None:
            continue
        parent_by_child[node].remove(node)
        removed_count += 1

    stripped_png = vlc.svg_to_png(ET.tostring(root, encoding="unicode"), scale=3.0, ppi=216.0)
    actual = Image.open(review_png_path).convert("RGBA")
    stripped = Image.open(io.BytesIO(stripped_png)).convert("RGBA")
    assert actual.size == stripped.size
    delta = ImageChops.difference(actual, stripped).convert("L")
    histogram = delta.histogram()
    changed_pixels = sum(histogram[11:])
    return removed_count, changed_pixels


def _write_retron43_config(
    tmp_path: Path,
    *,
    bad_complement: bool = False,
    folding_backend_name: str = "ViennaRNA",
    folding_executable: str | None = None,
    folding_interface: str = "cli",
    folding_python_module: str | None = None,
    folding_required: bool = False,
    folding_scope: str = "canonical_component_unit",
    emit_structure_plot: bool = False,
    viennarna_layout_algorithm: str = "naview",
) -> Path:
    complement = "aaaaaaaaaaaaaaaaaaa" if bad_complement else "tctctatcactgataggga"
    folding_block = ""
    if folding_executable is not None or folding_python_module is not None:
        folding_block = f"""
folding:
  enabled: true
  required: {"true" if folding_required else "false"}
  scope: {folding_scope}
  backend:
    name: {folding_backend_name}
    interface: {folding_interface}
    executable: {folding_executable}
    python_module: {folding_python_module or ""}
    backend_contract: secondary_structure_prediction_v1
    parameters:
      temperature_c: 37.0
  dna_policy:
    mode: convert_t_to_u_for_rna_backend
"""
    visual_emit = "    - sequence_evidence_map_v1\n"
    visual_plot_block = ""
    if emit_structure_plot:
        visual_emit += "    - viennarna_secondary_structure_svg_v1\n"
        visual_plot_block = f"""  viennarna_structure_plot:
    layout_algorithm: {viennarna_layout_algorithm}
"""
    config_path = tmp_path / "retron43_teto_manual_x8.yaml"
    config_path.write_text(
        f"""
contract: linear_ssdna_composition_v1
schema_version: 1
composition_id: retron43_teto_manual_x8
alphabet: dna
topology: linear_ssdna
coordinate_system: zero_based_half_open
canonicalization:
  compare_sequences_case_insensitive: true
  output_sequence_preserves_case: true
units:
  - unit_id: retron43_teto_unit
    repeat_count: 8
    segments:
      - segment_id: flank_5p
        role: flank_5p
        sequence: gtcagaaaaaaCAAG
        source:
          kind: literal
          label: manual_retron43_example
      - segment_id: payload_primary
        role: payload_primary
        sequence: tccctatcagtgatagaga
        source:
          kind: literal
          label: manual_teto_payload
      - segment_id: snapback_cap_segment
        role: snapback_cap_segment
        sequence: tCCTCAGcccGCTGAGGa
        source:
          kind: literal
          label: manual_snapback_43_cap
      - segment_id: payload_complement
        role: payload_complement
        sequence: {complement}
        transform:
          kind: reverse_complement
          source_segment_id: payload_primary
          assert_expected_sequence: true
        source:
          kind: derived
          from_segment_id: payload_primary
      - segment_id: flank_3p
        role: flank_3p
        sequence: CTCGacagtaactcaga
        source:
          kind: literal
          label: manual_retron43_example
    annotations:
      - annotation_id: stem_base_left
        role: stem_base_left
        location:
          basis: segment
          segment_id: flank_5p
          start: 11
          end: 15
      - annotation_id: teto_primary
        role: payload
        semantic_label: TetO
        location:
          basis: segment
          segment_id: payload_primary
          start: 0
          end: 19
      - annotation_id: snapback_cap
        role: snapback_cap
        location:
          basis: segment
          segment_id: snapback_cap_segment
          start: 0
          end: 18
      - annotation_id: teto_complement
        role: payload_complement
        semantic_label: TetO_reverse_complement
        location:
          basis: segment
          segment_id: payload_complement
          start: 0
          end: 19
      - annotation_id: stem_base_right
        role: stem_base_right
        location:
          basis: segment
          segment_id: flank_3p
          start: 0
          end: 4
    assertions:
      - assertion_id: payload_rc
        kind: reverse_complement
        left_segment_id: payload_primary
        right_segment_id: payload_complement
        severity: error
qa:
  require_no_unknown_bases: true
  allow_degenerate_bases: false
  require_segment_span_coverage: true
  require_non_overlapping_physical_segments: true
  require_annotation_bounds: true
  require_declared_transform_checks: true
  allow_cross_copy_intended_pairings: false
{folding_block}
visual:
  emit:
{visual_emit}{visual_plot_block}
  display_profile:
    title: Retron 43 TetO x8
    component_labels:
      flank_5p: "5' flank"
      payload_primary: TetO primary
      snapback_cap_segment: Snapback cap
      payload_complement: TetO complement
      flank_3p: "3' flank"
    annotation_labels:
      stem_base_left: Left stem base
      stem_base_right: Right stem base
    component_hues:
      flank_5p: "#4C78A8"
      flank_3p: "#72B7B2"
      payload_primary: "#F58518"
      payload_complement: "#E45756"
      snapback_cap_segment: "#54A24B"
      stem_base_left: "#B279A2"
      stem_base_right: "#9D755D"
    component_styles:
      flank_5p:
        fill: "#CBD5E1"
        alpha: 0.70
        edge_color: "#94A3B8"
      payload_primary:
        fill: "#34D399"
        alpha: 0.58
        edge_color: "#059669"
      snapback_cap_segment:
        fill: "#F472B6"
        alpha: 0.56
        edge_color: "#DB2777"
      payload_complement:
        fill: "#60A5FA"
        alpha: 0.58
        edge_color: "#2563EB"
      flank_3p:
        fill: "#CBD5E1"
        alpha: 0.70
        edge_color: "#94A3B8"
    base_highlight_color: "#111827"
benchling_export:
  enabled: true
  primary_format: genbank
  sidecars: [fasta, features_csv]
output:
  artifact_bundle: artifacts/retron43_teto_manual_x8
  usr:
    enabled: false
""",
        encoding="utf-8",
    )
    return config_path


def test_run_linear_ssdna_composition_writes_retron43_bundle(tmp_path: Path) -> None:
    config_path = _write_retron43_config(tmp_path)
    bundle = tmp_path / "artifacts" / "retron43_teto_manual_x8"
    stale_nested_plot = bundle / "folding" / "src" / "dnadesign" / "stale_plot.json"
    stale_nested_plot.parent.mkdir(parents=True)
    stale_nested_plot.write_text("{}\n", encoding="utf-8")
    stale_visual_contract = bundle / "visual" / "contracts" / "component_span_qa_sequence_evidence_map_v1.json"
    stale_visual_contract.parent.mkdir(parents=True)
    stale_visual_contract.write_text("{}\n", encoding="utf-8")

    result = run_linear_ssdna_composition(config_path)

    assert result.composition_id == "retron43_teto_manual_x8"
    assert result.sequence_length == 704
    assert result.artifact_bundle == bundle

    assembled = json.loads((bundle / "assembled_sequence.json").read_text(encoding="utf-8"))
    assert assembled["sequence"]["length"] == 704
    assert assembled["sequence"]["sequence"] == RETRON43_UNIT * 8
    assert assembled["unit_copies"][0]["span"] == {"start": 0, "end": 88}
    assert assembled["unit_copies"][7]["span"] == {"start": 616, "end": 704}
    assert assembled["assertions"] == [
        {"assertion_id": "payload_rc", "kind": "reverse_complement", "severity": "error", "status": "pass"}
    ]

    segment_spans = json.loads((bundle / "segment_spans.json").read_text(encoding="utf-8"))
    assert segment_spans["segments"][0] == {
        "copy_index": 0,
        "unit_id": "retron43_teto_unit",
        "segment_id": "flank_5p",
        "role": "flank_5p",
        "start": 0,
        "end": 15,
        "sequence": "gtcagaaaaaaCAAG",
    }
    assert segment_spans["segments"][-1]["start"] == 687
    assert segment_spans["segments"][-1]["end"] == 704

    annotation_spans = json.loads((bundle / "annotation_spans.json").read_text(encoding="utf-8"))
    assert annotation_spans["annotations"][4]["annotation_id"] == "stem_base_right"
    assert annotation_spans["annotations"][4]["start"] == 71
    assert annotation_spans["annotations"][9]["annotation_id"] == "stem_base_right"
    assert annotation_spans["annotations"][9]["start"] == 159

    validation = json.loads((bundle / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["status"] == "ok"
    assert validation["errors"] == []

    fasta = (bundle / "sequence.fa").read_text(encoding="utf-8")
    assert fasta.startswith(">retron43_teto_manual_x8 length=704 topology=linear_ssdna\n")
    assert RETRON43_UNIT in fasta

    with (bundle / "features.csv").open(newline="", encoding="utf-8") as handle:
        features = list(csv.DictReader(handle))
    assert features[0]["feature_id"] == "flank_5p"
    assert features[0]["display_label"] == "5' flank"
    assert features[0]["genbank_location"] == "1..15"
    assert features[1]["feature_id"] == "payload_primary"
    assert features[1]["display_label"] == "TetO primary"
    assert features[1]["genbank_location"] == "16..34"
    features_by_id = {(row["feature_kind"], row["feature_id"], int(row["copy_index"])): row for row in features}
    payload_complement_row = features_by_id[("segment", "payload_complement", 0)]
    assert payload_complement_row["strand"] == "-1"
    assert payload_complement_row["source_segment_id"] == "payload_primary"
    assert payload_complement_row["transform_kind"] == "reverse_complement"
    assert payload_complement_row["genbank_location"] == "complement(53..71)"
    teto_complement_row = features_by_id[("annotation", "teto_complement", 0)]
    assert teto_complement_row["strand"] == "-1"
    assert teto_complement_row["source_segment_id"] == "payload_primary"
    assert teto_complement_row["genbank_location"] == "complement(53..71)"

    genbank = (bundle / "sequence.gb").read_text(encoding="utf-8")
    assert "LOCUS       retron43_teto_manual_x8 704 bp ss-DNA linear SYN" in genbank
    assert '/label="Left stem base"' in genbank
    assert '/dnadesign_feature_id="stem_base_left"' in genbank
    assert '/dnadesign_copy_index="0"' in genbank
    assert "12..15" in genbank
    assert "complement(53..71)" in genbank
    with warnings.catch_warnings():
        warnings.simplefilter("error", BiopythonParserWarning)
        genbank_record = SeqIO.read(bundle / "sequence.gb", "genbank")
    assert str(genbank_record.seq).upper() == (RETRON43_UNIT * 8).upper()
    genbank_features = {
        (feature.qualifiers["label"][0], feature.qualifiers["dnadesign_copy_index"][0]): feature
        for feature in genbank_record.features
    }
    payload_primary = genbank_features[("TetO primary", "0")]
    assert int(payload_primary.location.start) == 15
    assert int(payload_primary.location.end) == 34
    assert payload_primary.location.strand == 1
    assert payload_primary.qualifiers["dnadesign_feature_id"] == ["payload_primary"]
    assert payload_primary.qualifiers["dnadesign_copy_index"] == ["0"]
    payload_complement = genbank_features[("TetO complement", "0")]
    assert int(payload_complement.location.start) == 52
    assert int(payload_complement.location.end) == 71
    assert payload_complement.location.strand == -1
    assert payload_complement.qualifiers["dnadesign_feature_id"] == ["payload_complement"]
    assert payload_complement.qualifiers["dnadesign_copy_index"] == ["0"]
    assert str(payload_complement.extract(genbank_record.seq)).upper() == "TCCCTATCAGTGATAGAGA"
    teto_complement = genbank_features[("TetO reverse complement", "0")]
    assert int(teto_complement.location.start) == 52
    assert int(teto_complement.location.end) == 71
    assert teto_complement.location.strand == -1
    assert teto_complement.qualifiers["dnadesign_feature_id"] == ["teto_complement"]
    assert teto_complement.qualifiers["dnadesign_copy_index"] == ["0"]
    assert str(teto_complement.extract(genbank_record.seq)).upper() == "TCCCTATCAGTGATAGAGA"

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    genbank_hint = manifest["operator_hints"]["genbank"]
    assert genbank_hint["path"] == "sequence.gb"
    assert genbank_hint["macos_finder_reveal"] == f"open -R {bundle / 'sequence.gb'}"

    visual = json.loads((bundle / "visual" / "sequence_evidence_map_v1.json").read_text(encoding="utf-8"))
    assert visual["contract_kind"] == "sequence_evidence_map_v1"
    assert visual["state_id"] == "retron43_teto_manual_x8.component_span_qa"
    assert visual["topology_kind"] == "linear_ssdna"
    assert visual["primary_sequence"] == RETRON43_UNIT
    assert visual["complement_sequence"] == RETRON43_UNIT_COMPLEMENT
    assert visual["display"]["title"] == "Retron 43 TetO x8 component span QA"
    assert visual["meta"]["visual_scope"]["mode"] == "canonical_component_span_qa"
    assert visual["meta"]["visual_scope"]["source_sequence_length"] == 704
    assert visual["meta"]["visual_scope"]["representative_copy_count"] == 1
    assert visual["meta"]["unit_copies"] == [
        {
            "copy_index": 0,
            "unit_id": "retron43_teto_unit",
            "span": {"start": 0, "end": 88},
            "source_span": {"start": 0, "end": 88},
        }
    ]
    assert [(owner["start"], owner["end"], owner["short_label"]) for owner in visual["owners"]] == [
        (0, 15, ""),
        (15, 34, ""),
        (34, 52, ""),
        (52, 71, ""),
        (71, 88, ""),
    ]
    assert [(tag["start"], tag["end"], tag["short_label"]) for tag in visual["effect_tags"]] == [
        (11, 15, ""),
        (71, 75, ""),
    ]
    assert visual["boundaries"] == []
    assert len(visual["pairings"]) == 1
    assert {label["text"] for label in visual["meta"]["segment_labels"]} == {
        "5' flank",
        "TetO primary",
        "Snapback cap",
        "TetO complement",
        "3' flank",
        "Left stem base",
        "Right stem base",
    }
    assert visual["meta"]["segment_label_gap_px"] == 6.0
    assert visual["meta"]["segment_label_tier_gap_px"] == 10.0
    assert visual["meta"]["interval_annotation_policy"] == "span_backdrops_only"
    assert visual["meta"]["render_pairing_links"] is False
    assert visual["meta"]["row_labels"] == {"primary": "Top", "complement": "Bottom"}
    stem_base_indices = [11, 12, 13, 14, 71, 72, 73, 74]
    assert visual["meta"]["base_highlights"] == {
        "primary": stem_base_indices,
        "complement": stem_base_indices,
    }
    assert visual["meta"]["base_highlight_color"] == {
        "primary": "#111827",
        "complement": "#111827",
    }
    assert [
        (
            backdrop["semantic"],
            backdrop["start"],
            backdrop["end"],
            backdrop["cover_rows"],
            backdrop["alpha"],
        )
        for backdrop in visual["meta"]["span_backdrops"]
    ] == [
        ("flank_5p", 0, 15, "both", 0.70),
        ("payload_primary", 15, 34, "both", 0.58),
        ("snapback_cap_segment", 34, 52, "both", 0.56),
        ("payload_complement", 52, 71, "both", 0.58),
        ("flank_3p", 71, 88, "both", 0.70),
    ]
    labels_by_text = {label["text"]: label for label in visual["meta"]["segment_labels"]}
    assert labels_by_text["TetO primary"]["label_side"] == "above"
    assert labels_by_text["Left stem base"]["label_side"] == "below"
    assert "component_span_qa_visual_contract" not in assembled["artifacts"]
    assert not (bundle / "folding" / "src").exists()
    assert not (bundle / "visual" / "contracts" / "component_span_qa_sequence_evidence_map_v1.json").exists()


def test_run_linear_ssdna_composition_writes_advisory_folding_artifacts_when_backend_missing(tmp_path: Path) -> None:
    config_path = _write_retron43_config(
        tmp_path,
        folding_executable="definitely-missing-rnafold-for-dnadesign-test",
    )

    result = run_linear_ssdna_composition(config_path)

    bundle = result.artifact_bundle
    request_path = bundle / "folding" / "secondary_structure_prediction_request_v1.yaml"
    prediction_path = bundle / "folding" / "secondary_structure_prediction_v1.json"
    preflight_path = bundle / "folding" / "folding_preflight.json"
    assert request_path.is_file()
    assert prediction_path.is_file()
    assert preflight_path.is_file()

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert request["input"]["sequence_artifact"] == "secondary_structure_input_sequence.json"
    assert request["input"]["sequence_id"] == "retron43_teto_manual_x8.component_span_qa"
    assert request["scope"] == {"mode": "canonical_component_unit"}
    assert preflight["status"] == "warning_optional_missing"
    assert prediction["status"] == "warning_optional_missing"
    assert prediction["input"]["length"] == 88
    assert prediction["input"]["sequence_sha256"] != result.sequence_sha256
    assert manifest["artifacts"]["folding_request"] == "folding/secondary_structure_prediction_request_v1.yaml"
    assert manifest["artifacts"]["folding_prediction"] == "folding/secondary_structure_prediction_v1.json"
    assert "viennarna_structure_plot" not in manifest["artifacts"]


def test_run_linear_ssdna_composition_fails_when_required_folding_backend_missing(tmp_path: Path) -> None:
    config_path = _write_retron43_config(
        tmp_path,
        folding_executable="definitely-missing-rnafold-for-dnadesign-test",
        folding_required=True,
    )

    with pytest.raises(ValidationError, match="folding failed"):
        run_linear_ssdna_composition(config_path)


def test_run_linear_ssdna_composition_can_use_viennarna_python_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.2"

class fold_compound:
    def __init__(self, sequence):
        self.sequence = sequence

    def mfe(self):
        return "." * len(self.sequence), -7.04

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_layout_circular(structure):
    return {"layout": "circular", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    if "U" in sequence or "T" not in sequence:
        return 0
    if layout != {"layout": "circular", "structure": structure}:
        return 0
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="240" height="80">\\n')
        handle.write('<g id="pairs"></g>\\n')
        handle.write('<g id="seq">\\n')
        for index, base in enumerate(sequence):
            handle.write(f'<text class="nucleotide" x="{index}" y="0">{base}</text>\\n')
        handle.write('</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    config_path = _write_retron43_config(
        tmp_path,
        folding_backend_name="ViennaRNA",
        folding_interface="python_api",
        folding_python_module="RNA",
        emit_structure_plot=True,
        viennarna_layout_algorithm="circular",
    )

    result = run_linear_ssdna_composition(config_path)

    prediction = json.loads(
        (result.artifact_bundle / "folding" / "secondary_structure_prediction_v1.json").read_text(encoding="utf-8")
    )
    assert prediction["status"] == "ok"
    assert prediction["backend"]["name"] == "ViennaRNA"
    assert prediction["backend"]["version"] == "2.7.2"
    assert prediction["result"]["dot_bracket"] == "." * 88

    manifest = json.loads((result.artifact_bundle / "manifest.json").read_text(encoding="utf-8"))
    plot_manifest_path = (
        result.artifact_bundle
        / "visual"
        / "viennarna_secondary_structure"
        / "viennarna_secondary_structure_svg_v1.json"
    )
    annotated_svg_path = (
        result.artifact_bundle / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg"
    )
    assert manifest["artifacts"]["viennarna_structure_plot"] == (
        "visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json"
    )
    assert plot_manifest_path.is_file()
    assert annotated_svg_path.is_file()
    plot_manifest = json.loads(plot_manifest_path.read_text(encoding="utf-8"))
    assert plot_manifest["contract_kind"] == "viennarna_secondary_structure_svg_v1"
    assert plot_manifest["layout_algorithm"] == "circular"
    assert plot_manifest["qa"]["nucleotide_node_count"] == 88
    annotated_svg = annotated_svg_path.read_text(encoding="utf-8")
    assert 'data-dnadesign-index0="15"' in annotated_svg
    assert 'data-dnadesign-owner-ids="retron43_teto_unit.payload_primary"' in annotated_svg
    assert 'id="dnadesign-secondary-structure-labels"' in annotated_svg
    assert 'data-dnadesign-section-label="TetO primary"' in annotated_svg
    assert 'data-dnadesign-stem-base-emphasis="true"' in annotated_svg
    annotation_manifest = json.loads(
        (
            result.artifact_bundle
            / "visual"
            / "viennarna_secondary_structure"
            / "secondary_structure.annotation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert {section["label"] for section in annotation_manifest["section_annotations"]} >= {
        "TetO primary",
        "Snapback cap",
        "TetO complement",
    }
    assert annotation_manifest["layout_normalization"]["requested_orientation"] == "cap_right"

    report = baserender.run_job(
        result.artifact_bundle / "baserender_jobs" / "component_span_qa_svg.yaml",
        kind="nucleotide_evidence_map_render_v3",
        strict=True,
        caller_root=result.artifact_bundle,
    )
    assert Path(report.outputs["images_path"]).is_file()

    review_manifest = publish_composition_review_svg(result.artifact_bundle)
    review_svg_path = result.artifact_bundle / review_manifest.artifacts.review_svg
    review_png_path = result.artifact_bundle / review_manifest.artifacts.review_png
    assert review_manifest.contract_kind == "composition_review_svg_v1"
    assert (
        review_manifest.sources.structure_svg
        == "visual/viennarna_secondary_structure/secondary_structure.annotated.svg"
    )
    assert review_manifest.layout.row_count == 2
    assert review_manifest.layout.panel_order == ["secondary_structure", "component_span"]
    assert review_manifest.layout.review_png_scale == 3.0
    assert review_manifest.layout.review_png_ppi == 216.0
    assert review_manifest.layout.component_nucleotide_font_size_px == 6.0
    assert review_manifest.layout.structure_fit_policy == "balanced_visual_weight"
    assert review_manifest.layout.structure_scale > 1.1
    assert review_manifest.layout.component_scale == pytest.approx(1.525, abs=0.01)
    assert review_manifest.layout.component_width_px > review_manifest.layout.structure_width_px
    assert review_manifest.layout.structure_to_component_width_ratio == pytest.approx(0.82, abs=0.01)
    assert review_manifest.layout.component_effective_nucleotide_font_size_px == pytest.approx(9.15, abs=0.01)
    assert review_manifest.layout.component_panel_emphasis == "bold_glyph_review"
    assert review_manifest.layout.component_source_title_policy == "omit_redundant_source_title"
    assert review_manifest.qa.subplot_visual_weight_balanced is True
    assert review_manifest.qa.component_panel_emphasis_applied is True
    assert review_manifest.qa.component_source_title_omitted is True
    assert review_manifest.qa.component_source_title_omitted_count >= 1
    assert review_manifest.qa.warnings == []
    assert review_svg_path == result.artifact_bundle / "visual" / "reviews" / "composition_overview.svg"
    assert review_png_path == result.artifact_bundle / "visual" / "reviews" / "composition_overview.png"
    assert review_png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    review_svg = review_svg_path.read_text(encoding="utf-8")
    assert 'data-dnadesign-panel="secondary_structure"' in review_svg
    assert 'data-dnadesign-source-svg="secondary_structure.annotated.svg"' in review_svg
    assert 'data-dnadesign-source-orientation="cap_right"' in review_svg
    assert "dnadesign-composition-review-structure-caption" not in review_svg
    assert 'data-dnadesign-panel="component_span"' in review_svg
    assert 'data-dnadesign-panel-row="1"' in review_svg
    assert 'data-dnadesign-panel-row="2"' in review_svg
    assert 'data-dnadesign-structure-fit-policy="balanced_visual_weight"' in review_svg
    assert 'data-dnadesign-structure-nucleotide-text-count="88"' in review_svg
    assert 'data-dnadesign-structure-nucleotide-text-font-policy="explicit_renderer_safe"' in review_svg
    assert review_svg.count('class="nucleotide') == 88
    assert review_svg.count('font-family="DejaVu Sans, Arial, sans-serif"') == 88
    assert review_svg.count('font-size="12.000px"') == 88
    removed_count, changed_pixels = _png_delta_without_secondary_structure_nucleotides(
        review_svg_path,
        review_png_path,
    )
    assert removed_count == 88
    assert changed_pixels > 20_000
    assert 'data-dnadesign-component-effective-nucleotide-font-size-px="9.150"' in review_svg
    assert 'data-dnadesign-component-panel-emphasis="bold_glyph_review"' in review_svg
    assert 'data-dnadesign-review-emphasis="component_span_bold_glyph"' in review_svg
    assert 'data-dnadesign-component-source-title-policy="omit_redundant_source_title"' in review_svg
    assert 'data-dnadesign-component-source-title-omitted-count="' in review_svg
    assert "component span QA" not in review_svg


def test_publish_composition_review_validates_target_font_size_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="target_nucleotide_font_size_px must be > 0"):
        publish_composition_review_svg(tmp_path / "missing_bundle", target_nucleotide_font_size_px=0)


def test_run_linear_ssdna_composition_requires_visual_emit_for_viennarna_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.2"

class fold_compound:
    def __init__(self, sequence):
        self.sequence = sequence

    def mfe(self):
        return "." * len(self.sequence), -7.04

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    raise AssertionError("plotting should require visual.emit opt-in")
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    config_path = _write_retron43_config(
        tmp_path,
        folding_backend_name="ViennaRNA",
        folding_interface="python_api",
        folding_python_module="RNA",
    )

    result = run_linear_ssdna_composition(config_path)

    manifest = json.loads((result.artifact_bundle / "manifest.json").read_text(encoding="utf-8"))
    prediction = json.loads(
        (result.artifact_bundle / "folding" / "secondary_structure_prediction_v1.json").read_text(encoding="utf-8")
    )
    assert prediction["status"] == "ok"
    assert "viennarna_structure_plot" not in manifest["artifacts"]


def test_run_linear_ssdna_composition_writes_baserender_component_span_job(tmp_path: Path) -> None:
    config_path = _write_retron43_config(tmp_path)

    result = run_linear_ssdna_composition(config_path)

    bundle = result.artifact_bundle
    render_job = bundle / "baserender_jobs" / "component_span_qa_svg.yaml"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["baserender_component_span_svg_job"] == "baserender_jobs/component_span_qa_svg.yaml"
    assert manifest["artifacts"]["visual_contract"] == "visual/sequence_evidence_map_v1.json"
    assert "component_span_qa_visual_contract" not in manifest["artifacts"]

    job = baserender.validate_job(
        render_job,
        kind="nucleotide_evidence_map_render_v3",
        caller_root=bundle,
    )
    assert job.input.path == bundle / "visual" / "sequence_evidence_map_v1.json"
    assert job.render.renderer == "nucleotide_evidence_map"
    assert job.render.style_overrides["connectors"] is True
    assert job.render.style_overrides["baseline_spacing"] == 28.0
    assert job.render.style_overrides["sequence"]["strand_gap_cells"] == 0.08
    assert job.render.style_overrides["connector_dash"] == []
    assert job.render.style_overrides["uniform_display_font_size"] is True
    assert job.render.style_overrides["font_size_seq"] == 6
    assert job.render.style_overrides["font_size_label"] == 6
    assert job.render.style_overrides["font_size_feature_label"] == 6
    assert job.render.style_overrides["font_size_annotation_label"] == 6
    assert job.render.style_overrides["font_size_span_link_label"] == 6
    assert job.render.style_overrides["legend_font_size"] == 6

    report = baserender.run_job(
        render_job,
        kind="nucleotide_evidence_map_render_v3",
        strict=True,
        caller_root=bundle,
    )
    svg_path = Path(report.outputs["images_path"])
    assert svg_path == bundle / "visual" / "renders" / "component_span_qa_svg" / "component_span_qa.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert svg_text.count('id="sequence_pair_connector:') == 88
    assert "stroke-dasharray" not in svg_text
    assert svg_text.count(":highlight") == 16
    assert 'id="sequence:fwd:11:C:highlight"' in svg_text
    assert 'id="sequence:rev:11:G:highlight"' in svg_text
    assert svg_text.count('id="sequence_backdrop:') == 5
    assert "intended RC" not in svg_text
    for label in ["5' flank", "TetO primary", "Snapback cap", "TetO complement", "3' flank"]:
        assert svg_text.count(f"<!-- {label} -->") == 1
    for raw_slug in ["payload_primary", "snapback_cap_segment", "payload_complement"]:
        assert f"<!-- {raw_slug} -->" not in svg_text


def test_run_linear_ssdna_composition_rejects_reverse_complement_mismatch(tmp_path: Path) -> None:
    config_path = _write_retron43_config(tmp_path, bad_complement=True)

    with pytest.raises(ValidationError, match="payload_complement reverse_complement does not match payload_primary"):
        run_linear_ssdna_composition(config_path)
