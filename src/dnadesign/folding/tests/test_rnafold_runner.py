"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_rnafold_runner.py

Runtime tests for backend-neutral secondary-structure folding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionBackendV1,
    SecondaryStructurePredictionDnaPolicyV1,
    SecondaryStructurePredictionInputV1,
    SecondaryStructurePredictionResultV1,
    SecondaryStructurePredictionV1,
)
from dnadesign.folding import (
    enrich_prediction_pairing_qa,
    parse_rnafold_stdout,
    preflight_request,
    publish_viennarna_structure_svg,
    run_prediction_request,
)
from dnadesign.folding.src.errors import FoldingConfigError
from dnadesign.folding.src.viennarna_svg import _expand_root_viewbox, _stem_metric_label

_CONTIGUOUS_STEM_SEQUENCE = "GACGATATCGTC"
_CONTIGUOUS_STEM_PAIR_MAP = [
    {"left": 0, "right": 11, "pair": "GC"},
    {"left": 1, "right": 10, "pair": "GU"},
    {"left": 2, "right": 9, "pair": "CG"},
    {"left": 3, "right": 8, "pair": "AU"},
    {"left": 4, "right": 7, "pair": "UA"},
    {"left": 5, "right": 6, "pair": "GT"},
]


def _write_assembled_sequence(tmp_path: Path, sequence: str = "GCAT") -> tuple[Path, str]:
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    artifact = tmp_path / "assembled_sequence.json"
    artifact.write_text(
        json.dumps(
            {
                "contract": "linear_ssdna_composition_v1",
                "schema_version": 1,
                "composition_id": "demo",
                "sequence": {
                    "id": "demo",
                    "length": len(sequence),
                    "sha256": sequence_sha256,
                    "sequence": sequence,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact, sequence_sha256


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    parts = [float(part) for part in str(root.attrib["viewBox"]).split()]
    assert len(parts) == 4
    return parts[0], parts[1], parts[2], parts[3]


def _rotate_test_point(
    point: tuple[float, float],
    *,
    center: tuple[float, float],
    angle_degrees: float,
) -> tuple[float, float]:
    import math

    angle = math.radians(angle_degrees)
    cos_v = math.cos(angle)
    sin_v = math.sin(angle)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return center[0] + dx * cos_v - dy * sin_v, center[1] + dx * sin_v + dy * cos_v


def _request(tmp_path: Path, *, executable: str, required: bool = False) -> SecondaryStructurePredictionRequestV1:
    artifact, sequence_sha256 = _write_assembled_sequence(tmp_path)
    return SecondaryStructurePredictionRequestV1.model_validate(
        {
            "contract": "secondary_structure_prediction_request_v1",
            "schema_version": 1,
            "request_id": "demo.rnafold.canonical_component_unit",
            "input": {
                "sequence_artifact": artifact.as_posix(),
                "sequence_id": "demo",
                "sequence_sha256": sequence_sha256,
                "alphabet": "dna",
                "topology": "linear_ssdna",
                "length": 4,
            },
            "scope": {"mode": "canonical_component_unit"},
            "backend": {
                "name": "ViennaRNA",
                "executable": executable,
                "parameters": {"temperature_c": 37.0},
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "output_coordinates": "original_dna_sequence",
                },
            },
            "policy": {
                "required": required,
                "fail_on_malformed_output": True,
                "fail_on_length_mismatch": True,
            },
        }
    )


def _python_api_request(
    tmp_path: Path,
    *,
    required: bool = False,
    sequence: str = "GCAT",
) -> SecondaryStructurePredictionRequestV1:
    artifact, sequence_sha256 = _write_assembled_sequence(tmp_path, sequence=sequence)
    return SecondaryStructurePredictionRequestV1.model_validate(
        {
            "contract": "secondary_structure_prediction_request_v1",
            "schema_version": 1,
            "request_id": "demo.viennarna.canonical_component_unit",
            "input": {
                "sequence_artifact": artifact.as_posix(),
                "sequence_id": "demo",
                "sequence_sha256": sequence_sha256,
                "alphabet": "dna",
                "topology": "linear_ssdna",
                "length": len(sequence),
            },
            "scope": {"mode": "canonical_component_unit"},
            "backend": {
                "name": "ViennaRNA",
                "interface": "python_api",
                "python_module": "RNA",
                "parameters": {"temperature_c": 37.0},
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "output_coordinates": "original_dna_sequence",
                },
            },
            "policy": {
                "required": required,
                "fail_on_malformed_output": True,
                "fail_on_length_mismatch": True,
            },
        }
    )


def _contiguous_stem_prediction(
    *,
    sequence_sha256: str = "abc",
    sequence_id: str = "demo",
) -> SecondaryStructurePredictionV1:
    return SecondaryStructurePredictionV1(
        prediction_id=f"{sequence_id}.viennarna.canonical_component_unit",
        status="ok",
        input=SecondaryStructurePredictionInputV1(
            sequence_id=sequence_id,
            sequence_sha256=sequence_sha256,
            alphabet="dna",
            topology="linear_ssdna",
            length=len(_CONTIGUOUS_STEM_SEQUENCE),
        ),
        backend=SecondaryStructurePredictionBackendV1(
            name="ViennaRNA",
            version="2.7.2",
            command=["RNA.fold_compound", "mfe"],
        ),
        dna_policy=SecondaryStructurePredictionDnaPolicyV1(
            mode="convert_t_to_u_for_rna_backend",
            submitted_alphabet="rna_surrogate",
            coordinates_mapped_to="original_dna_sequence",
        ),
        result=SecondaryStructurePredictionResultV1(
            dot_bracket="(((((())))))",
            mfe_kcal_mol=-4.1,
            pair_map=_CONTIGUOUS_STEM_PAIR_MAP,
        ),
    )


def _write_contiguous_stem_visual_contract(
    path: Path,
    *,
    include_segment_labels: bool,
) -> None:
    payload: dict[str, object] = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "demo",
        "topology_kind": "linear_ssdna",
        "alphabet": "dna",
        "primary_sequence": _CONTIGUOUS_STEM_SEQUENCE,
        "owners": [
            {
                "owner_id": "demo.payload_primary",
                "row_id": "primary",
                "start": 0,
                "end": 6,
                "display_label": "MSD TetO",
                "short_label": "",
            },
            {
                "owner_id": "demo.payload_complement",
                "row_id": "primary",
                "start": 6,
                "end": 12,
                "display_label": "MSD TetO complement",
                "short_label": "",
            },
        ],
        "pairings": [
            {
                "pairing_id": "demo.payload_rc",
                "primary_start": 0,
                "primary_end": 6,
                "complement_start": 6,
                "complement_end": 12,
                "display_label": "payload_rc",
                "short_label": "intended RC",
            }
        ],
    }
    if include_segment_labels:
        payload["meta"] = {
            "span_backdrops": [
                {
                    "semantic": "payload_primary",
                    "start": 0,
                    "end": 6,
                    "edge_color": "#F58518",
                },
                {
                    "semantic": "payload_complement",
                    "start": 6,
                    "end": 12,
                    "edge_color": "#E45756",
                },
            ],
            "segment_labels": [
                {"text": "MSD TetO", "start": 0, "end": 6, "label_side": "above"},
                {"text": "MSD TetO complement", "start": 6, "end": 12, "label_side": "above"},
            ],
        }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_expand_root_viewbox_keeps_normalized_geometry_centered_with_asymmetric_labels() -> None:
    root = ET.Element(
        "{http://www.w3.org/2000/svg}svg",
        {"width": "452", "height": "452"},
    )
    ET.SubElement(
        root,
        "{http://www.w3.org/2000/svg}rect",
        {
            "x": "0",
            "y": "0",
            "width": "452",
            "height": "452",
            "style": "stroke: white; fill: white",
        },
    )
    geometry_bbox = {
        "min_x": 60.0,
        "min_y": 100.0,
        "max_x": 500.0,
        "max_y": 225.0,
    }

    _expand_root_viewbox(
        root,
        layout_normalization={"geometry_bbox": geometry_bbox},
        label_boxes=[
            (70.0, 90.0, 110.0, 112.0),
            (500.0, 150.0, 560.0, 172.0),
        ],
    )

    x, _, width, _ = _parse_viewbox(root)
    left_margin = geometry_bbox["min_x"] - x
    right_margin = x + width - geometry_bbox["max_x"]
    assert left_margin == pytest.approx(right_margin, abs=0.01)
    assert left_margin >= 72.0


def test_parse_rnafold_stdout_maps_pairs_to_original_coordinates() -> None:
    parsed = parse_rnafold_stdout(
        stdout=">demo\nGCAU\n(()) (-2.30)\n",
        submitted_sequence="GCAU",
        input_length=4,
    )

    assert parsed.dot_bracket == "(())"
    assert parsed.mfe_kcal_mol == -2.30
    assert [pair.model_dump(mode="json") for pair in parsed.pair_map] == [
        {"left": 0, "right": 3, "pair": "GU"},
        {"left": 1, "right": 2, "pair": "CA"},
    ]


def test_preflight_reports_optional_missing_backend_without_silent_success(tmp_path: Path) -> None:
    request = _request(tmp_path, executable="definitely-missing-rnafold-for-dnadesign-test")

    preflight = preflight_request(request, output_dir=tmp_path / "folding")
    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")

    assert preflight.status == "warning_optional_missing"
    assert prediction.status == "warning_optional_missing"
    assert prediction.result is None
    assert "not available" in prediction.qa.warnings[0]
    assert (tmp_path / "folding" / "secondary_structure_prediction_v1.json").is_file()


def test_run_prediction_request_uses_rnafold_cli_output(tmp_path: Path) -> None:
    executable = tmp_path / "RNAfold"
    executable.write_text(
        """#!/bin/sh
if [ "$1" = "--version" ]; then
  printf "RNAfold 2.7.0\\n"
  exit 0
fi
cat >/dev/null
printf ">demo\\nGCAU\\n(()) (-2.30)\\n"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    request = _request(tmp_path, executable=executable.as_posix())

    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")

    assert prediction.status == "ok"
    assert prediction.backend is not None
    assert prediction.backend.version == "RNAfold 2.7.0"
    assert prediction.dna_policy is not None
    assert prediction.dna_policy.submitted_alphabet == "rna_surrogate"
    assert prediction.result is not None
    assert prediction.result.dot_bracket == "(())"
    assert prediction.result.pair_map[0].left == 0
    assert (tmp_path / "folding" / "RNAfold.stdout.txt").is_file()


def test_run_prediction_request_uses_viennarna_python_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.2"

class fold_compound:
    def __init__(self, sequence):
        self.sequence = sequence

    def mfe(self):
        return "(())", -2.3
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    request = _python_api_request(tmp_path)

    preflight = preflight_request(request, output_dir=tmp_path / "folding")
    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")

    assert preflight.status == "ok"
    assert preflight.backend_available
    assert prediction.status == "ok"
    assert prediction.backend is not None
    assert prediction.backend.name == "ViennaRNA"
    assert prediction.backend.version == "2.7.2"
    assert prediction.backend.command == ["RNA.fold_compound", "mfe"]
    assert prediction.dna_policy is not None
    assert prediction.dna_policy.submitted_alphabet == "rna_surrogate"
    assert prediction.result is not None
    assert prediction.result.dot_bracket == "(())"


def test_publish_viennarna_structure_svg_annotates_native_svg(
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
        return "(())", -2.3

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    if sequence != "GCAT":
        return 0
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg">\\n')
        handle.write('<g id="pairs"><line class="basepairs" id="1,4" x1="0" y1="0" x2="1" y2="1" /></g>\\n')
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
    request = _python_api_request(tmp_path)
    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": "GCAT",
                "owners": [
                    {
                        "owner_id": "demo.0.payload_primary",
                        "row_id": "primary",
                        "start": 0,
                        "end": 2,
                        "display_label": "Payload",
                        "short_label": "payload",
                    },
                    {
                        "owner_id": "demo.0.flank_3p",
                        "row_id": "primary",
                        "start": 2,
                        "end": 4,
                        "display_label": "Flank",
                        "short_label": "flank",
                    },
                ],
                "effect_tags": [
                    {
                        "tag_id": "demo.0.teto_primary",
                        "tag_kind": "payload",
                        "row_id": "primary",
                        "start": 0,
                        "end": 2,
                        "display_label": "TetO",
                        "short_label": "TetO",
                    }
                ],
                "meta": {
                    "span_backdrops": [
                        {
                            "semantic": "payload_primary",
                            "start": 0,
                            "end": 2,
                            "edge_color": "#F58518",
                        },
                        {
                            "semantic": "flank_3p",
                            "start": 2,
                            "end": 4,
                            "edge_color": "#0D9488",
                        },
                    ],
                    "segment_labels": [
                        {"text": "Payload primary", "start": 0, "end": 2, "label_side": "above"},
                        {"text": "3' flank", "start": 2, "end": 4, "label_side": "above"},
                    ],
                    "unit_copies": [
                        {"unit_id": "demo_unit", "copy_index": 0, "span": {"start": 0, "end": 2}},
                        {"unit_id": "demo_unit", "copy_index": 1, "span": {"start": 2, "end": 4}},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plot = publish_viennarna_structure_svg(
        prediction,
        assembled_sequence_path=tmp_path / "assembled_sequence.json",
        visual_contract_path=visual_contract,
        output_dir=tmp_path / "visual" / "viennarna_secondary_structure",
    )

    assert plot.contract_kind == "viennarna_secondary_structure_svg_v1"
    assert plot.qa.nucleotide_node_count == 4
    assert plot.qa.cross_copy_pair_count == 1
    assert plot.qa.length_matches_svg_nodes is True
    annotated = (tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg").read_text(
        encoding="utf-8"
    )
    assert 'data-dnadesign-index0="0"' in annotated
    assert 'data-dnadesign-owner-ids="demo.0.payload_primary"' in annotated
    assert 'data-dnadesign-effect-tags="payload"' in annotated
    assert 'data-dnadesign-left-index0="0"' in annotated
    assert ">T<" in annotated
    assert ">U<" not in annotated
    assert 'id="dnadesign-secondary-structure-labels"' in annotated
    assert 'data-dnadesign-section-label="Payload primary"' in annotated
    assert 'id="dnadesign-secondary-structure-semantic-edges"' in annotated
    assert 'data-dnadesign-edge-color-source="visual_contract_span_backdrops"' in annotated
    assert 'data-dnadesign-edge-semantic="payload_primary"' in annotated
    assert 'data-dnadesign-edge-semantic="flank_3p"' in annotated
    assert "stroke: #F58518" in annotated
    assert "stroke: #0D9488" in annotated
    annotation_manifest_path = (
        tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotation_manifest.json"
    )
    annotation_manifest = json.loads(annotation_manifest_path.read_text(encoding="utf-8"))
    assert annotation_manifest["nucleotides"][0]["display_index_1"] == 1
    assert annotation_manifest["nucleotides"][3]["base_dna"] == "T"
    assert annotation_manifest["nucleotides"][3]["base_submitted"] == "U"
    assert annotation_manifest["nucleotides"][0]["owner_ids"] == ["demo.0.payload_primary"]
    assert annotation_manifest["basepairs"][0]["is_cross_copy"] is True
    assert annotation_manifest["basepairs"][0]["left_copy_index"] == 0
    assert annotation_manifest["basepairs"][0]["right_copy_index"] == 1
    assert annotation_manifest["section_annotations"][0]["label"] == "Payload primary"
    assert annotation_manifest["layout_normalization"]["requested_orientation"] == "cap_right"


def test_publish_viennarna_structure_svg_requires_section_edge_colors(
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
        return "." * len(self.sequence), -1.0

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg">\\n')
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
    request = _python_api_request(tmp_path)
    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": "GCAT",
                "owners": [
                    {
                        "owner_id": "demo.payload_primary",
                        "row_id": "primary",
                        "start": 0,
                        "end": 4,
                        "display_label": "Payload",
                        "short_label": "payload",
                    },
                ],
                "meta": {
                    "segment_labels": [
                        {"text": "Payload primary", "start": 0, "end": 4, "label_side": "above"},
                    ],
                    "span_backdrops": [
                        {"semantic": "payload_primary", "start": 0, "end": 4},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FoldingConfigError, match="span_backdrops.*edge_color"):
        publish_viennarna_structure_svg(
            prediction,
            assembled_sequence_path=tmp_path / "assembled_sequence.json",
            visual_contract_path=visual_contract,
            output_dir=tmp_path / "visual" / "viennarna_secondary_structure",
        )


def test_publish_viennarna_structure_svg_can_normalize_cap_orientation(
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
        return "." * len(self.sequence), -1.0

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    coords = [(0, 0), (0, 10), (0, 80), (0, 90), (0, 20), (0, 30)]
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120">\\n')
        handle.write('<rect style="stroke: white; fill: white" height="120" x="0" y="0" width="120" />\\n')
        handle.write('<g id="structure">\\n')
        handle.write('<polyline class="backbone" id="outline" points="0,0 0,10 0,80 0,90 0,20 0,30" />\\n')
        handle.write('<g id="pairs">\\n')
        handle.write('<line class="basepairs" id="1,6" x1="0" y1="0" x2="0" y2="30" />\\n')
        handle.write('</g>\\n')
        handle.write('<g transform="translate(-4.6, 4)" id="seq">\\n')
        for index, base in enumerate(sequence):
            x, y = coords[index]
            handle.write(f'<text class="nucleotide" x="{x}" y="{y}">{base}</text>\\n')
        handle.write('</g>\\n</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    request = _python_api_request(tmp_path, sequence="GGCCAA")
    prediction = run_prediction_request(request, output_dir=tmp_path / "folding")
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": "GGCCAA",
                "owners": [
                    {
                        "owner_id": "demo.payload_primary",
                        "row_id": "primary",
                        "start": 0,
                        "end": 2,
                        "display_label": "Payload primary",
                        "short_label": "",
                    },
                    {
                        "owner_id": "demo.snapback_foldback_geometry",
                        "row_id": "primary",
                        "start": 2,
                        "end": 4,
                        "display_label": "Foldback",
                        "short_label": "",
                    },
                    {
                        "owner_id": "demo.payload_complement",
                        "row_id": "primary",
                        "start": 4,
                        "end": 6,
                        "display_label": "Payload complement",
                        "short_label": "",
                    },
                ],
                "effect_tags": [],
                "display": {"title": "retron43_teto_manual_x8 component span QA"},
                "meta": {
                    "structure_title": "Retron 43 TetO x8",
                    "component_palette": {
                        "payload_primary": "#F58518",
                        "payload_complement": "#E45756",
                        "snapback_foldback_geometry": "#64748B",
                        "snapback_cap": "#16A34A",
                        "stem_base_left": "#7C3AED",
                        "stem_base_right": "#A16207",
                    },
                    "span_backdrops": [
                        {
                            "semantic": "payload_primary",
                            "start": 0,
                            "end": 2,
                            "edge_color": "#F58518",
                        },
                        {
                            "semantic": "snapback_foldback_geometry",
                            "start": 2,
                            "end": 4,
                            "edge_color": "#64748B",
                        },
                        {
                            "semantic": "payload_complement",
                            "start": 4,
                            "end": 6,
                            "edge_color": "#E45756",
                        },
                    ],
                    "scar_nick": {
                        "left_base": "G",
                        "right_base": "A",
                        "profile_s3s2s1s0": "MXMM",
                    },
                    "segment_labels": [
                        {"text": "Left stem base", "start": 0, "end": 1, "label_side": "below"},
                        {"text": "TetO primary", "start": 0, "end": 2, "label_side": "above"},
                        {"text": "Foldback", "start": 2, "end": 4, "label_side": "above"},
                        {"text": "Cap", "start": 2, "end": 4, "label_side": "below"},
                        {"text": "TetO complement", "start": 4, "end": 6, "label_side": "above"},
                        {"text": "Right stem base", "start": 5, "end": 6, "label_side": "below"},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    publish_viennarna_structure_svg(
        prediction,
        assembled_sequence_path=tmp_path / "assembled_sequence.json",
        visual_contract_path=visual_contract,
        output_dir=tmp_path / "visual" / "viennarna_secondary_structure",
    )

    annotated = (tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg").read_text(
        encoding="utf-8"
    )
    annotation_manifest = json.loads(
        (
            tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert 'id="dnadesign-viennarna-normalized-layout"' in annotated
    assert 'id="dnadesign-secondary-structure-background"' in annotated
    assert 'style="stroke: white; fill: white"' not in annotated
    assert 'id="dnadesign-secondary-structure-title"' in annotated
    assert 'data-dnadesign-title-align="content_center"' in annotated
    assert 'data-dnadesign-upright-text="true"' in annotated
    assert 'id="dnadesign-viennarna-oriented-nucleotide-text"' in annotated
    assert 'data-dnadesign-text-position-policy="projected_to_cap_right_coordinates"' in annotated
    assert annotated.count('data-dnadesign-position-policy="projected_to_cap_right_coordinates"') == len("GGCCAA")
    assert annotated.count('data-dnadesign-anchor-policy="centered_on_projected_coordinate"') == len("GGCCAA")
    assert annotated.count('text-anchor="middle"') >= len("GGCCAA")
    assert annotated.count('dominant-baseline="middle"') >= len("GGCCAA")
    assert 'data-dnadesign-anchor-policy="centered_on_rotated_coordinate"' not in annotated
    root = ET.fromstring(annotated)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    backbone = root.find(".//svg:polyline[@class='backbone']", namespace)
    assert backbone is not None
    assert "stroke: #6B7280" in str(backbone.attrib.get("style", ""))
    assert "stroke-width: 2px" in str(backbone.attrib.get("style", ""))
    basepair = root.find(".//svg:line[@class='basepairs']", namespace)
    assert basepair is not None
    assert basepair.attrib["data-dnadesign-z-order"] == "hydrogen_bond_background"
    assert "stroke: #CBD5E1" in str(basepair.attrib.get("style", ""))
    assert "stroke-width: 1px" in str(basepair.attrib.get("style", ""))
    semantic_edge_layer = root.find(".//svg:g[@id='dnadesign-secondary-structure-semantic-edges']", namespace)
    assert semantic_edge_layer is not None
    assert semantic_edge_layer.attrib["data-dnadesign-edge-color-source"] == "visual_contract_span_backdrops"
    semantic_edges = semantic_edge_layer.findall("svg:line", namespace)
    assert len(semantic_edges) == len("GGCCAA") - 1
    assert {edge.attrib["data-dnadesign-edge-semantic"] for edge in semantic_edges} == {
        "payload_primary",
        "snapback_foldback_geometry",
        "payload_complement",
    }
    edge_styles = [str(edge.attrib.get("style", "")) for edge in semantic_edges]
    assert any("stroke: #F58518" in style for style in edge_styles)
    assert any("stroke: #64748B" in style for style in edge_styles)
    assert any("stroke: #E45756" in style for style in edge_styles)
    structure_group = root.find(".//svg:g[@id='structure']", namespace)
    assert structure_group is not None
    child_ids = [child.attrib.get("id") for child in list(structure_group)]
    assert child_ids.index("pairs") < child_ids.index("outline")
    projected_nodes = [
        node
        for node in root.iter()
        if node.attrib.get("data-dnadesign-position-policy") == "projected_to_cap_right_coordinates"
    ]
    assert len(projected_nodes) == len("GGCCAA")
    for node in projected_nodes:
        assert node.attrib["data-dnadesign-text-fill"] == "#111827"
        assert node.attrib["data-dnadesign-text-color-policy"] == "semantic_metadata_text_black"
        assert "fill: #111827" in str(node.attrib.get("style", ""))
    stem_base_nodes = [
        node for node in projected_nodes if "dnadesign-stem-base-nucleotide" in str(node.attrib.get("class", ""))
    ]
    assert stem_base_nodes
    for node in stem_base_nodes:
        style = str(node.attrib.get("style", ""))
        assert "font-weight: 500" in style
        assert "stroke: none" in style
        assert "paint-order: normal" in style
        assert "font-weight: 700" not in style
    native_structure_points = [(0, 0), (0, 10), (0, 80), (0, 90), (0, 20), (0, 30)]
    projected_center = (
        float(annotation_manifest["layout_normalization"]["center_x"]),
        float(annotation_manifest["layout_normalization"]["center_y"]),
    )
    projected_angle = float(annotation_manifest["layout_normalization"]["angle_degrees"])
    for node, native_point in zip(projected_nodes, native_structure_points, strict=True):
        expected_x, expected_y = _rotate_test_point(
            native_point,
            center=projected_center,
            angle_degrees=projected_angle,
        )
        assert float(node.attrib["x"]) == pytest.approx(expected_x, abs=0.001)
        assert float(node.attrib["y"]) == pytest.approx(expected_y, abs=0.001)
    assert 'id="dnadesign-secondary-structure-highlights"' in annotated
    assert 'data-dnadesign-section-kind="stem_base"' in annotated
    assert 'data-dnadesign-section-label="Left stem base"' in annotated
    assert 'data-dnadesign-section-label="Right stem base"' in annotated
    assert 'data-dnadesign-orientation="cap_right"' in annotated
    assert annotation_manifest["layout_normalization"]["applied"] is True
    assert (
        annotation_manifest["layout_normalization"]["nucleotide_text_orientation"]
        == "upright_projected_to_rotated_coordinates"
    )
    assert abs(annotation_manifest["layout_normalization"]["angle_degrees"]) > 1.0
    assert all(section["label_nucleotide_overlap_count"] == 0 for section in annotation_manifest["section_annotations"])
    assert all(section["label_reserved_overlap_count"] == 0 for section in annotation_manifest["section_annotations"])
    assert all(section["label_peer_overlap_count"] == 0 for section in annotation_manifest["section_annotations"])
    stem_sections = [
        section for section in annotation_manifest["section_annotations"] if section["section_kind"] == "stem_base"
    ]
    assert stem_sections
    assert all(
        ((section["label_x"] - section["anchor_x"]) ** 2 + (section["label_y"] - section["anchor_y"]) ** 2) ** 0.5
        <= 56.0
        for section in stem_sections
    )

    root = ET.fromstring(annotated)
    text_values = [node.text for node in root.findall(".//svg:text", namespace)]
    assert "Retron 43 TetO x8" in text_values
    assert "TetO payload | left G / right A | mismatch profile MXMM" in text_values
    assert "Cap CC (2 nt)" in text_values
    assert not any(
        str(value).startswith(("sections:", "components:", "snapback:", "scar_nick:")) for value in text_values
    )
    viewbox = [float(part) for part in root.attrib["viewBox"].split()]
    background = root.find("svg:rect[@id='dnadesign-secondary-structure-background']", namespace)
    assert background is not None
    assert [float(background.attrib[key]) for key in ("x", "y", "width", "height")] == viewbox
    title_node = root.find(".//svg:text[@class='dnadesign-structure-title']", namespace)
    assert title_node is not None
    assert title_node.text == "Retron 43 TetO x8"
    assert title_node.attrib["text-anchor"] == "middle"
    subtitle_nodes = root.findall(".//svg:text[@class='dnadesign-structure-subtitle']", namespace)
    assert subtitle_nodes
    assert all("font-size: 9px" in str(node.attrib.get("style", "")) for node in subtitle_nodes)
    stem_base_nodes = [
        node
        for node in root.findall(".//svg:text", namespace)
        if "stem_base" in str(node.attrib.get("data-dnadesign-effect-tags", ""))
    ]
    assert stem_base_nodes
    assert all(node.attrib.get("data-dnadesign-stem-base-emphasis") == "true" for node in stem_base_nodes)
    assert all("font-weight: 500" in str(node.attrib.get("style", "")) for node in stem_base_nodes)
    assert all("stroke: none" in str(node.attrib.get("style", "")) for node in stem_base_nodes)

    publish_viennarna_structure_svg(
        prediction,
        assembled_sequence_path=tmp_path / "assembled_sequence.json",
        visual_contract_path=visual_contract,
        output_dir=tmp_path / "visual" / "viennarna_secondary_structure_no_emphasis",
        emphasize_stem_base_nucleotides=False,
    )
    no_emphasis = (
        tmp_path / "visual" / "viennarna_secondary_structure_no_emphasis" / "secondary_structure.annotated.svg"
    ).read_text(encoding="utf-8")
    assert 'data-dnadesign-stem-base-emphasis="true"' not in no_emphasis


def test_publish_viennarna_structure_svg_labels_payload_with_contiguous_stem_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.2"

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="180" height="80">\\n')
        handle.write('<g id="pairs">\\n')
        for left, right in ((1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)):
            handle.write(f'<line class="basepairs" id="{left},{right}" x1="{left}" y1="0" x2="{right}" y2="20" />\\n')
        handle.write('</g><g id="seq">\\n')
        for index, base in enumerate(sequence):
            handle.write(f'<text class="nucleotide" x="{index * 8}" y="40">{base}</text>\\n')
        handle.write('</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    assembled, sequence_sha256 = _write_assembled_sequence(tmp_path, sequence=_CONTIGUOUS_STEM_SEQUENCE)
    prediction = _contiguous_stem_prediction(sequence_sha256=sequence_sha256)
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    _write_contiguous_stem_visual_contract(visual_contract, include_segment_labels=True)
    enriched = enrich_prediction_pairing_qa(prediction, visual_contract_path=visual_contract)

    publish_viennarna_structure_svg(
        enriched,
        assembled_sequence_path=assembled,
        visual_contract_path=visual_contract,
        output_dir=tmp_path / "visual" / "viennarna_secondary_structure",
    )

    annotated = (tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotated.svg").read_text(
        encoding="utf-8"
    )
    assert ">stem 3/6 bp<" in annotated
    assert 'data-dnadesign-section-label="MSD TetO"' in annotated
    assert 'data-dnadesign-stem-metric="contiguous_watson_crick"' in annotated
    assert 'data-dnadesign-contiguous-watson-crick-stem-bp="3"' in annotated
    assert 'data-dnadesign-predicted-watson-crick-pair-count="4"' in annotated
    annotation_manifest = json.loads(
        (
            tmp_path / "visual" / "viennarna_secondary_structure" / "secondary_structure.annotation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    primary_section = next(
        section for section in annotation_manifest["section_annotations"] if section["label"] == "MSD TetO"
    )
    assert primary_section["label_subtitle"] == "stem 3/6 bp"
    assert primary_section["contiguous_watson_crick_stem_bp"] == 3
    assert primary_section["predicted_watson_crick_pair_count"] == 4


def test_enrich_prediction_pairing_qa_classifies_cross_copy_and_intended_pairs(tmp_path: Path) -> None:
    request = _request(tmp_path, executable="unused")
    result = parse_rnafold_stdout(
        stdout=">demo\nGCAU\n(()) (-2.30)\n",
        submitted_sequence="GCAU",
        input_length=4,
    )
    from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
        SecondaryStructurePredictionBackendV1,
        SecondaryStructurePredictionDnaPolicyV1,
        SecondaryStructurePredictionV1,
    )

    prediction = SecondaryStructurePredictionV1(
        prediction_id=request.request_id,
        status="ok",
        input={
            "sequence_id": "demo",
            "sequence_sha256": request.input.sequence_sha256,
            "alphabet": "dna",
            "topology": "linear_ssdna",
            "length": 4,
        },
        backend=SecondaryStructurePredictionBackendV1(
            name="ViennaRNA",
            version="2.7.2",
            command=["RNA.fold_compound", "mfe"],
        ),
        dna_policy=SecondaryStructurePredictionDnaPolicyV1(
            mode="convert_t_to_u_for_rna_backend",
            submitted_alphabet="rna_surrogate",
            coordinates_mapped_to="original_dna_sequence",
        ),
        result=result,
    )
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": "GCAT",
                "pairings": [
                    {
                        "pairing_id": "demo.payload_rc",
                        "primary_start": 0,
                        "primary_end": 2,
                        "complement_start": 2,
                        "complement_end": 4,
                        "display_label": "payload_rc",
                        "short_label": "intended RC",
                    }
                ],
                "meta": {
                    "unit_copies": [
                        {"unit_id": "demo_unit", "copy_index": 0, "span": {"start": 0, "end": 2}},
                        {"unit_id": "demo_unit", "copy_index": 1, "span": {"start": 2, "end": 4}},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    enriched = enrich_prediction_pairing_qa(
        prediction,
        visual_contract_path=visual_contract,
        output_path=tmp_path / "secondary_structure_prediction_v1.json",
    )

    assert enriched.qa.pairing_summary is not None
    assert enriched.qa.pairing_summary.predicted_pair_count == 2
    assert enriched.qa.pairing_summary.cross_copy_pair_count == 2
    assert enriched.qa.pairing_summary.intended_recovered_count == 1
    assert enriched.qa.cross_copy_pairings[0]["left_index_0"] == 0
    assert enriched.qa.cross_copy_pairings[0]["right_index_0"] == 3
    assert enriched.qa.intended_pairings[0].pairing_id == "demo.payload_rc"
    assert enriched.qa.intended_pairings[0].status == "fully_recovered"
    written = json.loads((tmp_path / "secondary_structure_prediction_v1.json").read_text(encoding="utf-8"))
    assert written["qa"]["pairing_summary"]["cross_copy_pair_count"] == 2


def test_enrich_prediction_pairing_qa_reports_contiguous_watson_crick_stem_runs(tmp_path: Path) -> None:
    prediction = _contiguous_stem_prediction()
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    _write_contiguous_stem_visual_contract(visual_contract, include_segment_labels=False)

    enriched = enrich_prediction_pairing_qa(
        prediction,
        visual_contract_path=visual_contract,
    )

    intended = enriched.qa.intended_pairings[0]
    assert intended.predicted_pair_count == 6
    assert intended.predicted_watson_crick_pair_count == 4
    assert intended.contiguous_watson_crick_stem_bp == 3
    assert [run.length_bp for run in intended.contiguous_watson_crick_stem_runs] == [1, 3]
    assert intended.contiguous_watson_crick_stem_runs[-1].primary_start == 2
    assert intended.contiguous_watson_crick_stem_runs[-1].primary_end == 5
    assert intended.contiguous_watson_crick_stem_runs[-1].complement_start == 7
    assert intended.contiguous_watson_crick_stem_runs[-1].complement_end == 10


def test_enrich_prediction_pairing_qa_extends_payload_stem_metric_through_adjacent_primitives(
    tmp_path: Path,
) -> None:
    sequence = "ACGATGGAACCATCGT"
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": sequence,
                "effect_tags": [
                    {
                        "tag_id": "demo.stem_base_left",
                        "tag_kind": "stem_base_left",
                        "row_id": "primary",
                        "start": 0,
                        "end": 1,
                        "display_label": "Left Base",
                        "short_label": "",
                    },
                    {
                        "tag_id": "demo.snapback_retained_stem",
                        "tag_kind": "snapback_retained_stem",
                        "row_id": "primary",
                        "start": 5,
                        "end": 7,
                        "display_label": "Foldback stem",
                        "short_label": "",
                    },
                    {
                        "tag_id": "demo.snapback_cap",
                        "tag_kind": "snapback_cap",
                        "row_id": "primary",
                        "start": 7,
                        "end": 9,
                        "display_label": "Cap",
                        "short_label": "",
                    },
                    {
                        "tag_id": "demo.snapback_foldback_return",
                        "tag_kind": "snapback_foldback_return",
                        "row_id": "primary",
                        "start": 9,
                        "end": 11,
                        "display_label": "Foldback return",
                        "short_label": "",
                    },
                    {
                        "tag_id": "demo.stem_base_right",
                        "tag_kind": "stem_base_right",
                        "row_id": "primary",
                        "start": 15,
                        "end": 16,
                        "display_label": "Right Base",
                        "short_label": "",
                    },
                ],
                "pairings": [
                    {
                        "pairing_id": "demo.payload_rc",
                        "primary_start": 1,
                        "primary_end": 5,
                        "complement_start": 11,
                        "complement_end": 15,
                        "display_label": "payload_rc",
                        "short_label": "intended RC",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction = SecondaryStructurePredictionV1(
        prediction_id="demo.viennarna.canonical_component_unit",
        status="ok",
        input=SecondaryStructurePredictionInputV1(
            sequence_id="demo",
            sequence_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            alphabet="dna",
            topology="linear_ssdna",
            length=len(sequence),
        ),
        backend=SecondaryStructurePredictionBackendV1(
            name="ViennaRNA",
            version="2.7.2",
            command=["RNA.fold_compound", "mfe"],
        ),
        dna_policy=SecondaryStructurePredictionDnaPolicyV1(
            mode="convert_t_to_u_for_rna_backend",
            submitted_alphabet="rna_surrogate",
            coordinates_mapped_to="original_dna_sequence",
        ),
        result=SecondaryStructurePredictionResultV1(
            dot_bracket="(((((((())))))))",
            mfe_kcal_mol=-5.5,
            pair_map=[
                {"left": 0, "right": 15, "pair": "AT"},
                {"left": 1, "right": 14, "pair": "CG"},
                {"left": 2, "right": 13, "pair": "GC"},
                {"left": 3, "right": 12, "pair": "AT"},
                {"left": 4, "right": 11, "pair": "TA"},
                {"left": 5, "right": 10, "pair": "GC"},
                {"left": 6, "right": 9, "pair": "CG"},
                {"left": 7, "right": 8, "pair": "AT"},
            ],
        ),
    )

    enriched = enrich_prediction_pairing_qa(prediction, visual_contract_path=visual_contract)

    intended = enriched.qa.intended_pairings[0]
    assert intended.expected_pair_count == 4
    assert intended.predicted_pair_count == 4
    assert intended.predicted_watson_crick_pair_count == 7
    assert intended.contiguous_watson_crick_stem_bp == 7
    assert [run.length_bp for run in intended.contiguous_watson_crick_stem_runs] == [7]
    run = intended.contiguous_watson_crick_stem_runs[0]
    assert run.start_offset == -1
    assert run.end_offset == 6
    assert run.primary_start == 0
    assert run.primary_end == 7
    assert run.complement_start == 9
    assert run.complement_end == 16


def test_stem_metric_label_omits_payload_denominator_for_effective_extension() -> None:
    assert (
        _stem_metric_label(
            {
                "contiguous_watson_crick_stem_bp": 7,
                "expected_pair_count": 4,
            }
        )
        == "stem 7 bp"
    )
    assert (
        _stem_metric_label(
            {
                "contiguous_watson_crick_stem_bp": 3,
                "expected_pair_count": 4,
            }
        )
        == "stem 3/4 bp"
    )
