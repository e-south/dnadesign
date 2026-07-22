"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_cli.py

CLI tests for secondary-structure folding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.folding.cli import _plot_output_dir_for, app

_RUNNER = CliRunner()


def test_plot_output_dir_plain_relative_paths_resolve_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    prediction_dir = repo_root / "src" / "dnadesign" / "construct" / "workspace" / "outputs" / "bundle" / "folding"
    prediction_dir.mkdir(parents=True)
    prediction = prediction_dir / "secondary_structure_prediction_v1.json"
    prediction.write_text("{}\n", encoding="utf-8")
    requested = Path("src/dnadesign/construct/workspace/outputs/bundle/visual/viennarna_secondary_structure")
    monkeypatch.chdir(repo_root)

    output_dir = _plot_output_dir_for(prediction, requested)

    assert output_dir == (repo_root / requested).resolve()
    assert "/folding/src/" not in output_dir.as_posix()


def test_plot_output_dir_parent_relative_paths_stay_bundle_relative(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "bundle" / "folding"
    prediction_dir.mkdir(parents=True)
    prediction = prediction_dir / "secondary_structure_prediction_v1.json"
    prediction.write_text("{}\n", encoding="utf-8")

    output_dir = _plot_output_dir_for(prediction, Path("../visual/viennarna_secondary_structure"))

    assert output_dir == (tmp_path / "bundle" / "visual" / "viennarna_secondary_structure").resolve()


def _write_request(tmp_path: Path) -> Path:
    sequence = "GCAT"
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    artifact = tmp_path / "assembled_sequence.json"
    artifact.write_text(
        json.dumps(
            {
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
    request_path = tmp_path / "folding_request.yaml"
    request_path.write_text(
        yaml.safe_dump(
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
                    "length": len(sequence),
                },
                "scope": {"mode": "canonical_component_unit"},
                "backend": {
                    "name": "ViennaRNA",
                    "executable": "definitely-missing-rnafold-for-dnadesign-test",
                    "parameters": {},
                    "dna_policy": {
                        "mode": "convert_t_to_u_for_rna_backend",
                        "output_coordinates": "original_dna_sequence",
                    },
                },
                "policy": {
                    "required": False,
                    "fail_on_malformed_output": True,
                    "fail_on_length_mismatch": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return request_path


def test_folding_preflight_cli_reports_json_status(tmp_path: Path) -> None:
    request_path = _write_request(tmp_path)

    result = _RUNNER.invoke(app, ["preflight", "--request", request_path.as_posix(), "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "warning_optional_missing"
    assert payload["backend"]["name"] == "ViennaRNA"


@pytest.mark.parametrize(
    "contract",
    ("producer_folding_bundle_v1", "linear_ssdna_composition_bundle_manifest_v1"),
)
def test_folding_preflight_cli_accepts_producer_bundle(tmp_path: Path, contract: str) -> None:
    bundle = tmp_path / "bundle"
    folding_dir = bundle / "folding"
    folding_dir.mkdir(parents=True)
    _write_request(folding_dir)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "status": "ok",
                "artifacts": {"folding_request": "folding/folding_request.yaml"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["preflight", "--bundle", bundle.as_posix(), "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "warning_optional_missing"
    assert payload["output_dir"] == folding_dir.resolve().as_posix()


def test_folding_preflight_cli_rejects_uncontracted_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    folding_dir = bundle / "folding"
    folding_dir.mkdir(parents=True)
    _write_request(folding_dir)
    (bundle / "manifest.json").write_text(
        json.dumps({"status": "ok", "artifacts": {"folding_request": "folding/folding_request.yaml"}}) + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["preflight", "--bundle", bundle.as_posix()])

    assert result.exit_code == 1
    assert "Folding bundle manifest contract must be one of" in result.output


def test_folding_plot_cli_enriches_prediction_and_writes_viennarna_svg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.2"

def plot_layout_circular(structure):
    return {"layout": "circular", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    if sequence != "GCAT":
        return 0
    if layout != {"layout": "circular", "structure": structure}:
        return 0
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg">\\n')
        handle.write('<g id="pairs">\\n')
        handle.write('<line class="basepairs" id="1,4" x1="0" y1="0" x2="1" y2="1" />\\n')
        handle.write('<line class="basepairs" id="2,3" x1="0" y1="1" x2="1" y2="0" />\\n')
        handle.write('</g><g id="seq">\\n')
        for index, base in enumerate(sequence):
            handle.write(f'<text class="nucleotide" x="{index}" y="0">{base}</text>\\n')
        handle.write('</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
    sequence = "GCAT"
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    assembled = tmp_path / "assembled_sequence.json"
    assembled.write_text(
        json.dumps(
            {
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
    prediction = tmp_path / "secondary_structure_prediction_v1.json"
    prediction.write_text(
        json.dumps(
            {
                "contract": "secondary_structure_prediction_v1",
                "schema_version": 1,
                "prediction_id": "demo.viennarna.canonical_component_unit",
                "status": "ok",
                "input": {
                    "sequence_id": "demo",
                    "sequence_sha256": sequence_sha256,
                    "alphabet": "dna",
                    "topology": "linear_ssdna",
                    "length": len(sequence),
                },
                "backend": {
                    "name": "ViennaRNA",
                    "version": "2.7.2",
                    "command": ["RNA.fold_compound", "mfe"],
                    "parameters": {},
                },
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "submitted_alphabet": "rna_surrogate",
                    "coordinates_mapped_to": "original_dna_sequence",
                },
                "result": {
                    "dot_bracket": "(())",
                    "mfe_kcal_mol": -2.3,
                    "pair_map": [
                        {"left": 0, "right": 3, "pair": "GU"},
                        {"left": 1, "right": 2, "pair": "CA"},
                    ],
                },
                "qa": {"length_matches_input": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    visual_contract = tmp_path / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": sequence,
                "pairings": [
                    {
                        "pairing_id": "demo.payload_rc",
                        "primary_start": 0,
                        "primary_end": 2,
                        "complement_start": 2,
                        "complement_end": 4,
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

    result = _RUNNER.invoke(
        app,
        [
            "plot",
            "--prediction",
            prediction.as_posix(),
            "--assembled-sequence",
            assembled.as_posix(),
            "--visual-contract",
            visual_contract.as_posix(),
            "--output-dir",
            (tmp_path / "visual").as_posix(),
            "--python-module",
            "RNA",
            "--layout",
            "circular",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["layout_algorithm"] == "circular"
    assert payload["qa"]["cross_copy_pair_count"] == 2
    enriched_prediction = json.loads(prediction.read_text(encoding="utf-8"))
    assert enriched_prediction["qa"]["pairing_summary"]["intended_recovered_count"] == 1
    annotated = (tmp_path / "visual" / "secondary_structure.annotated.svg").read_text(encoding="utf-8")
    assert ">T<" in annotated
    assert ">U<" not in annotated


def test_folding_plot_cli_accepts_construct_bundle(
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
    bundle = tmp_path / "bundle"
    folding_dir = bundle / "folding"
    visual_dir = bundle / "visual"
    plot_dir = visual_dir / "viennarna_secondary_structure"
    folding_dir.mkdir(parents=True)
    visual_dir.mkdir(parents=True)

    sequence = "GCAT"
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    assembled = folding_dir / "secondary_structure_input_sequence.json"
    assembled.write_text(
        json.dumps(
            {
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
    prediction = folding_dir / "secondary_structure_prediction_v1.json"
    prediction.write_text(
        json.dumps(
            {
                "contract": "secondary_structure_prediction_v1",
                "schema_version": 1,
                "prediction_id": "demo.viennarna.canonical_component_unit",
                "status": "ok",
                "input": {
                    "sequence_id": "demo",
                    "sequence_sha256": sequence_sha256,
                    "alphabet": "dna",
                    "topology": "linear_ssdna",
                    "length": len(sequence),
                },
                "backend": {
                    "name": "ViennaRNA",
                    "version": "2.7.2",
                    "command": ["RNA.fold_compound", "mfe"],
                    "parameters": {},
                },
                "dna_policy": {
                    "mode": "convert_t_to_u_for_rna_backend",
                    "submitted_alphabet": "rna_surrogate",
                    "coordinates_mapped_to": "original_dna_sequence",
                },
                "result": {
                    "dot_bracket": "(())",
                    "mfe_kcal_mol": -2.3,
                    "pair_map": [{"left": 0, "right": 3, "pair": "GU"}],
                },
                "qa": {"length_matches_input": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    visual_contract = visual_dir / "sequence_evidence_map_v1.json"
    visual_contract.write_text(
        json.dumps(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "demo",
                "topology_kind": "linear_ssdna",
                "alphabet": "dna",
                "primary_sequence": sequence,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "contract": "linear_ssdna_composition_bundle_manifest_v1",
                "status": "ok",
                "artifacts": {
                    "folding_input_sequence": "folding/secondary_structure_input_sequence.json",
                    "folding_prediction": "folding/secondary_structure_prediction_v1.json",
                    "viennarna_structure_plot": (
                        "visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json"
                    ),
                    "visual_contract": "visual/sequence_evidence_map_v1.json",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "plot",
            "--bundle",
            bundle.as_posix(),
            "--python-module",
            "RNA",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["layout_algorithm"] == "naview"
    assert payload["artifacts"]["annotated_svg"] == "secondary_structure.annotated.svg"
    assert (plot_dir / "secondary_structure.annotated.svg").is_file()


def test_folding_plot_bundle_fails_fast_without_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    result = _RUNNER.invoke(app, ["plot", "--bundle", bundle.as_posix()])

    assert result.exit_code == 1
    assert "manifest.json" in result.output
