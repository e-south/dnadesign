"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_retron_msd_compiler.py

Tests for the Retron MSD design-id compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import csv
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from typer.testing import CliRunner

from dnadesign.studies.retron_hairpin_design.cli import app
from dnadesign.studies.retron_hairpin_design.msd_ids import (
    MsdDesignPartInput,
    MsdIdError,
    compute_scar_nick_profile,
    parse_msd_construct_label,
    parse_msd_design_parts,
)

_RUNNER = CliRunner()

_SCAR_NICK_HIT_LABELS = [
    "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
    "pES-retron-178-msd[TetR]; C26-LCAAG-RCTCG-MXMM",
    "pES-retron-179-msd[TetR]; C172-LAGTG-RCAAT-MXMM",
    "pES-retron-180-msd[TetR]; C172-LAGTG-RCATG-XWMM",
    "pES-retron-181-msd[TetR]; C172-LAGTG-RCTTT-MWXM",
    "pES-retron-182-msd[TetR]; C172-LAGTG-RCGAT-MXWM",
    "pES-retron-183-msd[TetR]; C172-LAATG-RCGTG-XMWM",
    "pES-retron-184-msd[TetR]; C172-LAGTG-RCATT-MWMM",
    "pES-retron-185-msd[TetR]; C172-LAATG-RCGTT-MMWM",
    "pES-retron-186-msd[TetR]; C172-LAGTG-RCGTT-MWWM",
    "pES-retron-187-msd[TetR]; C172-LAGTG-RCAAG-XXMM",
    "pES-retron-188-msd[TetR]; C172-LAATG-RCTTG-XMXM",
    "pES-retron-189-msd[TetR]; C172-LAATG-RCAGT-MXMM",
    "pES-retron-190-msd[TetR]; C172-LAGTG-RCAGT-MXMM",
    "pES-retron-191-msd[TetR]; C172-LAATG-RCAAT-MXMM",
    "pES-retron-192-msd[TetR]; C172-LAATG-RCACT-MXMM",
    "pES-retron-193-msd[TetR]; C172-LCTCT-RAGTG-MXMM",
    "pES-retron-194-msd[TetR]; C172-LCTCA-RTGTG-MXMM",
]
_TETO_PAYLOAD = "tccctatcagtgatagaga"
_SNAPBACK_CAP = "tCCTCAGcccGCTGAGGa"


def _install_fake_viennarna_python_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.0"

class fold_compound:
    def __init__(self, sequence):
        self.sequence = sequence

    def mfe(self):
        half = len(self.sequence) // 2
        structure = ["." for _ in self.sequence]
        for index in range(min(6, half)):
            structure[index] = "("
            structure[len(self.sequence) - index - 1] = ")"
        return "".join(structure), -1.0

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    if "U" in sequence or "T" not in sequence:
        return 0
    if layout != {"layout": "naview", "structure": structure}:
        return 0
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 80">\\n')
        handle.write('<g id="pairs">\\n')
        handle.write('<line class="basepairs" id="1,88" x1="0" y1="20" x2="220" y2="20" />\\n')
        handle.write('</g><g id="seq">\\n')
        for index, base in enumerate(sequence):
            handle.write(f'<text class="nucleotide" x="{index * 2}" y="50">{base}</text>\\n')
        handle.write('</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)


def _write_registry(tmp_path: Path) -> Path:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "msd_design_registry.yaml").write_text(
        """
contract: retron_msd_design_registry_v1
schema_version: 1
payloads:
  TetR:
    display_name: msd[teto]
caps:
  C172:
    source_construct: retron-172
constructs:
  pES-retron-177:
    source_notes: 26-derived base / 172-cap crossover; tests 172-cap permissiveness.
    scar_nick:
      route_status: note_only
      route_note: 26-derived base / 172-cap crossover
""",
        encoding="utf-8",
    )
    return study_dir


def test_compute_scar_nick_profile_uses_s3_to_s0_convention() -> None:
    assert compute_scar_nick_profile(left_base="CGGT", right_base="ACAG") == "MXMM"
    assert compute_scar_nick_profile(left_base="CTCT", right_base="AGTG") == "MXMM"
    assert compute_scar_nick_profile(left_base="AGTG", right_base="CAAG") == "XXMM"


def test_parse_msd_construct_label_infers_profile_when_missing() -> None:
    parsed = parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RACAG")

    assert parsed.construct_id == "pES-retron-177"
    assert parsed.payload_id == "TetR"
    assert parsed.cap_id == "C172"
    assert parsed.left_base == "CGGT"
    assert parsed.right_base == "ACAG"
    assert parsed.profile_s3s2s1s0 == "MXMM"
    assert parsed.msd_design_id == "msd-tetr-C172-LCGGT-RACAG-MXMM"


def test_parse_msd_design_parts_uses_same_static_lint_without_manual_label_syntax() -> None:
    parsed = parse_msd_design_parts(
        MsdDesignPartInput(
            construct_id="pES-retron-177",
            payload_id="TetR",
            cap_id="C172",
            left_base="CGGT",
            right_base="ACAG",
        )
    )

    assert parsed.construct_label == "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"
    assert parsed.profile_s3s2s1s0 == "MXMM"


def test_parse_msd_construct_label_rejects_wrong_profile() -> None:
    with pytest.raises(MsdIdError, match="provided profile"):
        parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MMMM")


def test_parse_msd_construct_label_rejects_non_ligatable_s0() -> None:
    with pytest.raises(MsdIdError, match="S0"):
        parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RCCAA")


def test_retron_msd_lint_cli_reports_reference_json(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["reference"]["msd_design_id"] == "msd-tetr-C172-LCGGT-RACAG-MXMM"
    assert payload["reference"]["scar_nick"]["route_status"] == "note_only"
    assert payload["next_step"].startswith("Input is complete")


def test_retron_msd_lint_cli_fails_fast_on_wrong_profile(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MMMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "MsdIdError"
    assert "provided profile" in payload["error"]
    assert "Correct the declared -MWX profile" in payload["next_step"]


def test_retron_msd_lint_cli_fails_fast_on_unknown_registry_part(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--id",
            "pES-retron-177-msd[TetR]; C999-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Unknown cap 'C999'" in payload["error"]
    assert "Route missing cap or shortening constraints to Snapback" in payload["next_step"]


def test_retron_msd_lint_spec_accepts_explicit_design_parts(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-177
    payload_id: TetR
    cap_id: C172
    left_base: CGGT
    right_base: ACAG
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172: tCCTCAGcccGCTGAGGa
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["record_count"] == 1
    assert payload["records"][0]["construct_label"] == "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"


def test_retron_msd_lint_spec_resolves_public_primitive_sources(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {"final_candidate": {"designed_sequence": "ACGTACGT"}},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scar_run = tmp_path / "scar_run"
    scar_table = scar_run / "export" / "table__scar_nick_candidates.csv"
    scar_table.parent.mkdir(parents=True)
    scar_table.write_text(
        "\n".join(
            [
                "rank,candidate_id,left_base,right_base,profile_s3s2s1s0,nickase_variant_id,nicked_strand,surviving_strand",
                "1,scar-rank-01,CGGT,ACAG,MXMM,Nb.BtsI,bottom,top",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-retron-public-source
    payload_id: TetR
    cap_id: C999
    stem_base_source:
      kind: scar_nick_stem_bases
      run_dir: {scar_run.as_posix()}
      selector:
        mode: rank
        rank: 1
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C999:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        mode: rank
        rank: 1
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)["records"][0]
    assert record["cap"]["id"] == "C999"
    assert record["cap"]["source_construct"] == "snapback-rank-01"
    assert record["scar_nick"]["left_base"] == "CGGT"
    assert record["scar_nick"]["right_base"] == "ACAG"
    assert record["scar_nick"]["route_status"] == "resolved"
    assert record["scar_nick"]["nick_orientation"] == "bottom"
    assert record["scar_nick"]["nickase"] == "Nb.BtsI"


def test_retron_msd_lint_spec_refuses_implicit_primitive_combinatorics(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {"final_candidate": {"designed_sequence": "ACGTACGT"}},
                    },
                    {
                        "rank": 2,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BsrDI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {"final_candidate": {"designed_sequence": "TGCATGCA"}},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        mode: all
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "selector mode=all is reserved for future explicit expansion contracts" in payload["error"]
    assert "no implicit combinatoric expansion" in payload["error"]


@pytest.mark.parametrize(
    ("selector_yaml", "selector_mode"),
    [
        ("mode: ranks\n        ranks: [1]", "ranks"),
        ("mode: range\n        start_rank: 1\n        end_rank: 1", "range"),
        ("mode: all", "all"),
    ],
)
def test_retron_msd_lint_spec_refuses_non_rank_selector_modes_for_single_option(
    tmp_path: Path,
    selector_yaml: str,
    selector_mode: str,
) -> None:
    study_dir = _write_registry(tmp_path)
    snapback_run = tmp_path / "snapback_run"
    snapback_report = snapback_run / "analysis" / "solve_report.json"
    snapback_report.parent.mkdir(parents=True)
    snapback_report.write_text(
        json.dumps(
            {
                "workflow": "snapback_released_solve",
                "hits": [
                    {
                        "rank": 1,
                        "hit_kind": "exact",
                        "nickase_variant_id": "Nb.BtsI",
                        "release_variant_id": "BspQI",
                        "target_search_hit": {"final_candidate": {"designed_sequence": "ACGTACGT"}},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        f"""
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
payload_sequences:
  TetR: tccctatcagtgatagaga
cap_sequences:
  C172:
    source:
      kind: snapback_released_solve_cap
      run_dir: {snapback_run.as_posix()}
      selector:
        {selector_yaml}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "lint",
            "--spec",
            spec_path.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert f"selector mode={selector_mode} is reserved for future explicit expansion contracts" in payload["error"]
    assert "Use selector mode=rank" in payload["error"]


def test_retron_msd_compile_cli_writes_catalog(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "compiled"

    result = _RUNNER.invoke(
        app,
        [
            "compile",
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

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    catalog_path = Path(payload["catalog_path"])
    reference_path = out_dir / "references" / "msd-tetr-C172-LCGGT-RACAG-MXMM.msd_design_reference_v1.json"
    assert catalog_path == out_dir / "msd_design_catalog_v1.json"
    assert Path(payload["output_dir"]) == out_dir
    assert Path(payload["references_dir"]) == out_dir / "references"
    assert Path(payload["index_path"]) == out_dir / "reference_index.tsv"
    assert Path(payload["manifest_path"]) == out_dir / "manifest.json"
    assert Path(payload["readme_path"]) == out_dir / "README.md"
    assert "run materialize with explicit payload/cap sequences" in payload["next_step"]
    assert "one GenBank/structure-review sequence bundle per MSD design" in payload["next_step"]
    assert not (out_dir / "assets").exists()
    assert reference_path.is_file()

    index_rows = list(
        csv.DictReader((out_dir / "reference_index.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t")
    )
    assert index_rows == [
        {
            "construct_id": "pES-retron-177",
            "msd_design_id": "msd-tetr-C172-LCGGT-RACAG-MXMM",
            "payload_id": "TetR",
            "cap_id": "C172",
            "left_base": "CGGT",
            "right_base": "ACAG",
            "profile_s3s2s1s0": "MXMM",
            "route_status": "note_only",
            "nick_orientation": "",
            "nickase": "",
            "reference_path": "references/msd-tetr-C172-LCGGT-RACAG-MXMM.msd_design_reference_v1.json",
        }
    ]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == "msd_design_catalog_bundle_v1"
    assert manifest["reference_count"] == 1
    assert manifest["references_dir"] == "references"
    assert manifest["layout"]["max_reference_depth"] == 1
    assert "Open first" in (out_dir / "README.md").read_text(encoding="utf-8")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["records"][0]["construct_id"] == "pES-retron-177"


def test_retron_msd_compile_text_reports_output_nudges(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "compiled"

    result = _RUNNER.invoke(
        app,
        [
            "compile",
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert f"output_dir: {out_dir}" in result.stdout
    assert f"references_dir: {out_dir / 'references'}" in result.stdout
    assert f"index_path: {out_dir / 'reference_index.tsv'}" in result.stdout
    assert f"manifest_path: {out_dir / 'manifest.json'}" in result.stdout
    assert f"readme_path: {out_dir / 'README.md'}" in result.stdout
    assert "next_step: Catalog bundle emitted" in result.stdout
    assert "run materialize with explicit payload/cap sequences" in result.stdout


def test_retron_msd_compile_cli_rejects_mixed_spec_and_label_sources(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
labels:
  - pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "compile",
            "--spec",
            spec_path.as_posix(),
            "--id",
            "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM",
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            (tmp_path / "compiled").as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "Use either --spec or --id/--input" in payload["error"]


def test_retron_msd_compile_refuses_legacy_assets_layout(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "compiled"
    (out_dir / "assets").mkdir(parents=True)

    result = _RUNNER.invoke(
        app,
        [
            "compile",
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
    assert "Legacy MSD compiler output layout" in payload["error"]
    assert "fresh --out-dir" in payload["next_step"]


def test_retron_msd_compile_refuses_stale_reference_files(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "compiled"
    references_dir = out_dir / "references"
    references_dir.mkdir(parents=True)
    (references_dir / "stale.msd_design_reference_v1.json").write_text("{}\n", encoding="utf-8")

    result = _RUNNER.invoke(
        app,
        [
            "compile",
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
    assert "Stale MSD design reference output" in payload["error"]
    assert "archive/remove unrelated generated output" in payload["next_step"]


def test_retron_msd_materialize_requires_concrete_sequences(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = _RUNNER.invoke(
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
    assert "Snapback" in payload["next_step"]


def test_retron_msd_materialize_requires_viennarna_for_deliverable_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    monkeypatch.setitem(sys.modules, "RNA", None)

    result = _RUNNER.invoke(
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
            f"TetR={_TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={_SNAPBACK_CAP}",
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
    _install_fake_viennarna_python_api(tmp_path, monkeypatch)
    monkeypatch.setenv("DNADESIGN_INKSCAPE", "__must_not_be_used__")
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"

    result = _RUNNER.invoke(
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
            f"TetR={_TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={_SNAPBACK_CAP}",
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
    assert (
        out_dir
        / "manifest"
        / "configs"
        / "composition"
        / "msd-tetr-C172-LCGGT-RACAG-MXMM.linear_ssdna_composition.yaml"
    ).is_file()
    assert payload["finder_open"] == f"open {out_dir}"
    assert "Single-unit MSD sequence bundle emitted" in payload["next_step"]

    variant = payload["variants"][0]
    variant_dir = out_dir / "variants" / "msd-tetr-C172-LCGGT-RACAG-MXMM"
    expected_variant = Path("variants/msd-tetr-C172-LCGGT-RACAG-MXMM")
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
    assert Path(variant["reverse_complement_genbank"]) == Path(
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/sequences/reverse_complement.gb"
    )
    assert Path(variant["composition_overview_svg"]) == Path(
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.svg"
    )
    assert Path(variant["composition_overview_png"]) == Path(
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.png"
    )
    assert Path(variant["secondary_structure_native_png"]) == Path(
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/secondary_structure.native.png"
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
    assert 'data-dnadesign-source-svg="secondary_structure.annotated.svg"' in composition_overview_svg
    assert 'data-dnadesign-source-orientation="cap_right"' in composition_overview_svg
    fills_by_semantic = {
        item["semantic"]: item["fill"]
        for item in visual_contract["meta"]["span_backdrops"]
        if item["semantic"] in {"flank_5p", "payload_primary", "snapback_cap_segment", "payload_complement", "flank_3p"}
    }
    assert len(set(fills_by_semantic.values())) == 5

    rows = list(
        csv.DictReader(
            (out_dir / "manifest" / "indexes" / "sequence_index.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    assert rows[0]["unit_count"] == "1"
    assert rows[0]["genbank"] == "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/sequences/forward.gb"
    assert rows[0]["reverse_complement_genbank"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/sequences/reverse_complement.gb"
    )
    assert rows[0]["composition_overview_svg"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.svg"
    )
    assert rows[0]["composition_overview_png"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.png"
    )
    assert rows[0]["secondary_structure_native_png"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/secondary_structure.native.png"
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
    unit_len = flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP) + len(_TETO_PAYLOAD) + flank_3p_len
    assert record["sequence"]["length"] == unit_len
    assert record["source"]["dnadesign_bundle"] == "variants/msd-tetr-C172-LCGGT-RACAG-MXMM"
    assert record["artifacts"]["genbank"] == "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/sequences/forward.gb"
    assert record["artifacts"]["reverse_complement_genbank"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/sequences/reverse_complement.gb"
    )
    assert record["artifacts"]["composition_overview_svg"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.svg"
    )
    assert record["artifacts"]["composition_overview_png"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/composition_overview.png"
    )
    assert record["artifacts"]["secondary_structure_native_png"] == (
        "variants/msd-tetr-C172-LCGGT-RACAG-MXMM/plots/secondary_structure.native.png"
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
    assert {"5' Flanking", "msd[teto]", "Cap", "msd[teto] complement"} <= set(features_by_label)
    features_by_id = {
        feature.qualifiers["dnadesign_feature_id"][0]: feature
        for feature in genbank_record.features
        if "dnadesign_feature_id" in feature.qualifiers
    }
    payload_complement = features_by_id["payload_complement"]
    assert int(payload_complement.location.start) == flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP)
    assert int(payload_complement.location.end) == (
        flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP) + len(_TETO_PAYLOAD)
    )
    assert payload_complement.location.strand == -1
    assert payload_complement.qualifiers["label"] == ["msd[teto] complement"]
    assert payload_complement.qualifiers["dnadesign_copy_index"] == ["0"]
    assert payload_complement.qualifiers["dnadesign_transform"] == ["reverse_complement"]
    assert str(payload_complement.extract(genbank_record.seq)).upper() == _TETO_PAYLOAD.upper()

    feature_rows = list(csv.DictReader(features_path.read_text(encoding="utf-8").splitlines()))
    annotation_ids = {row["feature_id"] for row in feature_rows if row["feature_kind"] == "annotation"}
    assert annotation_ids == {"stem_base_left", "stem_base_right"}
    assert {
        "5' Flanking",
        "Left Base",
        "msd[teto]",
        "Cap",
        "msd[teto] complement",
        "Right Base",
        "3' Flanking",
    } <= {row["display_label"] for row in feature_rows}
    assert not {"flank_5p", "payload_primary", "snapback_cap_segment", "payload_complement"} & {
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
    assert payload_complement_rows[0]["sequence"].upper() == str(Seq(_TETO_PAYLOAD).reverse_complement()).upper()

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
    study_dir = _write_registry(tmp_path)

    result = _RUNNER.invoke(
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

    assert result.exit_code != 0
    assert "repeat-count" in result.output


def test_retron_msd_materialize_refuses_flat_legacy_sequence_layout(tmp_path: Path) -> None:
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    out_dir.mkdir()
    (out_dir / "sequence_manifest.json").write_text("{}\n", encoding="utf-8")

    result = _RUNNER.invoke(
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
            f"TetR={_TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={_SNAPBACK_CAP}",
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
    study_dir = _write_registry(tmp_path)
    out_dir = tmp_path / "sequence_bundle"
    stale_plots_dir = out_dir / "variants" / "msd-tetr-C172-LCGGT-RACAG-MXMM" / "plots"
    stale_plots_dir.mkdir(parents=True)
    (stale_plots_dir / "component_span_and_folding.png").write_text("stale\n", encoding="utf-8")

    result = _RUNNER.invoke(
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
            f"TetR={_TETO_PAYLOAD}",
            "--cap-sequence",
            f"C172={_SNAPBACK_CAP}",
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


def test_checked_in_registry_compiles_planned_scar_nick_hits(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    input_file = study_dir / "msd_design_hit_labels.txt"
    out_dir = tmp_path / "compiled"

    result = _RUNNER.invoke(
        app,
        [
            "compile",
            "--input",
            input_file.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 18
    assert sorted(item.name for item in out_dir.iterdir()) == [
        "README.md",
        "manifest.json",
        "msd_design_catalog_v1.json",
        "reference_index.tsv",
        "references",
    ]
    reference_files = sorted((out_dir / "references").glob("*.msd_design_reference_v1.json"))
    assert len(reference_files) == 18
    assert all(path.is_file() for path in reference_files)
    assert not any(path.is_dir() for path in (out_dir / "references").iterdir())

    selected_labels = [
        line.strip()
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert selected_labels == _SCAR_NICK_HIT_LABELS
    top_nick_ids = {
        record["construct_id"] for record in payload["records"] if record["scar_nick"]["nick_orientation"] == "top"
    }
    assert top_nick_ids == {"pES-retron-193", "pES-retron-194"}


def test_retron_msd_compiler_is_not_exposed_as_top_level_project_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert "retron-msd" not in scripts
    assert all("retron_hairpin_design.cli" not in target for target in scripts.values())


def test_retron_msd_study_uses_public_tool_apis_only() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    study_paths = sorted((repo_root / "src" / "dnadesign" / "studies" / "retron_hairpin_design").glob("*.py"))
    imports: set[str] = set()
    for path in study_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert "dnadesign.construct" in imports
    assert "dnadesign.construct.src.composition" not in imports
    assert not any(name == "dnadesign.cruncher" or name.startswith("dnadesign.cruncher.src") for name in imports)
    assert not any(name.startswith("dnadesign.cruncher.workspaces") for name in imports)
    assert not any(name.startswith("dnadesign.folding.src") for name in imports)


def test_retron_msd_materialize_does_not_shell_out_to_inkscape() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    compiler_source = (repo_root / "src" / "dnadesign" / "studies" / "retron_hairpin_design" / "compiler.py").read_text(
        encoding="utf-8"
    )

    assert "inkscape" not in compiler_source.lower()
    assert "subprocess.run" not in compiler_source
