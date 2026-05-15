"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_retron_msd_compiler.py

Tests for the Retron MSD design-id compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from typer.testing import CliRunner

from dnadesign.studies.retron_hairpin_design.cli import app
from dnadesign.studies.retron_hairpin_design.msd_ids import (
    MsdIdError,
    compute_scar_nick_profile,
    parse_msd_construct_label,
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


def _write_registry(tmp_path: Path) -> Path:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "msd_design_registry.yaml").write_text(
        """
contract: retron_msd_design_registry_v1
schema_version: 1
payloads:
  TetR:
    display_name: TetR
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
    assert parsed.msd_design_id == "msd-tetr-c172-lcggt-racag-mxmm"


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
    assert payload["reference"]["msd_design_id"] == "msd-tetr-c172-lcggt-racag-mxmm"
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
    reference_path = out_dir / "references" / "msd-tetr-c172-lcggt-racag-mxmm.msd_design_reference_v1.json"
    assert catalog_path == out_dir / "msd_design_catalog_v1.json"
    assert Path(payload["output_dir"]) == out_dir
    assert Path(payload["references_dir"]) == out_dir / "references"
    assert Path(payload["index_path"]) == out_dir / "reference_index.tsv"
    assert Path(payload["manifest_path"]) == out_dir / "manifest.json"
    assert Path(payload["readme_path"]) == out_dir / "README.md"
    assert "run materialize with explicit payload/cap sequences" in payload["next_step"]
    assert "one GenBank/PNG sequence bundle per MSD design" in payload["next_step"]
    assert not (out_dir / "assets").exists()
    assert reference_path.is_file()

    index_rows = list(
        csv.DictReader((out_dir / "reference_index.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t")
    )
    assert index_rows == [
        {
            "construct_id": "pES-retron-177",
            "msd_design_id": "msd-tetr-c172-lcggt-racag-mxmm",
            "payload_id": "TetR",
            "cap_id": "C172",
            "left_base": "CGGT",
            "right_base": "ACAG",
            "profile_s3s2s1s0": "MXMM",
            "route_status": "note_only",
            "nick_orientation": "",
            "nickase": "",
            "reference_path": "references/msd-tetr-c172-lcggt-racag-mxmm.msd_design_reference_v1.json",
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


def test_retron_msd_materialize_writes_single_unit_genbank_png_and_reverse_complement(tmp_path: Path) -> None:
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
    assert Path(payload["sequence_manifest_path"]) == out_dir / "sequence_manifest.json"
    assert Path(payload["sequence_index_path"]) == out_dir / "sequence_index.tsv"
    assert Path(payload["variants_dir"]) == out_dir / "variants"
    assert Path(payload["composition_configs_dir"]) == out_dir / "composition_configs"
    assert payload["finder_open"] == f"open {out_dir}"
    assert "Single-unit MSD sequence bundle emitted" in payload["next_step"]

    variant = payload["variants"][0]
    variant_dir = out_dir / "variants" / "msd-tetr-c172-lcggt-racag-mxmm"
    genbank_path = variant_dir / "sequence.gb"
    features_path = variant_dir / "features.csv"
    png_path = variant_dir / "component_span_qa.png"
    assert variant["unit_count"] == 1
    assert Path(variant["genbank"]) == Path("variants/msd-tetr-c172-lcggt-racag-mxmm/sequence.gb")
    assert Path(variant["component_span_png"]) == Path("variants/msd-tetr-c172-lcggt-racag-mxmm/component_span_qa.png")
    assert genbank_path.is_file()
    assert features_path.is_file()
    assert png_path.is_file()
    assert png_path.stat().st_size > 0

    rows = list(
        csv.DictReader((out_dir / "sequence_index.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t")
    )
    assert rows[0]["unit_count"] == "1"
    assert rows[0]["genbank"] == "variants/msd-tetr-c172-lcggt-racag-mxmm/sequence.gb"
    assert rows[0]["component_span_png"] == "variants/msd-tetr-c172-lcggt-racag-mxmm/component_span_qa.png"
    assert rows[0]["finder_reveal"].startswith("open -R ")

    catalog = json.loads((out_dir / "msd_design_catalog_v1.json").read_text(encoding="utf-8"))
    record = catalog["records"][0]
    flank_5p_len = len("gtcagaaaaaa") + 4
    flank_3p_len = 4 + len("acagtaactcaga")
    unit_len = flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP) + len(_TETO_PAYLOAD) + flank_3p_len
    assert record["sequence"]["length"] == unit_len
    assert record["source"]["dnadesign_bundle"] == "variants/msd-tetr-c172-lcggt-racag-mxmm"
    assert record["artifacts"]["genbank"] == "variants/msd-tetr-c172-lcggt-racag-mxmm/sequence.gb"
    assert record["artifacts"]["component_span_png"] == (
        "variants/msd-tetr-c172-lcggt-racag-mxmm/component_span_qa.png"
    )

    genbank_record = next(SeqIO.parse(genbank_path, "genbank"))
    features_by_label = {
        feature.qualifiers["label"][0]: feature for feature in genbank_record.features if "label" in feature.qualifiers
    }
    payload_complement = features_by_label["payload_complement copy 0"]
    assert int(payload_complement.location.start) == flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP)
    assert int(payload_complement.location.end) == (
        flank_5p_len + len(_TETO_PAYLOAD) + len(_SNAPBACK_CAP) + len(_TETO_PAYLOAD)
    )
    assert payload_complement.location.strand == -1
    assert str(payload_complement.extract(genbank_record.seq)).upper() == _TETO_PAYLOAD.upper()

    feature_rows = list(csv.DictReader(features_path.read_text(encoding="utf-8").splitlines()))
    payload_complement_rows = [
        row for row in feature_rows if row["feature_kind"] == "annotation" and row["feature_id"] == "payload_complement"
    ]
    assert payload_complement_rows[0]["strand"] == "-1"
    assert payload_complement_rows[0]["source_segment_id"] == "payload_primary"
    assert payload_complement_rows[0]["transform_kind"] == "reverse_complement"
    assert payload_complement_rows[0]["genbank_location"].startswith("complement(")
    assert payload_complement_rows[0]["sequence"].upper() == str(Seq(_TETO_PAYLOAD).reverse_complement()).upper()


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
