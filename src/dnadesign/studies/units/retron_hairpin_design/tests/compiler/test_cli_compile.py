"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cli_compile.py

Compile command tests for the Retron MSD compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ..support.cli import RUNNER
from ..support.registry import write_minimal_retron_msd_registry


def test_retron_msd_compile_cli_writes_catalog(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "compiled"

    result = RUNNER.invoke(
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
            "payload_trim_id": "",
            "payload_trim_class": "",
            "parent_payload_id": "",
            "pwm_source_ref": "",
            "cap_id": "C172",
            "variant_role": "",
            "scaffold_context": "",
            "cap_selector_id": "",
            "stem_base_selector_id": "",
            "rt_mode": "",
            "decision_group": "",
            "control_id": "",
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


def test_retron_msd_compile_cli_allows_non_ligatable_s0_with_explicit_flag(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "compiled"

    result = RUNNER.invoke(
        app,
        [
            "compile",
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

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    record = payload["records"][0]
    assert record["msd_design_id"] == "msd-tetr-C172-LCGGG-RACAG-MXMX"
    assert record["scar_nick"]["profile_s3s2s1s0"] == "MXMX"
    assert record["scar_nick"]["s0_match_required"] is False


def test_retron_msd_compile_text_reports_output_nudges(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "compiled"

    result = RUNNER.invoke(
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
    study_dir = write_minimal_retron_msd_registry(tmp_path)
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

    result = RUNNER.invoke(
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
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "compiled"
    (out_dir / "assets").mkdir(parents=True)

    result = RUNNER.invoke(
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
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    out_dir = tmp_path / "compiled"
    references_dir = out_dir / "references"
    references_dir.mkdir(parents=True)
    (references_dir / "stale.msd_design_reference_v1.json").write_text("{}\n", encoding="utf-8")

    result = RUNNER.invoke(
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
