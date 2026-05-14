"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_retron_msd_compiler.py

Tests for the Retron MSD design-id compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
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
    assert "provided profile" in payload["error"]


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
    reference_path = out_dir / "assets" / "msd-tetr-c172-lcggt-racag-mxmm" / "msd_design_reference_v1.json"
    assert catalog_path == out_dir / "msd_design_catalog_v1.json"
    assert reference_path.is_file()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["records"][0]["construct_id"] == "pES-retron-177"


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
