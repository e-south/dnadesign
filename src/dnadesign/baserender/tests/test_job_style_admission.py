"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_job_style_admission.py

Verify shared CLI/API style admission without rendering or publishing outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dnadesign.baserender import SchemaError, validate_job

from .conftest import densegen_job_payload, write_job, write_parquet


def _payload(tmp_path: Path) -> dict:
    source = write_parquet(tmp_path / "input.parquet", [{"id": "r1", "sequence": "ACGT"}])
    return densegen_job_payload(
        parquet_path=source,
        bundle_path=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )


@pytest.mark.parametrize("surface", ["mapping", "path"])
@pytest.mark.parametrize(
    ("style", "message"),
    [
        ({"overrides": {"kmer": {"min_glyph_gap_px": -1}}}, "min_glyph_gap_px"),
        ({"overrides": {"kmer": {"unknown_padding": 1}}}, "Unknown style.kmer"),
        ({"preset": "not-a-packaged-style"}, "Style preset not found"),
    ],
)
def test_job_admission_rejects_invalid_styles_before_publication(tmp_path, surface, style, message) -> None:
    payload = _payload(tmp_path)
    payload["render"]["style"] = style
    source = payload if surface == "mapping" else write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match=message):
        validate_job(source, caller_root=tmp_path)

    assert not (tmp_path / "results").exists()


def test_cli_job_validation_uses_the_same_style_contract(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["render"]["style"]["overrides"] = {"kmer": {"min_glyph_gap_px": -1}}
    job_path = write_job(tmp_path / "job.yaml", payload)

    proc = subprocess.run(
        [sys.executable, "-B", "-m", "dnadesign.baserender.src.cli.app", "job", "validate", str(job_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "min_glyph_gap_px" in proc.stdout + proc.stderr
    assert not (tmp_path / "results").exists()


def test_job_validation_accepts_explicit_geometry_without_rendering(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["render"]["style"]["overrides"] = {"kmer": {"min_glyph_gap_px": 2, "pad_y_px": 3}}

    job = validate_job(payload, caller_root=tmp_path)

    assert job.render.style_overrides["kmer"] == {"min_glyph_gap_px": 2, "pad_y_px": 3}
    assert not (tmp_path / "results").exists()
