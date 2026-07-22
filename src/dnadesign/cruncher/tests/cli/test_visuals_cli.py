"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cli/test_visuals_cli.py

CLI contract tests for the generic Cruncher visual wrapper over BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_visuals_module_import_does_not_preload_matplotlib_or_emit_cache_warnings() -> None:
    code = "import sys\nfrom dnadesign.cruncher.cli.commands import visuals\nprint('matplotlib' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.strip().endswith("False")
    assert "Matplotlib" not in proc.stderr
    assert "MPLCONFIGDIR" not in proc.stderr


def test_cruncher_cli_app_import_does_not_preload_matplotlib_or_emit_cache_warnings() -> None:
    code = "import sys\nimport dnadesign.cruncher.cli.app\nprint('matplotlib' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.strip().endswith("False")
    assert "Matplotlib" not in proc.stderr
    assert "MPLCONFIGDIR" not in proc.stderr


def test_visuals_validate_and_run_delegate_through_public_baserender_surface(tmp_path: Path) -> None:
    contract_path = _write_json(
        tmp_path / "published" / "views" / "candidate.json",
        {
            "contract_kind": "yiu_topology_cartoon_v1",
            "state_id": "circularized_payload_candidate",
            "topology_kind": "circular_duplex",
            "sequence": "CCGATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
            "segments": [
                {"segment_id": "payload", "state_start": 0, "state_end": 15},
                {"segment_id": "retained", "state_start": 15, "state_end": 51},
            ],
            "annotations": [],
            "cuts": [],
            "junctions": [{"id": "junction", "join_index": 15}],
            "fragments": [],
            "display": {"title": "Circularized payload"},
            "meta": {"evidence_mode": "concrete_realization"},
        },
    )
    job_path = tmp_path / "published" / "baserender_jobs" / "candidate.job.yaml"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(
        "\n".join(
            [
                "version: 3",
                "results_root: ..",
                "input:",
                "  kind: json",
                f"  path: ../views/{contract_path.name}",
                "  adapter:",
                "    kind: yiu_topology_cartoon_v1",
                "  alphabet: DNA",
                "render:",
                "  renderer: topology_cartoon",
                "  style:",
                "    preset: null",
                "    overrides: {}",
                "outputs:",
                "  - kind: images",
                "    path: ../renders/candidate.pdf",
                "    fmt: pdf",
                "run:",
                "  strict: true",
                "  fail_on_skips: true",
                "  emit_report: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    validate_result = runner.invoke(app, ["visuals", "validate", "--job", str(job_path)], color=False)
    run_result = runner.invoke(app, ["visuals", "run", "--job", str(job_path)], color=False)

    assert validate_result.exit_code == 0
    assert "Render job kind -> render_job_v3" in validate_result.output
    assert "Renderer -> topology_cartoon" in validate_result.output
    assert run_result.exit_code == 0
    assert "Rendered job ->" in run_result.output
    assert (tmp_path / "published" / "renders" / "candidate.pdf").exists()
