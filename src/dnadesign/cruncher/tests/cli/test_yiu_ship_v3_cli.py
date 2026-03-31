"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_yiu_ship_v3_cli.py

CLI ship contracts for the canonical YIU v4 surface.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.cli.commands.yiu import run_yiu_render

runner = CliRunner()


def test_root_help_includes_yiu_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "yiu" in result.output


def test_yiu_help_removes_design_and_adds_render() -> None:
    result = runner.invoke(app, ["yiu", "--help"])

    assert result.exit_code == 0
    assert "design" not in result.output
    assert "trace" in result.output
    assert "render" in result.output


def test_yiu_render_reads_bundle_root_visual_inventory_and_reports_outputs(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "yiu" / "explicit" / "demo" / "abc123"
    visuals_dir = run_dir / "visuals"
    contracts_dir = run_dir / "contracts" / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "state_a.json").write_text("{}", encoding="utf-8")
    (contracts_dir / "state_b.json").write_text("{}", encoding="utf-8")
    (run_dir / "visual_inventory.json").write_text(
        json.dumps(
            {
                "bundle_kind": "explicit",
                "protocol_template": "yiu_circularized_payload_v1",
                "renderer_kind": "nucleotide_evidence_map",
                "views": [
                    {
                        "state_id": "state_a",
                        "contract_kind": "sequence_evidence_map_v1",
                        "view_contract_path": "contracts/visuals/state_a.json",
                        "render_artifact_path": "visuals/state_a.pdf",
                        "renderer_kind": "nucleotide_evidence_map",
                        "render_requested": False,
                        "render_completed": False,
                    },
                    {
                        "state_id": "state_b",
                        "contract_kind": "sequence_evidence_map_v1",
                        "view_contract_path": "contracts/visuals/state_b.json",
                        "render_artifact_path": "visuals/state_b.pdf",
                        "renderer_kind": "nucleotide_evidence_map",
                        "render_requested": False,
                        "render_completed": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def _fake_render(run: Path) -> dict[str, object]:
        calls.append(run)
        return {
            "run_dir": str(run.resolve()),
            "job_count": 2,
            "rendered_count": 2,
            "render_paths": [
                str((visuals_dir / "state_a.pdf").resolve()),
                str((visuals_dir / "state_b.pdf").resolve()),
            ],
        }

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.yiu.run_yiu_render",
        _fake_render,
        raising=False,
    )

    result = runner.invoke(app, ["yiu", "render", "--run", str(run_dir)])

    assert result.exit_code == 0
    assert calls == [run_dir]
    assert "Rendered jobs -> 2" in result.output
    assert "visual_inventory.json" in result.output


def test_yiu_init_workspace_scaffolds_only_canonical_v4_files(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_ship_v4"

    result = runner.invoke(app, ["yiu", "init-workspace", "--output", str(workspace)])

    assert result.exit_code == 0
    assert (workspace / "configs" / "yiu" / "example_reference_circularized.yiu.yaml").exists()
    assert (workspace / "configs" / "yiu" / "example_reference_circularized.yiu.solve.yaml").exists()
    assert not (workspace / "configs" / "yiu" / "compat").exists()

    explicit_payload = yaml.safe_load(
        (workspace / "configs" / "yiu" / "example_reference_circularized.yiu.yaml").read_text(encoding="utf-8")
    )
    solve_payload = yaml.safe_load(
        (workspace / "configs" / "yiu" / "example_reference_circularized.yiu.solve.yaml").read_text(encoding="utf-8")
    )
    assert explicit_payload["yiu"]["schema_version"] == 4
    assert explicit_payload["yiu"]["payload"]["target_sequence"] == "AGGTCTCACACCTATAGAG"
    assert explicit_payload["yiu"]["owner_lifecycle"][0]["owner_id"] == "source_fwd_primer_binding_region"
    assert explicit_payload["yiu"]["output"]["publish_contract_version"] == 4
    assert solve_payload["yiu_solve"]["scaffold_windows"][0]["owner_id"] == "sacrificial_region_long"
    assert solve_payload["yiu_solve"]["solve"]["max_solutions"] == 1


def test_run_yiu_render_updates_visual_inventory_after_success(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "yiu" / "solve" / "demo" / "abc123"
    visuals_dir = run_dir / "solution" / "visuals"
    contracts_dir = run_dir / "solution" / "contracts" / "visuals"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "hairpin_pcr_linear_insert.json").write_text("{}", encoding="utf-8")
    inventory_path = run_dir / "visual_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "bundle_kind": "solve",
                "protocol_template": "yiu_circularized_payload_v1",
                "renderer_kind": "nucleotide_evidence_map",
                "render_status": "not_requested",
                "render_count": 0,
                "last_rendered_at": None,
                "views": [
                    {
                        "state_id": "hairpin_pcr_linear_insert",
                        "contract_kind": "sequence_evidence_map_v1",
                        "view_contract_path": "solution/contracts/visuals/hairpin_pcr_linear_insert.json",
                        "render_artifact_path": "solution/visuals/hairpin_pcr_linear_insert.pdf",
                        "renderer_kind": "nucleotide_evidence_map",
                        "render_requested": False,
                        "render_completed": False,
                        "last_rendered_at": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.yiu.ensure_workspace_mpl_cache", lambda *_args, **_kwargs: None, raising=False
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.yiu.ensure_mpl_cache", lambda *_args, **_kwargs: None, raising=False
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.yiu.infer_workspace_root_from_output_artifact",
        lambda _path: run_dir,
        raising=False,
    )

    captured_jobs: list[dict[str, object]] = []

    def _fake_run_job(job: dict[str, object], *, kind: str, caller_root: Path) -> None:
        _ = kind, caller_root
        captured_jobs.append(job)
        outputs = job["outputs"][0]
        output_path = Path(str(outputs["path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("pdf", encoding="utf-8")

    fake_baserender = types.SimpleNamespace(run_job=_fake_run_job)
    monkeypatch.setitem(sys.modules, "dnadesign.baserender", fake_baserender)

    payload = run_yiu_render(run_dir)

    assert payload["job_count"] == 1
    assert payload["rendered_count"] == 1
    assert captured_jobs
    assert captured_jobs[0]["input"]["path"] == str((contracts_dir / "hairpin_pcr_linear_insert.json").resolve())
    assert captured_jobs[0]["outputs"][0]["path"] == str((visuals_dir / "hairpin_pcr_linear_insert.pdf").resolve())

    updated_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert updated_inventory["render_count"] == 1
    assert updated_inventory["render_status"] == "rendered"
    assert updated_inventory["last_rendered_at"] is not None
    assert all(view["render_completed"] is True for view in updated_inventory["views"])
    assert all(view["last_rendered_at"] is not None for view in updated_inventory["views"])
