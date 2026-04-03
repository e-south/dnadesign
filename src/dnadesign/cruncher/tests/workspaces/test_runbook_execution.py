"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/workspaces/test_runbook_execution.py

Validate machine runbook loading and fail-fast execution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dnadesign.cruncher.workspaces.runbook as runbook_module
from dnadesign.cruncher.workspaces.runbook import load_workspace_runbook, run_workspace_runbook


def _write_runbook(workspace: Path, payload: dict) -> Path:
    runbook_path = workspace / "configs" / "runbook.yaml"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(yaml.safe_dump(payload))
    return runbook_path


def _copytree_without_ds_store(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".DS_Store"))
    return dest


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runbook_rejects_unknown_keys() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [{"id": "lock", "run": ["lock"]}],
            "unexpected": True,
        }
    }
    with pytest.raises(ValueError):
        load_workspace_runbook(Path("runbook.yaml"), raw=payload)


def test_runbook_rejects_disallowed_cli_surface() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [{"id": "danger", "run": ["rm", "-rf", "/"]}],
        }
    }
    with pytest.raises(ValueError, match="disallowed cruncher command"):
        load_workspace_runbook(Path("runbook.yaml"), raw=payload)


def test_runbook_accepts_cassette_cli_surface() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [
                {
                    "id": "cassette_solve_fast",
                    "run": ["cassette", "solve", "--spec", "configs/cassettes/demo_hairpin_fast.cassette.solve.yaml"],
                }
            ],
        }
    }

    runbook = load_workspace_runbook(Path("runbook.yaml"), raw=payload)

    assert runbook.steps[0].run[0] == "cassette"


def test_runbook_accepts_yiu_cli_surface() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [
                {
                    "id": "yiu_render",
                    "run": ["yiu", "render", "--spec", "configs/yiu/example.yiu.yaml"],
                }
            ],
        }
    }

    runbook = load_workspace_runbook(Path("runbook.yaml"), raw=payload)

    assert runbook.steps[0].run[0] == "yiu"


def test_runbook_executes_selected_steps_in_runbook_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [
                    {"id": "clean", "run": ["workspaces", "clean-transient", "--root", ".", "--confirm"]},
                    {"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]},
                    {"id": "parse", "run": ["parse", "--force-overwrite", "-c", "configs/config.yaml"]},
                ],
            }
        },
    )

    calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        calls.append([str(item) for item in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    result = run_workspace_runbook(runbook_path, step_ids=["parse", "lock"])

    assert result.executed_step_ids == ["lock", "parse"]
    assert calls == [
        ["uv", "run", "cruncher", "lock", "-c", "configs/config.yaml"],
        ["uv", "run", "cruncher", "parse", "--force-overwrite", "-c", "configs/config.yaml"],
    ]


def test_runbook_streams_step_output_instead_of_capturing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]}],
            }
        },
    )

    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    run_workspace_runbook(runbook_path)

    assert len(call_kwargs) == 1
    kwargs = call_kwargs[0]
    assert kwargs.get("capture_output") is not True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_runbook_routes_step_output_to_log_file_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]}],
            }
        },
    )
    output_log = tmp_path / "runbook.log"
    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        _ = cmd
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    run_workspace_runbook(runbook_path, output_log_path=output_log)

    assert len(call_kwargs) == 1
    kwargs = call_kwargs[0]
    assert "stdout" in kwargs
    assert kwargs.get("stderr") == runbook_module.subprocess.STDOUT
    assert output_log.exists()


def test_runbook_fails_fast_on_step_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [
                    {"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]},
                    {"id": "parse", "run": ["parse", "--force-overwrite", "-c", "configs/config.yaml"]},
                    {"id": "sample", "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"]},
                ],
            }
        },
    )

    calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        calls.append([str(item) for item in cmd])
        if cmd[3] == "parse":
            return SimpleNamespace(returncode=2, stdout="", stderr="parse failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    with pytest.raises(RuntimeError, match="Runbook step failed"):
        run_workspace_runbook(runbook_path)

    assert calls == [
        ["uv", "run", "cruncher", "lock", "-c", "configs/config.yaml"],
        ["uv", "run", "cruncher", "parse", "--force-overwrite", "-c", "configs/config.yaml"],
    ]


def test_runbook_sets_writable_home_for_child_processes_when_home_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "sample", "run": ["sample", "-c", "configs/config.yaml"]}],
            }
        },
    )

    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        _ = cmd
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(runbook_module.os, "environ", {"HOME": "/tmp/unwritable-home"})
    monkeypatch.setattr(runbook_module, "_is_writable_directory", lambda _: False)

    run_workspace_runbook(runbook_path)

    assert len(call_kwargs) == 1
    env = call_kwargs[0].get("env")
    assert isinstance(env, dict)
    expected_home = (workspace / ".cruncher" / ".runtime_home").resolve()
    assert Path(str(env["HOME"])).resolve() == expected_home
    assert expected_home.is_dir()


def test_runbook_sets_workspace_local_mpl_cache_for_child_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "sample", "run": ["sample", "-c", "configs/config.yaml"]}],
            }
        },
    )

    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        _ = cmd
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    run_workspace_runbook(runbook_path)

    assert len(call_kwargs) == 1
    env = call_kwargs[0].get("env")
    assert isinstance(env, dict)
    expected_cache = (workspace / ".cruncher" / ".runtime_mplconfig").resolve()
    assert Path(str(env["MPLCONFIGDIR"])).resolve() == expected_cache
    assert expected_cache.is_dir()


def test_checked_in_yiu_demo_runbook_executes_end_to_end_without_matplotlib_cache_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_workspace = Path("src/dnadesign/cruncher/workspaces/demo_yiu_payload")
    workspace = _copytree_without_ds_store(source_workspace, tmp_path / "demo_yiu_payload")
    shutil.rmtree(workspace / "outputs", ignore_errors=True)
    shutil.rmtree(workspace / "bundles", ignore_errors=True)
    shutil.rmtree(workspace / ".cruncher", ignore_errors=True)
    runbook_path = workspace / "configs" / "runbook.yaml"
    output_log = tmp_path / "demo-runbook.log"
    runtime_home = tmp_path / "home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(runtime_home))
    monkeypatch.setenv("CRUNCHER_NONINTERACTIVE", "1")
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    result = run_workspace_runbook(runbook_path, output_log_path=output_log)

    assert result.executed_step_ids == [
        "yiu_validate",
        "yiu_render",
        "yiu_show",
        "tetr_validate",
        "tetr_render",
        "tetr_show",
    ]
    for bundle_name in ("example_payload", "tetr_monotypic_hit"):
        bundle_dir = workspace / "bundles" / bundle_name
        assert (bundle_dir / "visual_inventory.json").exists()
        inventory = _load_json(bundle_dir / "visual_inventory.json")
        assert inventory["render_status"] == "rendered"
        assert inventory["render_count"] == 3
        assert inventory["bundle_contract"] == "split_yiu_payload_bundle_v4"
        assert inventory["input_contract"] == "split_yiu_payload_rendering_v4"
        assert [view["view_id"] for view in inventory["views"]] == ["payload", "split_payload", "assembled_payload"]
        assert (bundle_dir / "payload_views.pdf").exists()
        assert not (bundle_dir / "payload.pdf").exists()
        assert not (bundle_dir / "split_payload.pdf").exists()
        assert not (bundle_dir / "assembled_payload.pdf").exists()
        assert not (bundle_dir / "inline_job").exists()

        payload = _load_json(bundle_dir / "payload_view.json")
        assembled = _load_json(bundle_dir / "assembled_payload_view.json")
        split_rows = _load_jsonl(bundle_dir / "split_payload_view.json")

        assert payload["state_id"] == "payload"
        assert payload["contract_kind"] == "yiu_payload_visual_v1"
        assert assembled["state_id"] == "assembled_payload"
        assert assembled["contract_kind"] == "sequence_evidence_map_v1"
        assert assembled["primary_sequence"] == payload["selected_payload_sequence"]
        assert [row["state_id"] for row in split_rows] == ["split_payload_left", "split_payload_right"]
        assert split_rows[0]["meta"]["panel_order"] == 0
        assert split_rows[0]["meta"]["fragment_side"] == "left"
        assert split_rows[0]["meta"]["sticky_end_orientation"] == "inward"
        assert split_rows[1]["meta"]["panel_order"] == 1
        assert split_rows[1]["meta"]["fragment_side"] == "right"
        assert split_rows[1]["meta"]["sticky_end_orientation"] == "inward"
        assert assembled["boundaries"] == []
        assert "junction_span" in assembled["meta"]
        assert "ligation_junction" not in json.dumps(assembled)
        assert "linearization_seam" not in json.dumps(assembled)

    bundle_dir = workspace / "bundles" / "example_payload"

    log_text = output_log.read_text(encoding="utf-8")
    assert "Matplotlib is building the font cache" not in log_text
    assert "MPLCONFIGDIR" not in log_text

    job_path = bundle_dir / "payload.job.yaml"
    job_path.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "results_root": ".",
                "input": {
                    "kind": "json",
                    "path": "payload_view.json",
                    "adapter": {"kind": "yiu_payload_visual_v1"},
                    "alphabet": "iupac_dna",
                },
                "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
                "outputs": [{"kind": "images", "path": "payload_replay.pdf", "fmt": "pdf"}],
                "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    base_env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    validate_proc = subprocess.run(
        [sys.executable, "-m", "dnadesign.cruncher.cli.app", "visuals", "validate", "--job", str(job_path)],
        cwd=workspace,
        env=base_env,
        check=True,
        capture_output=True,
        text=True,
    )
    render_proc = subprocess.run(
        [sys.executable, "-m", "dnadesign.cruncher.cli.app", "visuals", "run", "--job", str(job_path)],
        cwd=workspace,
        env=base_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Matplotlib" not in validate_proc.stderr
    assert "MPLCONFIGDIR" not in validate_proc.stderr
    assert "Matplotlib" not in render_proc.stderr
    assert "MPLCONFIGDIR" not in render_proc.stderr
    assert (bundle_dir / "payload.job" / "payload_replay.pdf").exists()
