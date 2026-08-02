"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_api_cli.py

Public-facade and CLI lifecycle tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.trijunction import api as api_module
from dnadesign.trijunction import preflight
from dnadesign.trijunction.cli import app
from dnadesign.trijunction.contracts.request import MAX_REQUEST_BYTES, parse_request
from dnadesign.trijunction.errors import TriJunctionBundleError, TriJunctionConfigError
from dnadesign.trijunction.tests.test_planner import _request_mapping

runner = CliRunner()


def _request_file(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    return path


def test_public_facade_is_lazy() -> None:
    code = (
        "import sys; import dnadesign.trijunction; "
        "assert 'typer' not in sys.modules; "
        "assert 'dnadesign.artifacts' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_preflight_api_creates_no_durable_artifacts(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = preflight(request)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result.status == "planned"
    assert result.validation_scope == "string_only"
    assert result.to_mapping()["validation_scope"] == "string_only"
    assert before == after == [Path("request.json")]


def test_cli_exposes_one_canonical_lifecycle() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "preflight" in result.stdout
    assert "plan" in result.stdout
    assert "build" in result.stdout
    assert "verify" in result.stdout


def test_cli_preflight_reports_planned_string_only_scope(tmp_path: Path) -> None:
    request = _request_file(tmp_path)

    result = runner.invoke(app, ["preflight", str(request), "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert payload["validation_scope"] == "string_only"


def test_cli_build_and_verify_json(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "design-v1"

    built = runner.invoke(
        app,
        ["build", str(request), "--output", str(destination), "--format", "json"],
    )
    verified = runner.invoke(app, ["verify", str(destination), "--format", "json"])

    assert built.exit_code == 0, built.output
    assert verified.exit_code == 0, verified.output
    assert json.loads(built.stdout)["status"] == "published"
    assert json.loads(verified.stdout)["status"] == "verified"


def test_cli_existing_destination_fails_without_overwrite(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "design-v1"
    first = runner.invoke(app, ["build", str(request), "--output", str(destination)])

    second = runner.invoke(app, ["build", str(request), "--output", str(destination)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 1
    assert "already exists and is immutable" in second.stderr


def test_build_rejects_existing_destination_before_design(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "design-v1"
    destination.mkdir()

    def unexpected_design(_request: object) -> None:
        raise AssertionError("design must not run for an invalid destination")

    monkeypatch.setattr(api_module, "design_trijunction", unexpected_design)

    with pytest.raises(TriJunctionBundleError, match="already exists and is immutable"):
        api_module.build(request, destination=destination)


def test_preflight_and_build_reject_oversized_in_memory_request_before_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = parse_request(_request_mapping())
    oversized = replace(
        request,
        order_policy=replace(request.order_policy, synthesis_scale="x" * MAX_REQUEST_BYTES),
    )
    destination = tmp_path / "design-v1"

    def unexpected_design(_request: object) -> None:
        raise AssertionError("design must not run for an oversized request")

    monkeypatch.setattr(api_module, "design_trijunction", unexpected_design)

    with pytest.raises(TriJunctionConfigError, match="canonical request exceeds.*input limit"):
        api_module.preflight(oversized)
    with pytest.raises(TriJunctionConfigError, match="canonical request exceeds.*input limit"):
        api_module.build(oversized, destination=destination)
    assert not destination.exists()


def test_cli_json_failure_is_machine_readable_and_non_retryable(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "design-v1"
    first = runner.invoke(app, ["build", str(request), "--output", str(destination)])

    second = runner.invoke(
        app,
        ["build", str(request), "--output", str(destination), "--format", "json"],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 1
    payload = json.loads(second.stderr)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "bundle_error"
    assert payload["error"]["retryable"] is False


def test_cli_rejects_output_format_before_publication(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "new-parent" / "design-v1"

    result = runner.invoke(
        app,
        ["build", str(request), "--output", str(destination), "--format", "bogus"],
    )

    assert result.exit_code == 2
    assert "Output format must be text or json" in result.output
    assert not destination.parent.exists()
