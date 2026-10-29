"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_api_cli.py

Public-facade and CLI lifecycle tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.junction import api as api_module
from dnadesign.junction import preflight, render_sequence_dissimilarity_svg
from dnadesign.junction.cli import app
from dnadesign.junction.contracts.request import MAX_REQUEST_BYTES, MAX_REQUEST_IDENTIFIER_BYTES, parse_request
from dnadesign.junction.errors import JunctionBundleError, JunctionConfigError, JunctionDesignError
from dnadesign.junction.sequence import reverse_complement
from dnadesign.junction.tests.test_planner import _request_mapping

runner = CliRunner()


def _request_file(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    return path


def test_public_facade_is_lazy() -> None:
    code = (
        "import sys; import dnadesign.junction; "
        "assert 'typer' not in sys.modules; "
        "assert 'dnadesign.artifacts' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_core_api_and_cli_imports_do_not_load_plotting_stack() -> None:
    code = (
        "import sys; import dnadesign.junction.api; import dnadesign.junction.cli; "
        "assert 'matplotlib' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules"
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


def test_sequence_comparison_svg_is_public_deterministic_and_write_free(tmp_path: Path) -> None:
    request = _request_file(tmp_path)
    planned = api_module.plan(request)
    group_id = planned.assembly_groups[0].assembly_group_id
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    first = render_sequence_dissimilarity_svg(planned, assembly_group_id=group_id)
    second = render_sequence_dissimilarity_svg(planned, assembly_group_id=group_id)

    assert first == second
    assert first.startswith(b"<?xml")
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def _boundary_request_file(tmp_path: Path, *, target_length: int) -> Path:
    raw = _request_mapping()
    raw["planning"]["search_range"] = 1  # type: ignore[index]
    target = raw["targets"][0]  # type: ignore[index]
    sequence = target["sequence"][:target_length]
    target["sequence"] = sequence
    target["recovery_primers"]["forward"]["binding_sequence"] = sequence[:8]
    target["recovery_primers"]["reverse"]["binding_sequence"] = reverse_complement(sequence[-8:])
    path = tmp_path / f"boundary-{target_length}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_empty_terminal_domain_fails_every_prepublication_surface_without_mutation(tmp_path: Path) -> None:
    request = _boundary_request_file(tmp_path, target_length=30)
    destination = tmp_path / "missing-parent" / "must-not-exist"

    with pytest.raises(JunctionDesignError, match="nonempty terminal domain"):
        api_module.plan(request)
    with pytest.raises(JunctionDesignError, match="nonempty terminal domain"):
        api_module.preflight(request)
    with pytest.raises(JunctionDesignError, match="nonempty terminal domain"):
        api_module.build(request, destination=destination)

    assert not destination.parent.exists()


def test_one_base_terminal_domain_builds_renders_and_verifies(tmp_path: Path) -> None:
    request = _boundary_request_file(tmp_path, target_length=31)
    destination = tmp_path / "design-v2"

    plan = api_module.plan(request)
    published = api_module.build(request, destination=destination)
    verified = api_module.verify(destination)

    assert plan.targets[0].fragments[-1].domain_end - plan.targets[0].fragments[-1].domain_start == 1
    assert published.plan_id == plan.plan_id
    assert verified.plan_id == plan.plan_id


def test_preflight_fails_closed_when_internal_thermodynamic_status_drifts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request_file(tmp_path)
    canonical = api_module.plan(request)
    mutated_assembly_groups = tuple(
        replace(
            assembly_group,
            search=replace(assembly_group.search, thermodynamic_screening="passed"),  # type: ignore[arg-type]
        )
        for assembly_group in canonical.assembly_groups
    )
    monkeypatch.setattr(
        api_module,
        "design_junction",
        lambda _request: replace(canonical, assembly_groups=mutated_assembly_groups),
    )

    with pytest.raises(JunctionDesignError, match="requires thermodynamic screening.*not_run"):
        api_module.preflight(request)


def test_cli_exposes_one_canonical_lifecycle() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "request" in result.stdout
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
    assert "already exists; publication is create-only" in second.stderr


def test_build_rejects_existing_destination_before_design(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request_file(tmp_path)
    destination = tmp_path / "design-v1"
    destination.mkdir()

    def unexpected_design(_request: object) -> None:
        raise AssertionError("design must not run for an invalid destination")

    monkeypatch.setattr(api_module, "design_junction", unexpected_design)

    with pytest.raises(JunctionBundleError, match="already exists; publication is create-only"):
        api_module.build(request, destination=destination)


def test_preflight_and_build_reject_oversized_in_memory_request_before_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = parse_request(_request_mapping())
    source_target = request.targets[0]
    sequence = "ACGT" * (MAX_REQUEST_BYTES // 4)
    recovery_primers = replace(
        source_target.recovery_primers,
        forward=replace(source_target.recovery_primers.forward, binding_sequence=sequence[:8]),
        reverse=replace(source_target.recovery_primers.reverse, binding_sequence="ACGTACGT"),
    )
    oversized = replace(
        request,
        targets=(replace(source_target, sequence=sequence, recovery_primers=recovery_primers),),
    )
    destination = tmp_path / "design-v1"

    def unexpected_design(_request: object) -> None:
        raise AssertionError("design must not run for an oversized request")

    monkeypatch.setattr(api_module, "design_junction", unexpected_design)

    with pytest.raises(JunctionConfigError, match="canonical request exceeds.*input limit"):
        api_module.preflight(oversized)
    with pytest.raises(JunctionConfigError, match="canonical request exceeds.*input limit"):
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


@pytest.mark.parametrize(
    ("suffix", "content", "message_kind"),
    [
        (".json", lambda value: f'{{"seed":{value}}}', "JSON"),
        (".yaml", lambda value: f"seed: {value}", "YAML"),
    ],
)
def test_cli_failure_wraps_parser_integer_limit_as_config_error(
    tmp_path: Path,
    suffix: str,
    content: Callable[[str], str],
    message_kind: str,
) -> None:
    request = tmp_path / f"request{suffix}"
    oversized_integer = "9" * 5_000
    request.write_text(content(oversized_integer), encoding="utf-8")

    result = runner.invoke(app, ["preflight", str(request), "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload == {
        "error": {
            "code": "config_error",
            "message": f"Invalid {message_kind} in junction request: {request}",
            "retryable": False,
        },
        "status": "error",
    }
    assert oversized_integer not in result.stderr


def test_cli_rejects_fraction_integer_too_large_for_float_without_echoing_it(tmp_path: Path) -> None:
    raw = _request_mapping()
    oversized_fraction = 10**400
    raw["planning"]["barcode_gc_min"] = oversized_fraction  # type: ignore[index]
    request = tmp_path / "request.json"
    request.write_text(json.dumps(raw), encoding="utf-8")

    result = runner.invoke(app, ["preflight", str(request), "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload == {
        "error": {
            "code": "config_error",
            "message": "planning.barcode_gc_min must be between 0 and 1",
            "retryable": False,
        },
        "status": "error",
    }
    assert str(oversized_fraction) not in result.stderr


def test_cli_rejects_oversized_identifier_without_echoing_it(tmp_path: Path) -> None:
    raw = _request_mapping()
    oversized_identifier = "a" * (MAX_REQUEST_IDENTIFIER_BYTES + 1)
    raw["targets"][0]["id"] = oversized_identifier  # type: ignore[index]
    request = tmp_path / "request.json"
    request.write_text(json.dumps(raw), encoding="utf-8")

    result = runner.invoke(app, ["preflight", str(request), "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload["error"] == {
        "code": "config_error",
        "message": "targets[0].id must not exceed 128 ASCII bytes",
        "retryable": False,
    }
    assert oversized_identifier not in result.stderr


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
