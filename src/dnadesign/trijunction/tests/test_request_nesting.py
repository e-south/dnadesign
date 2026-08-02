"""Regression coverage for deeply nested request documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.trijunction.cli import app
from dnadesign.trijunction.contracts.request import load_request
from dnadesign.trijunction.contracts.request.limits import MAX_REQUEST_BYTES
from dnadesign.trijunction.errors import TriJunctionConfigError

_NESTING_DEPTH = 20_000
_SENSITIVE_VALUE = "nested-content-must-not-leak"


def _nested_document(suffix: str) -> str:
    nested_value = "[" * _NESTING_DEPTH + json.dumps(_SENSITIVE_VALUE) + "]" * _NESTING_DEPTH
    if suffix == ".json":
        document = '{"schema":' + nested_value + "}"
    else:
        document = "schema: " + nested_value + "\n"
    assert len(document.encode("utf-8")) < MAX_REQUEST_BYTES
    return document


@pytest.mark.parametrize(("suffix", "format_name"), [(".json", "JSON"), (".yaml", "YAML")])
def test_load_request_rejects_excessive_document_nesting_as_config_error(
    tmp_path: Path,
    suffix: str,
    format_name: str,
) -> None:
    request = tmp_path / f"request{suffix}"
    request.write_text(_nested_document(suffix), encoding="utf-8")

    with pytest.raises(
        TriJunctionConfigError,
        match=rf"^Invalid {format_name} in TriJunction request: ",
    ) as caught:
        load_request(request)

    assert _SENSITIVE_VALUE not in str(caught.value)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_cli_reports_excessive_document_nesting_as_sanitized_json(
    tmp_path: Path,
    suffix: str,
) -> None:
    request = tmp_path / f"request{suffix}"
    request.write_text(_nested_document(suffix), encoding="utf-8")

    result = CliRunner().invoke(app, ["preflight", str(request), "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["error"] == {
        "code": "config_error",
        "message": f"Invalid {suffix[1:].upper()} in TriJunction request: {request}",
        "retryable": False,
    }
    assert _SENSITIVE_VALUE not in result.stderr
