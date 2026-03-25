"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_progress_builtin_registry.py

Focused tests for metadata-only status registry loading and lazy provider import.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dnadesign.ops.status.registry_loader import list_status_kind_specs


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _run_python(code: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_status_registry_fragments_load_provider_owned_specs() -> None:
    supported_specs = {spec.progress_kind: spec.provider_id for spec in list_status_kind_specs()}

    assert supported_specs["ops-audit-json"] == "builtin.ops"
    assert supported_specs["usr-dataset-state"] == "builtin.usr"
    assert supported_specs["promoter-study-preflight"] == "study.stress_promoter_ethanol_cipro"
    assert supported_specs["cluster-run-index"] == "builtin.cluster"
    assert supported_specs["opal-campaign-state"] == "builtin.opal"


def test_status_registry_loader_reads_metadata_without_provider_import() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from dnadesign.ops.status.registry_loader import list_status_kind_specs

list_status_kind_specs()
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith('dnadesign.ops.providers.')
    or name.startswith('dnadesign.studies.stress_promoter_ethanol_cipro')
)))
"""
    )

    assert imported_modules == []


def test_root_ops_cli_import_stays_on_lazy_dispatch_path() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from dnadesign.ops.cli import app

assert app is not None
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops._cli_legacy',
        'dnadesign.ops.cli.commands.catalog',
        'dnadesign.ops.cli.commands.progress',
        'dnadesign.ops.cli.commands.runbook',
    }
)))
"""
    )

    assert imported_modules == []


def test_catalog_list_cli_does_not_import_provider_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['catalog', 'list', '--repo-root', '.','--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith('dnadesign.ops.providers.')
    or name.startswith('dnadesign.studies.stress_promoter_ethanol_cipro')
)))
"""
    )

    assert imported_modules == []


def test_progress_kinds_cli_does_not_import_provider_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['progress', 'kinds', '--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith('dnadesign.ops.providers.')
    or name.startswith('dnadesign.studies.stress_promoter_ethanol_cipro')
)))
"""
    )

    assert imported_modules == []


def test_progress_kinds_cli_does_not_import_legacy_cli_module() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['progress', 'kinds', '--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'dnadesign.ops._cli_legacy'
)))
"""
    )

    assert imported_modules == []


def test_catalog_list_cli_does_not_import_runbook_execution_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['catalog', 'list', '--repo-root', '.','--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.orchestrator.execute',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
        'dnadesign.ops.runbooks.schema',
    }
)))
"""
    )

    assert imported_modules == []


def test_catalog_list_cli_does_not_import_status_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['catalog', 'list', '--repo-root', '.','--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.status.campaign',
        'dnadesign.ops.status.registry_loader',
        'dnadesign.ops.status.service',
    }
)))
"""
    )

    assert imported_modules == []


def test_progress_kinds_cli_does_not_import_runbook_execution_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['progress', 'kinds', '--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.orchestrator.execute',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
        'dnadesign.ops.runbooks.schema',
    }
)))
"""
    )

    assert imported_modules == []


def test_progress_kinds_cli_loads_registry_metadata_without_campaign_or_service_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['progress', 'kinds', '--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.status.campaign',
        'dnadesign.ops.status.service',
    }
)))
"""
    )

    assert imported_modules == []


def test_runbook_presets_cli_does_not_import_execution_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['runbook', 'presets'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.orchestrator.execute',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
        'dnadesign.ops.runbooks.schema',
    }
)))
"""
    )

    assert imported_modules == []


def test_progress_explain_cli_reads_metadata_without_campaign_or_provider_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(app, ['progress', 'explain', 'ops.control-plane.orchestration', '--json'])
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.status.campaign',
        'dnadesign.ops.status.service',
        'dnadesign.ops.providers.ops.provider',
        'dnadesign.ops.providers.usr.provider',
        'dnadesign.ops.providers.cluster.provider',
        'dnadesign.ops.providers.opal.provider',
        'dnadesign.studies.stress_promoter_ethanol_cipro.family',
        'dnadesign.studies.stress_promoter_ethanol_cipro.ops_provider',
    }
)))
"""
    )

    assert imported_modules == []


def test_progress_show_imports_only_requested_provider() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from typer.testing import CliRunner
from dnadesign.ops.cli import app

result = CliRunner().invoke(
    app,
    ['progress', 'show', 'ops.control-plane.orchestration', '--repo-root', '.', '--audit-json', 'missing/latest.json']
)
assert result.exit_code == 0, result.output
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.providers.ops.provider',
        'dnadesign.ops.providers.usr.provider',
        'dnadesign.ops.providers.cluster.provider',
        'dnadesign.ops.providers.opal.provider',
        'dnadesign.ops._cli_legacy',
        'dnadesign.studies.stress_promoter_ethanol_cipro.family',
        'dnadesign.studies.stress_promoter_ethanol_cipro.ops_provider',
    }
)))
"""
    )

    assert imported_modules == ["dnadesign.ops.providers.ops.provider"]
