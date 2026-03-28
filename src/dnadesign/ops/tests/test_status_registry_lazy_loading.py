"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_status_registry_lazy_loading.py

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
    supported_specs = {spec.status_kind: spec for spec in list_status_kind_specs()}

    assert supported_specs["ops-audit-json"].provider_ref == "dnadesign.ops.status_providers:provide_ops_audit_status"
    assert supported_specs["ops-audit-json"].owner_boundary == "ops"
    assert supported_specs["ops-audit-json"].observes_plane == "control"
    assert supported_specs["usr-dataset-state"].provider_ref == (
        "dnadesign.usr.ops.status_providers:provide_usr_dataset_state_status"
    )
    assert supported_specs["usr-dataset-state"].owner_boundary == "usr"
    assert supported_specs["usr-dataset-state"].observes_plane == "data"
    assert supported_specs["promoter-study-preflight"].provider_ref == (
        "dnadesign.studies.families.promoter.ops.provider:provide_promoter_preflight"
    )
    assert supported_specs["promoter-study-preflight"].owner_boundary == "usr"
    assert supported_specs["promoter-study-preflight"].observes_plane == "execution_readiness"
    assert supported_specs["cluster-run-index"].provider_ref == (
        "dnadesign.cluster.ops.status_providers:provide_cluster_run_index_status"
    )
    assert supported_specs["cluster-run-index"].owner_boundary == "cluster"
    assert supported_specs["cluster-run-index"].observes_plane == "data"
    assert supported_specs["opal-campaign-state"].provider_ref == (
        "dnadesign.opal.ops.status_providers:provide_opal_campaign_state_status"
    )
    assert supported_specs["opal-campaign-state"].owner_boundary == "opal"
    assert supported_specs["opal-campaign-state"].observes_plane == "control"


def test_status_registry_loader_reads_metadata_without_provider_import() -> None:
    imported_modules = _run_python(
        """
import json
import sys
from dnadesign.ops.status.registry_loader import list_status_kind_specs

list_status_kind_specs()
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.status_providers',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.opal.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.studies.families.promoter.adapter',
        'dnadesign.studies.families.promoter.ops.provider',
    }
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
        'dnadesign.ops.cli.commands.catalog',
        'dnadesign.ops.cli.commands.progress',
        'dnadesign.ops.cli.commands.runbook',
    }
)))
"""
    )

    assert imported_modules == []


def test_orchestrator_package_import_does_not_preload_gate_or_execution_modules() -> None:
    imported_modules = _run_python(
        """
import json
import sys
import dnadesign.ops.orchestrator as orchestrator

assert orchestrator is not None
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.orchestrator.execute',
        'dnadesign.ops.orchestrator.gates',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
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
    if name in {
        'dnadesign.ops.status_providers',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.opal.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.studies.families.promoter.adapter',
        'dnadesign.studies.families.promoter.ops.provider',
    }
)))
"""
    )

    assert imported_modules == []


def test_status_kinds_cli_does_not_import_provider_modules() -> None:
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
        'dnadesign.ops.status_providers',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.opal.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.studies.families.promoter.adapter',
        'dnadesign.studies.families.promoter.ops.provider',
    }
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


def test_status_kinds_cli_does_not_import_runbook_execution_modules() -> None:
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


def test_status_kinds_cli_loads_registry_metadata_without_campaign_or_service_modules() -> None:
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
        'dnadesign.ops.status_providers',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.opal.ops.status_providers',
        'dnadesign.studies.families.promoter.adapter',
        'dnadesign.studies.families.promoter.ops.provider',
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
        'dnadesign.ops.status_providers',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.opal.ops.status_providers',
        'dnadesign.studies.families.promoter.adapter',
        'dnadesign.studies.families.promoter.ops.provider',
    }
)))
"""
    )

    assert imported_modules == ["dnadesign.ops.status_providers"]
