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
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import dnadesign.ops.status.registry_loader as registry_loader
from dnadesign.ops.status.registry_loader import list_status_kind_specs


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _run_python(code: str) -> object:
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

    assert (
        supported_specs["ops-audit-json"].provider_ref
        == "dnadesign.ops.providers.builtin.status_provider:provide_ops_audit_status"
    )
    assert supported_specs["ops-audit-json"].owner_boundary == "ops"
    assert supported_specs["ops-audit-json"].observes_plane == "control"
    assert supported_specs["usr-dataset-state"].provider_ref == (
        "dnadesign.usr.ops.status_providers:provide_usr_dataset_state_status"
    )
    assert supported_specs["usr-dataset-state"].owner_boundary == "usr"
    assert supported_specs["usr-dataset-state"].observes_plane == "data"
    assert supported_specs["cluster-run-index"].provider_ref == (
        "dnadesign.cluster.ops.status_providers:provide_cluster_run_index_status"
    )
    assert supported_specs["cluster-run-index"].owner_boundary == "cluster"
    assert supported_specs["cluster-run-index"].observes_plane == "data"
    assert supported_specs["latentdna-workspace-snapshot"].provider_ref == (
        "dnadesign.latentdna.ops.status_providers:provide_latentdna_workspace_snapshot_status"
    )
    assert supported_specs["latentdna-workspace-snapshot"].owner_boundary == "latentdna"
    assert supported_specs["latentdna-workspace-snapshot"].observes_plane == "data"
    assert supported_specs["opal-campaign-state"].provider_ref == (
        "dnadesign.opal.src.ops.status_providers:provide_opal_campaign_state_status"
    )
    assert supported_specs["opal-campaign-state"].owner_boundary == "opal"
    assert supported_specs["opal-campaign-state"].observes_plane == "control"


def test_builtin_ops_status_registry_lives_under_provider_package() -> None:
    ops_root = _repo_root() / "src" / "dnadesign" / "ops"

    assert not (ops_root / "status.registry.yaml").exists()
    assert not (ops_root / "status_providers.py").exists()
    assert (ops_root / "providers" / "builtin" / "status.registry.yaml").is_file()
    assert (ops_root / "providers" / "builtin" / "status_provider.py").is_file()


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
        'dnadesign.ops.providers.builtin.status_provider',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.latentdna.ops.status_providers',
        'dnadesign.opal.src.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
    }
)))
"""
    )

    assert imported_modules == []


def test_status_registry_fragments_reject_unknown_top_level_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fragment = tmp_path / "status.registry.yaml"
    fragment.write_text(
        """
version: 1
provider_id: demo.provider
entries:
  - status_kind: demo-status
    owner_boundary: demo
    observes_plane: data
    provider_ref: dnadesign.demo.ops.status:provide_demo
extra_metadata: stale
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_loader, "_iter_status_registry_fragment_paths", lambda: (fragment,))
    registry_loader.list_status_kind_specs.cache_clear()
    try:
        with pytest.raises(ValueError, match="unknown key"):
            registry_loader.list_status_kind_specs()
    finally:
        registry_loader.list_status_kind_specs.cache_clear()


def test_status_registry_fragments_require_explicit_surface_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fragment = tmp_path / "status.registry.yaml"
    fragment.write_text(
        """
version: 1
provider_id: demo.provider
entries:
  - status_kind: demo-status
    owner_boundary: demo
    observes_plane: data
    provider_ref: dnadesign.demo.ops.status:provide_demo
    surface_type: artifact_state
    cost_class: cheap
    summary_scope: workspace
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_loader, "_iter_status_registry_fragment_paths", lambda: (fragment,))
    registry_loader.list_status_kind_specs.cache_clear()
    try:
        with pytest.raises(ValueError, match="missing required key\\(s\\) description"):
            registry_loader.list_status_kind_specs()
    finally:
        registry_loader.list_status_kind_specs.cache_clear()


def test_status_registry_fragments_reject_provider_ref_outside_fragment_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fragment = tmp_path / "status.registry.yaml"
    fragment.write_text(
        """
version: 1
provider_id: demo.provider
entries:
  - status_kind: demo-status
    owner_boundary: demo
    observes_plane: data
    provider_ref: dnadesign.other.ops.status:provide_demo
    description: Demo status.
    surface_type: artifact_state
    cost_class: cheap
    summary_scope: workspace
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_loader, "_iter_status_registry_fragment_paths", lambda: (fragment,))
    monkeypatch.setattr(
        registry_loader,
        "_expected_provider_ref_prefix",
        lambda *, fragment_path, dnadesign_root: "dnadesign.demo.ops.",
    )
    registry_loader.list_status_kind_specs.cache_clear()
    try:
        with pytest.raises(ValueError, match="provider_ref must stay under the fragment owner package"):
            registry_loader.list_status_kind_specs()
    finally:
        registry_loader.list_status_kind_specs.cache_clear()


def test_external_status_registry_entry_point_is_owner_confined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment = tmp_path / "status.registry.yaml"
    fragment.write_text(
        """
version: 1
provider_id: research-studies.demo
entries:
  - status_kind: private-study-status
    owner_boundary: studies
    observes_plane: record
    provider_ref: research_studies.demo.status:provide_status
    description: Read one private study record.
    surface_type: study_record
    cost_class: cheap
    summary_scope: repo
""",
        encoding="utf-8",
    )
    entry_point = SimpleNamespace(
        name="research-studies",
        value="research_studies.integrations.dnadesign_ops:status_registry_paths",
        load=lambda: lambda: (fragment,),
    )
    monkeypatch.setattr(registry_loader, "entry_points", lambda *, group: (entry_point,))

    fragments = registry_loader._load_external_status_registry_fragments()  # noqa: SLF001
    specs = registry_loader._load_status_kind_specs(  # noqa: SLF001
        fragment_paths=tuple(path for path, _ in fragments),
        dnadesign_root=_repo_root() / "src" / "dnadesign",
        provider_prefixes={path: prefix for path, prefix in fragments},
    )

    assert specs[0].provider_ref == "research_studies.demo.status:provide_status"

    fragment.write_text(
        fragment.read_text(encoding="utf-8").replace(
            "research_studies.demo.status:provide_status",
            "unrelated_package.demo:provide_status",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provider_ref must stay under the fragment owner package"):
        registry_loader._load_status_kind_specs(  # noqa: SLF001
            fragment_paths=tuple(path for path, _ in fragments),
            dnadesign_root=_repo_root() / "src" / "dnadesign",
            provider_prefixes={path: prefix for path, prefix in fragments},
        )


def test_status_registry_fragments_are_included_as_package_data() -> None:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    expected_patterns = {
        "dnadesign.cluster": ("ops/status.registry.yaml",),
        "dnadesign.latentdna": ("ops/status.registry.yaml",),
        "dnadesign.opal": ("src/ops/status.registry.yaml",),
        "dnadesign.ops": ("providers/*/status.registry.yaml",),
        "dnadesign.usr": ("ops/status.registry.yaml",),
    }
    for package_name, patterns in expected_patterns.items():
        for pattern in patterns:
            assert pattern in package_data[package_name]


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


def test_ops_contract_root_import_does_not_preload_sibling_tool_packages() -> None:
    imported_modules = _run_python(
        """
import json
import sys
import dnadesign.ops as ops

assert ops is not None
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.construct',
        'dnadesign.densegen',
        'dnadesign.infer',
    } or name.startswith('dnadesign.baserender')
)))
"""
    )

    assert imported_modules == []


def test_infer_fill_import_does_not_import_studies_package() -> None:
    imported_modules = _run_python(
        """
import importlib
import json
import sys

importlib.import_module('dnadesign.ops.orchestrator.infer_fill')
print(json.dumps(sorted(name for name in sys.modules if name.startswith('dnadesign.studies'))))
"""
    )

    assert imported_modules == []


def test_public_ops_api_import_stays_metadata_only_until_attribute_access() -> None:
    imported_modules = _run_python(
        """
import json
import sys
import dnadesign.ops.api as ops_api

assert 'build_batch_plan' in ops_api.__all__
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.ops.catalog.loader',
        'dnadesign.ops.orchestrator.execute',
        'dnadesign.ops.orchestrator.infer_fill',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
        'dnadesign.ops.preflight.support',
        'dnadesign.ops.status.campaign',
        'dnadesign.ops.status.registry_loader',
        'dnadesign.ops.status.service',
    }
)))
"""
    )

    assert imported_modules == []


def test_public_ops_preflight_import_stays_metadata_only_until_attribute_access() -> None:
    imported_modules = _run_python(
        """
import json
import sys
import dnadesign.ops.preflight as preflight

assert 'build_contract_preflight_checks' in preflight.__all__
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        'dnadesign.construct',
        'dnadesign.densegen',
        'dnadesign.infer',
        'dnadesign.ops.orchestrator.gates',
        'dnadesign.ops.orchestrator.mode_tools',
        'dnadesign.ops.orchestrator.plan',
        'dnadesign.ops.orchestrator.state',
        'dnadesign.ops.preflight.contract_checks',
        'dnadesign.ops.preflight.support',
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


def test_ops_package_roots_do_not_expose_dynamic_execution_facades() -> None:
    facade_presence = _run_python(
        """
import json
import dnadesign.ops as ops
import dnadesign.ops.orchestrator as orchestrator
import dnadesign.ops.status as status

print(json.dumps({
    'ops_api': hasattr(ops, 'api'),
    'orchestrator_build_batch_plan': hasattr(orchestrator, 'build_batch_plan'),
    'orchestrator_execute_batch_plan': hasattr(orchestrator, 'execute_batch_plan'),
    'status_build_status_inputs': hasattr(status, 'build_status_inputs'),
    'status_load_status_kind_spec': hasattr(status, 'load_status_kind_spec'),
    'status_run_status_kind': hasattr(status, 'run_status_kind'),
}, sort_keys=True))
"""
    )

    assert facade_presence == {
        "ops_api": False,
        "orchestrator_build_batch_plan": False,
        "orchestrator_execute_batch_plan": False,
        "status_build_status_inputs": False,
        "status_load_status_kind_spec": False,
        "status_run_status_kind": False,
    }


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
        'dnadesign.ops.providers.builtin.status_provider',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.latentdna.ops.status_providers',
        'dnadesign.opal.src.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
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
        'dnadesign.ops.providers.builtin.status_provider',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.latentdna.ops.status_providers',
        'dnadesign.opal.src.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
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
        'dnadesign.ops.providers.builtin.status_provider',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.opal.src.ops.status_providers',
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
        'dnadesign.ops.providers.builtin.status_provider',
        'dnadesign.usr.ops.status_providers',
        'dnadesign.cluster.ops.status_providers',
        'dnadesign.opal.src.ops.status_providers',
    }
)))
"""
    )

    assert imported_modules == ["dnadesign.ops.providers.builtin.status_provider"]
