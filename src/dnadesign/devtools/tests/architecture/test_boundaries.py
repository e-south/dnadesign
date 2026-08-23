"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/architecture/test_boundaries.py

Tests for cross-tool import boundary checks used in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from dnadesign.devtools.architecture.boundaries import (
    TOP_LEVEL_LEGACY_DIRECTORIES,
    TOP_LEVEL_ROOT_MODULES,
    TOP_LEVEL_SHARED_INFRA_PACKAGES,
    TOP_LEVEL_TOOL_BOUNDARY_PACKAGES,
    find_external_study_boundary_violations,
    find_legacy_surface_violations,
    find_review_surface_private_imports,
    find_top_level_layout_violations,
    find_undeclared_cross_tool_imports,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scaffold_top_level_layout(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    src_root.mkdir(parents=True, exist_ok=True)
    for name in TOP_LEVEL_ROOT_MODULES:
        _write(src_root / name, "")
    for name in TOP_LEVEL_TOOL_BOUNDARY_PACKAGES | TOP_LEVEL_SHARED_INFRA_PACKAGES:
        _write(src_root / name / "__init__.py", "")
    for name in TOP_LEVEL_LEGACY_DIRECTORIES:
        (src_root / name).mkdir(parents=True, exist_ok=True)


def test_dnadesign_source_does_not_import_external_reader_packages() -> None:
    source_root = Path("src/dnadesign")
    forbidden_roots = {"reader", "reader_workbench"}
    violations: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", maxsplit=1)[0] in forbidden_roots:
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.split(".", maxsplit=1)[0] in forbidden_roots:
                    violations.append(f"{path}:{node.lineno}: from {module} import ...")

    assert violations == []


def test_superseded_cruncher_hairpin_producers_do_not_return() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    retired_paths = (
        "src/dnadesign/cruncher/src/snapback",
        "src/dnadesign/cruncher/src/scar_nick",
        "src/dnadesign/cruncher/src/release_enzymes",
        "src/dnadesign/baserender/src/integrations/snapback",
        "src/dnadesign/baserender/src/integrations/scar_nick",
        "src/dnadesign/contracts/visual/snapback_visual_v1.py",
        "src/dnadesign/contracts/visual/scar_nick_visual_v1.py",
    )

    offenders: list[str] = []
    for path in retired_paths:
        candidate = repo_root / path
        if candidate.is_file():
            offenders.append(path)
        elif candidate.is_dir():
            offenders.extend(
                str(nested.relative_to(repo_root))
                for nested in candidate.rglob("*")
                if nested.is_file() and nested.suffix in {".md", ".py", ".yaml"}
            )

    assert offenders == []
    assert (repo_root / "src/dnadesign/cruncher/src/cassette").is_dir()
    assert (repo_root / "src/dnadesign/cruncher/src/nickases").is_dir()


def test_find_undeclared_cross_tool_imports_allows_declared_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges={("foo", "bar")},
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_reports_undeclared_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_reports_clean_absolute_import_target(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].import_target == "dnadesign.bar.api"


def test_find_undeclared_cross_tool_imports_allows_ops_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "ops" / "gates.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_ops_to_usr_non_contract_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "gates.py",
        "from dnadesign.usr.storage import DatasetStore\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "usr" / "storage.py", "class DatasetStore:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "usr"
    assert violations[0].import_target == "dnadesign.usr.storage"


def test_find_undeclared_cross_tool_imports_rejects_ops_to_studies_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "ops" / "gates.py", "from dnadesign.studies import StudyOpsContract\n")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "__init__.py", "class StudyOpsContract:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "studies"


def test_find_undeclared_cross_tool_imports_allows_construct_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "construct" / "runtime.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_opal_to_usr_public_facade(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "config.py", "from dnadesign.usr import require_explicit_usr_root\n"
    )
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "def require_explicit_usr_root():\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_opal_to_usr_internal_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "config.py",
        "from dnadesign.usr.src.cli.support.resolution.roots import require_explicit_usr_root\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "src" / "cli" / "support" / "resolution" / "roots.py",
        "def require_explicit_usr_root():\n    pass\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "opal"
    assert violations[0].imported_tool == "usr"


def test_find_undeclared_cross_tool_imports_allows_construct_to_folding_public_facade(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "construct" / "runtime.py", "import dnadesign.folding as folding\n")
    _write(tmp_path / "src" / "dnadesign" / "folding" / "__init__.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_construct_to_folding_non_facade(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "construct" / "runtime.py", "from dnadesign.folding.cli import app\n")
    _write(tmp_path / "src" / "dnadesign" / "folding" / "cli.py", "app = object()\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "construct"
    assert violations[0].imported_tool == "folding"
    assert violations[0].import_target == "dnadesign.folding.cli"


def test_find_undeclared_cross_tool_imports_allows_latentdna_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "latentdna" / "runtime.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_ops_to_infer_default_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "plan.py",
        "from dnadesign.infer import validate_runbook_gpu_resources\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "infer" / "__init__.py",
        "def validate_runbook_gpu_resources(**_kwargs):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_ops_to_infer_non_contract_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "plan.py",
        "from dnadesign.infer.runtime import infer_batch\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "infer" / "runtime.py", "def infer_batch():\n    return None\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "infer"
    assert violations[0].import_target == "dnadesign.infer.runtime"


def test_find_undeclared_cross_tool_imports_allows_permuter_to_infer_public_facade(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "permuter" / "evaluator.py",
        "from dnadesign.infer import run_extract\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "infer" / "__init__.py",
        "def run_extract():\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_permuter_to_infer_non_facade(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "permuter" / "evaluator.py",
        "from dnadesign.infer.runtime import run_extract\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "infer" / "runtime.py",
        "def run_extract():\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "permuter"
    assert violations[0].imported_tool == "infer"
    assert violations[0].import_target == "dnadesign.infer.runtime"


def test_find_undeclared_cross_tool_imports_allows_usr_cruncher_promoter_export_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "scripts" / "import_promoters.py",
        "from dnadesign.cruncher.ingest.promoters import load_promoter_export\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "ingest" / "promoters.py",
        "def load_promoter_export(_path):\n    return []\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_usr_cruncher_unspecified_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "scripts" / "import_promoters.py",
        "from dnadesign.cruncher.ingest.other import load_rows\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "ingest" / "other.py",
        "def load_rows(_path):\n    return []\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "usr"
    assert violations[0].imported_tool == "cruncher"
    assert violations[0].import_target == "dnadesign.cruncher.ingest.other"


def test_find_undeclared_cross_tool_imports_allows_shared_contract_package(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "baserender" / "api.py",
        "from dnadesign.contracts.visual import SequenceEvidenceMapV1\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "contracts" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "contracts" / "visual" / "__init__.py",
        "class SequenceEvidenceMapV1:\n    pass\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize(
    ("owner_tool", "imported_tool"),
    (
        ("notify", "densegen"),
        ("notify", "infer"),
        ("ops", "densegen"),
    ),
)
def test_find_undeclared_cross_tool_imports_allows_contract_owner_edges(
    tmp_path: Path,
    owner_tool: str,
    imported_tool: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / owner_tool / "api.py",
        f"from dnadesign.{imported_tool}.contracts import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / imported_tool / "contracts.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize("owner_tool", ("notify", "ops"))
def test_find_undeclared_cross_tool_imports_allows_construct_public_package_surface(
    tmp_path: Path,
    owner_tool: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / owner_tool / "api.py",
        "from dnadesign.construct import resolve_construct_usr_output_contract\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "construct" / "__init__.py",
        "def resolve_construct_usr_output_contract():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize("owner_tool", ("notify", "ops"))
def test_find_undeclared_cross_tool_imports_rejects_construct_contract_module_surface(
    tmp_path: Path,
    owner_tool: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / owner_tool / "api.py",
        "from dnadesign.construct.contracts import resolve_construct_usr_output_contract\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "construct" / "contracts.py",
        "def resolve_construct_usr_output_contract():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == owner_tool
    assert violations[0].imported_tool == "construct"
    assert violations[0].import_target == "dnadesign.construct.contracts"


def test_find_undeclared_cross_tool_imports_allows_ops_to_notify_public_package(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "notify.py",
        "from dnadesign.notify import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "__init__.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_ops_to_notify_internal_core_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "notify.py",
        "from dnadesign.notify.core.contracts import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "core" / "contracts.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "notify"
    assert violations[0].import_target == "dnadesign.notify.core.contracts"


def test_find_undeclared_cross_tool_imports_rejects_internal_src_target_even_for_allowed_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "foo" / "api.py",
        "from dnadesign.bar.src.runtime import run\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "bar" / "src" / "runtime.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges={("foo", "bar")},
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"
    assert violations[0].import_target == "dnadesign.bar.src.runtime"


def test_find_undeclared_cross_tool_imports_rejects_devtools_to_tool_internal(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "devtools" / "docs" / "checks.py",
        "from dnadesign.ops.orchestrator.state import ActiveJobResolution\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "orchestrator" / "state.py",
        "class ActiveJobResolution:\n    pass\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "devtools"
    assert violations[0].imported_tool == "ops"
    assert violations[0].import_target == "dnadesign.ops.orchestrator.state"


def test_find_undeclared_cross_tool_imports_allows_devtools_to_ops_public_surfaces(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "devtools" / "docs" / "checks.py",
        "\n".join(
            [
                "from dnadesign.ops.catalog import load_runbook_catalog",
                "from dnadesign.ops.runbooks import REPO_TRANSIENT_OPERATIONAL_DIR_NAMES",
                "from dnadesign.ops.status import list_status_kind_specs_for_repo",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "catalog" / "__init__.py",
        "def load_runbook_catalog():\n    return None\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "runbooks" / "__init__.py",
        "REPO_TRANSIENT_OPERATIONAL_DIR_NAMES = frozenset()\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "status" / "__init__.py",
        "def list_status_kind_specs_for_repo(_repo_root):\n    return ()\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_reports_relative_cross_tool_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ...bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_allows_relative_within_tool(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ..core.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "foo" / "core" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_reports_relative_import_without_module(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ... import bar\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "__init__.py", "")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_rejects_testsupport_imports_from_runtime_code(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "foo" / "api.py",
        "from dnadesign.devtools.tests.support.usr import ensure_registry\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "devtools" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "devtools" / "tests" / "support" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "devtools" / "tests" / "support" / "usr.py",
        "def ensure_registry():\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "devtools.tests.support"
    assert violations[0].import_target == "dnadesign.devtools.tests.support.usr"


def test_find_undeclared_cross_tool_imports_rejects_legacy_top_level_testsupport_imports(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.testsupport.usr import ensure_registry\n")
    _write(tmp_path / "src" / "dnadesign" / "testsupport" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "testsupport" / "usr.py", "def ensure_registry():\n    return None\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "testsupport"
    assert violations[0].import_target == "dnadesign.testsupport.usr"


def test_find_undeclared_cross_tool_imports_allows_relative_import_without_module_within_tool(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from .. import core\n")
    _write(tmp_path / "src" / "dnadesign" / "foo" / "core.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "__init__.py", "")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_ignores_test_files(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "tests" / "test_api.py", "from dnadesign.bar import api\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_review_surface_private_imports_allows_same_tool_internal_tests(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "tests" / "test_internal.py",
        "from dnadesign.ops.orchestrator.state import resolve_mode_decision\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "orchestrator" / "state.py",
        "def resolve_mode_decision():\n    return None\n",
    )

    violations = find_review_surface_private_imports(repo_root=tmp_path)

    assert violations == []


def test_find_review_surface_private_imports_allows_public_cross_tool_review_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "tests" / "test_public_surface.py",
        "from dnadesign.densegen import required_notebook_plot_ids\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py",
        "def required_notebook_plot_ids():\n    return []\n",
    )

    violations = find_review_surface_private_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_ignores_archived_and_prototypes(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")
    _write(
        tmp_path / "src" / "dnadesign" / "archived" / "legacy.py",
        "from dnadesign.bar.api import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "prototypes" / "draft.py",
        "from dnadesign.foo.api import run\n",
    )

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_legacy_surface_violations_flags_removed_repo_root_contract_paths(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "_contracts").mkdir(parents=True, exist_ok=True)
    legacy_usr_roots = tmp_path / "src" / "dnadesign" / "usr_roots.py"
    legacy_usr_roots.parent.mkdir(parents=True, exist_ok=True)
    legacy_usr_roots.write_text("# legacy\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/_contracts",
        "src/dnadesign/usr_roots.py",
    ]


def test_find_legacy_surface_violations_ignores_cache_only_legacy_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "src" / "dnadesign" / "_contracts" / "__pycache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "__init__.cpython-312.pyc").write_bytes(b"cache")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert violations == []


def test_find_legacy_surface_violations_flags_removed_junction_root_modules(tmp_path: Path) -> None:
    canonical = tmp_path / "src" / "dnadesign" / "junction" / "canonical.py"
    exports = tmp_path / "src" / "dnadesign" / "junction" / "exports.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("# removed identity module\n", encoding="utf-8")
    exports.write_text("# removed publication module\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/junction/canonical.py",
        "src/dnadesign/junction/exports.py",
    ]


def test_find_legacy_surface_violations_flags_removed_trijunction_package(tmp_path: Path) -> None:
    legacy_package = tmp_path / "src" / "dnadesign" / "trijunction"
    legacy_package.mkdir(parents=True, exist_ok=True)
    (legacy_package / "__init__.py").write_text("# removed product identity\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/trijunction",
    ]


def test_find_legacy_surface_violations_flags_removed_ops_study_paths(tmp_path: Path) -> None:
    legacy_path = tmp_path / "src" / "dnadesign" / "ops" / "promoter_preflight_coordinator.py"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("# legacy study-owned surface\n", encoding="utf-8")
    unexpected_cli_path = tmp_path / "src" / "dnadesign" / "ops" / "legacy_cli_bridge.py"
    unexpected_cli_path.write_text("# removed cli bridge\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/ops/legacy_cli_bridge.py",
        "src/dnadesign/ops/promoter_preflight_coordinator.py",
    ]


def test_find_legacy_surface_violations_allows_ops_provider_packages(tmp_path: Path) -> None:
    provider_root = tmp_path / "src" / "dnadesign" / "ops" / "providers" / "builtin"
    provider_root.mkdir(parents=True, exist_ok=True)
    (provider_root / "__init__.py").write_text("", encoding="utf-8")
    (provider_root / "status_provider.py").write_text("def provide_status():\n    return None\n", encoding="utf-8")
    (provider_root / "status.registry.yaml").write_text("version: 1\nentries: []\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert violations == []


def test_find_legacy_surface_violations_accepts_relative_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_cli_path = tmp_path / "src" / "dnadesign" / "ops" / "legacy_cli_bridge.py"
    legacy_cli_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_cli_path.write_text("# removed cli bridge\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    violations = find_legacy_surface_violations(repo_root=Path("."))

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/ops/legacy_cli_bridge.py",
    ]


def test_find_top_level_layout_violations_accepts_expected_inventory(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert violations == []


def test_find_top_level_layout_violations_accepts_missing_optional_legacy_dirs(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    for name in TOP_LEVEL_ROOT_MODULES:
        _write(src_root / name, "")
    for name in TOP_LEVEL_TOOL_BOUNDARY_PACKAGES | TOP_LEVEL_SHARED_INFRA_PACKAGES:
        _write(src_root / name / "__init__.py", "")

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert violations == []


def test_find_top_level_layout_violations_flags_unexpected_top_level_directory(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    unexpected_dir = tmp_path / "src" / "dnadesign" / "scratchpad"
    unexpected_dir.mkdir(parents=True, exist_ok=True)

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("unexpected top-level directory", "src/dnadesign/scratchpad"),
    ]


def test_architecture_checks_ignore_untracked_local_study_backup(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    _write(tmp_path / ".gitignore", "src/dnadesign/studies/\n")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "__init__.py", "# local backup\n")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)

    assert find_top_level_layout_violations(repo_root=tmp_path) == []
    assert find_external_study_boundary_violations(repo_root=tmp_path) == []


def test_external_study_boundary_rejects_tracked_code_under_artifact_named_directory(tmp_path: Path) -> None:
    _write(tmp_path / ".gitignore", "src/dnadesign/studies/\n")
    tracked_path = tmp_path / "src" / "dnadesign" / "studies" / "demo" / "workbench" / "impl.py"
    _write(tracked_path, "# tracked study code\n")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "--force", tracked_path.relative_to(tmp_path).as_posix()],
        cwd=tmp_path,
        check=True,
    )

    violations = find_external_study_boundary_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("live study packages must remain external to dnadesign", "src/dnadesign/studies")
    ]


def test_find_top_level_layout_violations_flags_unexpected_top_level_module(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    _write(tmp_path / "src" / "dnadesign" / "helpers.py", "# drift\n")

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("unexpected top-level module", "src/dnadesign/helpers.py"),
    ]


def test_external_study_boundary_accepts_absent_study_package(tmp_path: Path) -> None:
    assert find_external_study_boundary_violations(repo_root=tmp_path) == []


def test_external_study_boundary_rejects_public_study_package(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")

    violations = find_external_study_boundary_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("live study packages must remain external to dnadesign", "src/dnadesign/studies")
    ]


def test_main_fails_when_legacy_contract_surface_paths_exist(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "foo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "bar").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "_contracts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "foo" / "api.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "dnadesign" / "bar" / "api.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_main_fails_on_syntax_error_in_checked_file(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def broken(:\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1
