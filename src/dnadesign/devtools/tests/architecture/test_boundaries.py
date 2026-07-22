"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/architecture/test_boundaries.py

Tests for cross-tool import boundary checks used in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.devtools.architecture.boundaries import (
    TOP_LEVEL_LEGACY_DIRECTORIES,
    TOP_LEVEL_ROOT_MODULES,
    TOP_LEVEL_SHARED_INFRA_PACKAGES,
    TOP_LEVEL_TOOL_BOUNDARY_PACKAGES,
    find_legacy_surface_violations,
    find_review_surface_private_imports,
    find_studies_layout_violations,
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


def test_find_undeclared_cross_tool_imports_allows_studies_to_densegen_public_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "status" / "surface.py",
        "from dnadesign.densegen import inspect_analysis_surface\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py",
        "def inspect_analysis_surface(_path):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_studies_to_aligner_msa_public_api(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "alignment.py",
        "from dnadesign.aligner.msa import MsaRequest, run_msa\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "msa" / "__init__.py",
        "class MsaRequest:\n    pass\n\ndef run_msa(_request):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_studies_to_aligner_internal_backend(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "alignment.py",
        "from dnadesign.aligner.msa.backends.mafft import run_mafft\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "msa" / "backends" / "mafft.py",
        "def run_mafft(_request):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "aligner"
    assert violations[0].import_target == "dnadesign.aligner.msa.backends.mafft"


def test_find_undeclared_cross_tool_imports_allows_studies_to_permuter_public_api(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "candidate_expansion.py",
        "from dnadesign.permuter import CodingDnaDmsRequest, generate_variants\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "permuter" / "__init__.py",
        "class CodingDnaDmsRequest:\n    pass\n\ndef generate_variants(_request):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


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


def test_find_undeclared_cross_tool_imports_rejects_studies_to_permuter_internal_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "candidate_expansion.py",
        "from dnadesign.permuter.src.api.generate import generate_variants\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "permuter" / "src" / "api" / "generate.py",
        "def generate_variants(_request):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "permuter"


def test_find_undeclared_cross_tool_imports_rejects_studies_to_densegen_non_facade(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "status" / "surface.py",
        "from dnadesign.densegen.analysis import inspect_analysis_surface\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "analysis.py",
        "def inspect_analysis_surface(_path):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "densegen"
    assert violations[0].import_target == "dnadesign.densegen.analysis"


@pytest.mark.parametrize(
    "import_statement",
    (
        "from dnadesign.ops.catalog import discover_repo_root\n",
        "from dnadesign.ops.preflight import CommandExecution\n",
        "from dnadesign.ops.status import resolve_path_ref\n",
    ),
)
def test_find_undeclared_cross_tool_imports_allows_studies_to_ops_public_facades(
    tmp_path: Path,
    import_statement: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "status" / "surface.py",
        import_statement,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "catalog" / "__init__.py",
        "def discover_repo_root():\n    return None\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "ops" / "preflight" / "__init__.py", "class CommandExecution:\n    pass\n")
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "status" / "__init__.py", "def resolve_path_ref():\n    return None\n"
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize(
    "import_statement",
    (
        "from dnadesign.ops.status.path_ref import resolve_path_ref\n",
        "from dnadesign.ops.preflight.contract_checks import build_contract_preflight_checks\n",
        "from dnadesign.ops.orchestrator.state import ActiveJobResolution\n",
    ),
)
def test_find_undeclared_cross_tool_imports_rejects_studies_to_ops_internal_facades(
    tmp_path: Path,
    import_statement: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "status" / "surface.py",
        import_statement,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "status" / "path_ref.py", "def resolve_path_ref():\n    return None\n"
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "preflight" / "contract_checks.py",
        "def build_contract_preflight_checks():\n    return []\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "orchestrator" / "state.py",
        "class ActiveJobResolution:\n    pass\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "ops"


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


def test_find_undeclared_cross_tool_imports_allows_study_public_retron_producer_edges(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "retron_hairpin_design" / "compiler.py",
        "\n".join(
            [
                "import dnadesign.baserender as baserender",
                "import dnadesign.construct as construct",
                "from dnadesign.construct import run_linear_ssdna_composition",
                "from dnadesign.cruncher.scar_nick import load_scar_nick_stem_base_primitives",
                "from dnadesign.cruncher.snapback import load_released_solve_cap_primitives",
            ]
        ),
    )
    _write(tmp_path / "src" / "dnadesign" / "baserender" / "__init__.py", "def run_job():\n    return None\n")
    _write(
        tmp_path / "src" / "dnadesign" / "construct" / "__init__.py",
        "def run_linear_ssdna_composition():\n    return None\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "scar_nick" / "__init__.py",
        "def load_scar_nick_stem_base_primitives():\n    return []\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "snapback" / "__init__.py",
        "def load_released_solve_cap_primitives():\n    return []\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_study_unspecified_cruncher_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "retron_hairpin_design" / "compiler.py",
        "from dnadesign.cruncher.analysis import load_rows\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "analysis" / "__init__.py",
        "def load_rows():\n    return []\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "cruncher"
    assert violations[0].import_target == "dnadesign.cruncher.analysis"


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


@pytest.mark.parametrize("owner_tool", ("notify", "ops", "studies"))
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


@pytest.mark.parametrize("owner_tool", ("notify", "ops", "studies"))
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


def test_find_undeclared_cross_tool_imports_allows_ops_to_notify_core_contracts(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "notify.py",
        "from dnadesign.notify.core.contracts import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "core" / "contracts.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_ops_to_notify_non_contract_surface(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "notify.py",
        "from dnadesign.notify.contracts import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "contracts.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "notify"
    assert violations[0].import_target == "dnadesign.notify.contracts"


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
                "from dnadesign.ops.runbooks import PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR",
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
        "PACKAGED_RUNBOOK_PRESETS_RELATIVE_DIR = 'presets'\n",
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


def test_find_review_surface_private_imports_rejects_cross_tool_src_imports_in_tests(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "tests" / "test_public_surface.py",
        "from dnadesign.densegen.src.viz.plot_inventory import required_notebook_plot_ids\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "src" / "viz" / "plot_inventory.py",
        "def required_notebook_plot_ids():\n    return []\n",
    )

    violations = find_review_surface_private_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "densegen"
    assert violations[0].import_target == "dnadesign.densegen.src.viz.plot_inventory"


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


def test_find_review_surface_private_imports_rejects_ops_tests_importing_concrete_study_packages(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "tests" / "test_status.py",
        "from dnadesign.studies.units.demo_study.status.service import STUDY_STATUS_SERVICE\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "studies" / "units" / "demo_study" / "__init__.py", "")

    violations = find_review_surface_private_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "studies"
    assert violations[0].import_target == "dnadesign.studies.units.demo_study.status.service"


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


def test_find_undeclared_cross_tool_imports_allows_studies_to_opal_public_api(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "studies" / "demo" / "opal_handoff.py",
        "from dnadesign.opal import validate_x_parquet_column\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "__init__.py",
        "def validate_x_parquet_column():\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_studies_to_opal_private_api(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "studies" / "demo" / "opal_handoff.py",
        "from dnadesign.opal.src.storage.x_contracts import validate_x_parquet_column\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "src" / "storage" / "x_contracts.py",
        "def validate_x_parquet_column():\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "opal"
    assert violations[0].import_target == "dnadesign.opal.src.storage.x_contracts"


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


def test_find_undeclared_cross_tool_imports_scans_study_status_modules(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "src" / "runtime.py", "def run():\n    return 1\n")
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "demo_study" / "status" / "status.py",
        "from dnadesign.bar.src.runtime import run\n",
    )

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "bar"
    assert violations[0].import_target == "dnadesign.bar.src.runtime"


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


def test_find_legacy_surface_violations_flags_removed_ops_study_paths(tmp_path: Path) -> None:
    legacy_path = tmp_path / "src" / "dnadesign" / "ops" / "promoter_preflight_coordinator.py"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("# legacy study-owned surface\n", encoding="utf-8")
    legacy_family_path = tmp_path / "src" / "dnadesign" / "studies" / "families" / "demo" / "__init__.py"
    legacy_family_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_family_path.write_text("# legacy family surface\n", encoding="utf-8")
    unexpected_cli_path = tmp_path / "src" / "dnadesign" / "ops" / "legacy_cli_bridge.py"
    unexpected_cli_path.write_text("# removed cli bridge\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/ops/legacy_cli_bridge.py",
        "src/dnadesign/ops/promoter_preflight_coordinator.py",
        "src/dnadesign/studies/families",
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


def test_find_top_level_layout_violations_flags_unexpected_top_level_module(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    _write(tmp_path / "src" / "dnadesign" / "helpers.py", "# drift\n")

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("unexpected top-level module", "src/dnadesign/helpers.py"),
    ]


def test_find_studies_layout_violations_accepts_progressive_disclosure_layout(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")
    _write(studies_root / "README.md", "# studies\n")
    for name in ("assets", "core", "tests", "units"):
        (studies_root / name).mkdir(parents=True, exist_ok=True)
    (studies_root / "units" / "demo_study").mkdir(parents=True, exist_ok=True)
    _write(studies_root / "units" / "demo_study" / "tests" / "__init__.py", "")

    violations = find_studies_layout_violations(repo_root=tmp_path)

    assert violations == []


def test_find_studies_layout_violations_rejects_flat_concrete_study_package(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")
    _write(studies_root / "README.md", "# studies\n")
    for name in ("assets", "core", "tests", "units"):
        (studies_root / name).mkdir(parents=True, exist_ok=True)
    (studies_root / "demo_study").mkdir(parents=True, exist_ok=True)
    _write(studies_root / "units" / "existing_study" / "tests" / "__init__.py", "")

    violations = find_studies_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        (
            "concrete study package must live under src/dnadesign/studies/units",
            "src/dnadesign/studies/demo_study",
        )
    ]


def test_find_studies_layout_violations_rejects_missing_concrete_study_test_package(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")
    _write(studies_root / "README.md", "# studies\n")
    for name in ("assets", "core", "tests", "units"):
        (studies_root / name).mkdir(parents=True, exist_ok=True)
    (studies_root / "units" / "demo_study").mkdir(parents=True, exist_ok=True)

    violations = find_studies_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        (
            "concrete study tests must live inside the owning study unit",
            "src/dnadesign/studies/units/demo_study/tests",
        )
    ]


def test_find_studies_layout_violations_rejects_unscoped_concrete_study_test(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")
    _write(studies_root / "README.md", "# studies\n")
    for name in ("assets", "core", "tests", "units"):
        (studies_root / name).mkdir(parents=True, exist_ok=True)
    _write(studies_root / "units" / "demo_study" / "tests" / "__init__.py", "")
    _write(
        studies_root / "tests" / "test_demo_study_status.py",
        "from dnadesign.studies.units.demo_study.status import service\n",
    )

    violations = find_studies_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        (
            "study-specific test must live under src/dnadesign/studies/units/demo_study/tests",
            "src/dnadesign/studies/tests/test_demo_study_status.py",
        )
    ]


def test_find_studies_layout_violations_rejects_separate_concrete_study_test_package(tmp_path: Path) -> None:
    studies_root = tmp_path / "src" / "dnadesign" / "studies"
    _write(studies_root / "__init__.py", "")
    _write(studies_root / "README.md", "# studies\n")
    for name in ("assets", "core", "tests", "units"):
        (studies_root / name).mkdir(parents=True, exist_ok=True)
    _write(studies_root / "units" / "demo_study" / "tests" / "__init__.py", "")
    _write(studies_root / "tests" / "demo_study" / "__init__.py", "")

    violations = find_studies_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        (
            "study-specific tests must live under src/dnadesign/studies/units/demo_study/tests",
            "src/dnadesign/studies/tests/demo_study",
        )
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
