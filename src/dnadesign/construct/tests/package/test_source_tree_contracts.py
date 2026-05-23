"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/package/test_source_tree_contracts.py

Source-tree contracts for the construct package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

_ROOT_PUBLIC_PYTHON = {"__init__.py", "__main__.py"}

_SRC_NAMESPACES = {
    "annotations",
    "cli",
    "composition",
    "contracts",
    "interfaces",
    "orchestration",
    "persistence",
    "products",
    "realization",
    "seeding",
    "seeds",
    "sequences",
    "sources",
    "workspaces",
}

_REMOVED_FLAT_SRC_MODULES = {
    "annotations.py",
    "api.py",
    "composition.py",
    "composition_exports.py",
    "composition_review.py",
    "composition_visual.py",
    "config.py",
    "errors.py",
    "feature_retention.py",
    "focal_selectors.py",
    "orientation.py",
    "output_store.py",
    "runtime.py",
    "seed.py",
    "workspace.py",
}

_MAX_IMPLEMENTATION_MODULE_LINES = 500
_MAX_TEST_MODULE_LINES = 900


def _construct_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "construct"
    raise RuntimeError("repo root not found")


def test_construct_root_keeps_progressive_disclosure_directories() -> None:
    construct_root = _construct_root()
    assert (construct_root / "README.md").is_file()
    assert (construct_root / "docs").is_dir()
    assert (construct_root / "docs" / "reference").is_dir()
    assert (construct_root / "src").is_dir()
    assert (construct_root / "tests").is_dir()
    assert (construct_root / "workspaces").is_dir()


def test_construct_root_keeps_minimal_top_level_surface() -> None:
    construct_root = _construct_root()
    observed = {
        path.name
        for path in construct_root.iterdir()
        if path.name != "__pycache__" and not path.name.startswith(".") and path.name != "AGENTS.md"
    }
    assert observed == {
        "README.md",
        "__init__.py",
        "__main__.py",
        "docs",
        "src",
        "tests",
        "workspaces",
    }


def test_construct_root_python_surface_is_public_facades_only() -> None:
    construct_root = _construct_root()
    root_python = {path.name for path in construct_root.glob("*.py")}
    assert root_python == _ROOT_PUBLIC_PYTHON

    root_init = (construct_root / "__init__.py").read_text(encoding="utf-8")
    assert ".src.interfaces.api" in root_init
    assert ".src.interfaces.contracts" in root_init
    assert "ConstructUSROutputContract" in root_init


def test_construct_root_rejects_leaked_source_modules() -> None:
    construct_root = _construct_root()
    leaked_names = {"cli.py", "contracts.py", "main.py"}
    assert not (leaked_names & {path.name for path in construct_root.glob("*.py")})

    module_main = (construct_root / "__main__.py").read_text(encoding="utf-8")
    assert "from .src.cli import main" in module_main
    assert "from .cli import" not in module_main


def test_construct_src_root_has_no_flat_implementation_modules() -> None:
    construct_src = _construct_root() / "src"
    root_python = {path.name for path in construct_src.glob("*.py")}
    assert root_python == {"__init__.py"}
    assert not (_REMOVED_FLAT_SRC_MODULES & {path.name for path in construct_src.iterdir()})


def test_construct_src_uses_semantic_namespaces() -> None:
    construct_src = _construct_root() / "src"
    observed = {
        path.name
        for path in construct_src.iterdir()
        if path.is_dir() and path.name != "__pycache__" and not path.name.startswith(".")
    }
    assert observed == _SRC_NAMESPACES


def test_construct_internal_cli_is_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    cli_dir = construct_src / "cli"
    assert cli_dir.is_dir()
    assert (cli_dir / "__init__.py").is_file()
    assert (cli_dir / "app.py").is_file()
    assert (cli_dir / "commands").is_dir()


def test_construct_runtime_realization_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    realization_dir = construct_src / "realization"
    assert realization_dir.is_dir()
    assert (realization_dir / "__init__.py").is_file()
    assert (realization_dir / "assembly.py").is_file()
    assert (realization_dir / "normalize_anchor.py").is_file()
    assert (realization_dir / "parts.py").is_file()
    assert (realization_dir / "placement_guards.py").is_file()
    assert (realization_dir / "placement_models.py").is_file()
    assert (realization_dir / "placement_search.py").is_file()
    assert (realization_dir / "placement.py").is_file()
    assert (realization_dir / "sequences.py").is_file()
    assert (realization_dir / "slots.py").is_file()
    assert (realization_dir / "windows.py").is_file()


def test_construct_config_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    contracts_dir = construct_src / "contracts"
    assert contracts_dir.is_dir()
    assert (contracts_dir / "__init__.py").is_file()
    assert (contracts_dir / "base.py").is_file()
    assert (contracts_dir / "config.py").is_file()
    assert (contracts_dir / "datasets.py").is_file()
    assert (contracts_dir / "errors.py").is_file()
    assert (contracts_dir / "job_invariants.py").is_file()
    assert (contracts_dir / "job.py").is_file()
    assert (contracts_dir / "loader.py").is_file()
    assert (contracts_dir / "normalize_anchor.py").is_file()
    assert (contracts_dir / "output.py").is_file()
    assert (contracts_dir / "parts.py").is_file()
    assert (contracts_dir / "realization.py").is_file()
    assert (contracts_dir / "templates.py").is_file()


def test_construct_composition_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    composition_dir = construct_src / "composition"
    assert composition_dir.is_dir()
    assert (composition_dir / "__init__.py").is_file()
    assert (composition_dir / "baserender_jobs.py").is_file()
    assert (composition_dir / "bundle.py").is_file()
    assert (composition_dir / "exports.py").is_file()
    assert (composition_dir / "folding_runtime.py").is_file()
    assert (composition_dir / "models.py").is_file()
    assert (composition_dir / "review.py").is_file()
    assert (composition_dir / "review_assets.py").is_file()
    assert (composition_dir / "review_manifest.py").is_file()
    assert (composition_dir / "review_svg.py").is_file()
    assert (composition_dir / "runtime.py").is_file()
    assert (composition_dir / "svg_geometry.py").is_file()
    assert (composition_dir / "visual.py").is_file()


def test_construct_runtime_source_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    sources_dir = construct_src / "sources"
    assert sources_dir.is_dir()
    assert (sources_dir / "__init__.py").is_file()
    assert (sources_dir / "input_rows.py").is_file()
    assert (sources_dir / "paths.py").is_file()
    assert (sources_dir / "templates.py").is_file()


def test_construct_runtime_persistence_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    persistence_dir = construct_src / "persistence"
    assert persistence_dir.is_dir()
    assert (persistence_dir / "__init__.py").is_file()
    assert (persistence_dir / "records.py").is_file()
    assert (persistence_dir / "usr_registry.py").is_file()
    assert (persistence_dir / "write_session.py").is_file()


def test_construct_runtime_product_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    products_dir = construct_src / "products"
    assert products_dir.is_dir()
    assert (products_dir / "__init__.py").is_file()
    assert (products_dir / "classic.py").is_file()
    assert (products_dir / "normalize_anchor.py").is_file()
    assert (products_dir / "sequence_views.py").is_file()
    assert (products_dir / "specs.py").is_file()


def test_construct_workspace_contracts_are_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    workspaces_dir = construct_src / "workspaces"
    assert workspaces_dir.is_dir()
    assert (workspaces_dir / "__init__.py").is_file()
    assert (workspaces_dir / "models.py").is_file()
    assert (workspaces_dir / "registry.py").is_file()
    assert (workspaces_dir / "templates.py").is_file()


def test_construct_runtime_delegates_normalize_anchor_contracts() -> None:
    construct_src = _construct_root() / "src"
    runtime_source = (construct_src / "orchestration" / "runtime.py").read_text(encoding="utf-8")
    normalize_source = (construct_src / "realization" / "normalize_anchor.py").read_text(encoding="utf-8")
    products_source = (construct_src / "products" / "specs.py").read_text(encoding="utf-8")
    assert "load_annotation_features" not in runtime_source
    assert "resolve_focal_selection" not in runtime_source
    assert "classify_feature_retention" not in runtime_source
    assert "def build_normalize_spec_id" not in normalize_source
    assert "def build_normalize_spec_id" in products_source


def test_construct_runtime_delegates_source_loading_contracts() -> None:
    construct_src = _construct_root() / "src"
    runtime_source = (construct_src / "orchestration" / "runtime.py").read_text(encoding="utf-8")
    normalize_source = (construct_src / "realization" / "normalize_anchor.py").read_text(encoding="utf-8")
    assert "def _load_template_sequence" not in runtime_source
    assert "def _load_normalize_template" not in runtime_source
    assert "def _scan_usr_rows" not in runtime_source
    assert "Template FASTA" not in runtime_source
    assert "seq_annot__features" not in runtime_source
    assert "seq_annot__features" not in normalize_source


def test_construct_runtime_delegates_persistence_write_contracts() -> None:
    construct_src = _construct_root() / "src"
    runtime_source = (construct_src / "orchestration" / "runtime.py").read_text(encoding="utf-8")
    assert "def _write_output_records" not in runtime_source
    assert "def _write_planned_sequence_views" not in runtime_source
    assert "def _records_to_write" not in runtime_source
    assert "def _ensure_output_dataset" not in runtime_source
    assert "write_session()" not in runtime_source
    assert "load_sequence_view_index" not in runtime_source


def test_construct_runtime_delegates_product_lineage_contracts() -> None:
    construct_src = _construct_root() / "src"
    runtime_source = (construct_src / "orchestration" / "runtime.py").read_text(encoding="utf-8")
    assert "def _build_record" not in runtime_source
    assert "def _build_variant_record" not in runtime_source
    assert "def _build_normalize_record" not in runtime_source
    assert "SequenceViewRecord" not in runtime_source
    assert "construct__parts" not in runtime_source
    assert "def _spec_id" not in runtime_source


def test_construct_composition_runtime_delegates_bundle_publication_contracts() -> None:
    construct_src = _construct_root() / "src"
    runtime_source = (construct_src / "composition" / "runtime.py").read_text(encoding="utf-8")
    bundle_source = (construct_src / "composition" / "bundle.py").read_text(encoding="utf-8")
    assert "def _write_bundle" not in runtime_source
    assert "def _manifest_payload" not in runtime_source
    assert "write_folding_artifacts" not in runtime_source
    assert "visual_contract_payload" not in runtime_source
    assert "def write_composition_bundle" in bundle_source
    assert "Deprecated generated artifact directory" in bundle_source


def test_construct_runtime_stays_below_orchestration_line_budget() -> None:
    runtime_path = _construct_root() / "src" / "orchestration" / "runtime.py"
    assert len(runtime_path.read_text(encoding="utf-8").splitlines()) < 800


def test_construct_implementation_modules_stay_below_monolith_budget() -> None:
    construct_src = _construct_root() / "src"
    oversized = []
    for path in construct_src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count >= _MAX_IMPLEMENTATION_MODULE_LINES:
            oversized.append(f"{path.relative_to(construct_src)}:{line_count}")
    assert oversized == []


def test_construct_runtime_code_uses_explicit_contract_errors_instead_of_asserts() -> None:
    construct_src = _construct_root() / "src"
    offenders = []
    for path in construct_src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "assert " in source:
            offenders.append(path.relative_to(construct_src).as_posix())
    assert offenders == []


def test_construct_test_modules_stay_below_monolith_budget() -> None:
    construct_tests = _construct_root() / "tests"
    oversized = []
    for path in construct_tests.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count >= _MAX_TEST_MODULE_LINES:
            oversized.append(f"{path.relative_to(construct_tests)}:{line_count}")
    assert oversized == []


def test_construct_package_data_uses_workspace_shape_globs() -> None:
    repo_root = _construct_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"workspaces/*.md"' in pyproject
    assert '"workspaces/*/*"' in pyproject
    assert '"workspaces/*/inputs/*"' in pyproject
