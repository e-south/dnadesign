"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_integration_architecture.py

Test BaseRender integration boundaries and discovery contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_producer_implementations_live_under_integrations() -> None:
    root = _source_root()

    assert not any((root / "adapters").glob("*.py"))
    assert not (root / "styles" / "curated" / "cruncher_showcase.py").exists()
    assert not (root / "pipeline" / "attach_motifs_from_cruncher_lockfile.py").exists()
    assert not (root / "pipeline" / "attach_motifs_from_config.py").exists()
    assert not (root / "pipeline" / "attach_motifs_from_library.py").exists()
    assert not (root / "pipeline" / "sigma70.py").exists()


def test_job_parser_uses_integration_contracts_without_builtin_names() -> None:
    parser = (_source_root() / "config" / "render_job_v4.py").read_text(encoding="utf-8")

    for producer_term in (
        "attach_motifs_from_config",
        "attach_motifs_from_cruncher_lockfile",
        "attach_motifs_from_library",
        "cruncher",
        "densegen",
        "sigma70",
        "yiu",
    ):
        assert producer_term not in parser.lower()
    assert "normalize_transform_config" in parser
    assert "declared_transform_path_values" in parser

    contract_registry = (_source_root() / "config" / "job_contracts.py").read_text(encoding="utf-8").lower()
    assert "integrations.junction" not in contract_registry
    assert "usr_genbank_annotation_render_v1" not in contract_registry
    assert "registered_render_contracts" in contract_registry


def test_transform_loader_has_no_builtin_implementation_imports() -> None:
    path = _source_root() / "pipeline" / "transforms.py"
    source = path.read_text(encoding="utf-8").lower()

    assert "entry_points" not in source
    assert "if name ==" not in source
    assert not any("integrations." in module for module in _imported_modules(path))


def test_neutral_layers_do_not_import_named_integration_packages() -> None:
    root = _source_root()
    neutral_roots = ("config", "core", "execution", "io", "outputs", "pipeline", "public", "render")
    named_integrations = {
        path.name for path in (root / "integrations").iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    violations: list[str] = []

    for package in neutral_roots:
        for path in sorted((root / package).rglob("*.py")):
            for module in _imported_modules(path):
                if any(
                    module == f"integrations.{name}"
                    or module.startswith(f"integrations.{name}.")
                    or f".integrations.{name}" in module
                    for name in named_integrations
                ):
                    violations.append(f"{path.relative_to(root)} imports {module}")

    assert violations == []


def test_sequence_panel_facade_delegates_producer_defaults() -> None:
    source = (_source_root() / "public" / "sequence_panel.py").read_text(encoding="utf-8").lower()

    for producer_term in ("cruncher", "densegen", "usr_genbank"):
        assert producer_term not in source
    assert "sequence_panel_defaults" in source

    service = (_source_root() / "integrations" / "sequence_panels.py").read_text(encoding="utf-8").lower()
    for implementation_term in ("palette", "sigma70", "densegen", "usr_genbank"):
        assert implementation_term not in service
    assert "registered_sequence_panel" in service

    styles = (_source_root() / "integrations" / "styles.py").read_text(encoding="utf-8").lower()
    assert "registered_style_profile" in styles
    for producer_term in ("cruncher", "densegen", "usr_genbank"):
        assert producer_term not in styles


def test_builtin_integration_registry_is_unique_and_described() -> None:
    from dnadesign.baserender.src.integrations.registry import integration_providers

    providers = integration_providers()
    assert providers
    assert len({provider.name for provider in providers}) == len(providers)

    adapters = [adapter for provider in providers for adapter in provider.adapters]
    transforms = [transform for provider in providers for transform in provider.transforms]
    profiles = [profile for provider in providers for profile in provider.style_profiles]
    render_contracts = [contract for provider in providers for contract in provider.render_contracts]
    assert len({adapter.kind for adapter in adapters}) == len(adapters)
    assert len({transform.name for transform in transforms}) == len(transforms)
    assert len({profile.name for profile in profiles}) == len(profiles)
    assert len({contract.kind for contract in render_contracts}) == len(render_contracts)
    assert all(adapter.docs_slug for adapter in adapters)
    assert all(transform.docs_slug for transform in transforms)
    assert all(profile.docs_slug for profile in profiles)


def test_public_facade_has_no_producer_named_exports() -> None:
    public_facade = (_source_root().parent / "__init__.py").read_text(encoding="utf-8").lower()
    public_api = (_source_root() / "public" / "api.py").read_text(encoding="utf-8").lower()

    for producer_term in ("cruncher_showcase_style", "densegen_tfbs_required_keys"):
        assert producer_term not in public_facade
        assert producer_term not in public_api
    assert "style_profile_overrides" in public_facade


def test_integration_descriptors_do_not_eagerly_store_unused_schema_models() -> None:
    source = (_source_root() / "integrations" / "contracts.py").read_text(encoding="utf-8")

    assert "schema_model" not in source
