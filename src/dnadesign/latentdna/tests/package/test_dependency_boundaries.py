"""
Import-boundary tests for latentdna runtime packages.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _latentdna_src() -> Path:
    return _repo_root() / "src" / "dnadesign" / "latentdna" / "src"


def _module_name(path: Path) -> str:
    relative = path.relative_to(_latentdna_src()).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["dnadesign", "latentdna", "src", *parts])


def _resolve_imports(path: Path) -> list[str]:
    module_name = _module_name(path)
    package_parts = module_name.split(".")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    imports.append(node.module)
                continue
            anchor_parts = package_parts[: -node.level]
            if node.module is not None:
                anchor_parts.extend(node.module.split("."))
            if anchor_parts:
                imports.append(".".join(anchor_parts))
    return imports


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_latentdna_runtime_does_not_import_cluster_or_opal() -> None:
    offenders: list[str] = []
    for path in _python_files(_latentdna_src()):
        for imported in _resolve_imports(path):
            if imported.startswith("dnadesign.cluster") or imported.startswith("dnadesign.opal"):
                offenders.append(f"{path.relative_to(_repo_root())}: {imported}")

    assert offenders == []


def test_non_cli_modules_do_not_depend_on_cli_runtime() -> None:
    cli_root = _latentdna_src() / "cli"
    offenders: list[str] = []
    for path in _python_files(_latentdna_src()):
        if cli_root in path.parents:
            continue
        for imported in _resolve_imports(path):
            if imported.startswith("dnadesign.latentdna.src.cli"):
                offenders.append(f"{path.relative_to(_repo_root())}: {imported}")

    assert offenders == []
