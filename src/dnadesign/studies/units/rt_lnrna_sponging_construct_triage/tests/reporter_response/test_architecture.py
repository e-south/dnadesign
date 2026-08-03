"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/test_architecture.py

Architecture boundaries for the reporter-response package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_profile_contract_uses_the_bounded_package_layout() -> None:
    root = _reporter_response_root()

    assert (root / "profile").is_dir()
    assert not (root / "profile.py").exists()


def test_reporter_response_public_surface_uses_semantic_leaves() -> None:
    root = _reporter_response_root()
    budgets = {
        "building.py": 80,
        "comparison.py": 60,
        "parsing.py": 380,
        "serialization.py": 60,
    }

    assert not (root / "api.py").exists()
    for filename, line_budget in budgets.items():
        line_count = len((root / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_reporter_response_internal_import_graph_is_acyclic() -> None:
    root = _reporter_response_root()
    package = "dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response"
    modules = {_module_name(path, root=root, package=package): path for path in root.rglob("*.py")}
    graph = {name: set() for name in modules}

    for owner, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        owner_package = owner if path.name == "__init__.py" else owner.rsplit(".", maxsplit=1)[0]
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                target = _resolve_import_from(node, owner_package=owner_package)
                if target is not None:
                    targets.append(target)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            for target in targets:
                if target == package and owner != package:
                    raise AssertionError(f"{path} imports the reporter_response public facade")
                if target in modules and target != owner:
                    graph[owner].add(target)
                parent = target.rsplit(".", maxsplit=1)[0] if "." in target else ""
                while parent.startswith(package + "."):
                    if parent in modules and owner != parent and not owner.startswith(parent + "."):
                        graph[owner].add(parent)
                    parent = parent.rsplit(".", maxsplit=1)[0] if "." in parent else ""

    cycle = _first_import_cycle(graph)
    assert cycle is None, "reporter_response import cycle: " + " -> ".join(cycle or ())


def _reporter_response_root() -> Path:
    return Path(__file__).resolve().parents[2] / "reporter_response"


def _module_name(path: Path, *, root: Path, package: str) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((package, *parts)) if parts else package


def _resolve_import_from(node: ast.ImportFrom, *, owner_package: str) -> str | None:
    if node.level == 0:
        return node.module
    owner_parts = owner_package.split(".")
    keep = len(owner_parts) - (node.level - 1)
    if keep <= 0:
        return None
    suffix = node.module.split(".") if node.module else []
    return ".".join((*owner_parts[:keep], *suffix))


def _first_import_cycle(graph: dict[str, set[str]]) -> tuple[str, ...] | None:
    visiting: list[str] = []
    active: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> tuple[str, ...] | None:
        if module in active:
            start = visiting.index(module)
            return tuple((*visiting[start:], module))
        if module in visited:
            return None
        active.add(module)
        visiting.append(module)
        for dependency in sorted(graph[module]):
            cycle = visit(dependency)
            if cycle is not None:
                return cycle
        visiting.pop()
        active.remove(module)
        visited.add(module)
        return None

    for module in sorted(graph):
        cycle = visit(module)
        if cycle is not None:
            return cycle
    return None
