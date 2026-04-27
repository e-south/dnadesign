"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/architecture_boundaries.py

Enforces explicit cross-tool import boundaries for dnadesign package modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

TOP_LEVEL_ROOT_MODULES = {"__init__.py"}
TOP_LEVEL_TOOL_BOUNDARY_PACKAGES = {
    "aligner",
    "baserender",
    "billboard",
    "cluster",
    "construct",
    "cruncher",
    "densegen",
    "infer",
    "latentdna",
    "libshuffle",
    "nmf",
    "notify",
    "opal",
    "ops",
    "permuter",
    "studies",
    "tfkdanalysis",
    "usr",
}
TOP_LEVEL_SHARED_INFRA_PACKAGES = {"contracts", "devtools", "testsupport"}
TOP_LEVEL_LEGACY_DIRECTORIES = {"archived", "prototypes"}
_IGNORED_TOP_LEVEL_DIRECTORIES = {"__pycache__"}
_NON_TOOL_DIRS = TOP_LEVEL_SHARED_INFRA_PACKAGES | TOP_LEVEL_LEGACY_DIRECTORIES | _IGNORED_TOP_LEVEL_DIRECTORIES
_SKIPPED_PATH_SEGMENTS = {
    "tests",
    "notebooks",
    "docs",
    "workspaces",
    "jobs",
    "images",
    "assets",
    "datasets",
    "demo_material",
    "__pycache__",
}
_TEST_SUPPORT_PACKAGE = "testsupport"
_ALLOWED_CROSS_TOOL_IMPORTS: set[tuple[str, str]] = {
    ("billboard", "aligner"),
    ("cluster", "aligner"),
    ("cluster", "ops"),
    ("cluster", "usr"),
    ("construct", "usr"),
    ("cruncher", "baserender"),
    ("densegen", "baserender"),
    ("densegen", "cruncher"),
    ("densegen", "usr"),
    ("infer", "usr"),
    ("latentdna", "usr"),
    ("libshuffle", "aligner"),
    ("libshuffle", "billboard"),
    ("libshuffle", "nmf"),
    ("notify", "construct"),
    ("notify", "densegen"),
    ("notify", "infer"),
    ("opal", "ops"),
    ("ops", "construct"),
    ("ops", "densegen"),
    ("ops", "infer"),
    ("ops", "notify"),
    ("ops", "usr"),
    ("permuter", "infer"),
    ("studies", "infer"),
    ("studies", "ops"),
    ("studies", "densegen"),
    ("studies", "usr"),
    ("usr", "ops"),
}
_FORBIDDEN_LEGACY_SURFACE_PATHS = (
    Path("src/dnadesign/_contracts"),
    Path("src/dnadesign/usr_roots.py"),
    Path("src/dnadesign/usr/src/roots.py"),
    Path("src/dnadesign/ops/providers"),
    Path("src/dnadesign/ops/orchestrator/contracts.py"),
    Path("src/dnadesign/ops/promoter_study_context.py"),
    Path("src/dnadesign/ops/promoter_study_infer_runtime.py"),
    Path("src/dnadesign/ops/promoter_study_status_coordinator.py"),
    Path("src/dnadesign/ops/promoter_preflight_scope.py"),
    Path("src/dnadesign/ops/promoter_preflight_orchestration.py"),
    Path("src/dnadesign/ops/promoter_preflight_upstream.py"),
    Path("src/dnadesign/ops/promoter_preflight_infer.py"),
    Path("src/dnadesign/ops/promoter_preflight_coordinator.py"),
    Path("src/dnadesign/ops/tests/test_promoter_study_infer_runtime.py"),
    Path("src/dnadesign/ops/tests/test_promoter_study_status_coordinator.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_scope.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_orchestration.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_upstream.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_infer.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_coordinator.py"),
    Path("src/dnadesign/studies/promoter"),
    Path("docs/studies/promoter"),
    Path("src/dnadesign/studies/families/promoter/preflight_infer.py"),
    Path("src/dnadesign/studies/families/promoter/preflight_orchestration.py"),
    Path("src/dnadesign/studies/families/promoter/preflight_upstream.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_infer.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_orchestration.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_upstream.py"),
)
_ALLOWED_OPS_ROOT_CLI_PATHS = {
    Path("src/dnadesign/ops/cli"),
}
_ALLOWED_CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class ImportViolation:
    owner_tool: str
    imported_tool: str
    file_path: Path
    import_target: str


@dataclass(frozen=True)
class LegacySurfaceViolation:
    path: Path


@dataclass(frozen=True)
class TopLevelLayoutViolation:
    path: Path
    reason: str


def _discover_tools(repo_root: Path) -> set[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")
    return {
        path.name
        for path in src_root.iterdir()
        if path.is_dir() and path.name not in _NON_TOOL_DIRS and not path.name.startswith("_")
    }


def _iter_checked_python_files(repo_root: Path, tool_names: set[str]) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    files: list[Path] = []
    for path in src_root.rglob("*.py"):
        rel = path.relative_to(src_root)
        if not rel.parts:
            continue
        if rel.parts[0] not in tool_names:
            continue
        if any(segment in _SKIPPED_PATH_SEGMENTS for segment in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def _resolve_relative_import_base(*, node: ast.ImportFrom, package_parts: tuple[str, ...]) -> str | None:
    if node.level <= 0:
        return node.module

    parent_hops = node.level - 1
    if parent_hops > (len(package_parts) - 1):
        return None

    base_parts = package_parts[: len(package_parts) - parent_hops]
    return ".".join(base_parts)


def _iter_import_targets(module: ast.Module, *, package_parts: tuple[str, ...]) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative_import_base(node=node, package_parts=package_parts)
            if base is None:
                continue
            if node.module is not None:
                if node.level <= 0:
                    targets.append(node.module)
                    continue
                module_parts = tuple(part for part in node.module.split(".") if part)
                targets.append(".".join((*base.split("."), *module_parts)))
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                targets.append(f"{base}.{alias.name}")
    return targets


def find_undeclared_cross_tool_imports(
    *,
    repo_root: Path,
    allowed_edges: set[tuple[str, str]] | None = None,
) -> list[ImportViolation]:
    tool_names = _discover_tools(repo_root)
    allowed = _ALLOWED_CROSS_TOOL_IMPORTS if allowed_edges is None else allowed_edges
    src_root = repo_root / "src" / "dnadesign"
    violations: list[ImportViolation] = []

    for file_path in _iter_checked_python_files(repo_root, tool_names):
        rel_path = file_path.relative_to(src_root)
        owner_tool = rel_path.parts[0]
        package_parts = ("dnadesign", *rel_path.parts[:-1])
        source = file_path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            raise ValueError(f"Unable to parse Python file for boundary checks: {file_path}: {exc.msg}") from exc

        for target in _iter_import_targets(module, package_parts=package_parts):
            if not target.startswith("dnadesign."):
                continue
            parts = target.split(".")
            if len(parts) < 2:
                continue
            imported_tool = parts[1]
            if imported_tool == _TEST_SUPPORT_PACKAGE:
                violations.append(
                    ImportViolation(
                        owner_tool=owner_tool,
                        imported_tool=imported_tool,
                        file_path=file_path,
                        import_target=target,
                    )
                )
                continue
            if imported_tool not in tool_names or imported_tool == owner_tool:
                continue
            if target == f"dnadesign.{imported_tool}.src" or target.startswith(f"dnadesign.{imported_tool}.src."):
                violations.append(
                    ImportViolation(
                        owner_tool=owner_tool,
                        imported_tool=imported_tool,
                        file_path=file_path,
                        import_target=target,
                    )
                )
                continue
            if (owner_tool, imported_tool) in allowed:
                continue
            violations.append(
                ImportViolation(
                    owner_tool=owner_tool,
                    imported_tool=imported_tool,
                    file_path=file_path,
                    import_target=target,
                )
            )

    return sorted(
        violations,
        key=lambda item: (item.owner_tool, item.imported_tool, str(item.file_path), item.import_target),
    )


def find_legacy_surface_violations(*, repo_root: Path) -> list[LegacySurfaceViolation]:
    resolved_repo_root = repo_root.expanduser().resolve()
    violations: list[LegacySurfaceViolation] = []
    for relative_path in _FORBIDDEN_LEGACY_SURFACE_PATHS:
        candidate = (resolved_repo_root / relative_path).resolve()
        if candidate.exists() and not _is_cache_only_legacy_path(candidate):
            violations.append(LegacySurfaceViolation(path=candidate))
    ops_root = (resolved_repo_root / "src" / "dnadesign" / "ops").resolve()
    if ops_root.exists():
        for candidate in sorted(ops_root.iterdir()):
            relative_candidate = candidate.relative_to(resolved_repo_root)
            if relative_candidate in _ALLOWED_OPS_ROOT_CLI_PATHS:
                continue
            if candidate.is_file() and "cli" in candidate.stem:
                violations.append(LegacySurfaceViolation(path=candidate.resolve()))
    return sorted(violations, key=lambda item: str(item.path))


def find_top_level_layout_violations(*, repo_root: Path) -> list[TopLevelLayoutViolation]:
    resolved_repo_root = repo_root.expanduser().resolve()
    src_root = resolved_repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")

    required_directories = TOP_LEVEL_TOOL_BOUNDARY_PACKAGES | TOP_LEVEL_SHARED_INFRA_PACKAGES
    allowed_directories = required_directories | TOP_LEVEL_LEGACY_DIRECTORIES
    actual_directories = {
        path.name
        for path in src_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in _IGNORED_TOP_LEVEL_DIRECTORIES
    }
    actual_root_modules = {path.name for path in src_root.glob("*.py")}

    violations: list[TopLevelLayoutViolation] = []
    for name in sorted(required_directories - actual_directories):
        violations.append(
            TopLevelLayoutViolation(
                path=src_root / name,
                reason="sanctioned top-level directory missing",
            )
        )
    for name in sorted(actual_directories - allowed_directories):
        violations.append(
            TopLevelLayoutViolation(
                path=src_root / name,
                reason="unexpected top-level directory",
            )
        )
    for name in sorted(TOP_LEVEL_ROOT_MODULES - actual_root_modules):
        violations.append(
            TopLevelLayoutViolation(
                path=src_root / name,
                reason="sanctioned top-level module missing",
            )
        )
    for name in sorted(actual_root_modules - TOP_LEVEL_ROOT_MODULES):
        violations.append(
            TopLevelLayoutViolation(
                path=src_root / name,
                reason="unexpected top-level module",
            )
        )

    return violations


def _is_cache_only_legacy_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix in _ALLOWED_CACHE_FILE_SUFFIXES
    if not path.is_dir():
        return False

    descendants = list(path.rglob("*"))
    if not descendants:
        return False

    for descendant in descendants:
        if descendant.is_dir():
            if descendant.name != "__pycache__":
                return False
            continue
        if descendant.suffix not in _ALLOWED_CACHE_FILE_SUFFIXES:
            return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check dnadesign import and top-level layout boundaries.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        violations = find_undeclared_cross_tool_imports(repo_root=args.repo_root)
        legacy_surface_violations = find_legacy_surface_violations(repo_root=args.repo_root)
        top_level_layout_violations = find_top_level_layout_violations(repo_root=args.repo_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if not violations and not legacy_surface_violations and not top_level_layout_violations:
        print("Architecture boundary checks passed.")
        return 0

    print("Architecture boundary check failed.")
    for item in violations:
        print(f" - {item.file_path}: {item.owner_tool} -> {item.imported_tool} via '{item.import_target}'")
    for item in legacy_surface_violations:
        print(f" - forbidden legacy surface still exists: {item.path}")
    for item in top_level_layout_violations:
        print(f" - {item.reason}: {item.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
