"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/architecture/boundaries.py

Enforces explicit cross-tool import boundaries for dnadesign package modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

from dnadesign.devtools.architecture.lifecycle import find_installation_issues, load_lifecycle

TOP_LEVEL_ROOT_MODULES = {"__init__.py", "_entrypoints.py"}
TOP_LEVEL_SHARED_INFRA_PACKAGES = {"artifacts", "contracts", "devtools"}
_IGNORED_TOP_LEVEL_DIRECTORIES = {"__pycache__"}
_NON_TOOL_DIRS = TOP_LEVEL_SHARED_INFRA_PACKAGES | _IGNORED_TOP_LEVEL_DIRECTORIES
_RESOURCE_ROOT_DIRECTORIES = {
    "tests",
    "docs",
    "workspaces",
    "campaigns",
    "jobs",
    "images",
    "assets",
    "datasets",
    "demo_material",
    "__pycache__",
}
_REVIEW_SURFACE_PATH_SEGMENTS = {"tests", "notebooks", "docs", "jobs"}
_REVIEW_SURFACE_CHECKED_OWNERS = {"ops", "devtools"}
_TEST_SUPPORT_IMPORT_PREFIXES = ("dnadesign.devtools.tests.support", "dnadesign.testsupport")
_ALLOWED_CROSS_TOOL_IMPORTS: set[tuple[str, str]] = {
    ("cluster", "aligner"),
    ("cluster", "ops"),
    ("cluster", "usr"),
    ("construct", "folding"),
    ("construct", "usr"),
    ("cruncher", "baserender"),
    ("densegen", "baserender"),
    ("densegen", "cruncher"),
    ("densegen", "usr"),
    ("infer", "usr"),
    ("latentdna", "usr"),
    ("notify", "construct"),
    ("notify", "densegen"),
    ("notify", "infer"),
    ("opal", "ops"),
    ("opal", "baserender"),
    ("opal", "usr"),
    ("ops", "construct"),
    ("ops", "densegen"),
    ("ops", "infer"),
    ("ops", "notify"),
    ("ops", "usr"),
    ("permuter", "infer"),
    ("usr", "ops"),
}
_ALLOWED_CROSS_TOOL_EXACT_IMPORT_TARGETS: dict[tuple[str, str], tuple[str, ...]] = {
    ("notify", "construct"): ("dnadesign.construct",),
    ("ops", "construct"): ("dnadesign.construct",),
    ("ops", "densegen"): ("dnadesign.densegen.contracts",),
    ("ops", "infer"): ("dnadesign.infer", "dnadesign.infer.contracts"),
    ("ops", "notify"): ("dnadesign.notify",),
    ("ops", "usr"): ("dnadesign.usr",),
    ("opal", "usr"): ("dnadesign.usr",),
    ("opal", "baserender"): ("dnadesign.baserender",),
    ("permuter", "infer"): ("dnadesign.infer",),
    ("construct", "folding"): ("dnadesign.folding",),
    ("devtools", "ops"): (
        "dnadesign.ops.catalog",
        "dnadesign.ops.runbooks",
        "dnadesign.ops.status",
    ),
}
_ALLOWED_CROSS_TOOL_IMPORT_TARGET_PREFIXES: dict[tuple[str, str], tuple[str, ...]] = {
    ("usr", "cruncher"): ("dnadesign.cruncher.ingest.promoters",),
}


def affected_tool_consumers(changed: set[str], tool_names: set[str]) -> set[str]:
    """Include transitive consumers from the declared public dependency policy.

    Permissions conservatively bound impact; they are not inferred from import
    syntax. Restrict traversal to components present in the CI tool inventory.
    """
    reverse: dict[str, set[str]] = {}
    edges = (
        _ALLOWED_CROSS_TOOL_IMPORTS
        | _ALLOWED_CROSS_TOOL_EXACT_IMPORT_TARGETS.keys()
        | _ALLOWED_CROSS_TOOL_IMPORT_TARGET_PREFIXES.keys()
    )
    for consumer, provider in edges:
        if consumer in tool_names and provider in tool_names:
            reverse.setdefault(provider, set()).add(consumer)
    affected = set(changed)
    pending = list(changed)
    while pending:
        for consumer in reverse.get(pending.pop(), set()) - affected:
            affected.add(consumer)
            pending.append(consumer)
    return affected


_FORBIDDEN_LEGACY_SURFACE_PATHS = (
    Path("src/dnadesign/archived"),
    Path("src/dnadesign/prototypes"),
    Path("src/dnadesign/_contracts"),
    Path("src/dnadesign/usr_roots.py"),
    Path("src/dnadesign/usr/src/roots.py"),
    Path("src/dnadesign/ops/orchestrator/contracts.py"),
    Path("src/dnadesign/ops/promoter_study_context.py"),
    Path("src/dnadesign/ops/promoter_study_infer_runtime.py"),
    Path("src/dnadesign/ops/promoter_study_status_coordinator.py"),
    Path("src/dnadesign/ops/promoter_preflight_scope.py"),
    Path("src/dnadesign/ops/promoter_preflight_orchestration.py"),
    Path("src/dnadesign/ops/promoter_preflight_upstream.py"),
    Path("src/dnadesign/ops/promoter_preflight_infer.py"),
    Path("src/dnadesign/ops/promoter_preflight_coordinator.py"),
    Path("src/dnadesign/trijunction"),
    Path("src/dnadesign/junction/canonical.py"),
    Path("src/dnadesign/junction/exports.py"),
    Path("src/dnadesign/ops/tests/test_promoter_study_infer_runtime.py"),
    Path("src/dnadesign/ops/tests/test_promoter_study_status_coordinator.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_scope.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_orchestration.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_upstream.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_infer.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_coordinator.py"),
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
    resolved_repo_root = repo_root.expanduser().resolve()
    src_root = resolved_repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")
    return {
        path.name
        for path in src_root.iterdir()
        if path.is_dir()
        and path.name not in _NON_TOOL_DIRS
        and not path.name.startswith("_")
        and not _is_ignored_untracked_path(repo_root=resolved_repo_root, path=path)
    }


def _boundary_checked_owners(*, repo_root: Path, tool_names: set[str]) -> set[str]:
    checked = set(tool_names)
    checked.update(
        owner for owner in TOP_LEVEL_SHARED_INFRA_PACKAGES if (repo_root / "src" / "dnadesign" / owner).is_dir()
    )
    return checked


def _skip_checked_python_file(relative_path: Path) -> bool:
    parts = relative_path.parts
    if "__pycache__" in parts:
        return True
    if parts and parts[0] in TOP_LEVEL_SHARED_INFRA_PACKAGES:
        return len(parts) > 1 and parts[1] == "tests"
    # Resource shelves live at the tool root. The same names under src/ are
    # runtime packages and must receive the same checks as any other module.
    return len(parts) > 1 and parts[1] in _RESOURCE_ROOT_DIRECTORIES


def _iter_checked_python_files(repo_root: Path, tool_names: set[str]) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    files: list[Path] = []
    checked_owners = _boundary_checked_owners(repo_root=repo_root, tool_names=tool_names)
    for path in src_root.rglob("*.py"):
        rel = path.relative_to(src_root)
        if not rel.parts:
            continue
        if rel.parts[0] not in checked_owners:
            continue
        if _skip_checked_python_file(rel):
            continue
        files.append(path)
    return sorted(files)


def _iter_review_surface_python_files(repo_root: Path, tool_names: set[str]) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    files: list[Path] = []
    for path in src_root.rglob("*.py"):
        rel = path.relative_to(src_root)
        if not rel.parts:
            continue
        if rel.parts[0] not in _REVIEW_SURFACE_CHECKED_OWNERS:
            continue
        if not any(segment in _REVIEW_SURFACE_PATH_SEGMENTS for segment in rel.parts):
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


def _literal_dynamic_imports(module: ast.Module, *, package_parts: tuple[str, ...]) -> list[str]:
    """Resolve literal importlib/builtin calls without importing tool runtimes."""
    library_names = {"importlib"}
    loader_names = {"__import__"}
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            library_names.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "importlib":
            loader_names.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    targets: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_loader = (isinstance(function, ast.Name) and function.id in loader_names) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in library_names
            and function.attr == "import_module"
        )
        if not is_loader:
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        name = node.args[0] if node.args else keywords.get("name")
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            continue
        target = name.value
        if target.startswith("."):
            package = node.args[1] if len(node.args) > 1 else keywords.get("package")
            if isinstance(package, ast.Constant) and isinstance(package.value, str):
                package_name = package.value
            elif isinstance(package, ast.Name) and package.id == "__package__":
                package_name = ".".join(package_parts)
            else:
                continue
            target = resolve_name(target, package_name)
        targets.append(target)
    return targets


def _iter_import_targets(module: ast.Module, *, package_parts: tuple[str, ...], src_root: Path) -> list[str]:
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
                    base = node.module
                else:
                    base = f"{base}.{node.module}"
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                path = src_root.joinpath(*candidate.split(".")[1:])
                is_submodule = candidate.startswith("dnadesign.") and (
                    path.is_dir() or path.with_suffix(".py").is_file()
                )
                targets.append(candidate if alias.name != "*" and (node.module is None or is_submodule) else base)
    return list(dict.fromkeys(targets + _literal_dynamic_imports(module, package_parts=package_parts)))


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

        for target in _iter_import_targets(module, package_parts=package_parts, src_root=src_root):
            if not target.startswith("dnadesign."):
                continue
            parts = target.split(".")
            if len(parts) < 2:
                continue
            if target in _TEST_SUPPORT_IMPORT_PREFIXES or any(
                target.startswith(f"{prefix}.") for prefix in _TEST_SUPPORT_IMPORT_PREFIXES
            ):
                imported_tool = "devtools.tests.support" if target.startswith("dnadesign.devtools.") else "testsupport"
                violations.append(
                    ImportViolation(
                        owner_tool=owner_tool,
                        imported_tool=imported_tool,
                        file_path=file_path,
                        import_target=target,
                    )
                )
                continue
            imported_tool = parts[1]
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
            exact_targets = _ALLOWED_CROSS_TOOL_EXACT_IMPORT_TARGETS.get((owner_tool, imported_tool), ())
            if exact_targets:
                if target in exact_targets:
                    continue
                violations.append(
                    ImportViolation(
                        owner_tool=owner_tool,
                        imported_tool=imported_tool,
                        file_path=file_path,
                        import_target=target,
                    )
                )
                continue
            target_prefixes = _ALLOWED_CROSS_TOOL_IMPORT_TARGET_PREFIXES.get((owner_tool, imported_tool), ())
            if target_prefixes:
                if any(target == prefix or target.startswith(f"{prefix}.") for prefix in target_prefixes):
                    continue
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


def find_review_surface_private_imports(*, repo_root: Path) -> list[ImportViolation]:
    tool_names = _discover_tools(repo_root)
    src_root = repo_root / "src" / "dnadesign"
    violations: list[ImportViolation] = []

    for file_path in _iter_review_surface_python_files(repo_root, tool_names):
        rel_path = file_path.relative_to(src_root)
        owner_tool = rel_path.parts[0]
        package_parts = ("dnadesign", *rel_path.parts[:-1])
        source = file_path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            message = f"Unable to parse Python file for review-surface boundary checks: {file_path}: {exc.msg}"
            raise ValueError(message) from exc

        for target in _iter_import_targets(module, package_parts=package_parts, src_root=src_root):
            if not target.startswith("dnadesign."):
                continue
            parts = target.split(".")
            if len(parts) < 4:
                continue
            imported_tool = parts[1]
            if imported_tool == owner_tool:
                continue
            if imported_tool not in tool_names and imported_tool != "devtools":
                continue
            if parts[2] != "src":
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

    required_directories = {
        name for name, component in load_lifecycle(resolved_repo_root).items() if component.status != "retired"
    }
    allowed_directories = required_directories
    actual_directories = {
        path.name
        for path in src_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in _IGNORED_TOP_LEVEL_DIRECTORIES
        and not _is_ignored_untracked_path(repo_root=resolved_repo_root, path=path)
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


def find_external_study_boundary_violations(*, repo_root: Path) -> list[TopLevelLayoutViolation]:
    """Reject live study packages from the public dnadesign source tree."""
    resolved_repo_root = repo_root.expanduser().resolve()
    studies_root = resolved_repo_root / "src" / "dnadesign" / "studies"
    if not studies_root.exists():
        return []

    relative_root = studies_root.relative_to(resolved_repo_root)
    try:
        tracked_result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--", relative_root.as_posix()],
            cwd=resolved_repo_root,
            check=False,
            capture_output=True,
        )
    except OSError:
        tracked_files: tuple[bytes, ...] = ()
    else:
        tracked_files = tuple(path for path in tracked_result.stdout.split(b"\0") if path)
    if tracked_files:
        return [
            TopLevelLayoutViolation(
                path=studies_root,
                reason="live study packages must remain external to dnadesign",
            )
        ]

    if _is_ignored_untracked_path(repo_root=resolved_repo_root, path=studies_root):
        return []
    ignored_local_segments = {"__pycache__"}
    source_files = [
        path
        for path in studies_root.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and not ignored_local_segments.intersection(path.relative_to(studies_root).parts)
    ]
    if not source_files:
        return []
    return [
        TopLevelLayoutViolation(
            path=studies_root,
            reason="live study packages must remain external to dnadesign",
        )
    ]


def _is_ignored_untracked_path(*, repo_root: Path, path: Path) -> bool:
    """Return whether Git excludes a local path that is absent from its index."""

    try:
        relative_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative_path.as_posix()],
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


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
        review_surface_violations = find_review_surface_private_imports(repo_root=args.repo_root)
        legacy_surface_violations = find_legacy_surface_violations(repo_root=args.repo_root)
        top_level_layout_violations = find_top_level_layout_violations(repo_root=args.repo_root)
        external_study_violations = find_external_study_boundary_violations(repo_root=args.repo_root)
        installation_issues = find_installation_issues(args.repo_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if not (
        violations
        or review_surface_violations
        or legacy_surface_violations
        or top_level_layout_violations
        or external_study_violations
        or installation_issues
    ):
        print("Architecture boundary checks passed.")
        return 0

    print("Architecture boundary check failed.")
    for item in violations:
        print(f" - {item.file_path}: {item.owner_tool} -> {item.imported_tool} via '{item.import_target}'")
    for item in review_surface_violations:
        print(
            " - review surface imports private sibling internals: "
            f"{item.file_path}: {item.owner_tool} -> {item.imported_tool} via '{item.import_target}'"
        )
    for item in legacy_surface_violations:
        print(f" - forbidden legacy surface still exists: {item.path}")
    for item in top_level_layout_violations:
        print(f" - {item.reason}: {item.path}")
    for item in external_study_violations:
        print(f" - {item.reason}: {item.path}")
    for issue in installation_issues:
        print(f" - {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
