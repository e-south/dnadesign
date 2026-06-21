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
    "folding",
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
TOP_LEVEL_SHARED_INFRA_PACKAGES = {"contracts", "devtools"}
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
_REVIEW_SURFACE_PATH_SEGMENTS = {"tests", "notebooks", "docs", "jobs"}
_REVIEW_SURFACE_CHECKED_OWNERS = {"ops", "studies", "devtools"}
_TEST_SUPPORT_IMPORT_PREFIXES = ("dnadesign.devtools.tests.support", "dnadesign.testsupport")
_ALLOWED_CROSS_TOOL_IMPORTS: set[tuple[str, str]] = {
    ("billboard", "aligner"),
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
    ("studies", "aligner"),
    ("studies", "infer"),
    ("studies", "ops"),
    ("studies", "densegen"),
    ("studies", "opal"),
    ("studies", "permuter"),
    ("studies", "usr"),
    ("usr", "ops"),
}
_ALLOWED_CROSS_TOOL_EXACT_IMPORT_TARGETS: dict[tuple[str, str], tuple[str, ...]] = {
    ("notify", "construct"): ("dnadesign.construct",),
    ("ops", "construct"): ("dnadesign.construct",),
    ("ops", "densegen"): ("dnadesign.densegen.contracts",),
    ("ops", "infer"): ("dnadesign.infer", "dnadesign.infer.contracts"),
    ("ops", "notify"): ("dnadesign.notify.core.contracts",),
    ("ops", "usr"): ("dnadesign.usr",),
    ("permuter", "infer"): ("dnadesign.infer",),
    ("construct", "folding"): ("dnadesign.folding",),
    ("studies", "aligner"): ("dnadesign.aligner.msa",),
    ("studies", "baserender"): ("dnadesign.baserender",),
    ("studies", "construct"): ("dnadesign.construct",),
    ("studies", "densegen"): ("dnadesign.densegen",),
    ("studies", "infer"): ("dnadesign.infer", "dnadesign.infer.contracts"),
    ("studies", "opal"): ("dnadesign.opal",),
    ("studies", "permuter"): ("dnadesign.permuter",),
    ("studies", "ops"): (
        "dnadesign.ops.catalog",
        "dnadesign.ops.preflight",
        "dnadesign.ops.status",
    ),
    ("studies", "usr"): ("dnadesign.usr",),
    ("devtools", "ops"): (
        "dnadesign.ops.catalog",
        "dnadesign.ops.runbooks",
        "dnadesign.ops.status",
    ),
}
_ALLOWED_CROSS_TOOL_IMPORT_TARGET_PREFIXES: dict[tuple[str, str], tuple[str, ...]] = {
    ("studies", "cruncher"): ("dnadesign.cruncher.scar_nick", "dnadesign.cruncher.snapback"),
    ("usr", "cruncher"): ("dnadesign.cruncher.ingest.promoters",),
}
_FORBIDDEN_LEGACY_SURFACE_PATHS = (
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
    Path("src/dnadesign/ops/tests/test_promoter_study_infer_runtime.py"),
    Path("src/dnadesign/ops/tests/test_promoter_study_status_coordinator.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_scope.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_orchestration.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_upstream.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_infer.py"),
    Path("src/dnadesign/ops/tests/test_promoter_preflight_coordinator.py"),
    Path("src/dnadesign/studies/families"),
    Path("src/dnadesign/studies/promoter"),
    Path("docs/studies/promoter"),
    Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/preflight_infer.py"),
    Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/preflight_orchestration.py"),
    Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/preflight_upstream.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_infer.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_orchestration.py"),
    Path("src/dnadesign/studies/tests/test_promoter_preflight_upstream.py"),
)
_ALLOWED_OPS_ROOT_CLI_PATHS = {
    Path("src/dnadesign/ops/cli"),
}
_ALLOWED_STUDIES_ROOT_DIRECTORIES = {"assets", "core", "tests", "units"}
_ALLOWED_STUDIES_ROOT_FILES = {"README.md", "__init__.py"}
_ALLOWED_CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}
_CONCRETE_STUDIES_PACKAGE_NAME = "units"


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


def _boundary_checked_owners(*, repo_root: Path, tool_names: set[str]) -> set[str]:
    checked = set(tool_names)
    if (repo_root / "src" / "dnadesign" / "devtools").is_dir():
        checked.add("devtools")
    return checked


def _skip_checked_python_file(relative_path: Path) -> bool:
    if relative_path.parts and relative_path.parts[0] == "devtools":
        return "tests" in relative_path.parts or "__pycache__" in relative_path.parts
    return any(segment in _SKIPPED_PATH_SEGMENTS for segment in relative_path.parts)


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

        for target in _iter_import_targets(module, package_parts=package_parts):
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
            if (
                owner_tool in {"ops", "devtools"}
                and imported_tool == "studies"
                and len(parts) >= 4
                and parts[2] == _CONCRETE_STUDIES_PACKAGE_NAME
            ):
                violations.append(
                    ImportViolation(
                        owner_tool=owner_tool,
                        imported_tool=imported_tool,
                        file_path=file_path,
                        import_target=target,
                    )
                )
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


def _discover_concrete_study_ids(studies_root: Path) -> tuple[str, ...]:
    study_units_root = studies_root / _CONCRETE_STUDIES_PACKAGE_NAME
    if not study_units_root.is_dir():
        return ()
    return tuple(
        candidate.name
        for candidate in sorted(study_units_root.iterdir())
        if candidate.is_dir() and not candidate.name.startswith(".") and candidate.name != "__pycache__"
    )


def _study_test_filename_prefixes(study_id: str) -> tuple[str, ...]:
    parts = tuple(part for part in study_id.split("_") if part)
    prefixes = {f"test_{study_id}"}
    if len(parts) >= 2:
        prefixes.add(f"test_{parts[0]}_{parts[1]}")
    if parts and len(parts[0]) >= 5:
        prefixes.add(f"test_{parts[0]}")
    return tuple(sorted(prefixes, key=lambda item: (len(item), item)))


def _root_test_imports_concrete_study(test_path: Path, *, study_id: str) -> bool:
    source = test_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(test_path))
    except SyntaxError as exc:
        raise ValueError(f"Unable to parse Python file for studies layout checks: {test_path}: {exc.msg}") from exc

    import_prefix = f"dnadesign.studies.units.{study_id}"
    for target in _iter_import_targets(module, package_parts=("dnadesign", "studies", "tests")):
        if target == import_prefix or target.startswith(f"{import_prefix}."):
            return True
    return False


def _root_test_matches_concrete_study(test_path: Path, *, study_id: str) -> bool:
    stem = test_path.stem
    return stem.startswith(_study_test_filename_prefixes(study_id)) or _root_test_imports_concrete_study(
        test_path,
        study_id=study_id,
    )


def find_studies_layout_violations(*, repo_root: Path) -> list[TopLevelLayoutViolation]:
    resolved_repo_root = repo_root.expanduser().resolve()
    studies_root = resolved_repo_root / "src" / "dnadesign" / "studies"
    if not studies_root.exists():
        raise FileNotFoundError(f"Expected studies source root at {studies_root}")

    violations: list[TopLevelLayoutViolation] = []
    for candidate in sorted(studies_root.iterdir()):
        if candidate.name.startswith(".") or candidate.name == "__pycache__":
            continue
        if candidate.is_dir():
            if candidate.name not in _ALLOWED_STUDIES_ROOT_DIRECTORIES:
                violations.append(
                    TopLevelLayoutViolation(
                        path=candidate,
                        reason="concrete study package must live under src/dnadesign/studies/units",
                    )
                )
            continue
        if candidate.is_file() and candidate.name not in _ALLOWED_STUDIES_ROOT_FILES:
            violations.append(
                TopLevelLayoutViolation(
                    path=candidate,
                    reason="unexpected studies package root file",
                )
            )

    concrete_study_ids = _discover_concrete_study_ids(studies_root)
    tests_root = studies_root / "tests"
    study_units_root = studies_root / _CONCRETE_STUDIES_PACKAGE_NAME
    for study_id in concrete_study_ids:
        study_tests_root = study_units_root / study_id / "tests"
        if not study_tests_root.is_dir():
            violations.append(
                TopLevelLayoutViolation(
                    path=study_tests_root,
                    reason="concrete study tests must live inside the owning study unit",
                )
            )
            continue
        if not (study_tests_root / "__init__.py").is_file():
            violations.append(
                TopLevelLayoutViolation(
                    path=study_tests_root / "__init__.py",
                    reason="concrete study tests package missing __init__.py",
                )
            )

    if tests_root.is_dir():
        for candidate in sorted(tests_root.iterdir()):
            if candidate.is_dir() and candidate.name in concrete_study_ids:
                violations.append(
                    TopLevelLayoutViolation(
                        path=candidate,
                        reason=(
                            f"study-specific tests must live under src/dnadesign/studies/units/{candidate.name}/tests"
                        ),
                    )
                )
        for test_path in sorted(tests_root.glob("test_*.py")):
            for study_id in concrete_study_ids:
                if _root_test_matches_concrete_study(test_path, study_id=study_id):
                    violations.append(
                        TopLevelLayoutViolation(
                            path=test_path,
                            reason=f"study-specific test must live under src/dnadesign/studies/units/{study_id}/tests",
                        )
                    )
                    break
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
        review_surface_violations = find_review_surface_private_imports(repo_root=args.repo_root)
        legacy_surface_violations = find_legacy_surface_violations(repo_root=args.repo_root)
        top_level_layout_violations = find_top_level_layout_violations(repo_root=args.repo_root)
        studies_layout_violations = find_studies_layout_violations(repo_root=args.repo_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if not (
        violations
        or review_surface_violations
        or legacy_surface_violations
        or top_level_layout_violations
        or studies_layout_violations
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
    for item in studies_layout_violations:
        print(f" - {item.reason}: {item.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
