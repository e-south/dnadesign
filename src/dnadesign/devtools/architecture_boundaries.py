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

_NON_TOOL_DIRS = {"devtools", "__pycache__"}
_SKIPPED_PATH_SEGMENTS = {
    "tests",
    "notebooks",
    "docs",
    "workspaces",
    "jobs",
    "images",
    "datasets",
    "demo_material",
    "__pycache__",
}
_ALLOWED_CROSS_TOOL_IMPORTS: set[tuple[str, str]] = {
    ("billboard", "aligner"),
    ("cluster", "aligner"),
    ("cluster", "usr"),
    ("cruncher", "baserender"),
    ("densegen", "cruncher"),
    ("densegen", "usr"),
    ("infer", "usr"),
    ("libshuffle", "aligner"),
    ("libshuffle", "billboard"),
    ("libshuffle", "nmf"),
    ("notify", "densegen"),
    ("permuter", "infer"),
}


@dataclass(frozen=True)
class ImportViolation:
    owner_tool: str
    imported_tool: str
    file_path: Path
    import_target: str


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
            if imported_tool not in tool_names or imported_tool == owner_tool:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check dnadesign cross-tool import boundaries.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        violations = find_undeclared_cross_tool_imports(repo_root=args.repo_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if not violations:
        print("Architecture boundary checks passed.")
        return 0

    print("Architecture boundary check failed: undeclared cross-tool imports detected.")
    for item in violations:
        print(f" - {item.file_path}: {item.owner_tool} -> {item.imported_tool} via '{item.import_target}'")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
