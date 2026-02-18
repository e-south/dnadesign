"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/ci_changes.py

Computes CI core/external integration lane scope from changed files in pull request workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from .tool_coverage import load_baseline

_EXTERNAL_INTEGRATION_MARKERS = ("fimo", "integration")
_EXTERNAL_INTEGRATION_GLOBAL_FILES = {
    ".github/workflows/ci.yaml",
    "pixi.toml",
    "pixi.lock",
    "pyproject.toml",
    "uv.lock",
}
_FULL_CORE_EXACT_FILES = {
    ".github/workflows/ci.yaml",
    "pixi.toml",
    "pixi.lock",
    "pyproject.toml",
    "uv.lock",
    ".github/tool-coverage-baseline.json",
}
_NON_TOOL_DIRS = {"devtools", "__pycache__"}


@dataclass(frozen=True)
class ChangeScope:
    run_external_integration: bool
    run_full_core: bool
    affected_tools: list[str]
    external_integration_tools: list[str]

    @property
    def run_coverage_gate(self) -> bool:
        return self.run_full_core or bool(self.affected_tools)

    @property
    def affected_tools_csv(self) -> str:
        return ",".join(self.affected_tools)

    @property
    def external_integration_tools_csv(self) -> str:
        return ",".join(self.external_integration_tools)


def _collect_pytest_marker_roots(module: ast.Module) -> tuple[set[str], set[str]]:
    pytest_roots = {"pytest"}
    mark_roots: set[str] = set()

    for statement in module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "pytest":
                    pytest_roots.add(alias.asname or alias.name)

        if isinstance(statement, ast.ImportFrom) and statement.module == "pytest":
            for alias in statement.names:
                if alias.name == "mark":
                    mark_roots.add(alias.asname or alias.name)
                if alias.name == "pytest":
                    pytest_roots.add(alias.asname or alias.name)

    return pytest_roots, mark_roots


def _is_pytest_marker_node(node: ast.AST, marker_names: set[str], pytest_roots: set[str], mark_roots: set[str]) -> bool:
    marker_node = node.func if isinstance(node, ast.Call) else node
    if not isinstance(marker_node, ast.Attribute):
        return False
    if marker_node.attr not in marker_names:
        return False

    marker_root = marker_node.value
    if isinstance(marker_root, ast.Name):
        return marker_root.id in mark_roots

    if not isinstance(marker_root, ast.Attribute):
        return False
    if marker_root.attr != "mark":
        return False
    if not isinstance(marker_root.value, ast.Name):
        return False
    return marker_root.value.id in pytest_roots


def discover_repo_tools(*, repo_root: Path) -> set[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")
    return {
        path.name
        for path in src_root.iterdir()
        if path.is_dir() and path.name not in _NON_TOOL_DIRS and not path.name.startswith("_")
    }


def _module_has_external_integration_markers(module: ast.Module, marker_names: set[str]) -> bool:
    pytest_roots, mark_roots = _collect_pytest_marker_roots(module)

    for node in ast.walk(module):
        if _is_pytest_marker_node(node, marker_names, pytest_roots, mark_roots):
            return True
    return False


def _test_file_has_external_integration_markers(*, test_path: Path, marker_names: set[str]) -> bool:
    source = test_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(test_path))
    except SyntaxError as exc:
        raise ValueError(
            f"Unable to parse test file for external integration marker detection: {test_path}: {exc.msg}"
        ) from exc
    return _module_has_external_integration_markers(module, marker_names)


def discover_external_integration_tools(*, repo_root: Path, tool_names: set[str]) -> set[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")

    marker_names = set(_EXTERNAL_INTEGRATION_MARKERS)
    external_integration_tools: set[str] = set()
    for tool_name in sorted(tool_names):
        tests_root = src_root / tool_name / "tests"
        if not tests_root.exists():
            continue
        for test_path in tests_root.rglob("*.py"):
            if not (test_path.name.startswith("test_") or test_path.name == "conftest.py"):
                continue
            if _test_file_has_external_integration_markers(test_path=test_path, marker_names=marker_names):
                external_integration_tools.add(tool_name)
                break
    return external_integration_tools


def validate_tool_baseline(*, baseline_tools: set[str], repo_tools: set[str]) -> None:
    missing_from_baseline = sorted(repo_tools - baseline_tools)
    extra_in_baseline = sorted(baseline_tools - repo_tools)
    if not missing_from_baseline and not extra_in_baseline:
        return

    messages: list[str] = ["Tool coverage baseline mismatch with repository tools under src/dnadesign/."]
    if missing_from_baseline:
        messages.append(f"Missing baseline entries: {', '.join(missing_from_baseline)}")
    if extra_in_baseline:
        messages.append(f"Unknown baseline entries: {', '.join(extra_in_baseline)}")
    raise ValueError(" ".join(messages))


def determine_scope(
    *,
    event_name: str,
    changed_files: list[str],
    tool_names: set[str],
    external_integration_tool_names: set[str],
) -> ChangeScope:
    sorted_tools = sorted(tool_names)
    sorted_external_integration_tools = sorted(external_integration_tool_names)
    if event_name != "pull_request":
        run_external_integration = bool(sorted_external_integration_tools)
        return ChangeScope(
            run_external_integration=run_external_integration,
            run_full_core=True,
            affected_tools=sorted_tools,
            external_integration_tools=sorted_external_integration_tools,
        )

    run_external_integration = False
    run_full_core = False
    affected_tools: set[str] = set()

    for path in changed_files:
        path = path.strip()
        if not path:
            continue

        if path in _EXTERNAL_INTEGRATION_GLOBAL_FILES:
            run_external_integration = True

        if path in _FULL_CORE_EXACT_FILES:
            run_full_core = True

        if path.startswith("src/dnadesign/"):
            parts = Path(path).parts
            if len(parts) < 3:
                run_full_core = True
                run_external_integration = True
                continue
            tool_name = parts[2]
            if tool_name in tool_names:
                affected_tools.add(tool_name)
                if tool_name in external_integration_tool_names:
                    run_external_integration = True
            else:
                run_full_core = True
                run_external_integration = True

    if run_full_core:
        affected_tools = set(sorted_tools)

    external_integration_tools = sorted(tool for tool in affected_tools if tool in external_integration_tool_names)
    if run_external_integration and not external_integration_tools:
        run_external_integration = False

    return ChangeScope(
        run_external_integration=run_external_integration,
        run_full_core=run_full_core,
        affected_tools=sorted(affected_tools),
        external_integration_tools=external_integration_tools,
    )


def _load_baseline_tools(path: Path) -> set[str]:
    baseline = load_baseline(path)
    return set(baseline)


def _load_changed_files(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Changed-files input is missing: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute CI lane scope from changed files.")
    parser.add_argument("--event-name", required=True, help="GitHub event name (e.g. pull_request, push).")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--changed-files-file", type=Path, default=None)
    parser.add_argument("--github-output", type=Path, default=None)
    return parser


def _emit_outputs(scope: ChangeScope, github_output: Path | None) -> None:
    outputs = {
        "run_external_integration": str(scope.run_external_integration).lower(),
        "run_full_core": str(scope.run_full_core).lower(),
        "run_coverage_gate": str(scope.run_coverage_gate).lower(),
        "affected_tools_csv": scope.affected_tools_csv,
        "external_integration_tools_csv": scope.external_integration_tools_csv,
    }

    for key, value in outputs.items():
        print(f"{key}={value}")

    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        baseline_tools = _load_baseline_tools(args.baseline_json)
        repo_tools = discover_repo_tools(repo_root=args.repo_root)
        validate_tool_baseline(baseline_tools=baseline_tools, repo_tools=repo_tools)
        external_integration_tool_names = discover_external_integration_tools(
            repo_root=args.repo_root,
            tool_names=repo_tools,
        )
        changed_files = _load_changed_files(args.changed_files_file)
        if args.event_name == "pull_request" and not changed_files:
            raise ValueError("No changed files detected for pull_request event; refusing to infer empty CI scope.")
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    scope = determine_scope(
        event_name=args.event_name,
        changed_files=changed_files,
        tool_names=repo_tools,
        external_integration_tool_names=external_integration_tool_names,
    )
    _emit_outputs(scope, args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
