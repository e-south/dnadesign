"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/ci/test_targets.py

Resolves CI pytest target directories from an affected tool list.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

_STUDIES_TOOL_NAME = "studies"
_ADDITIONAL_TOOL_TEST_DIRS = {
    "cluster": ("src/cli/tests",),
}


def parse_tools_csv(value: str) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()

    for raw_name in value.split(","):
        tool_name = raw_name.strip()
        if not tool_name or tool_name in seen:
            continue
        seen.add(tool_name)
        tools.append(tool_name)

    if not tools:
        raise ValueError("--affected-tools-csv must include at least one tool name.")
    return tools


def _load_changed_files(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Changed-files input is missing: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _study_unit_test_dirs(*, studies_root: Path, changed_files: list[str]) -> list[Path]:
    units_root = studies_root / "units"
    if not units_root.is_dir():
        return []

    changed_study_ids: set[str] = set()
    shared_studies_change = not changed_files
    for raw_path in changed_files:
        parts = Path(raw_path).parts
        if parts[:2] == ("docs", "studies"):
            if len(parts) >= 3 and (units_root / parts[2]).is_dir():
                changed_study_ids.add(parts[2])
            else:
                shared_studies_change = True
            continue
        if parts[:3] != ("src", "dnadesign", "studies"):
            continue
        if len(parts) >= 5 and parts[3] == "units":
            changed_study_ids.add(parts[4])
            continue
        shared_studies_change = True

    if shared_studies_change:
        unit_roots = sorted(path for path in units_root.iterdir() if path.is_dir())
    else:
        unit_roots = [units_root / study_id for study_id in sorted(changed_study_ids)]

    return [unit_root / "tests" for unit_root in unit_roots if (unit_root / "tests").is_dir()]


def _append_existing_target(targets: list[str], target: Path) -> None:
    if target.is_dir():
        target_value = str(target)
        if target_value not in targets:
            targets.append(target_value)


def resolve_test_targets(
    *, repo_root: Path, tool_names: list[str], changed_files: list[str] | None = None
) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")

    changed_files = changed_files or []
    targets: list[str] = []
    for tool_name in tool_names:
        tool_root = src_root / tool_name
        if not tool_root.is_dir():
            raise ValueError(f"Unknown tool in affected set: {tool_name}")
        _append_existing_target(targets, tool_root / "tests")
        for relative_test_dir in _ADDITIONAL_TOOL_TEST_DIRS.get(tool_name, ()):
            _append_existing_target(targets, tool_root / relative_test_dir)
        if tool_name == _STUDIES_TOOL_NAME:
            for test_dir in _study_unit_test_dirs(studies_root=tool_root, changed_files=changed_files):
                _append_existing_target(targets, test_dir)

    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve affected tool test directories for CI pytest invocations.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--affected-tools-csv", required=True)
    parser.add_argument("--changed-files-file", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        tool_names = parse_tools_csv(args.affected_tools_csv)
        changed_files = _load_changed_files(args.changed_files_file)
        targets = resolve_test_targets(repo_root=args.repo_root, tool_names=tool_names, changed_files=changed_files)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    for target in targets:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
