"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/ci_test_targets.py

Resolves CI pytest target directories from an affected tool list.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path


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


def resolve_test_targets(*, repo_root: Path, tool_names: list[str]) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected dnadesign source root at {src_root}")

    targets: list[str] = []
    for tool_name in tool_names:
        tool_root = src_root / tool_name
        if not tool_root.is_dir():
            raise ValueError(f"Unknown tool in affected set: {tool_name}")
        test_dir = tool_root / "tests"
        if test_dir.is_dir():
            targets.append(str(test_dir))

    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve affected tool test directories for CI pytest invocations.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--affected-tools-csv", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        tool_names = parse_tools_csv(args.affected_tools_csv)
        targets = resolve_test_targets(repo_root=args.repo_root, tool_names=tool_names)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    for target in targets:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
