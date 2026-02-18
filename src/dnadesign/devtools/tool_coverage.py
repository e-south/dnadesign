"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tool_coverage.py

Aggregates coverage.py JSON output by dnadesign tool and enforces baseline floors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CoverageRegression:
    tool: str
    baseline: float
    actual: float


def _split_coverage_path(path_value: str) -> tuple[str, ...]:
    normalized = path_value.replace("\\", "/")
    return tuple(part for part in normalized.split("/") if part)


def _extract_tool_name(path_value: str) -> str | None:
    parts = _split_coverage_path(path_value)
    try:
        base_idx = parts.index("dnadesign")
    except ValueError:
        return None

    tool_idx = base_idx + 1
    if tool_idx >= len(parts):
        return None

    tool_name = parts[tool_idx]
    if "tests" in parts[tool_idx + 1 :]:
        return None
    return tool_name


def iter_tool_file_summaries(coverage_data: dict, *, tool_names: set[str]) -> Iterator[tuple[str, int, int]]:
    for file_path, file_data in coverage_data.get("files", {}).items():
        tool_name = _extract_tool_name(file_path)
        if tool_name is None or tool_name not in tool_names:
            continue

        summary = file_data.get("summary", {})
        covered_lines = int(summary.get("covered_lines", 0))
        num_statements = int(summary.get("num_statements", 0))
        if num_statements <= 0:
            continue

        yield tool_name, covered_lines, num_statements


def summarize_tool_coverage(coverage_data: dict, tool_names: set[str]) -> dict[str, float]:
    totals: dict[str, dict[str, int]] = {tool: {"covered": 0, "statements": 0} for tool in sorted(tool_names)}
    for tool_name, covered_lines, num_statements in iter_tool_file_summaries(coverage_data, tool_names=tool_names):
        totals[tool_name]["covered"] += covered_lines
        totals[tool_name]["statements"] += num_statements

    coverage_by_tool: dict[str, float] = {}
    for tool_name, counts in totals.items():
        statements = counts["statements"]
        covered = counts["covered"]
        if statements == 0:
            coverage_by_tool[tool_name] = 0.0
            continue
        coverage_by_tool[tool_name] = round((covered / statements) * 100.0, 2)

    return coverage_by_tool


def find_regressions(baseline: dict[str, float], actual: dict[str, float]) -> list[CoverageRegression]:
    regressions: list[CoverageRegression] = []
    for tool_name in sorted(baseline):
        baseline_value = float(baseline[tool_name])
        actual_value = float(actual.get(tool_name, 0.0))
        if actual_value < baseline_value:
            regressions.append(
                CoverageRegression(
                    tool=tool_name,
                    baseline=baseline_value,
                    actual=actual_value,
                )
            )
    return regressions


def load_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file not found: {path}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {path} must be an object.")
    return payload


def load_baseline(path: Path) -> dict[str, float]:
    raw = load_json(path)
    if not raw:
        raise ValueError("Coverage baseline JSON must include at least one tool.")

    normalized: dict[str, float] = {}
    for tool, value in raw.items():
        tool_name = str(tool).strip()
        if not tool_name:
            raise ValueError("Coverage baseline contains an empty tool name.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Coverage baseline for '{tool_name}' must be numeric.") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"Coverage baseline for '{tool_name}' must be finite.")
        if numeric_value < 0.0 or numeric_value > 100.0:
            raise ValueError(f"Coverage baseline for '{tool_name}' must be between 0 and 100.")
        normalized[tool_name] = numeric_value
    return normalized


def validate_coverage_payload(coverage_data: dict, *, source: str) -> None:
    files_payload = coverage_data.get("files")
    if not isinstance(files_payload, dict):
        raise ValueError(f"Coverage JSON must include a 'files' object: {source}")

    for file_path, file_payload in files_payload.items():
        if not isinstance(file_payload, dict):
            raise ValueError(f"Coverage file entry must be an object for {file_path}: {source}")

        summary = file_payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Coverage file summary must be an object for {file_path}: {source}")

        for key in ("covered_lines", "num_statements"):
            value = summary.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Coverage summary field '{key}' must be an integer for {file_path}: {source}")
            if value < 0:
                raise ValueError(f"Coverage summary field '{key}' must be non-negative for {file_path}: {source}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check dnadesign per-tool coverage baselines.")
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument(
        "--only-tools",
        type=str,
        default=None,
        help="Comma-separated tool names to scope baseline checks (default: all tools in baseline).",
    )
    return parser


def _resolve_selected_tools(*, baseline: dict[str, float], only_tools: str | None) -> set[str]:
    if only_tools is None:
        return set(baseline)

    selected = {item.strip() for item in only_tools.split(",") if item.strip()}
    if not selected:
        raise ValueError("--only-tools must include at least one tool name.")

    unknown = sorted(selected - set(baseline))
    if unknown:
        unknown_csv = ",".join(unknown)
        raise ValueError(f"--only-tools includes unknown tools: {unknown_csv}")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        baseline = load_baseline(args.baseline_json)
        selected_tools = _resolve_selected_tools(baseline=baseline, only_tools=args.only_tools)
        coverage_data = load_json(args.coverage_json)
        validate_coverage_payload(coverage_data, source=str(args.coverage_json))
    except ValueError as exc:
        print(str(exc))
        return 1

    baseline_selected = {tool_name: baseline[tool_name] for tool_name in sorted(selected_tools)}
    actual = summarize_tool_coverage(coverage_data=coverage_data, tool_names=set(baseline_selected))
    regressions = find_regressions(baseline=baseline_selected, actual=actual)

    for tool_name in sorted(actual):
        print(f"{tool_name}: {actual[tool_name]:.2f}% (baseline {baseline_selected[tool_name]:.2f}%)")

    if regressions:
        print("Coverage regressions detected:")
        for item in regressions:
            print(f"- {item.tool}: {item.actual:.2f}% < {item.baseline:.2f}%")
        return 1

    print("Coverage baselines satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
