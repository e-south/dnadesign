"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/coverage_summary.py

Builds per-tool coverage summary payloads for quality score inputs in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tool_coverage import (
    iter_tool_file_summaries,
    load_baseline,
    load_json,
    summarize_tool_coverage,
    validate_coverage_payload,
)


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


def _overall_coverage_percent(*, coverage_data: dict, selected_tools: set[str]) -> float:
    covered_total = 0
    statements_total = 0
    for _tool_name, covered_lines, num_statements in iter_tool_file_summaries(coverage_data, tool_names=selected_tools):
        covered_total += covered_lines
        statements_total += num_statements

    if statements_total == 0:
        return 0.0
    return round((covered_total / statements_total) * 100.0, 2)


def build_coverage_summary_payload(
    *, coverage_data: dict, baseline: dict[str, float], selected_tools: set[str]
) -> dict[str, object]:
    if not selected_tools:
        raise ValueError("selected_tools must include at least one tool.")

    unknown = sorted(selected_tools - set(baseline))
    if unknown:
        unknown_csv = ",".join(unknown)
        raise ValueError(f"selected_tools includes unknown tools: {unknown_csv}")

    actual_by_tool = summarize_tool_coverage(coverage_data=coverage_data, tool_names=selected_tools)
    tools_payload: dict[str, dict[str, object]] = {}
    gate_passed_tools = 0
    for tool_name in sorted(selected_tools):
        actual = round(float(actual_by_tool[tool_name]), 2)
        baseline_value = round(float(baseline[tool_name]), 2)
        gate = "pass" if actual >= baseline_value else "fail"
        if gate == "pass":
            gate_passed_tools += 1
        tools_payload[tool_name] = {
            "actual": actual,
            "baseline": baseline_value,
            "gate": gate,
        }

    return {
        "overall_core": _overall_coverage_percent(coverage_data=coverage_data, selected_tools=selected_tools),
        "gate_passed_tools": gate_passed_tools,
        "gate_total_tools": len(selected_tools),
        "tools": tools_payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-tool coverage summary payload for CI quality scoring.")
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--only-tools", type=str, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        coverage_data = load_json(args.coverage_json)
        validate_coverage_payload(coverage_data, source=str(args.coverage_json))
        baseline = load_baseline(args.baseline_json)
        selected_tools = _resolve_selected_tools(baseline=baseline, only_tools=args.only_tools)
        payload = build_coverage_summary_payload(
            coverage_data=coverage_data,
            baseline=baseline,
            selected_tools=selected_tools,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Coverage summary written to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
