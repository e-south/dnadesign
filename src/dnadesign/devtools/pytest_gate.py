"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/pytest_gate.py

Validates pytest JUnit reports to ensure CI lanes execute real non-skipped tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JUnitSummary:
    tests: int
    skipped: int
    non_skipped: int
    non_skipped_by_tool: dict[str, int]


def _read_non_negative_int_attr(*, suite: ET.Element, attr_name: str, report_path: Path) -> int:
    raw = suite.get(attr_name)
    if raw is None:
        raise ValueError(f"{report_path}: missing required JUnit testsuite attribute '{attr_name}'.")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{report_path}: JUnit testsuite attribute '{attr_name}' must be an integer.") from exc
    if value < 0:
        raise ValueError(f"{report_path}: JUnit testsuite attribute '{attr_name}' must be >= 0.")
    return value


def _extract_tool_from_testcase(testcase: ET.Element) -> str | None:
    classname = (testcase.get("classname") or "").strip()
    if classname:
        parts = [part for part in classname.split(".") if part]
        if "dnadesign" in parts:
            idx = parts.index("dnadesign")
            if idx + 1 < len(parts):
                return parts[idx + 1]

    file_attr = (testcase.get("file") or "").strip().replace("\\", "/")
    if not file_attr:
        return None
    file_parts = [part for part in file_attr.split("/") if part]
    if "dnadesign" not in file_parts:
        return None
    idx = file_parts.index("dnadesign")
    if idx + 1 >= len(file_parts):
        return None
    return file_parts[idx + 1]


def _parse_required_tools_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def evaluate_junit_report(*, report_path: Path, required_tools: set[str] | None = None) -> JUnitSummary:
    if not report_path.exists():
        raise FileNotFoundError(f"JUnit report file is missing: {report_path}")

    root = ET.parse(report_path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
        if not suites:
            raise ValueError(f"{report_path}: JUnit report has no testsuite entries.")
    else:
        raise ValueError(f"{report_path}: unsupported JUnit root tag '{root.tag}'.")

    total_tests = 0
    total_skipped = 0
    non_skipped_by_tool: dict[str, int] = {}
    for suite in suites:
        total_tests += _read_non_negative_int_attr(suite=suite, attr_name="tests", report_path=report_path)
        total_skipped += _read_non_negative_int_attr(suite=suite, attr_name="skipped", report_path=report_path)
        for testcase in suite.findall("testcase"):
            if testcase.find("skipped") is not None:
                continue
            tool_name = _extract_tool_from_testcase(testcase)
            if tool_name is None:
                continue
            non_skipped_by_tool[tool_name] = non_skipped_by_tool.get(tool_name, 0) + 1

    if total_skipped > total_tests:
        raise ValueError(f"{report_path}: skipped count cannot exceed tests count ({total_skipped}>{total_tests}).")
    if total_tests == 0:
        raise ValueError(f"{report_path}: no tests were collected.")

    non_skipped = total_tests - total_skipped
    if non_skipped == 0:
        raise ValueError(f"{report_path}: all collected tests were skipped.")

    required = set() if required_tools is None else {item.strip() for item in required_tools if item.strip()}
    if required:
        missing = sorted(tool for tool in required if non_skipped_by_tool.get(tool, 0) == 0)
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(f"{report_path}: required tool(s) without executed non-skipped tests: {missing_csv}.")

    return JUnitSummary(
        tests=total_tests,
        skipped=total_skipped,
        non_skipped=non_skipped,
        non_skipped_by_tool=non_skipped_by_tool,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate pytest JUnit report has non-skipped executed tests.")
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--lane-name", default="test")
    parser.add_argument("--required-tools-csv", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = evaluate_junit_report(
            report_path=args.junit_xml,
            required_tools=_parse_required_tools_csv(args.required_tools_csv),
        )
    except (FileNotFoundError, ET.ParseError, ValueError) as exc:
        print(str(exc))
        return 1

    print(
        f"{args.lane_name} lane JUnit gate passed: "
        f"tests={summary.tests}, skipped={summary.skipped}, non_skipped={summary.non_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
