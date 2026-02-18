"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/quality_score.py

Builds CI-backed quality score inputs from coverage summary, baselines, and lane results.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .tool_coverage import load_baseline, load_json

_ALLOWED_LANE_RESULTS = {"success", "failure", "cancelled", "skipped"}
_ALLOWED_TOOL_GATES = {"pass", "fail"}


def _validate_lane_result(*, value: str, lane_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _ALLOWED_LANE_RESULTS:
        allowed = ", ".join(sorted(_ALLOWED_LANE_RESULTS))
        raise ValueError(f"{lane_name} lane result must be one of: {allowed}")
    return normalized


def _validate_coverage_summary(coverage_summary: dict) -> None:
    for key in ("overall_core", "gate_passed_tools", "gate_total_tools", "tools"):
        if key not in coverage_summary:
            raise ValueError(f"Coverage summary missing required field: {key}")
    if not isinstance(coverage_summary["tools"], dict):
        raise ValueError("Coverage summary field 'tools' must be an object.")


def _read_numeric_field(*, row: dict, tool_name: str, field_name: str) -> float:
    if field_name not in row:
        raise ValueError(f"Coverage summary tool entry for '{tool_name}' is missing required field '{field_name}'.")
    value = row[field_name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Coverage summary tool field '{field_name}' must be numeric for '{tool_name}'.")
    return float(value)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_quality_score_inputs(
    *,
    coverage_summary: dict,
    baseline: dict[str, float],
    core_lane_result: str,
    external_integration_lane_result: str,
    publish_lane_result: str,
) -> dict[str, object]:
    _validate_coverage_summary(coverage_summary)
    core = _validate_lane_result(value=core_lane_result, lane_name="core")
    external_integration = _validate_lane_result(
        value=external_integration_lane_result,
        lane_name="external integration",
    )
    publish = _validate_lane_result(value=publish_lane_result, lane_name="publish")

    summary_tools = set(coverage_summary["tools"])
    baseline_tools = set(baseline)
    if summary_tools != baseline_tools:
        missing = sorted(baseline_tools - summary_tools)
        extra = sorted(summary_tools - baseline_tools)
        details: list[str] = []
        if missing:
            details.append(f"missing tools in coverage summary: {', '.join(missing)}")
        if extra:
            details.append(f"unknown tools in coverage summary: {', '.join(extra)}")
        raise ValueError(f"Coverage summary/baseline tool mismatch: {'; '.join(details)}")

    tools_payload: list[dict[str, object]] = []
    for tool_name in sorted(baseline):
        summary_row = coverage_summary["tools"].get(tool_name)
        if not isinstance(summary_row, dict):
            raise ValueError(f"Coverage summary tool entry must be an object for '{tool_name}'.")

        actual = _read_numeric_field(row=summary_row, tool_name=tool_name, field_name="actual")
        summary_baseline = _read_numeric_field(row=summary_row, tool_name=tool_name, field_name="baseline")
        summary_gate = summary_row.get("gate")
        if not isinstance(summary_gate, str):
            raise ValueError(f"Coverage summary tool field 'gate' must be a string for '{tool_name}'.")
        normalized_summary_gate = summary_gate.strip().lower()
        if normalized_summary_gate not in _ALLOWED_TOOL_GATES:
            allowed = ", ".join(sorted(_ALLOWED_TOOL_GATES))
            raise ValueError(f"Coverage summary tool field 'gate' must be one of: {allowed} (tool '{tool_name}').")

        baseline_value = float(baseline[tool_name])
        gate = "pass" if actual >= baseline_value else "fail"
        if round(summary_baseline, 2) != round(baseline_value, 2):
            raise ValueError(
                f"Coverage summary baseline mismatch for '{tool_name}': "
                f"{summary_baseline:.2f} (summary) vs {baseline_value:.2f} (baseline file)."
            )
        if normalized_summary_gate != gate:
            raise ValueError(
                f"Coverage summary gate mismatch for '{tool_name}': "
                f"{normalized_summary_gate} (summary) vs {gate} (computed)."
            )

        tools_payload.append(
            {
                "tool": tool_name,
                "actual": round(actual, 2),
                "baseline": round(baseline_value, 2),
                "gate": gate,
                "meets_baseline": gate == "pass",
            }
        )

    return {
        "generated_at_utc": _utc_now_iso(),
        "lanes": {
            "core": core,
            "external_integration": external_integration,
            "publish": publish,
        },
        "coverage": {
            "overall_core": float(coverage_summary["overall_core"]),
            "gate_passed_tools": int(coverage_summary["gate_passed_tools"]),
            "gate_total_tools": int(coverage_summary["gate_total_tools"]),
        },
        "tools": tools_payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CI-backed quality score inputs.")
    parser.add_argument("--coverage-summary-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--core-lane-result", required=True)
    parser.add_argument("--external-integration-lane-result", required=True)
    parser.add_argument("--publish-lane-result", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        coverage_summary = load_json(args.coverage_summary_json)
        baseline = load_baseline(args.baseline_json)
        payload = build_quality_score_inputs(
            coverage_summary=coverage_summary,
            baseline=baseline,
            core_lane_result=args.core_lane_result,
            external_integration_lane_result=args.external_integration_lane_result,
            publish_lane_result=args.publish_lane_result,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Quality score inputs written to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
