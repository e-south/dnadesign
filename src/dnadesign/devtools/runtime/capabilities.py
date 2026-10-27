"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/runtime/capabilities.py

Record capability execution from CI test reports without probing local hardware.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def collect_capabilities(report: Path, *, revision: str, run_id: int, run_attempt: int) -> dict:
    """Bind representative video-output evidence and deferred GPU status to a CI attempt."""
    cases = []
    if report.is_file():
        root = ET.parse(report).getroot()
        if root.tag not in {"testsuite", "testsuites"}:
            raise ValueError(f"Unsupported JUnit report root: {root.tag}")
        cases = [
            case
            for case in root.iter("testcase")
            if (case.get("classname") or "").endswith(".baserender.tests.test_outputs_explicit")
            and case.get("name") == "test_video_output_does_not_produce_images"
        ]
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    status = "failed" if failed else "passed" if cases and not skipped else "not_run"
    return {
        "schema_version": 1,
        "revision": revision,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "capabilities": {
            "video_output": {
                "status": status,
                "tests": len(cases),
                "executed": len(cases) - skipped,
                "skipped": skipped,
                "failed": failed,
            },
            "gpu_inference": {"status": "not_run", "reason": "GPU qualification is deferred."},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = collect_capabilities(
        args.junit_xml, revision=args.revision, run_id=args.run_id, run_attempt=args.run_attempt
    )
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    for name, capability in evidence["capabilities"].items():
        print(f"{name}: {capability['status']}")


if __name__ == "__main__":
    main()
