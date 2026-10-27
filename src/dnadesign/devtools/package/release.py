"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/package/release.py

Require successful CI on the exact main commit before publishing a release.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_successful_ci(payload: dict, revision: str) -> dict:
    """Return the latest main-push CI run, rejecting missing or failed evidence.

    Raises
    ------
    ValueError
        If the exact revision lacks a successful canonical CI run.

    Returns
    -------
    dict
        The matching GitHub Actions run retained in release provenance.
    """
    runs = [
        run
        for run in payload["workflow_runs"]
        if run.get("head_sha") == revision
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("path") == ".github/workflows/ci.yaml"
        and (run.get("head_repository") or {}).get("full_name") == "e-south/dnadesign"
    ]
    if not runs:
        message = "No canonical main-push CI run for the release commit"
        raise ValueError(message)
    latest = max(runs, key=lambda run: int(run["id"]))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        message = "The latest main-push CI run has not completed successfully"
        raise ValueError(message)
    return latest


def require_capabilities(evidence: dict, *, revision: str, run: dict) -> None:
    """Require video execution in the same CI attempt; GPU qualification remains deferred."""
    expected = {"schema_version": 1, "revision": revision, "run_id": run["id"], "run_attempt": run["run_attempt"]}
    if any(type(evidence.get(key)) is not type(value) or evidence[key] != value for key, value in expected.items()):
        raise ValueError("Capability evidence does not match the release revision and CI attempt")
    capabilities = evidence.get("capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != {"video_output", "gpu_inference"}:
        raise ValueError("Capability evidence must include video_output and gpu_inference")
    for name, capability in capabilities.items():
        if not isinstance(capability, dict) or capability.get("status") not in {"passed", "failed", "not_run"}:
            raise ValueError(f"Invalid capability state: {name}")
        if capability["status"] == "failed":
            raise ValueError(f"Capability failed: {name}")
    if capabilities["video_output"]["status"] != "passed":
        raise ValueError("Release requires executed video_output validation")
    video = capabilities["video_output"]
    if (
        any(type(video.get(key)) is not int for key in ("tests", "executed", "skipped", "failed"))
        or video["tests"] < 1
        or video["executed"] != video["tests"]
        or video["skipped"] != 0
        or video["failed"] != 0
    ):
        raise ValueError("video_output evidence must contain successful, non-skipped test execution")


def main() -> None:
    """Validate downloaded CI evidence and print its qualifying run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    evidence_mode = parser.add_mutually_exclusive_group(required=True)
    evidence_mode.add_argument(
        "--select-ci-only", action="store_true", help="Select the CI run before downloading evidence"
    )
    evidence_mode.add_argument("--capabilities", type=Path, help="Qualify the downloaded capability evidence")
    args = parser.parse_args()
    run = require_successful_ci(json.loads(args.checks.read_text()), args.revision)
    if args.capabilities is not None:
        evidence = json.loads(args.capabilities.read_text())
        require_capabilities(evidence, revision=args.revision, run=run)
        print(json.dumps({"ci": run, "capabilities": evidence}, indent=2))
    else:
        print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
