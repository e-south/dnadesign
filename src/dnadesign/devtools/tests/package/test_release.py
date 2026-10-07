"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_release.py

Reject release evidence from other commits, branches or workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest


def _run(**overrides: object) -> dict:
    return {
        "id": 1,
        "head_sha": "abc123",
        "event": "push",
        "head_branch": "main",
        "path": ".github/workflows/ci.yaml",
        "head_repository": {"full_name": "e-south/dnadesign"},
        "status": "completed",
        "conclusion": "success",
        **overrides,
    }


def _check(runs: list[dict]) -> dict:
    from dnadesign.devtools.package.release import require_successful_ci

    return require_successful_ci({"workflow_runs": runs}, "abc123")


def test_exact_main_success_qualifies() -> None:
    assert _check([_run()])["id"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"head_sha": "different"},
        {"event": "pull_request"},
        {"head_branch": "feature"},
        {"path": ".github/workflows/unrelated.yml"},
        {"head_repository": {"full_name": "another/dense-arrays"}},
    ],
)
def test_other_run_cannot_qualify(changes: dict) -> None:
    with pytest.raises(ValueError, match="No canonical"):
        _check([_run(**changes)])


@pytest.mark.parametrize("changes", [{"conclusion": "failure"}, {"status": "in_progress"}])
def test_earlier_success_cannot_mask_latest_failure(changes: dict) -> None:
    with pytest.raises(ValueError, match="latest"):
        _check([_run(), _run(id=2, **changes)])
