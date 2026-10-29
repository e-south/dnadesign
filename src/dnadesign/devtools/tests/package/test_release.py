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
        "run_attempt": 1,
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


def _capabilities() -> dict:
    return {
        "schema_version": 1,
        "revision": "abc123",
        "run_id": 1,
        "run_attempt": 1,
        "capabilities": {
            "video_output": {"status": "passed", "tests": 1, "executed": 1, "skipped": 0, "failed": 0},
            "gpu_inference": {"status": "not_run"},
        },
    }


def test_release_accepts_video_execution_with_gpu_deferred() -> None:
    from dnadesign.devtools.package.release import require_capabilities

    require_capabilities(_capabilities(), revision="abc123", run=_run())


@pytest.mark.parametrize("status", ["not_run", "failed", "unknown"])
def test_release_rejects_unqualified_video(status: str) -> None:
    from dnadesign.devtools.package.release import require_capabilities

    evidence = _capabilities()
    evidence["capabilities"]["video_output"]["status"] = status
    with pytest.raises(ValueError, match="video_output"):
        require_capabilities(evidence, revision="abc123", run=_run())


@pytest.mark.parametrize(
    ("key", "value"),
    [("revision", "other"), ("run_id", 9), ("run_attempt", 2), ("schema_version", 2), ("capabilities", {})],
)
def test_release_rejects_missing_or_mismatched_capability_evidence(key: str, value: object) -> None:
    from dnadesign.devtools.package.release import require_capabilities

    evidence = _capabilities()
    evidence[key] = value
    with pytest.raises(ValueError):
        require_capabilities(evidence, revision="abc123", run=_run())


@pytest.mark.parametrize(
    ("key", "value"), [("tests", 0), ("executed", 0), ("skipped", 1), ("failed", 1), ("tests", True)]
)
def test_passed_label_cannot_hide_missing_or_failed_execution(key: str, value: object) -> None:
    from dnadesign.devtools.package.release import require_capabilities

    evidence = _capabilities()
    evidence["capabilities"]["video_output"][key] = value
    with pytest.raises(ValueError, match="video_output"):
        require_capabilities(evidence, revision="abc123", run=_run())


@pytest.mark.parametrize("valid_banner", [True, False])
def test_workflow_checks_built_readme_banner(tmp_path, monkeypatch, valid_banner):
    """A branch-relative README is valid; the published metadata must be version-bound."""
    import io
    import urllib.request
    from pathlib import Path
    from zipfile import ZipFile

    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load((root / ".github/workflows/release.yaml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    step = next(item for item in steps if item.get("name") == "Verify immutable package-page banner")
    program = step["run"].split("<<'PYBANNER'\n", 1)[1].rsplit("PYBANNER", 1)[0]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "9.0.0"\n')
    (tmp_path / "README.md").write_text("![toolkit](assets/dnadesign-banner.png)\n")
    (tmp_path / "assets").mkdir()
    banner = b"\x89PNG\r\n\x1a\nexample"
    (tmp_path / "assets/dnadesign-banner.png").write_bytes(banner)
    (tmp_path / "dist").mkdir()
    url = "https://raw.githubusercontent.com/e-south/dnadesign/v9.0.0/assets/dnadesign-banner.png"
    with ZipFile(tmp_path / "dist/example.whl", "w") as archive:
        archive.writestr("example.dist-info/METADATA", f"Metadata-Version: 2.4\nVersion: 9.0.0\n\n![toolkit]({url})\n")

    def fetch(requested, *, timeout):
        assert requested == url
        return io.BytesIO(banner if valid_banner else b"wrong-content")

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    if valid_banner:
        exec(compile(program, "release-banner-check", "exec"), {})
    else:
        with pytest.raises(AssertionError, match="Published banner differs"):
            exec(compile(program, "release-banner-check", "exec"), {})
