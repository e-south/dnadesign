"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/security/test_tracked_text_privacy.py

Tests for tracked text and operator-config privacy checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from dnadesign.devtools.security.tracked_text_privacy import find_privacy_issues, main


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _track(root: Path, relative_path: str, content: bytes) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    subprocess.run(["git", "-C", str(root), "add", relative_path], check=True)


def test_find_privacy_issues_rejects_known_personal_tokens(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(
        tmp_path,
        "docs/bu-scc/operator.md",
        "".join(
            (
                "workstation=/Users/" + "Shockwing/project\n",
                "email=ericjohn" + "south@gmail.com\n",
                "remote=esouth@" + "scc1.bu.edu\n",
                "project=/project/dunlop/" + "esouth/tool\n",
                "projectnb=/projectnb/dunlop/" + "esouth/tool\n",
            )
        ).encode(),
    )

    issues = find_privacy_issues(tmp_path)

    assert {issue.token_name for issue in issues} == {
        "personal_gmail",
        "personal_macos_home",
        "personal_scc_login",
        "personal_scc_project_root",
        "personal_scc_projectnb_root",
    }
    assert {issue.path.as_posix() for issue in issues} == {"docs/bu-scc/operator.md"}


def test_find_privacy_issues_rejects_tracked_active_remotes_config(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(
        tmp_path,
        "src/dnadesign/usr/remotes.yaml",
        b"remotes:\n  cluster:\n    user: example-user\n",
    )

    issues = find_privacy_issues(tmp_path)

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [
        ("src/dnadesign/usr/remotes.yaml", "tracked_active_remotes_config")
    ]


def test_find_privacy_issues_covers_study_and_tool_operator_routes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "/projectnb/dunlop/" + "esouth/tool"
    _track(
        tmp_path,
        "docs/studies/demo/operations/runtime/command-groups/pipeline.yaml",
        f"command: {token}\n".encode(),
    )
    _track(
        tmp_path,
        "src/dnadesign/infer/docs/operations/runbook.md",
        f"path: {token}\n".encode(),
    )
    _track(
        tmp_path,
        "src/dnadesign/ops/runbooks/presets/demo.yaml",
        f"path: {token}\n".encode(),
    )

    assert [(issue.path.as_posix(), issue.token_name) for issue in find_privacy_issues(tmp_path)] == [
        (
            "docs/studies/demo/operations/runtime/command-groups/pipeline.yaml",
            "personal_scc_projectnb_root",
        ),
        ("src/dnadesign/infer/docs/operations/runbook.md", "personal_scc_projectnb_root"),
        ("src/dnadesign/ops/runbooks/presets/demo.yaml", "personal_scc_projectnb_root"),
    ]


def test_find_privacy_issues_allows_examples_and_skips_binary_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(
        tmp_path,
        "src/dnadesign/usr/remotes.example.yaml",
        b"remotes:\n  cluster:\n    user: example-user\n    local_repo_root: /Users/example/dnadesign\n",
    )
    _track(tmp_path, "docs/bu-scc/example.md", b"Use /Users/example/dnadesign for examples.\n")
    _track(tmp_path, "docs/bu-scc/scientific.bin", b"\x00/Users/" + b"Shockwing\x00")

    assert find_privacy_issues(tmp_path) == ()


def test_find_privacy_issues_limits_legacy_provenance_allowance_to_exact_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "/Users/" + "Shockwing"
    allowed_path = (
        "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "reader_spop_msd_structure_panel_v1/manifest.yaml"
    )
    future_path = (
        "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "reader_spop_msd_structure_panel_v1/reports/future.yaml"
    )
    _track(tmp_path, allowed_path, f"path: {token}\n".encode())
    _track(tmp_path, future_path, f"path: {token}\n".encode())
    _track(tmp_path, "docs/studies/demo/outputs/status.yaml", f"path: {token}\n".encode())
    _track(tmp_path, "src/dnadesign/infer/docs/dev/journal.md", f"path: {token}\n".encode())

    issues = find_privacy_issues(tmp_path, today=date(2026, 7, 30))

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [(future_path, "personal_macos_home")]


def test_find_privacy_issues_does_not_allow_other_tokens_at_legacy_path(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    allowed_path = (
        "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "reader_spop_msd_structure_panel_v1/manifest.yaml"
    )
    content = "".join(
        (
            "path=/Users/" + "Shockwing/project\n",
            "email=ericjohn" + "south@gmail.com\n",
        )
    )
    _track(tmp_path, allowed_path, content.encode())

    issues = find_privacy_issues(tmp_path, today=date(2026, 7, 30))

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [(allowed_path, "personal_gmail")]


def test_find_privacy_issues_expires_legacy_provenance_allowance(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    allowed_path = (
        "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "reader_spop_msd_structure_panel_v1/reports/discrepancies.yaml"
    )
    _track(tmp_path, allowed_path, b"status: regenerated_without_private_paths\n")

    issues = find_privacy_issues(tmp_path, today=date(2026, 10, 1))

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [
        (allowed_path, "expired_legacy_path_allowance")
    ]


def test_main_fails_closed_when_a_tracked_token_is_present(tmp_path: Path, capsys) -> None:
    _init_repo(tmp_path)
    _track(tmp_path, "docs/bu-scc/operator.md", b"remote=esouth@" + b"scc1.bu.edu\n")

    assert main(["--repo-root", str(tmp_path)]) == 1
    assert "docs/bu-scc/operator.md:1: personal_scc_login" in capsys.readouterr().err


def test_study_records_never_reference_a_tracked_active_remote_profile() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
    records = sorted((repo_root / "docs" / "studies").glob("*/record/datasets.yaml"))

    assert records
    for record in records:
        text = record.read_text(encoding="utf-8")
        assert "usr_remotes_path: src/dnadesign/usr/remotes.yaml" not in text

    stress_record = repo_root / "docs/studies/stress_ethanol_cipro_growth/record/datasets.yaml"
    assert "usr_remotes_path: n/a" in stress_record.read_text(encoding="utf-8")
    stress_record_readme = repo_root / "docs/studies/stress_ethanol_cipro_growth/record/README.md"
    readme_text = stress_record_readme.read_text(encoding="utf-8")
    assert "--remotes-config" in readme_text
    assert "USR_REMOTES_PATH" in readme_text
