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


def test_find_privacy_issues_covers_operator_routes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "/projectnb/dunlop/" + "esouth/tool"
    _track(
        tmp_path,
        "docs/operations/runbook.yaml",
        f"command: {token}\n".encode(),
    )
    _track(
        tmp_path,
        "src/dnadesign/infer/docs/operations/runbook.md",
        f"path: {token}\n".encode(),
    )
    _track(
        tmp_path,
        "docs/templates/runbook.yaml",
        f"path: {token}\n".encode(),
    )

    assert [(issue.path.as_posix(), issue.token_name) for issue in find_privacy_issues(tmp_path)] == [
        (
            "docs/operations/runbook.yaml",
            "personal_scc_projectnb_root",
        ),
        ("docs/templates/runbook.yaml", "personal_scc_projectnb_root"),
        ("src/dnadesign/infer/docs/operations/runbook.md", "personal_scc_projectnb_root"),
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


def test_find_privacy_issues_scans_non_doc_fixtures_for_home_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(
        tmp_path,
        "src/package/fixtures/provenance.json",
        b'{"source": "/Users/' + b'private-operator/project/output.json"}\n',
    )

    issues = find_privacy_issues(tmp_path)

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [
        ("src/package/fixtures/provenance.json", "absolute_home_path")
    ]


def test_find_privacy_issues_rejects_private_study_identifiers_in_content_and_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    private_identifier = "stress_" + "ethanol_cipro_growth"
    _track(tmp_path, "docs/reference.md", f"study_id: {private_identifier}\n".encode())
    _track(tmp_path, f"docs/{private_identifier}/README.md", b"private study\n")

    issues = find_privacy_issues(tmp_path)

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [
        ("docs/reference.md", "private_study_identifier"),
        (f"docs/{private_identifier}/README.md", "private_study_identifier"),
    ]


def test_find_privacy_issues_rejects_personal_paths_in_provenance(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "/Users/" + "Shockwing"
    provenance_path = "docs/provenance/structure/manifest.yaml"
    _track(tmp_path, provenance_path, f"path: {token}\n".encode())
    issues = find_privacy_issues(tmp_path)

    assert [(issue.path.as_posix(), issue.token_name) for issue in issues] == [(provenance_path, "personal_macos_home")]


def test_main_fails_closed_when_a_tracked_token_is_present(tmp_path: Path, capsys) -> None:
    _init_repo(tmp_path)
    _track(tmp_path, "docs/bu-scc/operator.md", b"remote=esouth@" + b"scc1.bu.edu\n")

    assert main(["--repo-root", str(tmp_path)]) == 1
    assert "docs/bu-scc/operator.md:1: personal_scc_login" in capsys.readouterr().err


def test_public_tree_contains_no_tracked_study_implementations() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    tracked_paths = tuple(path.decode() for path in result.stdout.split(b"\0") if path)

    forbidden_prefixes = ("docs/studies/", "src/dnadesign/studies/")
    assert not [path for path in tracked_paths if path.startswith(forbidden_prefixes)]
