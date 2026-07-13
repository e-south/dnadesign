"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_test_targets.py

Tests for CI helper that resolves affected tool test directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.ci.test_targets import main, resolve_test_targets


def test_resolve_test_targets_returns_existing_test_dirs(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    dense_tests = tmp_path / "src" / "dnadesign" / "densegen" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    dense_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["usr", "densegen"])

    assert targets == [str(usr_tests), str(dense_tests)]


def test_resolve_test_targets_skips_tools_without_tests(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "usr").mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["usr"])

    assert targets == []


def test_resolve_test_targets_includes_cluster_owned_cli_tests(tmp_path: Path) -> None:
    cluster_tests = tmp_path / "src" / "dnadesign" / "cluster" / "tests"
    cluster_cli_tests = tmp_path / "src" / "dnadesign" / "cluster" / "src" / "cli" / "tests"
    cluster_tests.mkdir(parents=True, exist_ok=True)
    cluster_cli_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["cluster"])

    assert targets == [str(cluster_tests), str(cluster_cli_tests)]


def test_resolve_test_targets_includes_changed_study_unit_tests(tmp_path: Path) -> None:
    shared_tests = tmp_path / "src" / "dnadesign" / "studies" / "tests"
    stress_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "stress_ethanol_cipro_growth" / "tests"
    retron_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design" / "tests"
    shared_tests.mkdir(parents=True, exist_ok=True)
    stress_tests.mkdir(parents=True, exist_ok=True)
    retron_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(
        repo_root=tmp_path,
        tool_names=["studies"],
        changed_files=[
            "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/candidate_table.py",
        ],
    )

    assert targets == [str(shared_tests), str(stress_tests)]


def test_resolve_test_targets_includes_all_study_units_for_shared_studies_change(tmp_path: Path) -> None:
    shared_tests = tmp_path / "src" / "dnadesign" / "studies" / "tests"
    retron_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design" / "tests"
    stress_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "stress_ethanol_cipro_growth" / "tests"
    shared_tests.mkdir(parents=True, exist_ok=True)
    retron_tests.mkdir(parents=True, exist_ok=True)
    stress_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(
        repo_root=tmp_path,
        tool_names=["studies"],
        changed_files=["src/dnadesign/studies/README.md"],
    )

    assert targets == [str(shared_tests), str(retron_tests), str(stress_tests)]


def test_resolve_test_targets_includes_all_study_units_when_changed_file_context_is_missing(tmp_path: Path) -> None:
    shared_tests = tmp_path / "src" / "dnadesign" / "studies" / "tests"
    stress_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "stress_ethanol_cipro_growth" / "tests"
    shared_tests.mkdir(parents=True, exist_ok=True)
    stress_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["studies"])

    assert targets == [str(shared_tests), str(stress_tests)]


def test_main_fails_for_unknown_tool(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "usr" / "tests").mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "usr,ghost",
        ]
    )

    assert rc == 1


def test_main_fails_for_empty_tool_list(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "usr" / "tests").mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "",
        ]
    )

    assert rc == 1


def test_main_prints_one_target_per_line(tmp_path: Path, capsys) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "usr",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"{usr_tests}\n"


def test_main_uses_changed_file_context_for_study_unit_targets(tmp_path: Path, capsys) -> None:
    shared_tests = tmp_path / "src" / "dnadesign" / "studies" / "tests"
    stress_tests = tmp_path / "src" / "dnadesign" / "studies" / "units" / "stress_ethanol_cipro_growth" / "tests"
    shared_tests.mkdir(parents=True, exist_ok=True)
    stress_tests.mkdir(parents=True, exist_ok=True)
    changed_files = tmp_path / "changed-files.txt"
    changed_files.write_text(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/snapshot.py\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "studies",
            "--changed-files-file",
            str(changed_files),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"{shared_tests}\n{stress_tests}\n"
