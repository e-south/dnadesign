"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/quality/test_entropy.py

Tests for quality entropy reporting and stale-doc evidence checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

from dnadesign.devtools.quality.entropy import main


def test_module_help_runs_without_site_packages() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-S", "-m", "dnadesign.devtools.quality.entropy", "--help"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Generate quality entropy report." in result.stdout


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_minimal_quality_score(
    path: Path,
    *,
    evidence_path: str = "docs/README.md",
    last_verified: str | None = None,
) -> None:
    verified = dt.date.today().isoformat() if last_verified is None else last_verified
    path.write_text(
        "\n".join(
            [
                "# QUALITY SCORE",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {verified}",
                "",
                "## Quality scorecard",
                "| Area | Axis | Score (0-4) | Trend | Gate | Evidence | Owner | Last verified | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                f"| `usr` | domain | 3 | stable | enforced | `{evidence_path}` | maintainers | {verified} | none |",
                "",
                "## Gap tracker",
                "| Gap | Impact | Tracking artifact | Exit criteria |",
                "| --- | --- | --- | --- |",
                "| gap one | medium | `QUALITY_SCORE.md` | close it |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_main_reports_old_sor_reviews_as_nonblocking_advisories(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write_minimal_quality_score(tmp_path / "QUALITY_SCORE.md")
    _write(tmp_path / "ARCHITECTURE.md", "**Owner:** maintainers\n**Last verified:** 2020-01-01\n")
    _write(tmp_path / "DESIGN.md", "**Owner:** maintainers\n**Last verified:** 2020-01-01\n")
    _write(tmp_path / "SECURITY.md", "**Owner:** maintainers\n**Last verified:** 2020-01-01\n")
    _write(tmp_path / "RELIABILITY.md", "**Owner:** maintainers\n**Last verified:** 2020-01-01\n")
    _write(tmp_path / "PLANS.md", "**Owner:** maintainers\n**Last verified:** 2020-01-01\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--review-age-advisory-days",
            "30",
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 0
    report = report_path.read_text(encoding="utf-8")
    assert "Review-age advisories: 5" in report
    assert "review age is" in report


def test_main_reports_old_scorecard_review_as_nonblocking_advisory(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write_minimal_quality_score(
        tmp_path / "QUALITY_SCORE.md",
        last_verified="2020-01-01",
    )
    for name in ("ARCHITECTURE.md", "DESIGN.md", "SECURITY.md", "RELIABILITY.md", "PLANS.md"):
        _write(tmp_path / name, f"**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--review-age-advisory-days",
            "30",
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 0
    report = report_path.read_text(encoding="utf-8")
    assert "Review-age advisories: 2" in report
    assert "scorecard row 1 (`usr`)" in report


def test_main_reports_broken_evidence_links_and_fails(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write_minimal_quality_score(tmp_path / "QUALITY_SCORE.md", evidence_path="docs/missing.md")
    _write(tmp_path / "ARCHITECTURE.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "DESIGN.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "SECURITY.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "RELIABILITY.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "PLANS.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 1
    report = report_path.read_text(encoding="utf-8")
    assert "Broken evidence links: 1" in report


def test_main_reports_unresolved_gaps_but_passes_without_critical_findings(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write_minimal_quality_score(tmp_path / "QUALITY_SCORE.md", evidence_path="docs/README.md")
    _write(tmp_path / "ARCHITECTURE.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "DESIGN.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "SECURITY.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "RELIABILITY.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(tmp_path / "PLANS.md", f"**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 0
    report = report_path.read_text(encoding="utf-8")
    assert "Unresolved gaps: 1" in report


def test_main_fails_when_quality_scorecard_section_is_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "QUALITY_SCORE.md",
        "\n".join(
            [
                "# QUALITY SCORE",
                "",
                "**Type:** system-of-record",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
                "",
                "## Gap tracker",
                "| Gap | Impact | Tracking artifact | Exit criteria |",
                "| --- | --- | --- | --- |",
                "| gap one | medium | `QUALITY_SCORE.md` | close it |",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "DESIGN.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "SECURITY.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n"
    )
    _write(
        tmp_path / "RELIABILITY.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "PLANS.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 1
    assert not report_path.exists()


def test_main_fails_when_gap_tracker_section_is_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write(
        tmp_path / "QUALITY_SCORE.md",
        "\n".join(
            [
                "# QUALITY SCORE",
                "",
                "**Type:** system-of-record",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
                "",
                "## Quality scorecard",
                "| Area | Axis | Score (0-4) | Trend | Gate | Evidence | Owner | Last verified | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                f"| `usr` | domain | 3 | stable | enforced | `docs/README.md` | maintainers | {today} | none |",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "DESIGN.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "SECURITY.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n"
    )
    _write(
        tmp_path / "RELIABILITY.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "PLANS.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 1
    assert not report_path.exists()


def test_main_fails_when_scorecard_row_owner_is_empty(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write(
        tmp_path / "QUALITY_SCORE.md",
        "\n".join(
            [
                "# QUALITY SCORE",
                "",
                "**Type:** system-of-record",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
                "",
                "## Quality scorecard",
                "| Area | Axis | Score (0-4) | Trend | Gate | Evidence | Owner | Last verified | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                f"| `usr` | domain | 3 | stable | enforced | `docs/README.md` |  | {today} | none |",
                "",
                "## Gap tracker",
                "| Gap | Impact | Tracking artifact | Exit criteria |",
                "| --- | --- | --- | --- |",
                "| gap one | medium | `QUALITY_SCORE.md` | close it |",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "DESIGN.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "SECURITY.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n"
    )
    _write(
        tmp_path / "RELIABILITY.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "PLANS.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 1
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Scorecard contract issues: 1" in report


def test_main_fails_when_scorecard_row_last_verified_is_invalid(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    _write(
        tmp_path / "QUALITY_SCORE.md",
        "\n".join(
            [
                "# QUALITY SCORE",
                "",
                "**Type:** system-of-record",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
                "",
                "## Quality scorecard",
                "| Area | Axis | Score (0-4) | Trend | Gate | Evidence | Owner | Last verified | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "| `usr` | domain | 3 | stable | enforced | `docs/README.md` | maintainers | not-a-date | none |",
                "",
                "## Gap tracker",
                "| Gap | Impact | Tracking artifact | Exit criteria |",
                "| --- | --- | --- | --- |",
                "| gap one | medium | `QUALITY_SCORE.md` | close it |",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "DESIGN.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "SECURITY.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n"
    )
    _write(
        tmp_path / "RELIABILITY.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "PLANS.md", f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n")

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--report-file",
            str(report_path),
        ]
    )

    assert rc == 1
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Scorecard contract issues: 1" in report
