"""
--------------------------------------------------------------------------------
densegen project
src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py

Contract checks for BU SCC operator guidance discoverability from DenseGen docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

DENSEGEN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
DENSEGEN_TUTORIALS = DENSEGEN_ROOT / "docs" / "tutorials"
BU_SCC_DOCS = REPO_ROOT / "docs" / "bu-scc"
NOTIFY_DOCS = REPO_ROOT / "docs" / "notify"
FIXTURES = BU_SCC_DOCS / "fixtures"
QSTAT_HIGH_PRESSURE_FIXTURE = FIXTURES / "qstat_high_pressure.fixture"
SGE_OPERATOR_BRIEF_SCRIPT = Path.home() / ".agents" / "skills" / "sge-hpc-ops" / "scripts" / "sge-operator-brief.sh"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text()


def test_stress_tutorial_links_to_bu_scc_operational_docs() -> None:
    text = _read(DENSEGEN_TUTORIALS / "study_stress_ethanol_cipro.md")
    assert "[BU SCC Quickstart]" in text
    assert "[BU SCC Batch + Notify runbook]" in text
    assert "[BU SCC job templates]" in text


def test_bu_scc_quickstart_contains_status_first_submission_gate() -> None:
    text = _read(BU_SCC_DOCS / "quickstart.md")
    assert "running_jobs > 3" in text
    assert 'qstat -u "$USER"' in text
    assert "-hold_jid" in text


def test_bu_scc_readme_avoids_external_skill_link_dependency() -> None:
    text = _read(BU_SCC_DOCS / "README.md")
    assert "https://github.com/e-south/agent-skills/tree/main/sge-hpc-ops" not in text


def test_notify_docs_avoid_external_skill_link_dependency() -> None:
    text = _read(NOTIFY_DOCS / "README.md")
    assert "https://github.com/e-south/agent-skills/tree/main/sge-hpc-ops" not in text


def test_bu_scc_docs_use_current_densegen_runtime_field_names() -> None:
    batch_notify = _read(BU_SCC_DOCS / "batch-notify.md")
    jobs_readme = _read(BU_SCC_DOCS / "jobs" / "README.md")
    docs_bundle = "\n".join((batch_notify, jobs_readme))

    assert "densegen.solver.solver_attempt_timeout_seconds" in docs_bundle
    assert "densegen.solver.time_limit_seconds" not in docs_bundle
    assert "densegen.runtime.max_seconds_per_plan" not in docs_bundle


def test_status_first_queue_fair_guidance_matches_operator_brief_high_pressure_fixture() -> None:
    if not SGE_OPERATOR_BRIEF_SCRIPT.exists():
        pytest.skip(f"Missing centralized sge operator brief script: {SGE_OPERATOR_BRIEF_SCRIPT}")

    assert QSTAT_HIGH_PRESSURE_FIXTURE.exists(), f"Missing fixture: {QSTAT_HIGH_PRESSURE_FIXTURE}"
    command = [
        str(SGE_OPERATOR_BRIEF_SCRIPT),
        "--qstat-file",
        str(QSTAT_HIGH_PRESSURE_FIXTURE),
        "--planned-submits",
        "8",
        "--warn-over-running",
        "3",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    operator_brief = result.stdout

    assert "Submit Gate: confirm" in operator_brief
    assert "Advisor: array" in operator_brief
    assert "Queue Policy: respect queue, do not skip the line" in operator_brief
    assert "Next Action: Ask for explicit user confirmation" in operator_brief

    quickstart = _read(BU_SCC_DOCS / "quickstart.md")
    batch_notify = _read(BU_SCC_DOCS / "batch-notify.md")
    cheat_sheet = _read(BU_SCC_DOCS / "agent-cheatsheet.md")
    docs_bundle = "\n".join((quickstart, batch_notify, cheat_sheet)).lower()

    assert "running_jobs > 3" in docs_bundle
    assert "qsub -t" in docs_bundle
    assert "-hold_jid" in docs_bundle
    assert "respect the queue and do not skip the line" in docs_bundle
