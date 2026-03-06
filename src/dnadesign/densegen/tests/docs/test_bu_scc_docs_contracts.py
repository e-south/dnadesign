"""
--------------------------------------------------------------------------------
densegen project
src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py

Contract checks for BU SCC operator guidance discoverability from DenseGen docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

DENSEGEN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
DENSEGEN_TUTORIALS = DENSEGEN_ROOT / "docs" / "tutorials"
BU_SCC_DOCS = REPO_ROOT / "docs" / "bu-scc"
NOTIFY_DOCS = REPO_ROOT / "docs" / "notify"
TOP_LEVEL_SYSTEM_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "DESIGN.md",
    REPO_ROOT / "RELIABILITY.md",
    REPO_ROOT / "SECURITY.md",
    REPO_ROOT / "PLANS.md",
    REPO_ROOT / "QUALITY_SCORE.md",
    REPO_ROOT / "docs" / "README.md",
)


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


def test_densegen_analysis_qsub_is_plot_only_without_notebook_generation() -> None:
    qsub_script = _read(BU_SCC_DOCS / "jobs" / "densegen-analysis.qsub")
    assert (
        'DENSEGEN_ANALYSIS_PLOTS="${DENSEGEN_ANALYSIS_PLOTS:-stage_a_summary,placement_map,run_health,tfbs_usage}"'
        in qsub_script
    )
    assert 'uv run dense plot -c "$DENSEGEN_CONFIG" --only "$DENSEGEN_ANALYSIS_PLOTS"' in qsub_script
    assert "dense_array_video_showcase requires ffmpeg executable in PATH." in qsub_script
    assert 'ATTEMPTS_PARQUET="$TABLES_DIR/attempts.parquet"' in qsub_script
    assert 'COMPOSITION_PARQUET="$TABLES_DIR/composition.parquet"' in qsub_script
    assert "requires attempts artifacts" in qsub_script
    assert "requires composition artifacts" in qsub_script
    assert "resolve_run_root" in qsub_script
    assert 'dirname "$DENSEGEN_CONFIG"' not in qsub_script
    assert "dense notebook generate" not in qsub_script
    assert "DENSEGEN_NOTEBOOK_FORCE" not in qsub_script


def test_status_first_queue_fair_guidance_present_in_bu_scc_docs_bundle() -> None:
    quickstart = _read(BU_SCC_DOCS / "quickstart.md")
    batch_notify = _read(BU_SCC_DOCS / "batch-notify.md")
    cheat_sheet = _read(BU_SCC_DOCS / "agent-cheatsheet.md")
    docs_bundle = "\n".join((quickstart, batch_notify, cheat_sheet)).lower()

    assert "running_jobs > 3" in docs_bundle
    assert "qsub -t" in docs_bundle
    assert "-hold_jid" in docs_bundle
    assert "respect the queue and do not skip the line" in docs_bundle


def test_bu_scc_notify_webhook_examples_require_owner_only_secret_file_permissions() -> None:
    quickstart = _read(BU_SCC_DOCS / "quickstart.md")
    batch_notify = _read(BU_SCC_DOCS / "batch-notify.md")
    jobs_readme = _read(BU_SCC_DOCS / "jobs" / "README.md")
    docs_bundle = "\n".join((quickstart, batch_notify, jobs_readme))

    assert 'touch "$WEBHOOK_FILE"' in docs_bundle
    assert 'chmod 600 "$WEBHOOK_FILE"' in docs_bundle


def test_top_level_docs_do_not_reference_removed_repo_local_sge_skill_path() -> None:
    removed_repo_local_path = "docs/bu-scc/sge-hpc-ops/SKILL.md"
    for path in TOP_LEVEL_SYSTEM_DOCS:
        text = _read(path)
        assert removed_repo_local_path not in text
