"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_cruncher_study_docs_and_skill_contracts.py

Docs and repo-local skill contracts for the checked-in Cruncher shortening study.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_snapback_shortening_study_is_visible_through_docs_and_agents() -> None:
    docs_index = _read("docs/README.md")
    study_registry = _read("docs/studies/index.yaml")
    studies_index = _read("docs/studies/README.md")
    cruncher_docs = _read("src/dnadesign/cruncher/docs/README.md")
    root_agents = _read("AGENTS.md")
    cruncher_agents = _read("src/dnadesign/cruncher/AGENTS.md")
    dev_docs = _read("docs/dev/README.md")

    assert "cruncher-study-status.md" in docs_index
    assert "cruncher-study-preflight.md" in docs_index
    assert "studies/snapback_shortening_effort/status.md" in docs_index
    assert "pin the desired record with `--study-dir docs/studies/<study-id>`" in study_registry
    assert ".agents/skills/snapback-hairpin-study/SKILL.md" in studies_index
    assert "docs/studies/snapback_shortening_effort" in studies_index
    assert "selector untouched and pin that study with `--study-dir docs/studies/<study-id>`" in studies_index
    assert "snapback_shortening_effort/status.md" in cruncher_docs
    assert "snapback_shortening_effort/routes.md" in cruncher_docs
    assert ".agents/skills/snapback-hairpin-study/SKILL.md" in cruncher_docs
    assert ".agents/skills/snapback-hairpin-study/SKILL.md" in root_agents
    assert ".agents/skills/snapback-hairpin-study/SKILL.md" in cruncher_agents
    assert ".agents/skills/snapback-hairpin-study/scripts/audit-snapback-hairpin-study-skill.sh" in dev_docs


def test_snapback_shortening_study_record_and_skill_keep_boundary_language_explicit() -> None:
    skill = _read(".agents/skills/snapback-hairpin-study/SKILL.md")
    route_matrix = _read(".agents/skills/snapback-hairpin-study/references/route-matrix.md")
    refresh_loop = _read(".agents/skills/snapback-hairpin-study/references/refresh-loop.md")
    study_surfaces = _read(".agents/skills/snapback-hairpin-study/references/study-surfaces.md")
    status = _read("docs/studies/snapback_shortening_effort/status.md")
    routes = _read("docs/studies/snapback_shortening_effort/routes.md")
    pipeline = _read("docs/studies/snapback_shortening_effort/pipeline.yaml")
    ops_study = _read("docs/studies/snapback_shortening_effort/ops.study.yaml")

    assert "released-product Snapback" in status
    assert "exposed post-release bottom strand" in status
    assert "Current phase: `snapback_released_solve`" in status
    assert "src/dnadesign/cruncher/workspaces/de033/runbook.md" in status
    assert "Next-scope preflight stays read-only" in status
    assert "FREQUENT_CUTTER" in status
    assert "YIU" in status
    assert "Repo-local study shortcut" in status
    assert "canonical post-probe handoff" in status
    assert "released_snapback_artifacts.md" in status
    assert "This page keeps the study-owned handoff map in one place." in routes
    assert "Ordered post-probe handoff" in routes
    assert "Open `pipeline.yaml` only when the task needs machine-readable command-group" in routes
    assert "src/dnadesign/cruncher/workspaces/de033" in routes
    assert "--nick-preset neb_nicking_v1" in routes
    assert "--nick-additional-preset thermo_nicking_v1" in routes
    assert "--release-preset type_iis_release_v1" in routes
    assert "exposed post-release bottom strand" in routes
    assert "whole-catalog released" in routes
    assert "plots/released_hit_triptych.pdf" in routes
    assert "Treat `released-design` and `released-show` as an optional audit path only." in routes
    assert "is expected to report" in routes
    assert "`invalid_precursor` under the degenerate-prefix-aware nonnegative-origin" in routes
    assert "single contiguous fully degenerate `N` block" in status
    assert "contiguous fully degenerate `N` block" in routes
    assert "Pair with:" in routes
    assert "repo:.agents/skills/snapback-hairpin-study/SKILL.md" in pipeline
    assert "--nick-additional-preset thermo_nicking_v1" in pipeline
    assert "manifest:pipeline.yaml" not in pipeline
    assert "pair_with:" in pipeline
    assert "harness-engineering" in pipeline
    assert "pragmatic-programming-principles" in pipeline
    assert "id: snapback_released_solve" in ops_study
    assert "status: in_progress" in ops_study
    assert "snapback_released_solve: [study_record, snapback_workspace, snapback_probe]" in ops_study
    assert "skill_ref: repo:.agents/skills/snapback-hairpin-study/SKILL.md" not in ops_study
    assert "repo_local_skill" not in ops_study
    assert "study.skill.present" not in ops_study
    assert "harness-engineering" in skill
    assert "pragmatic-programming-principles" in skill
    assert "knowledge-integrity" in skill
    assert "autonomy-capability" in skill
    assert "architecture-invariants" in skill
    assert "exposed-bottom-strand geometry lane" in skill
    assert "FREQUENT_CUTTER" in skill
    assert "contiguous fully degenerate `N` block" in skill
    assert "do not require `pipeline.yaml` or `ops.study.yaml`" in skill
    assert "docs/studies/snapback_shortening_effort/status.md" in skill
    assert "cruncher.data-plane.cruncher-study-status" in route_matrix
    assert "cruncher.data-plane.cruncher-study-preflight" in route_matrix
    assert "Blank-thread bootstrap" in refresh_loop
    assert "Open `routes.md` after the record or blocker answer is settled." in refresh_loop
    assert "Pair with `harness-engineering`" in refresh_loop
    assert "Pair with `pragmatic-programming-principles`" in refresh_loop
    assert "canonical" in study_surfaces
    assert "docs/studies/snapback_shortening_effort/status.md" in study_surfaces
    assert ".agents/skills/snapback-hairpin-study/SKILL.md" in study_surfaces


def test_repo_local_snapback_hairpin_skill_audit_is_present_and_passes() -> None:
    skill_root = _repo_root() / ".agents" / "skills" / "snapback-hairpin-study"

    assert (skill_root / "SKILL.md").exists()
    assert (skill_root / "references" / "route-matrix.md").exists()
    assert (skill_root / "references" / "refresh-loop.md").exists()
    assert (skill_root / "references" / "study-surfaces.md").exists()
    assert (skill_root / "references" / "external-sources.md").exists()
    assert (skill_root / "scripts" / "audit-snapback-hairpin-study-skill.sh").exists()

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(skill_root / "scripts" / "audit-snapback-hairpin-study-skill.sh")],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
