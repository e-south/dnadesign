"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_cruncher_study_docs_and_skill_contracts.py

Docs and repo-local skill contracts for the checked-in Cruncher retron hairpin study.

Module Author(s): Eric J. South
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


def test_retron_hairpin_study_is_visible_through_docs_and_agents() -> None:
    docs_index = _read("docs/README.md")
    study_registry = _read("docs/studies/index.yaml")
    studies_index = _read("docs/studies/README.md")
    cruncher_docs = _read("src/dnadesign/cruncher/docs/README.md")
    root_agents = _read("AGENTS.md")
    cruncher_agents = _read("src/dnadesign/cruncher/AGENTS.md")
    dev_docs = _read("docs/dev/README.md")
    retired_study_id = "snapback" + "_shortening_effort"

    assert "cruncher-study-status.md" in docs_index
    assert "cruncher-study-preflight.md" in docs_index
    assert "studies/retron_hairpin_design/status.md" in docs_index
    assert "study_id: retron_hairpin_design" in study_registry
    assert retired_study_id not in study_registry
    assert "pin the desired record with `--study-dir docs/studies/<study-id>`" in study_registry
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in studies_index
    assert "docs/studies/retron_hairpin_design" in studies_index
    assert retired_study_id not in studies_index
    assert "selector untouched and pin that study with `--study-dir docs/studies/<study-id>`" in studies_index
    assert "retron_hairpin_design/status.md" in cruncher_docs
    assert "retron_hairpin_design/routes.md" in cruncher_docs
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in cruncher_docs
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in root_agents
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in cruncher_agents
    assert ".agents/skills/retron-hairpin-study/scripts/audit-retron-hairpin-study-skill.sh" in dev_docs


def test_retron_hairpin_study_record_and_skill_keep_boundary_language_explicit() -> None:
    skill = _read(".agents/skills/retron-hairpin-study/SKILL.md")
    route_matrix = _read(".agents/skills/retron-hairpin-study/references/route-matrix.md")
    refresh_loop = _read(".agents/skills/retron-hairpin-study/references/refresh-loop.md")
    study_surfaces = _read(".agents/skills/retron-hairpin-study/references/study-surfaces.md")
    scar_nick_context = _read("docs/studies/retron_hairpin_design/scar-nick-base-junction.md")
    status = _read("docs/studies/retron_hairpin_design/status.md")
    routes = _read("docs/studies/retron_hairpin_design/routes.md")
    pipeline = _read("docs/studies/retron_hairpin_design/pipeline.yaml")
    ops_study = _read("docs/studies/retron_hairpin_design/ops.study.yaml")

    assert "released-product Snapback" in status
    assert "scar-nick" in status
    assert "profile-diverse `S0=M` scar analogs" in status
    assert "retained-active released-product policy" in status
    assert "retained top and bottom product routes" in status
    assert "Current phase: `snapback_released_solve`" in status
    assert "src/dnadesign/cruncher/workspaces/de033/runbook.md" in status
    assert "Next-scope preflight stays read-only" in status
    assert "FREQUENT_CUTTER" in status
    assert "YIU" in status
    assert "Repo-local study shortcut" in status
    assert "canonical post-probe handoff" in status
    assert "released_snapback_artifacts.md" in status
    assert "BbsI-HF retains 10/256 strict scars" in status
    assert "Exact B26 `MXMX` remains a biological control architecture" in status
    assert "This page keeps the study-owned handoff map in one place." in routes
    assert "Ordered post-probe handoff" in routes
    assert "Open `pipeline.yaml` only when the task needs machine-readable command-group" in routes
    assert "### Context route: scar-nick base-junction" in routes
    assert "src/dnadesign/cruncher/workspaces/de033" in routes
    assert "--nick-preset neb_nicking_v1" in routes
    assert "--nick-additional-preset thermo_nicking_v1" in routes
    assert "--release-preset type_iis_release_v1" in routes
    assert "retained active top and bottom products" in routes
    assert "whole-catalog released" in routes
    assert "plots/released_hit_triptych.pdf" in routes
    assert "Treat `released-design` and `released-show` as an optional audit path only." in routes
    assert "is expected to report" in routes
    assert "`invalid_precursor` under the degenerate-prefix-aware nonnegative-origin" in routes
    assert "single contiguous fully degenerate `N` block" in status
    assert "contiguous fully degenerate `N` block" in routes
    assert "Pair with:" in routes
    assert "repo:.agents/skills/retron-hairpin-study/SKILL.md" in pipeline
    assert "repo:docs/studies/retron_hairpin_design/scar-nick-base-junction.md" in pipeline
    assert "--nick-additional-preset thermo_nicking_v1" in pipeline
    assert "manifest:pipeline.yaml" not in pipeline
    assert "pair_with:" in pipeline
    assert "harness-engineering" in pipeline
    assert "pragmatic-programming-principles" in pipeline
    assert "id: snapback_released_solve" in ops_study
    assert "id: scar_nick_base_junction_context" in ops_study
    assert "artifact: scar_nick_base_junction_note" in ops_study
    assert "status: in_progress" in ops_study
    assert "snapback_released_solve: [study_record, snapback_workspace, snapback_probe]" in ops_study
    assert "skill_ref: repo:.agents/skills/retron-hairpin-study/SKILL.md" not in ops_study
    assert "repo_local_skill" not in ops_study
    assert "study.skill.present" not in ops_study
    assert "harness-engineering" in skill
    assert "code-change-discipline" in skill
    assert "scar-nick base-junction" in skill
    assert "S0=M" in skill
    assert "FREQUENT_CUTTER" in skill
    assert "docs/studies/retron_hairpin_design/status.md" in skill
    assert "docs/studies/retron_hairpin_design/scar-nick-base-junction.md" in skill
    assert "cruncher.data-plane.cruncher-study-status" in route_matrix
    assert "cruncher.data-plane.cruncher-study-preflight" in route_matrix
    assert "scar-nick route in `routes.md`" in route_matrix
    assert "Blank-thread bootstrap" in refresh_loop
    assert "Open `routes.md` after the record or blocker answer is settled." in refresh_loop
    assert "Pair with `harness-engineering`" in refresh_loop
    assert "Pair with `code-change-discipline`" in refresh_loop
    assert "canonical" in study_surfaces
    assert "docs/studies/retron_hairpin_design/status.md" in study_surfaces
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in study_surfaces
    assert "exact terminal nick" in scar_nick_context
    assert "top or bottom nick allowed" in scar_nick_context
    assert "BbsI-HF" in scar_nick_context
    assert "PaqCI" in scar_nick_context
    assert "BsaI-HFv2" in scar_nick_context
    assert "nicked_strand" in scar_nick_context


def test_repo_local_retron_hairpin_skill_audit_is_present_and_passes() -> None:
    skill_root = _repo_root() / ".agents" / "skills" / "retron-hairpin-study"

    assert (skill_root / "SKILL.md").exists()
    assert (skill_root / "references" / "route-matrix.md").exists()
    assert (skill_root / "references" / "refresh-loop.md").exists()
    assert (skill_root / "references" / "study-surfaces.md").exists()
    assert (skill_root / "references" / "external-sources.md").exists()
    assert (skill_root / "scripts" / "audit-retron-hairpin-study-skill.sh").exists()

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(skill_root / "scripts" / "audit-retron-hairpin-study-skill.sh")],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
