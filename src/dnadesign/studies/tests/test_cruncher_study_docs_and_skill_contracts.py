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

import yaml


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def _skill_frontmatter() -> dict[str, object]:
    skill = _read(".agents/skills/retron-hairpin-study/SKILL.md")
    frontmatter = skill.split("---", 2)[1]
    payload = yaml.safe_load(frontmatter)
    assert isinstance(payload, dict)
    return payload


def test_retron_hairpin_study_is_visible_through_docs_and_agents() -> None:
    docs_index = _read("docs/README.md")
    study_registry = _read("docs/studies/index.yaml")
    studies_index = _read("docs/studies/README.md")
    cruncher_docs = _read("src/dnadesign/cruncher/docs/README.md")
    root_agents = _read("AGENTS.md")
    cruncher_agents = _read("src/dnadesign/cruncher/AGENTS.md")
    dev_docs = _read("docs/dev/README.md")
    retired_study_id = "snapback" + "_shortening_effort"

    assert "Navigate a checked-in study without exposing study-specific routes here" in docs_index
    assert "docs/studies/<study-id>/routes.md" in docs_index
    assert "cruncher-study-status.md" not in docs_index
    assert "cruncher-study-preflight.md" not in docs_index
    assert "studies/retron_hairpin_design/status.md" not in docs_index
    assert "study_id: retron_hairpin_design" in study_registry
    assert retired_study_id not in study_registry
    assert "pin the desired record with `--study-dir docs/studies/<study-id>`" in study_registry
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in studies_index
    assert "docs/studies/retron_hairpin_design" in studies_index
    assert "Study status and preflight surfaces" in studies_index
    assert "Study record authoring" in studies_index
    assert "`cruncher-study-status` and `cruncher-study-preflight` commands only for" in studies_index
    assert "explicit status or readiness questions" in studies_index
    assert retired_study_id not in studies_index
    assert "keep the selector" in studies_index
    assert "retron_hairpin_design/status.md" not in cruncher_docs
    assert "retron_hairpin_design/routes.md" in cruncher_docs
    assert ".agents/skills/retron-hairpin-study/SKILL.md" not in cruncher_docs
    assert "only for explicit status or readiness questions" in cruncher_docs
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in root_agents
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in cruncher_agents
    assert ".agents/skills/retron-hairpin-study/scripts/audit-retron-hairpin-study-skill.sh" in dev_docs


def test_retron_hairpin_skill_frontmatter_is_yaml_safe_and_discovery_scoped() -> None:
    frontmatter = _skill_frontmatter()
    description = frontmatter["description"]
    metadata = frontmatter["metadata"]

    assert frontmatter["name"] == "retron-hairpin-study"
    assert isinstance(description, str)
    assert len(description) <= 220
    assert "Use for MSD IDs" in description
    assert "generic Cruncher/snapback" in description
    assert "Snapback/scar-nick/YIU" not in description
    assert isinstance(metadata, dict)
    assert metadata["version"] == "0.7.5"


def test_retron_hairpin_skill_naive_agent_discovery_and_prompt_surface_contract() -> None:
    frontmatter = _skill_frontmatter()
    skill = _read(".agents/skills/retron-hairpin-study/SKILL.md")
    test_matrix = _read(".agents/skills/retron-hairpin-study/references/test-matrix.md")
    external_sources = _read(".agents/skills/retron-hairpin-study/references/external-sources.md")
    discovery_surface = "\n".join(
        [
            str(frontmatter["name"]),
            str(frontmatter["description"]),
            skill.split("## Trigger Tests", 1)[1],
            test_matrix,
        ]
    )

    for phrase in (
        "Use for MSD IDs",
        "single-unit MSD sequence bundles",
        "design catalogs",
        "GenBank/native-structure PNG",
        "missing MSD parts",
        "generic Cruncher/snapback",
    ):
        assert phrase in discovery_surface
    assert "Complete MSD label or complete parts" in skill
    assert "Load only the needed surfaces" in skill
    assert "OpenAI Developers guidance" in external_sources
    assert "https://developers.openai.com/api/docs/guides/prompt-engineering#coding" in external_sources

    positive_prompts = (
        "Compile pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM into a design catalog.",
        "Generate one MSD sequence plus GenBank and PNG for this Retron MSD payload and cap.",
        "Which primitive route owns this missing Retron MSD part?",
    )
    negative_prompts = (
        "Run a generic Cruncher snapback search for another project.",
        "Explain retron biology broadly.",
        "Design a wet-lab retron protocol.",
    )
    for prompt in positive_prompts:
        assert _naive_skill_match(prompt, discovery_surface), prompt
    for prompt in negative_prompts:
        assert not _naive_skill_match(prompt, discovery_surface), prompt


def _naive_skill_match(prompt: str, discovery_surface: str) -> bool:
    prompt_norm = prompt.lower()
    surface_norm = discovery_surface.lower()
    if "generic cruncher snapback" in prompt_norm:
        return False
    if "wet-lab" in prompt_norm or "biology broadly" in prompt_norm:
        return False
    positive_terms = ("retron msd", "msd", "single-unit", "genbank", "design catalog")
    return any(term in prompt_norm and term in surface_norm for term in positive_terms)


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

    assert "Retron MSD product work as a genetic compiler" in status
    assert "scar-nick" in status
    assert "profile-diverse `S0=M` scar analogs" in status
    assert "Complete labels or complete part sets should compile directly" in status
    assert "caller-chosen transient directories" in status
    assert "retained-active released-product policy" in status
    assert "retained top and bottom product routes" in status
    assert "Current phase:" not in status
    assert "snapback_released_solve" not in status
    assert "src/dnadesign/cruncher/workspaces/de033/runbook.md" in status
    assert "FREQUENT_CUTTER" in status
    assert "YIU" in status
    assert "Repo-local study shortcut" in status
    assert "released_snapback_artifacts.md" in status
    assert "BbsI-HF retains 6/256 strict scars" in status
    assert "Exact B26 `MXMX` remains a biological control architecture" in status
    assert "docs/studies/retron_hairpin_design/msd_design_registry.yaml" in status
    assert "docs/studies/retron_hairpin_design/msd_design_hit_labels.txt" in status
    assert "starts from the user's provided parts and desired" in routes
    assert "Primitive route handoff" in routes
    assert "Do not run study status or preflight first." in routes
    assert "### Study route: MSD design references" in routes
    assert "msd_design_hit_labels.txt" in routes
    assert "not registered as a top-level `uv run retron-msd` tool" in routes
    assert "Reader should not parse Construct, Folding, BaseRender, or Cruncher internals" in routes
    assert "Materialize command" in routes
    assert "manifest/sequence_manifest.json" in routes
    assert "manifest/sequence_index.tsv" in routes
    assert "secondary_structure.native.png" in routes
    assert "composition_overview.svg" in routes
    assert "Visible GenBank/CSV labels should be display labels" in routes
    assert "repeat-count flag" in routes
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
    assert 'primary_lane: "study-owned Retron MSD design-reference compilation"' in pipeline
    assert 'state_label: "product route"' in pipeline
    assert "Do not answer compiler-style requests by defaulting to study phase" in pipeline
    assert "repo:docs/studies/retron_hairpin_design/scar-nick-base-junction.md" in pipeline
    assert "--nick-additional-preset thermo_nicking_v1" in pipeline
    assert "manifest:pipeline.yaml" not in pipeline
    assert "pair_with:" in pipeline
    assert "harness-engineering" in pipeline
    assert "pragmatic-programming-principles" not in pipeline
    assert "id: msd_design_reference_catalog" in ops_study
    assert "mode: tracks" in ops_study
    assert "track_order:" in ops_study
    assert "current_track:\n    strategy: explicit\n    id: msd_design_reference_catalog" in ops_study
    assert "group_track_bindings:" in ops_study
    assert "target_track_groups:" in ops_study
    assert "current_phase:" not in ops_study
    assert "phase_order:" not in ops_study
    assert "\nphases:" not in ops_study
    assert "group_phase_bindings:" not in ops_study
    assert "target_phase_groups:" not in ops_study
    assert "msd_design_reference_catalog: [study_record, msd_reference_compile]" in ops_study
    assert "id: snapback_released_solve" in ops_study
    assert "id: scar_nick_base_junction_context" in ops_study
    assert "artifact: scar_nick_base_junction_note" in ops_study
    assert "status: in_progress" in ops_study
    assert "skill_ref: repo:.agents/skills/retron-hairpin-study/SKILL.md" not in ops_study
    assert "repo_local_skill" not in ops_study
    assert "study.skill.present" not in ops_study
    assert "harness-engineering" in skill
    assert "code-change-discipline" in skill
    assert "Route Retron MSD work as a genetic compiler" in skill
    assert "Input completeness classification" in skill
    assert 'Do not say "snapshot posture"' in skill
    assert "whether the answer came from snapshot posture" not in skill
    assert "current phase and next route" not in skill
    assert "scar-nick base-junction" in skill
    assert "S0=M" in skill
    assert "msd_design_registry.yaml" in skill
    assert "src/dnadesign/studies/retron_hairpin_design/compiler.py" in study_surfaces
    assert "msd_design_hit_labels.txt" in pipeline
    assert "msd_design_reference_catalog" in pipeline
    assert "msd_design_hit_labels" in ops_study
    assert "Start with input completeness, not study phase" in route_matrix
    assert "Complete labels should lint/compile directly." in route_matrix
    assert "msd_design_reference_v1" in study_surfaces
    assert "reference_index.tsv" in pipeline
    assert "flat references" in pipeline
    assert "id: msd_single_unit_materialize" in pipeline
    assert "sequence_manifest.json" in pipeline
    assert "secondary_structure.native.png" in pipeline
    assert "composition_overview.svg" in pipeline
    assert "shallow output-bundle layout" in study_surfaces
    assert "single-unit sequence artifact generation" in study_surfaces
    assert "materialize` GenBank/native-structure-PNG route" in study_surfaces
    assert "full component spans" in skill
    assert "same-span annotations" in skill
    assert "scar-nick route in `routes.md`" in route_matrix
    assert "cruncher-study-status --study-dir docs/studies/retron_hairpin_design" in route_matrix
    assert "cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design" in route_matrix
    assert "Compiler Bootstrap" in refresh_loop
    assert "Use study status only when the" in refresh_loop
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
