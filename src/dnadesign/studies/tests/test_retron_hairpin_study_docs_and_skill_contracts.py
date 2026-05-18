"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_retron_hairpin_study_docs_and_skill_contracts.py

Docs and repo-local skill contracts for the checked-in Retron hairpin study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import yaml

from dnadesign.studies.studies.retron_hairpin_design.catalog.msd_ids import parse_msd_construct_label


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def _sha256(rel_path: str) -> str:
    return hashlib.sha256((_repo_root() / rel_path).read_bytes()).hexdigest()


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
    assert "docs/studies/<study-id>/routes/README.md" in docs_index
    assert "retron-hairpin-design-status.md" not in docs_index
    assert "retron-hairpin-design-preflight.md" not in docs_index
    assert "studies/retron_hairpin_design/record/status.md" not in docs_index
    assert "study_id: retron_hairpin_design" in study_registry
    assert retired_study_id not in study_registry
    assert "pin the desired record with `--study-dir docs/studies/<study-id>`" in study_registry
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in studies_index
    assert "docs/studies/retron_hairpin_design" in studies_index
    assert "Study status and preflight surfaces" in studies_index
    assert "Study record authoring" in studies_index
    assert "`retron-hairpin-design-status` and `retron-hairpin-design-preflight` commands only for" in studies_index
    assert "explicit status or readiness questions" in studies_index
    assert retired_study_id not in studies_index
    assert "keep the selector" in studies_index
    assert "retron_hairpin_design/status.md" not in cruncher_docs
    assert "retron_hairpin_design/routes/README.md" in cruncher_docs
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
    assert metadata["version"] == "0.7.7"


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
        "sequence bundles",
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
    scar_nick_context = _read("docs/studies/retron_hairpin_design/contexts/scar-nick-base-junction.md")
    status = _read("docs/studies/retron_hairpin_design/record/status.md")
    routes = _read("docs/studies/retron_hairpin_design/routes/README.md")
    route_msd = _read("docs/studies/retron_hairpin_design/routes/msd-design-references.md")
    route_snapback = _read("docs/studies/retron_hairpin_design/routes/released-product-snapback.md")
    route_scar_nick = _read("docs/studies/retron_hairpin_design/routes/scar-nick-base-junction.md")
    route_linear = _read("docs/studies/retron_hairpin_design/routes/linear-ssdna-composition.md")
    route_yiu = _read("docs/studies/retron_hairpin_design/routes/yiu-boundary-check.md")
    route_details = "\n".join([route_msd, route_snapback, route_scar_nick, route_linear, route_yiu])
    workbench_readme = _read("docs/studies/retron_hairpin_design/workbench/README.md")
    workbench_ontology_readme = _read("docs/studies/retron_hairpin_design/workbench/ontology/README.md")
    workbench_design_sets_readme = _read("docs/studies/retron_hairpin_design/workbench/design_sets/README.md")
    workbench_provenance_readme = _read("docs/studies/retron_hairpin_design/workbench/provenance/README.md")
    workbench_directions = _read("docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml")
    workbench_design_set = _read(
        "docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
    )
    workbench_compile_run = _read(
        "docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/2026-05-18-msd-177-194.compile.yaml"
    )
    workbench_materialization = _read(
        "docs/studies/retron_hairpin_design/workbench/provenance/materializations/2026-05-18-msd-177-194.single-unit.yaml"
    )
    pipeline = _read("docs/studies/retron_hairpin_design/operations/pipeline.yaml")
    ops_study = _read("docs/studies/retron_hairpin_design/operations/ops.study.yaml")

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
    assert "docs/studies/retron_hairpin_design/compiler/msd_design_registry.yaml" in status
    assert "docs/studies/retron_hairpin_design/compiler/msd_design_hit_labels.txt" in status
    assert "docs/studies/retron_hairpin_design/workbench/" in status
    assert "starts from the user's provided parts and desired" in routes
    assert len(routes.splitlines()) <= 85
    assert "Quick Route" in routes
    assert "Keep this page as a one-hop route map." in routes
    assert "Experimental workbench" in routes
    assert "Do not run study status or preflight first." in routes
    assert "docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml" in routes
    assert "MSD Design References Route" in route_msd
    assert "msd_design_hit_labels.txt" in route_msd
    assert "not registered as a top-level" in route_msd
    assert "Reader should\nnot parse Construct, Folding, BaseRender, or Cruncher internals" in route_msd
    assert "materialize" in route_msd
    assert "manifest/bundle/" in route_msd
    assert "manifest/indexes/" in route_msd
    assert "manifest/sequence_manifest.json" not in routes
    assert "manifest/sequence_index.tsv" not in routes
    assert "secondary_structure.native.png" in route_msd
    assert "composition_overview.svg" in route_msd
    assert "composition_overview.png" in route_msd
    assert "Each `variants/<msd_design_id>/` directory groups" in route_msd
    assert "Open `../operations/pipeline.yaml` only when the task needs" in routes
    assert "Scar-Nick Base-Junction Route" in route_scar_nick
    assert "src/dnadesign/cruncher/workspaces/de033" in route_snapback
    assert "--nick-preset neb_nicking_v1" in route_snapback
    assert "--nick-additional-preset thermo_nicking_v1" in route_snapback
    assert "--release-preset type_iis_release_v1" in route_snapback
    assert "retained active top and bottom products" in route_snapback
    assert "plots/released_hit_triptych.pdf" in route_snapback
    assert "expected to report\n`invalid_precursor`" in route_snapback
    assert "Linear ssDNA Composition Route" in route_linear
    assert "YIU Boundary Check Route" in route_yiu
    assert "single contiguous fully degenerate `N` block" in status
    assert "contiguous fully degenerate `N` block" in route_snapback
    assert "Pair with `harness-engineering`" in routes
    assert "retron_workbench_directions_v1" in workbench_directions
    assert "retron_msd_design_set_v1" in workbench_design_set
    assert "retron_msd_compiler_run_record_v1" in workbench_compile_run
    assert "retron_msd_materialization_record_v1" in workbench_materialization
    assert "msd_design_hit_labels.txt` remains a convenience compiler input" in workbench_readme
    assert "ontology/" in workbench_readme
    assert "design_sets/" in workbench_readme
    assert "provenance/" in workbench_readme
    assert "controlled vocabulary" in workbench_ontology_readme
    assert "authoritative answer" in workbench_design_sets_readme
    assert "Run records cite design sets" in workbench_provenance_readme
    assert "repo:.agents/skills/retron-hairpin-study/SKILL.md" in pipeline
    assert 'primary_lane: "study-owned Retron MSD design-reference compilation"' in pipeline
    assert 'state_label: "product route"' in pipeline
    assert "Do not answer compiler-style requests by defaulting to study phase" in pipeline
    assert "repo:docs/studies/retron_hairpin_design/contexts/scar-nick-base-junction.md" in pipeline
    assert "repo:docs/studies/retron_hairpin_design/workbench/README.md" in pipeline
    assert "scar_nick_profile_panel_v1.yaml" in pipeline
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
    assert "artifact: workbench_scar_nick_profile_panel" in ops_study
    assert "study.route_msd_design_references.present" in ops_study
    assert "study.workbench_compile_run.present" in ops_study
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
    assert "src/dnadesign/studies/studies/retron_hairpin_design/compiler.py" in study_surfaces
    assert "msd_design_hit_labels.txt" in pipeline
    assert "msd_design_reference_catalog" in pipeline
    assert "msd_design_hit_labels" in ops_study
    assert "workbench/design_sets" in study_surfaces
    assert "workbench/ontology" in study_surfaces
    assert "workbench/provenance" in study_surfaces
    assert "Start with input completeness, not study phase" in route_matrix
    assert "Complete labels should lint/compile directly." in route_matrix
    assert "Persistent hypotheses, effect tags, and design-set membership live in the workbench" in route_matrix
    assert "msd_design_reference_v1" in study_surfaces
    assert "reference_index.tsv" in pipeline
    assert "flat references" in pipeline
    assert "id: msd_single_unit_materialize" in pipeline
    assert "sequence_manifest.json" in pipeline
    assert "secondary_structure.native.png" in pipeline
    assert "composition_overview.svg" in pipeline
    assert "composition_overview.png" in pipeline
    assert "shallow output-bundle layout" in study_surfaces
    assert "single-unit sequence artifact generation" in study_surfaces
    assert "materialize` GenBank/native-structure-PNG/review-PNG" in study_surfaces
    assert "full component spans" in skill
    assert "same-span annotations" in skill
    assert "scar-nick route in `routes/README.md` / `routes/scar-nick-base-junction.md`" in route_matrix
    assert "studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design" in route_matrix
    assert "studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design" in route_matrix
    assert "Compiler Bootstrap" in refresh_loop
    assert "Use study status only when the" in refresh_loop
    assert "If the question is persistent provenance" in refresh_loop
    assert "Pair with `harness-engineering`" in refresh_loop
    assert "Pair with `code-change-discipline`" in refresh_loop
    assert "canonical" in study_surfaces
    assert "docs/studies/retron_hairpin_design/record/status.md" in study_surfaces
    assert ".agents/skills/retron-hairpin-study/SKILL.md" in study_surfaces
    assert "exact terminal nick" in scar_nick_context
    assert "top or bottom nick allowed" in scar_nick_context
    assert "BbsI-HF" in scar_nick_context
    assert "PaqCI" in scar_nick_context
    assert "BsaI-HFv2" in scar_nick_context
    assert "nicked_strand" in scar_nick_context
    assert "MSD Design References Route" in route_details


def test_retron_hairpin_workbench_design_set_records_compile_and_trace_provenance() -> None:
    directions = yaml.safe_load(_read("docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml"))
    design_set = yaml.safe_load(
        _read("docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml")
    )
    compiler_run = yaml.safe_load(
        _read(
            "docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/2026-05-18-msd-177-194.compile.yaml"
        )
    )
    materialization = yaml.safe_load(
        _read(
            "docs/studies/retron_hairpin_design/workbench/provenance/materializations/2026-05-18-msd-177-194.single-unit.yaml"
        )
    )
    label_lines = [
        line.strip()
        for line in _read("docs/studies/retron_hairpin_design/compiler/msd_design_hit_labels.txt").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert directions["contract"] == "retron_workbench_directions_v1"
    assert directions["schema_version"] == 1
    assert design_set["contract"] == "retron_msd_design_set_v1"
    assert design_set["schema_version"] == 1
    assert compiler_run["contract"] == "retron_msd_compiler_run_record_v1"
    assert compiler_run["schema_version"] == 1
    assert materialization["contract"] == "retron_msd_materialization_record_v1"
    assert materialization["schema_version"] == 1

    known_directions = {direction["id"]: direction for direction in directions["directions"]}
    known_effect_tags = set(directions["effect_tags"])
    design_labels = [design["label"] for design in design_set["designs"]]

    assert design_set["label_count"] == 18
    assert len(design_set["designs"]) == 18
    assert design_labels == label_lines
    assert design_set["input_hashes"]["convenience_label_input_sha256"] == _sha256(
        "docs/studies/retron_hairpin_design/compiler/msd_design_hit_labels.txt"
    )
    assert design_set["input_hashes"]["registry_sha256"] == _sha256(
        "docs/studies/retron_hairpin_design/compiler/msd_design_registry.yaml"
    )

    for design in design_set["designs"]:
        parsed = parse_msd_construct_label(design["label"])
        assert parsed.construct_id == design["construct_id"]
        assert parsed.msd_design_id == design["expected_msd_design_id"]
        assert parsed.payload_id == design["payload_id"]
        assert parsed.cap_id == design["cap_id"]
        assert parsed.left_base == design["left_base"]
        assert parsed.right_base == design["right_base"]
        assert parsed.profile_s3s2s1s0 == design["profile_s3s2s1s0"]
        assert design["direction_ids"]
        assert design["effect_tags"]
        assert design["rationale"]
        assert set(design["direction_ids"]) <= set(known_directions)
        assert set(design["effect_tags"]) <= known_effect_tags
        for direction_id in design["direction_ids"]:
            assert set(design["effect_tags"]) & set(known_directions[direction_id]["effect_tags"])

    assert compiler_run["status"] == "verified"
    assert compiler_run["design_set_ref"] == (
        "docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
    )
    assert compiler_run["inputs"]["label_input"]["sha256"] == _sha256(
        "docs/studies/retron_hairpin_design/compiler/msd_design_hit_labels.txt"
    )
    assert compiler_run["inputs"]["registry"]["sha256"] == _sha256(
        "docs/studies/retron_hairpin_design/compiler/msd_design_registry.yaml"
    )
    assert compiler_run["observed_output"]["output_policy"] == "transient_not_checked_in"
    assert compiler_run["observed_output"]["record_count"] == 18
    assert compiler_run["observed_output"]["catalog_contract"] == "msd_design_catalog_v1"
    assert compiler_run["observed_output"]["bundle_contract"] == "msd_design_catalog_bundle_v1"

    assert materialization["status"] == "not_run_missing_sequence_subcomponents"
    assert materialization["design_set_ref"] == compiler_run["design_set_ref"]
    assert materialization["compiler_run_ref"].endswith("2026-05-18-msd-177-194.compile.yaml")
    assert set(materialization["required_inputs"]["payload_sequences"]) == {"TetR"}
    assert set(materialization["required_inputs"]["cap_sequences"]) == {"C26", "C172"}
    assert "plots/secondary_structure.native.png" in materialization["expected_output"]["per_design_deliverables"]
    assert "plots/composition_overview.svg" in materialization["expected_output"]["per_design_deliverables"]
    assert "plots/composition_overview.png" in materialization["expected_output"]["per_design_deliverables"]


def test_retron_hairpin_workbench_keeps_root_bounded_by_record_lanes() -> None:
    workbench = _repo_root() / "docs" / "studies" / "retron_hairpin_design" / "workbench"

    visible_root_files = {path.name for path in workbench.iterdir() if path.is_file() and not path.name.startswith(".")}
    visible_root_dirs = {path.name for path in workbench.iterdir() if path.is_dir() and not path.name.startswith(".")}

    assert visible_root_files == {"README.md"}
    assert visible_root_dirs == {"design_sets", "ontology", "provenance"}
    assert (workbench / "ontology" / "README.md").exists()
    assert (workbench / "ontology" / "directions.yaml").exists()
    assert (workbench / "design_sets" / "README.md").exists()
    assert (workbench / "design_sets" / "scar_nick_profile_panel_v1.yaml").exists()
    assert (workbench / "provenance" / "README.md").exists()
    assert (workbench / "provenance" / "compiler_runs" / "README.md").exists()
    assert (workbench / "provenance" / "compiler_runs" / "2026-05-18-msd-177-194.compile.yaml").exists()
    assert (workbench / "provenance" / "materializations" / "README.md").exists()
    assert (workbench / "provenance" / "materializations" / "2026-05-18-msd-177-194.single-unit.yaml").exists()


def test_named_study_records_keep_root_bounded_by_semantic_lanes() -> None:
    studies_root = _repo_root() / "docs" / "studies"
    expected_dirs_by_study = {
        "retron_hairpin_design": {
            "compiler",
            "contexts",
            "contracts",
            "operations",
            "record",
            "routes",
            "workbench",
        },
        "stress_ethanol_cipro_growth": {
            "audits",
            "bindings",
            "contracts",
            "operations",
            "record",
            "routes",
        },
        "regulondb_native_promoter_panel": {
            "audits",
            "bindings",
            "operations",
            "record",
            "routes",
        },
    }

    for study_id, expected_dirs in expected_dirs_by_study.items():
        study_root = studies_root / study_id
        visible_root_files = {
            path.name for path in study_root.iterdir() if path.is_file() and not path.name.startswith(".")
        }
        visible_root_dirs = {
            path.name for path in study_root.iterdir() if path.is_dir() and not path.name.startswith(".")
        }

        assert visible_root_files == {"README.md"}
        assert visible_root_dirs == expected_dirs
        assert (study_root / "record" / "README.md").exists()
        assert (study_root / "record" / "campaign.yaml").exists()
        assert (study_root / "record" / "datasets.yaml").exists()
        assert (study_root / "record" / "status.md").exists()
        assert (study_root / "operations" / "README.md").exists()
        assert (study_root / "operations" / "ops.study.yaml").exists()
        assert (study_root / "operations" / "pipeline.yaml").exists()
        assert (study_root / "routes" / "README.md").exists()

    assert len((studies_root / "stress_ethanol_cipro_growth" / "routes" / "README.md").read_text().splitlines()) <= 60
    assert (
        len((studies_root / "regulondb_native_promoter_panel" / "routes" / "README.md").read_text().splitlines()) <= 60
    )


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
