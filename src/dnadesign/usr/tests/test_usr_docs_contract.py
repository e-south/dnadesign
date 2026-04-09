"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_usr_docs_contract.py

Contracts for USR sync syntax and DenseGen-to-Notify event-boundary docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
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
    path = _repo_root() / rel_path
    return path.read_text(encoding="utf-8")


def test_usr_sync_docs_use_positional_remote_for_sync_commands() -> None:
    readme = _read("src/dnadesign/usr/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    combined = f"{readme}\n{sync_ops}"

    stale = re.compile(r"usr\s+(?:pull|push|diff|status)\s+[^\n]*--remote\b")
    assert stale.search(combined) is None
    assert "uv run usr diff my_dataset bu-scc" in sync_ops
    assert "uv run usr pull my_dataset bu-scc -y" in sync_ops


def test_bu_scc_runbook_uses_positional_usr_pull_example() -> None:
    runbook = _read("docs/bu-scc/batch-notify.md")
    stale = re.compile(r"uv run usr pull [^\n]*--remote\b")
    assert stale.search(runbook) is None
    assert "uv run usr pull densegen/demo_hpc bu-scc -y" in runbook


def test_usr_notify_boundary_docs_keep_events_contract_explicit() -> None:
    notify_doc = _read("docs/notify/usr-events.md")
    usr_event_doc = _read("src/dnadesign/usr/docs/reference/event-log.md")

    assert ".events.log" in notify_doc
    assert "outputs/meta/events.jsonl" in notify_doc
    assert ".events.log" in usr_event_doc
    assert "not Notify input" in usr_event_doc


def test_usr_sync_docs_follow_progressive_disclosure_flow() -> None:
    sync_router = _read("src/dnadesign/usr/docs/operations/sync.md")
    quickstart = _read("src/dnadesign/usr/docs/operations/sync-quickstart.md")
    troubleshooting = _read("src/dnadesign/usr/docs/operations/sync-troubleshooting.md")

    assert "sync-quickstart.md" in sync_router
    assert "sync-modes.md" in sync_router
    assert "sync-troubleshooting.md" in sync_router
    assert "Quick path" in quickstart
    assert "Failure diagnosis sequence" in troubleshooting


def test_usr_sync_docs_cover_iterative_hpc_clone_safety_loop() -> None:
    quickstart = _read("src/dnadesign/usr/docs/operations/sync-quickstart.md")

    assert "Iterative batch loop (HPC clone -> local clone)" in quickstart
    assert "uv run usr diff my_dataset bu-scc" in quickstart
    assert "uv run usr pull my_dataset bu-scc -y" in quickstart
    assert "uv run usr push my_dataset bu-scc -y" in quickstart
    assert "fails fast when remote `records.parquet` is missing" in quickstart
    assert "fails fast when local `records.parquet` is missing" in quickstart
    assert "skip transfer when no changes are detected" in quickstart
    assert "shared remote dataset lock (`.usr.lock`)" in quickstart
    assert "--verify-sidecars" in quickstart
    assert "--no-verify-sidecars" in quickstart
    assert "--verify-derived-hashes" in quickstart
    assert "--no-verify-derived-hashes" in quickstart
    assert (
        "defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks"
        in quickstart
    )
    assert "--strict-bootstrap-id" in quickstart
    assert "USR_SYNC_STRICT_BOOTSTRAP_ID=1" in quickstart
    assert "stage into a temporary directory and only promote after verification" in quickstart
    assert "reject symlink and unsupported entry types before promotion" in quickstart
    assert "post-action sync audit summary" in quickstart


def test_usr_sync_router_declares_route_metadata() -> None:
    sync_router = _read("src/dnadesign/usr/docs/operations/sync.md")

    assert "**Type:** route" in sync_router
    assert "**Plane:** data-plane" in sync_router
    assert "**Owner-boundary:** usr" in sync_router


def test_docs_index_links_progressive_usr_sync_workflows() -> None:
    docs_index = _read("docs/README.md")
    assert "Choose a workflow" in docs_index
    assert "Inspect available work" in docs_index
    assert "Start with:" in docs_index
    assert "Quick notes" not in docs_index
    assert "runbooks/README.md" in docs_index
    assert "#### Single-tool workflows" in docs_index
    assert "#### Cross-tool dataset workflows" in docs_index
    assert "#### Scheduler and environment workflows" in docs_index
    assert "Use these when one tool owns the next step" in docs_index
    assert "Use these when data moves through more than one tool" in docs_index
    assert "Use these when the next step is orchestration, environment setup, or audit output" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-audit-loop.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index
    assert "uv run dense workspace list" in docs_index
    assert "uv run construct workspace list" in docs_index
    assert "uv run infer workspace list" in docs_index
    assert "uv run cluster workspace list" in docs_index
    assert "uv run usr ls --root <usr-root>" in docs_index


def test_docs_index_exposes_task_first_workflow_map() -> None:
    docs_index = _read("docs/README.md")
    assert "Choose a workflow" in docs_index
    assert "Sync iterative HPC outputs to local analysis safely" in docs_index
    assert "Run cross-machine sync with stricter failure checks" in docs_index
    assert "Chain DenseGen -> USR -> Infer -> USR updates" in docs_index
    assert "chosen as `X` or export a flattened matrix" in docs_index
    assert "Hand one construct-backed dataset to infer and downstream watchers" in docs_index
    assert "Run BU SCC batch jobs with notifications" in docs_index
    assert "Plan and execute deterministic DenseGen/Infer HPC orchestration runbooks" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index


def test_start_here_doc_exposes_lightweight_workflow_navigation() -> None:
    readme = _read("README.md")
    docs_index = _read("docs/README.md")

    assert "start-here.md" not in docs_index
    assert "docs/start-here.md" not in readme


def test_start_here_doc_is_not_part_of_docs_surface() -> None:
    assert not (_repo_root() / "docs" / "start-here.md").exists()


def test_top_level_readme_exposes_workflow_docs_map() -> None:
    readme = _read("README.md")

    assert "## Documentation" in readme
    assert "docs/README.md" in readme
    assert "Use the docs index to choose a workflow, inspect existing work, or jump to a tool." in readme
    assert "cluster` exploration and OPAL active learning" not in readme
    assert "Workflow and docs map" not in readme
    assert "## Repository map" not in readme


def test_docs_index_includes_progressive_entrypoint_ladders() -> None:
    docs_index = _read("docs/README.md")

    assert "Choose a workflow" in docs_index
    assert "Design a sequence library in a workspace" in docs_index
    assert "Run model inference and write outputs back to datasets" in docs_index
    assert "Construct -> USR -> Infer shared dataset runbook" in docs_index
    assert "Sync iterative HPC outputs to local analysis safely" in docs_index
    assert "Run cross-machine sync with stricter failure checks" in docs_index
    assert "src/dnadesign/usr/docs/operations/workflow-map.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/promoter-evo2-journey.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/promoter-study-status-contract.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index
    assert "studies/README.md" in docs_index


def test_docs_index_routes_study_root_semantics_to_study_records() -> None:
    docs_index = _read("docs/README.md")
    studies_index = _read("docs/studies/README.md")

    assert "Naming rule for study work:" not in docs_index
    assert "Check study dataset-root semantics and affiliated-dataset registry terms" in docs_index
    assert "workspace_local_export" in studies_index
    assert "shared" in studies_index
    assert "external_usr" in studies_index


def test_usr_top_readme_is_lightweight_router() -> None:
    usr_readme = _read("src/dnadesign/usr/README.md")

    assert "## Documentation" in usr_readme
    assert "docs/README.md" in usr_readme
    assert "docs/getting-started/cli-quickstart.md" in usr_readme
    assert "docs/operations/README.md" in usr_readme
    assert "docs/reference/README.md" in usr_readme
    assert "## Start" not in usr_readme
    assert "## Command-line quickstart (run from anywhere)" not in usr_readme
    assert "## Remote synchronization (secure shell)" not in usr_readme
    assert "## Package boundary" not in usr_readme
    assert "## CLI" not in usr_readme
    assert "### Schema contract" not in usr_readme


def test_usr_agent_and_sync_docs_prefer_explicit_remotes_config() -> None:
    usr_agents = _read("src/dnadesign/usr/AGENTS.md")
    sync_setup = _read("src/dnadesign/usr/docs/operations/sync-setup.md")
    sync_skill = _read("src/dnadesign/usr/skills/bu-scc-usr-sync/SKILL.md")

    assert "--remotes-config <remotes.yaml>" in usr_agents
    assert "USR_REMOTES_PATH" in usr_agents
    assert "fallback" in usr_agents
    assert "--remotes-config" in sync_setup
    assert "USR_REMOTES_PATH" in sync_setup
    assert "Prefer `uv run usr --remotes-config <remotes.yaml> ...`" in sync_skill


def test_usr_docs_index_exposes_getting_started_and_reference_paths() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")

    assert "### Choose a task" in usr_docs
    assert "getting-started/README.md" in usr_docs
    assert "getting-started/cli-quickstart.md" in usr_docs
    assert "operations/README.md" in usr_docs
    assert "reference/README.md" in usr_docs
    assert "reference/schema-contract.md" in usr_docs
    assert "reference/event-log.md" in usr_docs
    assert "operations/multi-source-shared-dataset-assembly.md" in usr_docs
    assert "operations/construct-infer-shared-dataset-runbook.md" in usr_docs
    assert "operations/promoter-evo2-journey.md" in usr_docs
    assert "operations/promoter-study-status-contract.md" in usr_docs
    assert "operations/promoter-characterization-feature-matrix.md" in usr_docs
    assert "choose cluster or prepare OPAL" in usr_docs


def test_promoter_study_index_and_status_are_checked_in_for_stress_ethanol_cipro_growth() -> None:
    datasets = _read("docs/studies/stress_ethanol_cipro_growth/datasets.yaml")
    status = _read("docs/studies/stress_ethanol_cipro_growth/status.md")
    pipeline = _read("docs/studies/stress_ethanol_cipro_growth/pipeline.yaml")

    assert "promoter/stress_ethanol_cipro_anchor_set" in datasets
    assert "promoter/stress_ethanol_cipro_construct_contexts" in datasets
    assert "pipeline.yaml" in status
    assert "construct_workspace:" in pipeline
    assert "study_stress_ethanol_cipro_pdual10" in pipeline
    assert "infer_batch_7b_with_notify:" in pipeline
    assert "anchor_only:" in pipeline


def test_promoter_study_ops_contract_marks_default_notify_submit_path_as_required() -> None:
    payload = yaml.safe_load(_read("docs/studies/stress_ethanol_cipro_growth/ops.study.yaml"))
    checks = payload["preflight"]["checks"]["infer_batch_preparation"]
    by_id = {row["check_id"]: row for row in checks}

    assert by_id["notify.environment.webhook"]["required"] is True
    assert by_id["notify.environment.tls"]["required"] is True
    assert by_id["notify.profile.anchor_only_20b"]["required"] is True
    assert by_id["notify.resolve_events.anchor_only_20b"]["required"] is True
    assert by_id["infer.batch.20b.anchor_only.plan"]["required"] is True
    assert by_id["infer.batch.queue"]["required"] is False


def test_usr_promoter_journey_doc_links_cross_tool_owner_surfaces() -> None:
    journey = _read("src/dnadesign/usr/docs/operations/promoter-evo2-journey.md")

    assert "ops catalog show usr.data-plane.promoter-feature-matrix" in journey
    assert "multi-source-shared-dataset-assembly.md" in journey
    assert "construct-infer-shared-dataset-runbook.md" in journey
    assert "evo2-promoter-features.md" in journey
    assert "evo2-provider.md" in journey
    assert "promoter-study-status-contract.md" in journey
    assert "docs/notify/README.md" in journey
    assert "usr-infer-x-active-learning.md" in journey
    assert "only after one explicit `infer__...` column is chosen as `X`" in journey


def test_promoter_study_status_contract_documents_manifest_and_refresh_loop() -> None:
    contract = _read("src/dnadesign/usr/docs/operations/promoter-study-status-contract.md")
    docs_index = _read("docs/README.md")
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")
    studies_index = _read("docs/studies/README.md")
    promoter_index = _read("docs/studies/index.yaml")
    index_template = _read("docs/templates/promoter-study-index.yaml")
    datasets_template = _read("docs/templates/promoter-study-datasets.yaml")
    template = _read("docs/templates/promoter-study-status.md")
    templates_index = _read("docs/templates/README.md")
    root_agents = _read("AGENTS.md")
    usr_agents = _read("src/dnadesign/usr/AGENTS.md")
    skill = _read(".agents/skills/promoter-study-status/SKILL.md")

    assert "promoter-study-status-contract.md" in docs_index
    assert "promoter-study-status-contract.md" in usr_docs
    assert "promoter-study-status-contract.md" in ops_index
    assert "studies/README.md" in docs_index
    assert "promoter-study-index.yaml" in templates_index
    assert "promoter-study-datasets.yaml" in templates_index
    assert "promoter-study-status.md" in templates_index
    assert ".agents/skills/promoter-study-status/SKILL.md" in root_agents
    assert ".agents/skills/promoter-study-status/SKILL.md" in usr_agents
    assert "docs/studies/index.yaml" in root_agents
    assert "docs/studies/index.yaml" in usr_agents
    assert "docs/studies/promoter/index.yaml" not in root_agents
    assert "docs/studies/promoter/index.yaml" not in usr_agents
    assert "docs/studies/README.md" in skill
    assert "docs/studies/index.yaml" in skill
    assert "docs/studies/<study-id>/campaign.yaml" in skill
    assert "docs/studies/<study-id>/datasets.yaml" in skill
    assert "docs/studies/<study-id>/status.md" in skill
    assert "docs/studies/<study-id>/ops.study.yaml" in skill
    assert "docs/studies/index.yaml" in contract
    assert "docs/studies/<study-id>/" in contract
    assert "docs/studies/README.md" in contract
    assert "docs/templates/promoter-study-index.yaml" in contract
    assert "docs/templates/promoter-study-datasets.yaml" in contract
    assert "docs/templates/promoter-study-status.md" in contract
    assert "ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix" in contract
    assert "ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml" in contract
    assert "ops progress show usr.data-plane.promoter-feature-matrix" in contract
    assert "docs/studies/index.yaml" in studies_index
    assert "active_study_id: stress_ethanol_cipro_growth" in promoter_index
    assert "record_root: docs/studies/stress_ethanol_cipro_growth" in promoter_index
    assert "version: 1" in index_template
    assert "active_study_id: <study-id>" in index_template
    assert "record_root: docs/studies/<study-id>" in index_template
    assert "docs/studies/<study-id>/datasets.yaml" in studies_index
    assert "docs/studies/<study-id>/ops.study.yaml" in studies_index
    assert "If `docs/studies/index.yaml` is missing" in studies_index
    assert "cp docs/templates/promoter-study-index.yaml docs/studies/index.yaml" in studies_index
    assert "cp docs/templates/promoter-study-datasets.yaml docs/studies/<study-id>/datasets.yaml" in studies_index
    assert "cp docs/templates/promoter-study-ops.study.yaml docs/studies/<study-id>/ops.study.yaml" in studies_index
    assert "Read `docs/studies/index.yaml` first." in contract
    assert "If the registry already exists, edit it in place instead of replacing it." in contract
    assert "docs/studies/<study-id>/" in studies_index
    assert "cp docs/templates/promoter-study-status.md docs/studies/<study-id>/status.md" in studies_index
    assert "dataset registry" in studies_index
    assert "role: densegen_anchor" in datasets_template
    assert "role: feature_matrix" in datasets_template
    assert "root_kind: shared|workspace_local_export|external_usr" in datasets_template
    assert "status: present|planned" in datasets_template
    assert "onboard_mode: existing_local|existing_remote|existing_both|create_new" in datasets_template
    assert "authority: local|remote|shared" in datasets_template
    assert "notes: <how this location relates to the shared study root>" in datasets_template
    assert "default_direction: pull|push|bidirectional|none" in datasets_template
    assert "audit_json:" in datasets_template
    assert "strict_bootstrap_id: true" in datasets_template
    assert "remote_root_kind: shared|workspace_local_export|external_usr" in datasets_template
    assert "remote_path: n/a" in datasets_template
    assert "Target row count:" in template
    assert "Current shared feature dataset:" in template
    assert "Current feature-dataset row count:" in template
    assert "Affiliated dataset registry: `datasets.yaml`" in template
    assert "DenseGen anchor shared dataset:" in template
    assert "Wildtype or manual dataset:" in template
    assert "Construct template seed dataset:" in template
    assert "anchor_only" in template
    assert "anchor_plus_template" in template
    assert "full_lane_set" in template
    assert "uv run infer prune --usr <dataset> --usr-root <usr-root>" in template
    assert "uv run usr maintenance overlay-remove <dataset> --namespace infer --mode archive" in template
    assert "uv run usr maintenance overlay-compact <dataset> --namespace densegen" in template
    assert "notify usr-events watch --events <usr-root>/<feature-dataset>/.events.log --dry-run" in template
    assert "usr.data-plane.hpc-sync" in contract
    assert "root_kind" in contract
    assert "workspace_local_export" in contract
    assert "If `status.md` still marks the shared feature dataset as `n/a`, skip this" in contract
    assert "uv run usr --root <usr-root> info <dataset-id> --format json" in contract
    assert "--audit-json-out docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json" in contract
    assert "ops progress show usr.data-plane.hpc-sync --sync-audit-json" in contract
    assert "strict_bootstrap_id: true" in contract
    assert "onboard_mode: existing_remote" in skill
    assert "docs/studies/<study-id>/datasets.yaml" in skill
    assert "usr.data-plane.hpc-sync" in skill
    assert "source-assembly mode" in skill
    assert "promoter-study-preflight --scope next --json" in skill
    assert not (_repo_root() / "docs" / "studies" / "promoter").exists()
    assert not (_repo_root() / "src/dnadesign/usr/skills/promoter-study-status/SKILL.md").exists()


def test_repo_local_promoter_skill_audit_is_documented_and_present() -> None:
    dev_docs = _read("docs/dev/README.md")
    skill_root = _repo_root() / ".agents" / "skills" / "promoter-study-status"

    assert ".agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh" in dev_docs
    assert (skill_root / "SKILL.md").exists()
    assert (skill_root / "scripts" / "audit-promoter-study-status-skill.sh").exists()


def test_repo_local_promoter_skill_audit_passes() -> None:
    skill_root = _repo_root() / ".agents" / "skills" / "promoter-study-status"
    result = subprocess.run(
        [shutil.which("bash") or "bash", str(skill_root / "scripts" / "audit-promoter-study-status-skill.sh")],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_usr_docs_index_avoids_anchor_coupling_to_top_readme() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")

    assert "../README.md#" not in usr_docs


def test_promoter_study_record_is_checked_in_for_stress_ethanol_cipro_growth() -> None:
    index_yaml = _read("docs/studies/index.yaml")
    campaign = _read("docs/studies/stress_ethanol_cipro_growth/campaign.yaml")
    datasets = _read("docs/studies/stress_ethanol_cipro_growth/datasets.yaml")
    ops_study = _read("docs/studies/stress_ethanol_cipro_growth/ops.study.yaml")
    status = _read("docs/studies/stress_ethanol_cipro_growth/status.md")

    assert "active_study_id: stress_ethanol_cipro_growth" in index_yaml
    assert "family: promoter" in index_yaml
    assert "record_root: docs/studies/stress_ethanol_cipro_growth" in index_yaml
    assert "promoter/stress_ethanol_cipro_feature_matrix" in campaign
    assert "role: densegen_anchor" in datasets
    assert "remote_name: cluster" in datasets
    assert "root_kind: shared" in datasets
    assert "role: densegen_anchor" in datasets
    assert "remote_name: cluster" in datasets
    assert "mg1655_promoters" in datasets
    assert "plasmids" in datasets
    assert "version: 2" in ops_study
    assert "record_sources:" in ops_study
    assert "lifecycle:" in ops_study
    assert "execution_surfaces:" in ops_study
    assert "checks:" in ops_study
    assert "summary_scope: repo" in ops_study
    assert "group_phase_bindings:" in ops_study
    assert "runtime_shared_groups: [notify_environment]" in ops_study
    assert "densegen/study_stress_ethanol_cipro" in status
    assert "157160" in status
    assert "157164" in status
    assert "100000" in status
    assert "`anchor_only_7b=1024`, `anchor_plus_template_7b=128`" in status
    assert "`anchor_only_20b=256`, `anchor_plus_template_20b=48`" in status
    assert "`h_rt=24:00:00`" in status


def test_usr_reference_docs_cover_core_contracts() -> None:
    ref_index = _read("src/dnadesign/usr/docs/reference/README.md")
    schema = _read("src/dnadesign/usr/docs/reference/schema-contract.md")
    overlays = _read("src/dnadesign/usr/docs/reference/overlay-and-registry.md")
    events = _read("src/dnadesign/usr/docs/reference/event-log.md")
    api = _read("src/dnadesign/usr/docs/reference/python-api.md")

    assert "schema-contract.md" in ref_index
    assert "overlay-and-registry.md" in ref_index
    assert "event-log.md" in ref_index
    assert "python-api.md" in ref_index
    assert "Required columns" in schema
    assert "Overlay merge semantics" in overlays
    assert "Namespace registry (required)" in overlays
    assert "Payload fields" in events
    assert "Notify expects at minimum" in events
    assert "from dnadesign.usr import Dataset" in api


def test_usr_docs_index_exposes_sync_runbooks() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md")
    construct_handoff = _read("src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")

    assert "operations/README.md" in usr_docs
    assert "architecture-introspection.md" in usr_docs
    assert "sync.md" in ops_index
    assert "sync-audit-loop.md" in ops_index
    assert "hpc-agent-sync-flow.md" in ops_index
    assert "chained-densegen-infer-sync-runbook.md" in ops_index
    assert "construct-infer-shared-dataset-runbook.md" in ops_index
    assert "sync-fidelity-drills.md" in ops_index
    assert "continue to cluster or prepare OPAL after choosing one explicit `X` column" in ops_index
    assert "pressure-test-loop-mock-batch--adversarial-schemas" in ops_index
    assert "hpc-agent-sync-flow.md" in usr_docs
    assert "sync-audit-loop.md" in usr_docs
    assert "chained-densegen-infer-sync-runbook.md" in usr_docs
    assert "construct-infer-shared-dataset-runbook.md" in usr_docs
    assert "sync-fidelity-drills.md" in usr_docs
    assert "**Type:** runbook" in runbook
    assert "**Plane:** data-plane" in runbook
    assert "**Owner-boundary:** usr" in runbook
    assert "Preflight" in runbook
    assert "Run loop" in runbook
    assert "Verify loop" in runbook
    assert "uv run usr diff" in runbook
    assert "uv run usr pull" in runbook
    assert "uv run usr push" in runbook
    assert "Full chained loop" in chained
    assert "# Chained DenseGen and Infer Sync Runbook" in chained
    assert "**Type:** runbook" in chained
    assert "**Plane:** data-plane" in chained
    assert "**Owner-boundary:** usr" in chained
    assert "uv run infer run --preset evo2/extract_logits_ll --usr" in chained
    assert '--usr-root "$LOCAL_USR_ROOT"' in chained
    assert "qsub -P <project>" in chained
    assert "_derived changed" in chained
    assert "Construct -> USR -> Infer Shared Dataset Runbook" in construct_handoff
    assert "uv run construct workspace run-project" in construct_handoff
    assert "uv run infer validate usr-registry" in construct_handoff
    assert 'export DNADESIGN_REPO_ROOT="$(git rev-parse --show-toplevel)"' in construct_handoff
    assert (
        'cp "$DNADESIGN_REPO_ROOT/src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml"'
        in construct_handoff
    )
    assert "notify usr-events watch" in construct_handoff
    assert "Drill 1: Pull must fail when `_derived` payload is missing" in fidelity
    assert "Drill 2: Push must fail when remote misses local overlays" in fidelity
    assert "Drill 3: Overlay schema attack surface" in fidelity
    assert 'export LOCAL_USR_ROOT="src/dnadesign/usr/datasets"' in fidelity
    assert '--usr-root "$LOCAL_USR_ROOT"' in fidelity
    assert "--verify-sidecars" in fidelity
    assert "--no-verify-sidecars" in fidelity
    assert "--verify-derived-hashes" in fidelity
    assert "post-pull-sidecars" in fidelity
    assert "post-push-sidecars" in fidelity
    assert 'export LOCAL_USR_ROOT="src/dnadesign/usr/datasets"' in chained
    assert 'export DATASET_ID="my_dataset"' in chained
    assert "HPC_USR_ROOT" not in chained
    assert "No extra HPC-side `pull` is required" in chained
    assert "hpc-agent-sync-flow.md" in sync_ops
    assert "sync-audit-loop.md" in sync_ops
    assert "chained-densegen-infer-sync-runbook.md" in sync_ops
    assert "sync-fidelity-drills.md" in sync_ops


def test_multi_source_runbook_makes_upstream_dataset_mapping_explicit() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md")

    assert "### 1b) Map those ids to real upstream datasets before validation" in runbook
    assert "does not create the extra upstream dataset for you" in runbook
    assert 'export PRIMARY_INPUT_DATASET="anchor_parts_demo"' in runbook
    assert 'export EXTRA_INPUT_DATASET="<existing_densegen_or_manual_usr_dataset>"' in runbook


def test_usr_sync_docs_are_split_into_progressive_runbooks() -> None:
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")
    sync_router = _read("src/dnadesign/usr/docs/operations/sync.md")
    quickstart = _read("src/dnadesign/usr/docs/operations/sync-quickstart.md")
    setup = _read("src/dnadesign/usr/docs/operations/sync-setup.md")
    modes = _read("src/dnadesign/usr/docs/operations/sync-modes.md")
    troubleshooting = _read("src/dnadesign/usr/docs/operations/sync-troubleshooting.md")

    assert "sync-quickstart.md" in ops_index
    assert "sync-setup.md" in ops_index
    assert "sync-modes.md" in ops_index
    assert "sync-troubleshooting.md" in ops_index

    assert "sync-quickstart.md" in sync_router
    assert "sync-setup.md" in sync_router
    assert "sync-modes.md" in sync_router
    assert "sync-troubleshooting.md" in sync_router

    assert "Minimum command loop" in quickstart
    assert "Configure a USR remote" in setup
    assert "Dataset directory mode" in modes
    assert "Common failure signatures" in troubleshooting


def test_usr_docs_include_sync_audit_runbook_with_chained_commands() -> None:
    docs_index = _read("docs/README.md")
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    audit = _read("src/dnadesign/usr/docs/operations/sync-audit-loop.md")

    assert "src/dnadesign/usr/docs/operations/sync-audit-loop.md" in docs_index
    assert "sync-audit-loop.md" in usr_docs
    assert "sync-audit-loop.md" in ops_index
    assert "sync-audit-loop.md" in sync_ops
    assert "--audit-json-out" in audit
    assert "jq -r" in audit
    assert "uv run usr diff" in audit
    assert "uv run usr pull" in audit
    assert "uv run usr push" in audit
    assert "local_only" in audit
    assert "remote_only" in audit
    assert "usr_output_version" in audit


def test_usr_workflow_map_runbook_is_indexed_with_command_chains() -> None:
    docs_index = _read("docs/README.md")
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")
    workflow_map = _read("src/dnadesign/usr/docs/operations/workflow-map.md")

    assert "workflow-map.md" in docs_index
    assert "workflow-map.md" in usr_docs
    assert "workflow-map.md" in ops_index
    assert "multi-source-shared-dataset-assembly.md" in docs_index
    assert "multi-source-shared-dataset-assembly.md" in usr_docs
    assert "multi-source-shared-dataset-assembly.md" in ops_index
    assert "promoter-characterization-feature-matrix.md" in docs_index
    assert "promoter-characterization-feature-matrix.md" in usr_docs
    assert "promoter-characterization-feature-matrix.md" in ops_index
    assert "Bootstrap from remote -> local clone" in workflow_map
    assert "Iterative HPC batch loop" in workflow_map
    assert "DenseGen -> USR -> Infer -> USR chained loop" in workflow_map
    assert "Multi-source USR assembly -> Construct -> Infer" in workflow_map
    assert "Construct -> USR -> Infer shared dataset loop" in workflow_map
    assert "Promoter feature matrix -> Cluster or OPAL prep" in workflow_map
    assert "short summaries, not full procedures" in workflow_map
    assert "## Context preamble" in workflow_map
    assert 'WORKFLOW_ROOT="${WORKFLOW_ROOT:-$PWD}"' in workflow_map
    assert 'ARTIFACT_ROOT="${ARTIFACT_ROOT:-$WORKFLOW_ROOT/outputs/logs/usr-workflow-map}"' in workflow_map
    assert "Pressure-test loop (mock batch + adversarial schemas)" in workflow_map
    assert 'uv run usr diff "$DATASET_ID" bu-scc' in workflow_map
    assert 'uv run usr pull "$DATASET_ID" bu-scc -y' in workflow_map
    assert 'uv run usr push "$DATASET_ID" bu-scc -y' in workflow_map
    assert (
        'uv run infer run --preset evo2/extract_logits_ll --usr "$DATASET_ID" '
        '--usr-root "$LOCAL_USR_ROOT" --field sequence --device cpu --write-back' in workflow_map
    )
    assert (
        'uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window' in workflow_map
    )
    assert (
        'uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window' in workflow_map
    )
    assert 'DATASET_ID="anchor_template_shared_dataset_demo"' in workflow_map
    assert (
        'uv run notify usr-events watch --events "$USR_ROOT/$DATASET_ID/.events.log" --provider generic '
        "--dry-run --no-advance-cursor-on-dry-run" in workflow_map
    )
    assert 'uv run cluster fit --dataset "$FEATURE_DATASET"' in workflow_map
    assert "infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean" in workflow_map
    assert 'uv run opal validate -c "$OPAL_WORKDIR/configs/campaign.yaml"' in workflow_map
    assert "run_usr_harness_cycle.sh" in workflow_map
    assert "test_sync_schema_adversarial.py" in workflow_map
    assert "--audit-json-out" in workflow_map
    assert "primary_changed: .primary.changed" in workflow_map
    assert "derived_changed: ._derived.changed" in workflow_map
    assert "aux_changed: ._auxiliary.changed" in workflow_map
    assert "/tmp/usr-sync-audit.json" not in workflow_map
    assert '"$ARTIFACT_ROOT/usr-sync-audit.json"' in workflow_map
    assert '"$ARTIFACT_ROOT/usr-harness-report.json"' in workflow_map
    assert '"$ARTIFACT_ROOT/usr-sync-audit-drill-report.json"' in workflow_map
    assert '"$ARTIFACT_ROOT/usr-sync-audit-drill"' in workflow_map


def test_multi_source_source_of_truth_runbook_routes_to_shared_downstream_handoff() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md")

    assert 'uv run usr --root "$USR_ROOT" maintenance merge' in runbook
    assert "--carry-namespace usr_label" in runbook
    assert (
        'uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --runtime'
        in runbook
    )
    assert 'uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window' in runbook
    assert 'export DATASET_ID="$DOWNSTREAM_DATASET"' in runbook
    assert (
        "construct-infer-shared-dataset-runbook.md#5-shared-downstream-continuation-prepare-infer-handoff-"
        "against-the-construct-dataset" in runbook
    )
    assert (
        "construct-infer-shared-dataset-runbook.md#6-shared-downstream-continuation-verify-downstream-event-"
        "consumption" in runbook
    )
    assert "promoter-characterization-feature-matrix.md" in runbook
    assert "../../../../../docs/operations/orchestration-runbooks.md" in runbook


def test_promoter_feature_matrix_runbook_uses_extract_ops_and_dataset_placeholders() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md")

    assert "operation: extract" in runbook
    assert "dataset: <anchor-only-feature-dataset>" in runbook
    assert "dataset: <construct-expanded-feature-dataset>" in runbook


def test_construct_source_of_truth_runbook_documents_construct_notify_resolver_modes() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md")

    assert "anchor-template-shared-dataset-demo" in runbook
    assert 'export CONSTRUCT_CONFIG="$WORKSPACE_ROOT/config.slot_a.window.yaml"' in runbook
    assert 'export DATASET_ID="anchor_template_shared_dataset_demo"' in runbook
    assert 'config["jobs"][0]["ingest"]["dataset"] = os.environ["DATASET_ID"]' in runbook
    assert "uv run python - <<'PY'" in runbook
    assert "perl -0pi -e" not in runbook
    assert 'notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG"' in runbook
    assert (
        'notify setup resolve-events --tool construct --workspace "shared_dataset_demo:slot_a_window" --json' in runbook
    )
    assert "CONSTRUCT_WORKSPACE_ROOT" in runbook
    assert "DNADESIGN_REPO_ROOT" in runbook
    assert "promoter-characterization-feature-matrix.md" in runbook
    assert "is not supported today" not in runbook
    assert "--project slot_b_window" in runbook


def test_usr_merge_docs_make_overlay_limit_explicit() -> None:
    maintenance = _read("src/dnadesign/usr/docs/reference/maintenance.md")
    runbook = _read("src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md")

    assert "does not implicitly copy source overlay namespaces or `_derived` sidecars" in maintenance
    assert "--carry-namespace <namespace>" in maintenance
    assert "only `id`-keyed overlays are supported" in maintenance
    assert "rewrites canonical base rows only" in runbook
    assert "--carry-namespace <namespace>" in runbook
    assert "--carry-namespace usr_label" in runbook


def test_notify_and_ops_routes_link_construct_source_of_truth_runbook() -> None:
    notify_index = _read("docs/notify/README.md")
    notify_runbook = _read("docs/notify/usr-events.md")
    ops_index = _read("docs/operations/README.md")
    docs_index = _read("docs/README.md")

    assert "construct-infer-shared-dataset-runbook.md" in notify_index
    assert "construct-infer-shared-dataset-runbook.md" in notify_runbook
    assert "construct-infer-shared-dataset-runbook.md" in ops_index
    assert "multi-source-shared-dataset-assembly.md" in docs_index
    assert "multi-source-shared-dataset-assembly.md" in ops_index
    assert "promoter-characterization-feature-matrix.md" in docs_index
    assert "promoter-characterization-feature-matrix.md" in ops_index


def test_top_level_docs_surfaces_avoid_meta_routing_jargon() -> None:
    for rel_path in ("README.md", "docs/README.md", "docs/runbooks/README.md"):
        text = _read(rel_path).lower()
        assert "authoritative" not in text
        assert "canonical" not in text
        assert "progressive disclosure" not in text


def test_promoter_feature_matrix_runbook_routes_to_cluster_and_opal() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md")

    assert 'uv run usr --root "$USR_ROOT" maintenance merge' in runbook
    assert 'uv run infer run --config "$INFER_CONFIG_7B" --dry-run' in runbook
    assert 'uv run infer run --config "$INFER_CONFIG_7B"' in runbook
    assert "../../../cluster/docs/workflows/exploratory-clustering.md" in runbook
    assert "../../../opal/docs/workflows/usr-infer-x-active-learning.md" in runbook
    assert "fit -> umap -> analyze" in runbook
    assert "data.location.kind: usr" in runbook


def test_usr_harness_script_is_documented_in_workflow_map() -> None:
    workflow_map = _read("src/dnadesign/usr/docs/operations/workflow-map.md")
    script_path = _repo_root() / "src/dnadesign/usr/scripts/run_usr_harness_cycle.sh"

    assert script_path.exists()
    assert "run_usr_harness_cycle.sh" in workflow_map
    assert "preflight -> run -> verify" in workflow_map
    assert "USR_HARNESS_REPORT_PATH" in workflow_map
    assert "USR_HARNESS_RUN_SYNC_AUDIT_DRILL" in workflow_map
    assert "USR_HARNESS_SYNC_AUDIT_REPORT_PATH" in workflow_map


def test_usr_sync_audit_drill_script_is_documented_in_workflow_map() -> None:
    workflow_map = _read("src/dnadesign/usr/docs/operations/workflow-map.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")
    script_path = _repo_root() / "src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py"

    assert script_path.exists()
    assert "run_usr_sync_audit_drill.py" in workflow_map
    assert "--report-json" in workflow_map
    assert "diff/pull/push" in workflow_map
    assert "run_usr_sync_audit_drill.py" in ops_index


def test_usr_introspection_doc_covers_lifecycle_and_config_mapping() -> None:
    introspection = _read("src/dnadesign/usr/docs/architecture-introspection.md")

    assert "Intent and use-case map" in introspection
    assert "Lifecycle model" in introspection
    assert "Architecture view stack" in introspection
    assert "Config-schema to behavior mapping" in introspection
    assert "Interaction map" in introspection
    assert "Evidence ledger" in introspection
    assert "Open questions and risk notes" in introspection


def test_usr_sync_runbooks_avoid_agent_or_human_labeling_language() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    audit = _read("src/dnadesign/usr/docs/operations/sync-audit-loop.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    combined = "\n".join([usr_docs, sync_ops, runbook, audit, fidelity]).lower()

    banned = [
        "agent-oriented",
        "agent runbook",
        "agent checklist",
        "agentic",
        "for agents",
        "for humans",
    ]
    for token in banned:
        assert token not in combined


def test_usr_hpc_and_chained_runbooks_use_default_hash_sync_contract_examples() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md")

    assert "--verify auto" not in runbook
    assert "--verify parquet" not in runbook
    assert "--verify auto" not in chained
    assert "--verify parquet" not in chained
    assert "--no-verify-derived-hashes" in runbook
    assert "--no-verify-derived-hashes" in chained
    assert (
        "defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks"
        in runbook
    )
    assert (
        "defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks"
        in chained
    )


def test_usr_sync_docs_include_auxiliary_file_audit_contract() -> None:
    sync_ops = _read("src/dnadesign/usr/docs/operations/sync.md")
    sync_quickstart = _read("src/dnadesign/usr/docs/operations/sync-quickstart.md")
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    hpc = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")

    assert "_auxiliary" in sync_ops
    assert "_auxiliary" in sync_quickstart
    assert "_auxiliary" in chained
    assert "_auxiliary" in fidelity
    assert "strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks" in sync_ops
    assert "strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks" in chained
    assert "strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks" in hpc
    assert "--audit-json-out" in sync_quickstart


def test_promoter_feature_matrix_routes_to_downstream_owner_workflows_without_repeating_commands() -> None:
    matrix = _read("src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md")

    assert "cluster/docs/workflows/exploratory-clustering.md" in matrix
    assert "opal/docs/workflows/usr-infer-x-active-learning.md" in matrix
    assert "uv run cluster fit \\" not in matrix
    assert "uv run cluster umap \\" not in matrix
    assert "uv run opal validate" not in matrix
    assert "uv run opal run" not in matrix


def test_usr_storage_policy_docs_distinguish_workspace_defaults_and_explicit_external_roots() -> None:
    architecture = _read("ARCHITECTURE.md")
    design = _read("DESIGN.md")
    setup = _read("src/dnadesign/usr/docs/operations/sync-setup.md")
    quickstart = _read("src/dnadesign/usr/docs/operations/sync-quickstart.md")

    assert "Curated study-facing workspaces that enable USR sinks should default those" in architecture
    assert "Workspace-local export roots remain allowed only as explicit opt-in producer" in architecture
    assert "Curated study-facing workspace and runbook examples should default USR sinks" in design
    assert "Workspace-local export roots and external USR roots remain allowed only when" in design
    assert "Shared repo-local datasets should live under `src/dnadesign/usr/datasets`." in setup
    assert "External dataset roots are still allowed for ad-hoc sync or mirror workflows" in setup
    assert "The canonical repo-local datasets root is `src/dnadesign/usr/datasets`" in quickstart


def test_construct_and_multi_source_runbooks_mark_workspace_local_paths_as_tracer_bullets() -> None:
    construct_handoff = _read("src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md")
    multi_source = _read("src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md")

    assert "uses the packaged construct workspace as a local tracer bullet" in construct_handoff
    assert "declared shared USR root authoritative" in construct_handoff
    assert "local tracer-bullet USR root" in construct_handoff
    assert "uses a packaged construct workspace as a local tracer bullet" in multi_source
    assert "declared shared USR root authoritative" in multi_source


def test_hpc_sync_runbook_covers_bootstrap_from_either_side() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")

    assert "Bootstrap from either side" in runbook
    assert "HPC has dataset, local does not" in runbook
    assert "Local has dataset, HPC does not" in runbook
    assert 'uv run usr pull "$DATASET_ID" bu-scc -y' in runbook
    assert 'uv run usr push "$DATASET_ID" bu-scc -y' in runbook
