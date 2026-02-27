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
from pathlib import Path


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
    assert "uv run usr diff densegen/my_dataset bu-scc" in sync_ops
    assert "uv run usr pull densegen/my_dataset bu-scc -y" in sync_ops


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
    assert "uv run usr diff densegen/my_dataset bu-scc" in quickstart
    assert "uv run usr pull densegen/my_dataset bu-scc -y" in quickstart
    assert "uv run usr push densegen/my_dataset bu-scc -y" in quickstart
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


def test_docs_index_links_progressive_usr_sync_workflows() -> None:
    docs_index = _read("docs/README.md")
    assert "Workflow lanes" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-audit-loop.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index


def test_docs_index_exposes_task_first_workflow_map() -> None:
    docs_index = _read("docs/README.md")
    assert "Workflow lanes" in docs_index
    assert "Sync iterative HPC outputs to local analysis safely" in docs_index
    assert "Run cross-machine sync with stricter failure checks" in docs_index
    assert "Chain DenseGen -> USR -> Infer -> USR updates" in docs_index
    assert "Run BU SCC batch jobs with notifications" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md" in docs_index
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
    assert "Workflow and docs map" not in readme
    assert "## Repository map" not in readme


def test_docs_index_includes_progressive_entrypoint_ladders() -> None:
    docs_index = _read("docs/README.md")

    assert "Workflow lanes" in docs_index
    assert "Design a sequence library in a workspace" in docs_index
    assert "Run model inference and write outputs back to datasets" in docs_index
    assert "Sync iterative HPC outputs to local analysis safely" in docs_index
    assert "Run cross-machine sync with stricter failure checks" in docs_index
    assert "src/dnadesign/usr/docs/operations/workflow-map.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md" in docs_index
    assert "src/dnadesign/usr/docs/operations/sync-fidelity-drills.md" in docs_index


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


def test_usr_docs_index_exposes_getting_started_and_reference_paths() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")

    assert "getting-started/README.md" in usr_docs
    assert "getting-started/cli-quickstart.md" in usr_docs
    assert "operations/README.md" in usr_docs
    assert "reference/README.md" in usr_docs
    assert "reference/schema-contract.md" in usr_docs
    assert "reference/event-log.md" in usr_docs


def test_usr_docs_index_avoids_anchor_coupling_to_top_readme() -> None:
    usr_docs = _read("src/dnadesign/usr/docs/README.md")

    assert "../README.md#" not in usr_docs


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
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")
    fidelity = _read("src/dnadesign/usr/docs/operations/sync-fidelity-drills.md")
    ops_index = _read("src/dnadesign/usr/docs/operations/README.md")

    assert "operations/README.md" in usr_docs
    assert "architecture-introspection.md" in usr_docs
    assert "sync.md" in ops_index
    assert "sync-audit-loop.md" in ops_index
    assert "hpc-agent-sync-flow.md" in ops_index
    assert "chained-densegen-infer-sync-demo.md" in ops_index
    assert "sync-fidelity-drills.md" in ops_index
    assert "pressure-test-loop-mock-batch--adversarial-schemas" in ops_index
    assert "hpc-agent-sync-flow.md" in usr_docs
    assert "sync-audit-loop.md" in usr_docs
    assert "chained-densegen-infer-sync-demo.md" in usr_docs
    assert "sync-fidelity-drills.md" in usr_docs
    assert "Preflight" in runbook
    assert "Run loop" in runbook
    assert "Verify loop" in runbook
    assert "uv run usr diff" in runbook
    assert "uv run usr pull" in runbook
    assert "uv run usr push" in runbook
    assert "Full chained loop" in chained
    assert "uv run infer run --preset evo2/extract_logits_ll --usr" in chained
    assert "qsub -P <project>" in chained
    assert "_derived changed" in chained
    assert "Drill 1: Pull must fail when `_derived` payload is missing" in fidelity
    assert "Drill 2: Push must fail when remote misses local overlays" in fidelity
    assert "Drill 3: Overlay schema attack surface" in fidelity
    assert "--verify-sidecars" in fidelity
    assert "--no-verify-sidecars" in fidelity
    assert "--verify-derived-hashes" in fidelity
    assert "post-pull-sidecars" in fidelity
    assert "post-push-sidecars" in fidelity
    assert "hpc-agent-sync-flow.md" in sync_ops
    assert "sync-audit-loop.md" in sync_ops
    assert "chained-densegen-infer-sync-demo.md" in sync_ops
    assert "sync-fidelity-drills.md" in sync_ops


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
    assert "Bootstrap from remote -> local clone" in workflow_map
    assert "Iterative HPC batch loop" in workflow_map
    assert "DenseGen -> USR -> Infer -> USR chained loop" in workflow_map
    assert "Pressure-test loop (mock batch + adversarial schemas)" in workflow_map
    assert 'uv run usr diff "$DATASET_ID" bu-scc' in workflow_map
    assert 'uv run usr pull "$DATASET_ID" bu-scc -y' in workflow_map
    assert 'uv run usr push "$DATASET_ID" bu-scc -y' in workflow_map
    assert "run_usr_harness_cycle.sh" in workflow_map
    assert "test_sync_schema_adversarial.py" in workflow_map
    assert "--audit-json-out" in workflow_map


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
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")

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
    chained = _read("src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md")
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


def test_hpc_sync_runbook_covers_bootstrap_from_either_side() -> None:
    runbook = _read("src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md")

    assert "Bootstrap from either side" in runbook
    assert "HPC has dataset, local does not" in runbook
    assert "Local has dataset, HPC does not" in runbook
    assert 'uv run usr pull "$DATASET_ID" bu-scc -y' in runbook
    assert 'uv run usr push "$DATASET_ID" bu-scc -y' in runbook
