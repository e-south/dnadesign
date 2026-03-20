"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_ops_docs_progressive_disclosure_contracts.py

Progressive-disclosure contract tests for Ops package and top-level Ops docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _read(path: Path) -> str:
    assert path.exists(), f"Missing markdown file: {path}"
    return path.read_text(encoding="utf-8")


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def test_ops_module_readme_has_banner_narrative_and_doc_map() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "ops" / "README.md")
    _assert_token_order(
        text,
        [
            "![Ops banner](assets/ops-banner.svg)",
            "## Common entrypoints",
            "## Documentation",
        ],
        label="src/dnadesign/ops/README.md",
    )
    assert "cross-tool orchestration control plane" in text
    assert "Use Ops when:" in text
    assert "Do not use Ops when:" in text
    assert "shared catalog view" in text
    assert "uv run ops catalog list" in text
    assert "Typical flow: browse the catalog" in text
    assert "uv run ops catalog show <registry-id>" in text
    assert "uv run ops progress explain <registry-id>" in text
    assert "docs/README.md" in text
    assert "docs/how-to-use-ops.md" in text
    assert "../../../docs/runbooks/README.md" in text
    assert "../../../docs/operations/README.md" in text
    assert "../../../docs/operations/orchestration-runbooks.md" in text
    assert "runbooks/presets" in text
    assert "../../../docs/README.md" in text
    assert "## Entrypoint contract" not in text
    assert "## Boundary reminder" not in text
    assert "progressive disclosure" not in text.lower()


def test_ops_package_local_docs_index_routes_to_shared_runbook_surface() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "ops" / "docs" / "README.md")
    _assert_token_order(
        text,
        [
            "### Start here",
            "### Package-local surfaces",
            "### Boundary reminders",
        ],
        label="src/dnadesign/ops/docs/README.md",
    )
    assert "../../../../docs/runbooks/README.md" in text
    assert "shared catalog view over" in text
    assert "how-to-use-ops.md" in text
    assert "../../../../docs/operations/README.md" in text
    assert "../../../../docs/operations/orchestration-runbooks.md" in text
    assert "../runbooks/presets" in text
    assert "../../../../docs/README.md" in text
    assert "uv run ops catalog list" in text
    assert "prints YAML to stdout unless you pass `--out`" in text


def test_ops_how_to_doc_carries_quick_usage_commands() -> None:
    text = _read(_repo_root() / "src" / "dnadesign" / "ops" / "docs" / "how-to-use-ops.md")
    _assert_token_order(
        text,
        [
            "### Quick terms",
            "### Discover the right runbook",
            "### Inspect one registered procedure",
            "### Check status and build manifests",
            "### Continue reading",
        ],
        label="src/dnadesign/ops/docs/how-to-use-ops.md",
    )
    assert "registry id" in text
    assert "related route" in text
    assert "progress surface" in text
    assert "campaign manifest" in text
    assert "uv run ops catalog list" in text
    assert "uv run ops catalog list --simple" in text
    assert 'uv run ops catalog list --query "promoter feature matrix"' in text
    assert "uv run ops catalog show <registry-id>" in text
    assert "--plane data-plane --query infer" in text
    assert "--section tool-sources" in text
    assert "--related-to usr.data-plane.promoter-feature-matrix" in text
    assert "typed related procedures" in text
    assert "typed related tool docs" in text
    assert "exact deep docs when declared" in text
    assert "uv run ops progress explain <registry-id>" in text
    assert (
        "uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>"
        in text
    )
    assert "inspect the required progress flags before you run `progress show`" in text
    assert "uv run ops progress scaffold <registry-id> ..." in text
    assert "prints YAML to stdout unless you pass `--out`" in text
    assert "can cross tool boundaries" in text
    assert "uv run ops progress scaffold --related-to <registry-id>" in text
    assert "uv run ops progress campaign --manifest <manifest.yaml>" in text
    assert "../../../../docs/runbooks/README.md" in text
    assert "../../../../docs/operations/README.md" in text
    assert "../../../../docs/operations/orchestration-runbooks.md" in text


def test_ops_docs_index_has_progressive_disclosure_routes() -> None:
    text = _read(_repo_root() / "docs" / "operations" / "README.md")
    _assert_token_order(
        text,
        [
            "### What Ops is for",
            "### Start here",
            "### Shell routes",
            "### Orchestration routes",
            "### Contracts",
            "### Status and manifest routes",
            "### Verification loop",
            "### Operator quickstart",
        ],
        label="docs/operations/README.md",
    )
    assert "runbook init command contract" in text
    assert "runbook plan command contract" in text
    assert "runbook execute command contract" in text
    assert "**Type:** route" in text
    assert "**Plane:** control-plane" in text
    assert "**Owner-boundary:** ops" in text
    assert "ops runbook init --workflow" in text
    assert "uv run ops catalog list" in text
    assert "uv run ops catalog list --simple" in text
    assert 'uv run ops catalog list --query "promoter feature matrix"' in text
    assert "uv run ops catalog show <registry-id>" in text
    assert "uv run ops progress explain <registry-id>" in text
    assert "related procedures around one path" in text
    assert "--plane data-plane --query infer" in text
    assert "--section tool-sources" in text
    assert "--related-to usr.data-plane.promoter-feature-matrix" in text
    assert "related tool docs around one path" in text
    assert "exact deep docs when declared" in text
    assert (
        "uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>"
        in text
    )
    assert "uv run ops progress scaffold <registry-id> ..." in text
    assert "prints YAML to stdout unless you pass `--out`" in text
    assert "`ops progress show` and `ops progress campaign` are read-only" in text
    assert "uv run ops progress scaffold --related-to <registry-id>" in text
    assert "uv run ops progress campaign --manifest <manifest.yaml>" in text
    assert "--project <project>" in text
    assert "project dunlop" not in text
    assert "orchestration-runbooks.md" in text
    assert "../runbooks/README.md" in text
    assert "multi-source-source-of-truth-assembly.md" in text
    assert "../README.md" in text
    assert "../../src/dnadesign/ops/README.md" in text
    assert "progressive disclosure" not in text.lower()


def test_orchestration_runbook_doc_keeps_run_order_and_contract_sections() -> None:
    text = _read(_repo_root() / "docs" / "operations" / "orchestration-runbooks.md")
    _assert_token_order(
        text,
        [
            "### Why this exists",
            "### Runbook bootstrap path",
            "### 2-minute dry-run path",
            "### Orchestration workflow ids",
            "### Runbook schema (v1)",
            "### Planner and executor commands",
            "### Contract rules",
        ],
        label="docs/operations/orchestration-runbooks.md",
    )
    assert "uv run ops runbook init" in text
    assert "--project <project>" in text
    assert "uv run ops runbook presets" in text
    assert "uv run ops runbook active-jobs" in text
    assert "Infer scaffolds also include notify by default" in text
    assert "**Type:** runbook" in text
    assert "**Plane:** control-plane" in text
    assert "**Owner-boundary:** ops" in text
    assert "It does not own durable USR-backed data-plane workflows" in text
    assert "default is `300`" in text
    assert "operator and agent review" not in text
    assert "--command-timeout-seconds" in text
    assert "mode=auto" in text
    assert "none -> fresh" in text
    assert "resume_ready -> resume" in text
    assert "partial -> contract error" in text
    assert "<workspace-root>/outputs/logs/ops/audit/<file>.json" in text
    assert "<path-to-audit.json>" not in text
    assert "Only workspace-scoped audit paths are accepted" in text
    assert "prune-ops-logs" in text
    assert "logging.retention.keep_last" in text
    assert "logging.retention.max_age_days" in text
    assert "outputs/logs/ops/runtime" in text
    assert "usr-overlay-guard" in text
    assert "usr-records-part-guard" in text
    assert "usr-archived-overlay-guard" in text
    assert "densegen-overlay-guard" not in text
    assert "densegen.overlay_guard.overlay_namespace" in text
    assert "densegen.overlay_guard.namespace" not in text
    assert "transient operational working directories at repo root" in text
    assert "/scratch" in text
    assert "resources.gpu_memory_gib" in text
    assert "model.parallelism" in text
    assert "gpu_capability=8.9 -> 45.0 GiB" in text
    assert "gpu_capability=9.0 -> 80.0 GiB" in text
    assert "including overlays that arrived through explicit USR merge carry" in text
    assert "passes `--overwrite` to `infer run`" in text
    assert "without implicitly pruning the namespace" in text
    assert "--mode fresh --allow-fresh-reset" in text
    assert "--no-discover-active-jobs" in text
    assert "operator-visible warning" in text
    assert "infer.overlay_guard.overlay_namespace` is fixed to `infer`" in text
    assert "densegen_batch_with_notify" in text
    assert "infer_batch_with_notify" in text
    assert "project: <project>" in text
    assert "project: dunlop" not in text
    assert "with_notify_slack" not in text
    assert "precedents" not in text


def test_repo_docs_index_exposes_ops_tool_and_operations_route() -> None:
    text = _read(_repo_root() / "docs" / "README.md")
    assert "### Workflow routes" in text
    assert "### Quick terms" in text
    assert "### New here?" in text
    assert "### Shell routes" not in text
    assert "### Workflow lanes" not in text
    assert "[Workflow routes](#workflow-routes)" in text
    assert "[Runbook catalog](runbooks/README.md)" in text
    assert "uv run ops catalog list --simple" in text
    assert "uv run ops catalog show <registry-id>" not in text
    assert "uv run ops progress explain <registry-id>" not in text
    assert "exact deep docs when declared" not in text
    assert "uv run ops catalog list --section tool-sources" not in text
    assert "uv run ops catalog list --related-to <registry-id>" not in text
    assert "uv run ops progress scaffold <registry-id> ..." not in text
    assert "[Ops orchestration index](operations/README.md)" in text
    assert "| `ops` | `uv run ops --help` | [ops README](../src/dnadesign/ops/README.md) |" in text


def test_runbook_catalog_covers_cross_tool_inventory_without_relocating_owners() -> None:
    text = _read(_repo_root() / "docs" / "runbooks" / "README.md")

    assert "## Runbook Catalog" in text
    assert "uv run ops catalog list" in text
    assert "uv run ops catalog list --simple" in text
    assert "### Shell decision table" in text
    assert "--plane data-plane --query infer" in text
    assert "--section tool-sources" in text
    assert "--related-to usr.data-plane.promoter-feature-matrix" in text
    assert "uv run ops catalog show <registry-id>" in text
    assert "required progress inputs" in text
    assert "exact deep docs when declared" in text
    assert "next shell commands" in text
    assert "uv run ops progress explain <registry-id>" in text
    assert "uv run ops progress show <registry-id> ..." in text
    assert "uv run ops progress scaffold <registry-id> ..." in text
    assert "uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix" in text
    assert "uv run ops progress campaign --manifest <manifest.yaml>" in text
    assert "prints to stdout unless you pass `--out`" in text
    assert "### Discovery shortcuts" in text
    assert "### Authoritative cross-tool procedures" in text
    assert "### Tool-local runbook sources" in text
    assert "### Progress surface glossary" in text
    assert "### Explicit campaign manifest shape" in text
    assert "### Boundary reminders" in text
    assert "ops.control-plane.orchestration" in text
    assert "usr.data-plane.hpc-sync" in text
    assert "usr.data-plane.chained-densegen-infer-sync" in text
    assert "usr.data-plane.multi-source-source-of-truth" in text
    assert "usr.data-plane.construct-infer-source-of-truth" in text
    assert "usr.data-plane.promoter-feature-matrix" in text
    assert "cluster.downstream.exploratory-clustering" in text
    assert "opal.downstream.usr-infer-x-active-learning" in text
    assert "../operations/orchestration-runbooks.md" in text
    assert "../../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md" in text
    assert "../../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md" in text
    assert "../../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md" in text
    assert "../../src/dnadesign/densegen/docs/README.md" in text
    assert "../../src/dnadesign/construct/docs/README.md" in text
    assert "../../src/dnadesign/infer/docs/README.md" in text
    assert "It does not replace the owner-local runbook or workflow" in text
    assert "drift is a docs-check failure" in text
    assert "This is still not an inferred global campaign engine." in text
    assert "Relative artifact paths in the manifest resolve from the manifest directory" in text
    assert "`ops-audit-json`" in text
    assert "`opal-campaign-state`" in text
    assert "Ops is not" not in text


def test_repo_root_readme_lists_ops_in_docs_and_tool_catalog() -> None:
    text = _read(_repo_root() / "README.md")
    assert "## New here?" not in text
    assert "uv run ops catalog list" not in text
    assert "uv run ops progress explain" not in text
    assert "[Docs index](docs/README.md)" in text
    assert "best place to start if you are orienting to the repo" in text
    assert "[Docs workflow routes](docs/README.md#workflow-routes)" not in text
    assert "including the downstream split between" not in text
    assert "[Ops operations](docs/operations/README.md)" not in text
    assert "[Notify operations](docs/notify/README.md)" not in text
    assert "[Workflow lanes](docs/README.md#workflow-lanes)" not in text
    assert (
        "[Cross-tool information architecture contract](ARCHITECTURE.md#cross-tool-information-architecture)"
        not in text
    )
    assert "[Boundary rules](DESIGN.md#toolpackage-boundaries)" not in text
    assert "| [**ops**](src/dnadesign/ops/README.md) |" in text
    assert "DenseGen/Infer + Notify batch workflows" not in text


def test_root_ops_row_is_tool_agnostic() -> None:
    text = _read(_repo_root() / "README.md")
    expected_row = (
        "| [**ops**](src/dnadesign/ops/README.md) | "
        "Runbook-driven orchestration for deterministic batch workflows across tools. |"
    )
    assert expected_row in text


def test_dev_docs_index_is_action_oriented() -> None:
    text = _read(_repo_root() / "docs" / "dev" / "README.md")
    _assert_token_order(
        text,
        [
            "## Developer Documentation",
            "### Start here",
            "### Day-to-day tasks",
            "### CI and quality checks",
            "### Planning and decisions",
        ],
        label="docs/dev/README.md",
    )
    assert "journal.md" in text
    assert "uv run python -m dnadesign.devtools.docs_checks --repo-root ." in text
    assert "for agents" not in text.lower()
    assert "for humans" not in text.lower()


def test_core_docs_avoid_contrived_doc_language() -> None:
    targets = [
        "README.md",
        "docs/README.md",
        "docs/dev/README.md",
        "docs/notify/README.md",
        "docs/operations/README.md",
        "src/dnadesign/ops/README.md",
        "src/dnadesign/ops/docs/README.md",
        "src/dnadesign/notify/README.md",
        "src/dnadesign/notify/docs/README.md",
        "src/dnadesign/usr/README.md",
        "src/dnadesign/usr/docs/README.md",
        "ARCHITECTURE.md",
        "DESIGN.md",
        "RELIABILITY.md",
        "QUALITY_SCORE.md",
    ]
    banned_tokens = ("progressive disclosure", "canonical", "for agents", "for humans")
    repo_root = _repo_root()
    for rel in targets:
        text = _read(repo_root / rel).lower()
        for token in banned_tokens:
            assert token not in text, f"{rel}: contains banned token {token!r}"


def test_ops_docs_remove_legacy_presets_and_workflow_alias_terms() -> None:
    docs_targets = [
        _repo_root() / "docs" / "operations" / "README.md",
        _repo_root() / "docs" / "operations" / "orchestration-runbooks.md",
        _repo_root() / "src" / "dnadesign" / "ops" / "README.md",
        _repo_root() / "src" / "dnadesign" / "ops" / "docs" / "README.md",
    ]
    for path in docs_targets:
        text = _read(path)
        assert "precedents" not in text
        assert "with_notify_slack" not in text
