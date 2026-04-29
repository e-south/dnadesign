"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_docs_information_architecture_contracts.py

Information-architecture contract tests for infer docs progressive disclosure.

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


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def test_infer_top_readme_is_lightweight_router() -> None:
    readme = _read("src/dnadesign/infer/README.md")

    _assert_token_order(
        readme,
        [
            "## Documentation",
        ],
        label="src/dnadesign/infer/README.md",
    )
    assert "docs/README.md" in readme
    assert "docs/index.md" in readme
    assert "docs/getting-started/cli-quickstart.md" in readme
    assert "workspaces/README.md" in readme
    assert "docs/operations/evo2-sequence-features.md" in readme
    assert "docs/operations/pressure-test-agnostic-models.md" in readme
    assert "docs/reference/README.md" in readme
    assert "../../../docs/README.md" in readme
    assert "## Choose a task" not in readme
    assert "## Shared handoffs before infer runs" not in readme
    assert "## Entrypoint contract" not in readme
    assert "## Boundary reminder" not in readme


def test_infer_docs_readme_keeps_workflow_then_type_progressive_disclosure() -> None:
    docs_readme = _read("src/dnadesign/infer/docs/README.md")

    _assert_token_order(
        docs_readme,
        [
            "### Read order",
            "### Documentation by workflow",
            "### Shared dataset handoffs into infer",
            "### Documentation by type",
        ],
        label="src/dnadesign/infer/docs/README.md",
    )
    assert "getting-started/README.md" in docs_readme
    assert "getting-started/cli-quickstart.md" in docs_readme
    assert "operations/README.md" in docs_readme
    assert "../workspaces/README.md" in docs_readme
    assert "operations/evo2-sequence-features.md" in docs_readme
    assert "operations/pressure-test-agnostic-models.md" in docs_readme
    assert "reference/evo2-provider.md" in docs_readme
    assert "reference/feature-schema.md" in docs_readme
    assert "../workspaces/evo2_feature_bundle_smoke/README.md" in docs_readme
    assert "tutorials/demo_pressure_test_usr_ops_notify.md" in docs_readme
    assert "../../usr/docs/operations/multi-source-shared-dataset-assembly.md" in docs_readme
    assert "../../usr/docs/operations/construct-infer-shared-dataset-runbook.md" in docs_readme
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" in docs_readme
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" in docs_readme
    assert "../../cluster/docs/workflows/exploratory-clustering.md" in docs_readme
    assert "file or USR dataset" in docs_readme
    assert "data.location.kind: usr" in docs_readme
    assert "[Multi-source shared dataset assembly]" in docs_readme
    assert "[Construct -> USR -> Infer shared dataset runbook]" in docs_readme
    assert "[cluster exploratory clustering workflow]" in docs_readme
    assert "reference/README.md" in docs_readme
    assert "architecture/README.md" in docs_readme
    assert "dev/README.md" in docs_readme
    assert "dev/journal.md" in docs_readme
    pressure_test_section = docs_readme.split("#### Pressure-test agnostic model writes into USR", maxsplit=1)[1]
    pressure_test_section = pressure_test_section.split("#### Continue after infer-derived `X` exists", maxsplit=1)[0]
    assert "../../usr/docs/operations/multi-source-shared-dataset-assembly.md" not in pressure_test_section
    assert "../../usr/docs/operations/construct-infer-shared-dataset-runbook.md" not in pressure_test_section
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" not in pressure_test_section
    shared_handoff_section = docs_readme.split("### Shared dataset handoffs into infer", maxsplit=1)[1]
    shared_handoff_section = shared_handoff_section.split("### Documentation by type", maxsplit=1)[0]
    assert "../../cluster/docs/workflows/exploratory-clustering.md" not in shared_handoff_section
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" not in shared_handoff_section
    by_type = docs_readme.split("### Documentation by type", maxsplit=1)[1]
    assert "../../usr/docs/operations/multi-source-shared-dataset-assembly.md" not in by_type
    assert "../../usr/docs/operations/construct-infer-shared-dataset-runbook.md" not in by_type
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" not in by_type
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" not in by_type
    assert "../../cluster/docs/workflows/exploratory-clustering.md" not in by_type
    assert "[Section index](index.md)" in docs_readme


def test_infer_docs_index_exists_and_points_back_to_docs_readme() -> None:
    docs_index = _read("src/dnadesign/infer/docs/index.md")

    assert "docs/README.md" in docs_index or "README.md" in docs_index
    assert "### Getting started" in docs_index
    assert "### Tutorials" in docs_index
    assert "### Cross-tool handoff routes" in docs_index
    assert "### Operations" in docs_index
    assert "### Reference" in docs_index
    assert "### Developer notes" in docs_index
    assert "../../usr/docs/operations/multi-source-shared-dataset-assembly.md" in docs_index
    assert "../../usr/docs/operations/construct-infer-shared-dataset-runbook.md" in docs_index
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" in docs_index
    assert "../../cluster/docs/workflows/exploratory-clustering.md" in docs_index
    assert "cluster exploratory workflow" in docs_index
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" in docs_index
    assert "operations/evo2-sequence-features.md" in docs_index
    assert "reference/evo2-provider.md" in docs_index
    assert "reference/feature-schema.md" in docs_index


def test_infer_operations_index_links_pressure_test_demo_and_runbook() -> None:
    ops_index = _read("src/dnadesign/infer/docs/operations/README.md")
    assert "evo2-sequence-features.md" in ops_index
    assert "pressure-test-agnostic-models.md" in ops_index
    assert "../tutorials/demo_pressure_test_usr_ops_notify.md" in ops_index
    assert "scc-evo2-gpu-uv-runbook.md" in ops_index
    assert "../../../usr/docs/operations/construct-infer-shared-dataset-runbook.md" in ops_index


def test_infer_pressure_test_tutorial_covers_local_and_ops_paths() -> None:
    tutorial = _read("src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md")
    assert "uv run infer validate config --config" in tutorial
    assert "uv run infer workspace init --id demo_usr_pressure --profile usr-pressure" in tutorial
    assert 'export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets"' in tutorial
    assert "uv run infer run --config" in tutorial
    assert 'uv run infer prune --usr "$DATASET_ID" --usr-root "$USR_ROOT"' in tutorial
    assert "layer` values `mid` and `final`" in tutorial
    assert "uv run ops runbook init" in tutorial
    assert "uv run ops runbook execute" in tutorial
    assert 'export OPS_RUNBOOK="$WORKSPACE_ROOT/outputs/logs/ops/runbooks/infer-pressure.runbook.yaml"' in tutorial
    assert "--no-submit" in tutorial
    assert "--submit" in tutorial
    assert "uv run usr --root" in tutorial


def test_infer_workspaces_readme_mentions_workspace_inventory_command() -> None:
    workspaces = _read("src/dnadesign/infer/workspaces/README.md")
    assert "uv run infer workspace list" in workspaces


def test_stress_ethanol_cipro_workspace_readme_uses_repo_root_placeholder_for_runbook_plan() -> None:
    readme = _read("src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md")

    assert "--repo-root <repo-root>" in readme
    assert "/project/dunlop/esouth/dnadesign" not in readme


def test_infer_docs_excluding_journal_avoid_legacy_flat_module_paths() -> None:
    docs_root = _repo_root() / "src" / "dnadesign" / "infer" / "docs"
    legacy_tokens = [
        "src/dnadesign/infer/adapter_dispatch.py",
        "src/dnadesign/infer/adapter_runtime.py",
        "src/dnadesign/infer/cli_builders.py",
        "src/dnadesign/infer/cli_ingest.py",
        "src/dnadesign/infer/cli_requests.py",
        "src/dnadesign/infer/tests/test_",
    ]
    offenders: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        if path.resolve() == (docs_root / "dev" / "journal.md").resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in legacy_tokens):
            offenders.append(str(path.relative_to(_repo_root())))
    assert offenders == []


def test_infer_docs_examples_and_workspaces_avoid_hardcoded_personal_usr_roots() -> None:
    repo_root = _repo_root()
    targets = [
        repo_root / "src/dnadesign/infer/docs",
        repo_root / "src/dnadesign/infer/workspaces",
    ]
    offenders: list[str] = []
    for target in targets:
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == (repo_root / "src/dnadesign/infer/docs/dev/journal.md").resolve():
                continue
            if path.suffix not in {".md", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "/projectnb/dunlop/esouth/outputs/usr_datasets" in text:
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []
