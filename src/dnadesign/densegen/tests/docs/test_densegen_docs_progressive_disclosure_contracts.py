"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/docs/test_densegen_docs_progressive_disclosure_contracts.py

Contract checks that DenseGen operator docs keep progressive-disclosure structure.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
TUTORIALS = DOCS_ROOT / "tutorials"
HOWTO = DOCS_ROOT / "howto"

RUNBOOK_TUTORIALS = (
    "demo_tfbs_baseline.md",
    "demo_sampling_baseline.md",
    "study_constitutive_sigma_panel.md",
)
ANALYSIS_NOTEBOOK_COMMAND = {
    "demo_tfbs_baseline.md": 'uv run dense notebook run -c "$PWD/config.yaml"',
    "demo_sampling_baseline.md": 'pixi run dense notebook run -c "$PWD/config.yaml"',
    "study_constitutive_sigma_panel.md": 'pixi run dense notebook run -c "$PWD/config.yaml"',
}


def _read(path: Path) -> str:
    assert path.exists(), f"Missing markdown file: {path}"
    return path.read_text()


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def test_densegen_tutorials_keep_progressive_disclosure_flow() -> None:
    for name in RUNBOOK_TUTORIALS:
        path = TUTORIALS / name
        text = _read(path)
        _assert_token_order(
            text,
            [
                "### Runbook command",
                "### Prerequisites",
                "### Key config sections",
                "### Step-by-step commands",
                "### Expected outputs",
                "### Related docs",
            ],
            label=name,
        )


def test_densegen_usr_notify_tutorial_keeps_walkthrough_progression() -> None:
    path = TUTORIALS / "demo_usr_notify.md"
    text = _read(path)
    _assert_token_order(
        text,
        [
            "### What this tutorial demonstrates",
            "### Prerequisites",
            "### Key config knobs",
            "### Walkthrough",
            "### Expected outputs",
            "### Troubleshooting",
        ],
        label=path.name,
    )
    assert 'notify setup resolve-events --tool densegen --config "$CONFIG" --json' in text
    assert (
        'notify usr-events watch --tool densegen --config "$CONFIG" --provider generic --url "$NOTIFY_WEBHOOK" '
        "--dry-run --no-advance-cursor-on-dry-run" in text
    )
    assert (
        'notify setup slack --tool densegen --config "$CONFIG" --secret-source env --url-env NOTIFY_WEBHOOK' not in text
    )


def test_densegen_docs_route_to_shared_multi_source_runbook() -> None:
    text = _read(DOCS_ROOT / "README.md")
    assert "../../usr/docs/operations/assembly/multi-source-shared-dataset.md" in text
    assert "../../usr/docs/operations/promoter/characterization-feature-matrix.md" in text
    _assert_token_order(
        text,
        [
            "#### Run with Notify",
            "tutorials/demo_usr_notify.md",
            "concepts/runtime/observability-and-events.md",
            "#### Continue into shared downstream data-plane flows",
            "../../usr/docs/operations/assembly/multi-source-shared-dataset.md",
            "../../usr/docs/operations/promoter/characterization-feature-matrix.md",
        ],
        label="densegen/docs/README.md",
    )


def test_densegen_top_level_readme_routes_to_downstream_shared_flows() -> None:
    text = _read(ROOT / "README.md")
    assert "## Documentation" in text
    assert "docs/README.md" in text
    assert "workspaces/README.md" in text
    assert "docs/reference/cli.md" in text
    assert "## Start here" not in text
    assert "## Continue after generation" not in text
    assert "## Boundary reminder" not in text

    docs_text = _read(DOCS_ROOT / "README.md")
    assert "../../usr/docs/operations/assembly/multi-source-shared-dataset.md" in docs_text
    assert "../../usr/docs/operations/promoter/characterization-feature-matrix.md" in docs_text


def test_densegen_docs_index_keeps_cross_tool_handoff_routes_separate_from_tutorials() -> None:
    text = _read(DOCS_ROOT / "index.md")
    _assert_token_order(
        text,
        [
            "### Tutorials",
            "tutorials/demo_usr_notify.md",
            "### Cross-tool handoff routes",
            "../../usr/docs/operations/assembly/multi-source-shared-dataset.md",
            "../../usr/docs/operations/promoter/characterization-feature-matrix.md",
            "### Workspace docs",
        ],
        label="densegen/docs/index.md",
    )


def test_densegen_howto_guides_keep_scope_sentence() -> None:
    for path in sorted(HOWTO.glob("*.md")):
        text = _read(path)
        assert "Read it when" in text, f"{path}: missing scope sentence using 'Read it when'"


def test_densegen_tutorials_include_analysis_only_existing_outputs_path() -> None:
    for name in RUNBOOK_TUTORIALS:
        path = TUTORIALS / name
        text = _read(path)
        assert "### If outputs already exist (analysis mode)" in text
        assert "./runbook.sh --mode analysis" in text
        assert ANALYSIS_NOTEBOOK_COMMAND[name] in text


def test_hpc_howto_exposes_core_batch_and_analysis_flows() -> None:
    path = HOWTO / "hpc.md"
    text = _read(path)
    _assert_token_order(
        text,
        [
            "### Core generation flow (run shell or interactive session)",
            "### Scheduler submission flow (batch wrapper)",
            "### Post-run analysis flow",
        ],
        label=path.name,
    )
    assert "### Config-change guardrails for resume safety" in text
    assert "Config changed beyond plan quotas." in text
    assert "outputs/meta/run.lock" in text
