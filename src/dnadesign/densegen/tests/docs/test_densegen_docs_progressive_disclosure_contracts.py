"""
--------------------------------------------------------------------------------
densegen project
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
    "study_stress_ethanol_cipro.md",
)
ANALYSIS_NOTEBOOK_COMMAND = {
    "demo_tfbs_baseline.md": 'uv run dense notebook run -c "$PWD/config.yaml"',
    "demo_sampling_baseline.md": 'pixi run dense notebook run -c "$PWD/config.yaml"',
    "study_constitutive_sigma_panel.md": 'pixi run dense notebook run -c "$PWD/config.yaml"',
    "study_stress_ethanol_cipro.md": 'pixi run dense notebook run -c "$PWD/config.yaml"',
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


def test_densegen_howto_guides_keep_scope_sentence() -> None:
    for path in sorted(HOWTO.glob("*.md")):
        text = _read(path)
        assert "Read it when" in text, f"{path}: missing scope sentence using 'Read it when'"


def test_densegen_tutorials_include_analysis_only_existing_outputs_path() -> None:
    for name in RUNBOOK_TUTORIALS:
        path = TUTORIALS / name
        text = _read(path)
        assert "### If outputs already exist (analysis-only)" in text
        assert "./runbook.sh --analysis-only" in text
        assert ANALYSIS_NOTEBOOK_COMMAND[name] in text


def test_stress_tutorial_exposes_core_batch_analysis_modes_and_resume_guardrails() -> None:
    path = TUTORIALS / "study_stress_ethanol_cipro.md"
    text = _read(path)
    _assert_token_order(
        text,
        [
            "#### Mode 1: Core generation run (interactive or OnDemand shell)",
            "#### Mode 2: BU SCC batch loop (target quota)",
            "#### Mode 3: Post-run analysis only",
        ],
        label=path.name,
    )
    assert "backend: GUROBI" in text
    assert "1,000,000" in text
    assert "Config changed beyond plan quotas." in text
    assert "dense run --resume --no-plot" in text
    assert "outputs/meta/run.lock" in text
    assert "usr maintenance merge" in text


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
