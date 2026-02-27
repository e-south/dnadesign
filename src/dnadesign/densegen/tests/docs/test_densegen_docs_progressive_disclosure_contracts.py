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
