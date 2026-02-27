"""
--------------------------------------------------------------------------------
densegen project
src/dnadesign/densegen/tests/docs/test_densegen_docs_snippet_contracts.py

Contract checks for DenseGen runbook/tutorial snippet comment and YAML annotation
style in operator-facing docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
WORKSPACES = ROOT / "workspaces"

RUNBOOK_DOCS = (
    DOCS_ROOT / "howto" / "hpc.md",
    DOCS_ROOT / "howto" / "bu-scc.md",
    DOCS_ROOT / "howto" / "cruncher_pwm_pipeline.md",
    WORKSPACES / "demo_tfbs_baseline" / "runbook.md",
    WORKSPACES / "demo_sampling_baseline" / "runbook.md",
    WORKSPACES / "study_constitutive_sigma_panel" / "runbook.md",
    WORKSPACES / "study_stress_ethanol_cipro" / "runbook.md",
)

DEMO_DOCS = (
    DOCS_ROOT / "tutorials" / "demo_tfbs_baseline.md",
    DOCS_ROOT / "tutorials" / "demo_sampling_baseline.md",
    DOCS_ROOT / "tutorials" / "study_constitutive_sigma_panel.md",
    DOCS_ROOT / "tutorials" / "study_stress_ethanol_cipro.md",
    DOCS_ROOT / "tutorials" / "demo_usr_notify.md",
)

SHELL_FENCE_PATTERN = re.compile(r"```bash\n(?P<body>.*?)\n```", flags=re.DOTALL)
YAML_FENCE_PATTERN = re.compile(r"```yaml\n(?P<body>.*?)\n```", flags=re.DOTALL)
ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
COMMAND_PATTERN = re.compile(r"^(?:\./|uv\b|pixi\b|mkdir\b|cp\b|cd\b|export\b|ls\b|python\b)")


def _read(path: Path) -> str:
    assert path.exists(), f"missing markdown file: {path}"
    return path.read_text()


def _extract_fenced_blocks(text: str, *, pattern: re.Pattern[str]) -> list[list[str]]:
    return [match.group("body").splitlines() for match in pattern.finditer(text)]


def _extract_indented_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("    "):
            current.append(raw[4:])
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _iter_shell_blocks(path: Path) -> Iterable[list[str]]:
    text = _read(path)
    yield from _extract_fenced_blocks(text, pattern=SHELL_FENCE_PATTERN)
    if path.parent.name.startswith(("demo_", "study_")) and path.name == "runbook.md":
        yield from _extract_indented_blocks(text)


def _looks_like_command(line: str) -> bool:
    stripped = line.strip()
    if stripped == "":
        return False
    if stripped.startswith("#"):
        return False
    if stripped in {"fi", "then", "do", "done", "else", "esac"}:
        return False
    if stripped.startswith(("if ", "for ", "while ", "case ")):
        return True
    if ASSIGNMENT_PATTERN.match(stripped):
        return True
    return COMMAND_PATTERN.match(stripped) is not None


def _assert_shell_block_has_command_comments(*, path: Path, block: list[str], block_label: str) -> None:
    previous_nonempty = ""
    heredoc_end: str | None = None
    for idx, raw in enumerate(block, start=1):
        stripped = raw.strip()
        if stripped == "":
            continue
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            previous_nonempty = stripped
            continue
        if stripped.startswith("#"):
            previous_nonempty = stripped
            continue
        if "<<" in stripped and _looks_like_command(stripped):
            marker = stripped.split("<<", 1)[1].strip()
            marker = marker.strip("'").strip('"')
            heredoc_end = marker or None
        if _looks_like_command(stripped):
            assert previous_nonempty.startswith("#"), (
                f"{path}:{block_label}:{idx}: command lacks contextual comment: {stripped!r}"
            )
        previous_nonempty = stripped


def _assert_yaml_block_has_inline_comments(*, path: Path, block: list[str], block_label: str) -> None:
    for idx, raw in enumerate(block, start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        if stripped.endswith(":"):
            continue
        if " #" not in raw:
            raise AssertionError(f"{path}:{block_label}:{idx}: yaml value line missing inline comment: {stripped!r}")
        comment_sep = raw.split("#", 1)[0]
        assert comment_sep.endswith("  "), (
            f"{path}:{block_label}:{idx}: yaml inline comments must be separated by two spaces"
        )


def test_densegen_runbook_and_demo_shell_snippets_comment_each_command() -> None:
    targets = sorted(set(RUNBOOK_DOCS) | set(DEMO_DOCS))
    for path in targets:
        for block_idx, block in enumerate(_iter_shell_blocks(path), start=1):
            _assert_shell_block_has_command_comments(path=path, block=block, block_label=f"shell-block-{block_idx}")


def test_densegen_demo_yaml_snippets_keep_inline_context_comments() -> None:
    for path in DEMO_DOCS:
        blocks = _extract_fenced_blocks(_read(path), pattern=YAML_FENCE_PATTERN)
        for block_idx, block in enumerate(blocks, start=1):
            _assert_yaml_block_has_inline_comments(path=path, block=block, block_label=f"yaml-block-{block_idx}")
