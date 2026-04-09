"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workspace_service.py

Scaffold the payload-centric YIU workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.cruncher.app.yiu_workspace_blueprints import (
    ADVANCED_SPEC_FILENAME,
    DEFAULT_DEMO_SEQUENCE,
    DEMO_SPEC_FILENAME,
    PWM_CONTEXT_FILENAME,
    STARTER_JUNCTION_MODES,
    advanced_pwm_spec_text,
    canonical_spec_text,
    example_pwm_context_text,
    runbook_markdown,
    runbook_payload,
)

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXACT_DNA_SEQUENCE_RE = re.compile(r"^[ACGT]+$")


@dataclass(frozen=True)
class YiuWorkspaceScaffoldResult:
    workspace_root: Path
    runbook_path: Path
    runbook_doc_path: Path
    spec_path: Path


def _workspace_gitignore_text() -> str:
    return "\n".join(
        [
            ".cruncher/",
            "outputs/",
            ".DS_Store",
            "",
        ]
    )


def _repo_root_from(start: Path) -> Path | None:
    cursor = start.resolve()
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_cruncher_workspaces_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is None:
        raise ValueError(
            "Unable to determine the standard Cruncher workspaces root. Pass --root or --output explicitly."
        )
    return (repo_root / "src" / "dnadesign" / "cruncher" / "workspaces").resolve()


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("YIU workspace name must be non-empty.")
    if "/" in raw or "\\" in raw:
        raise ValueError("YIU workspace name must be a simple directory name or use --output.")
    if _WORKSPACE_NAME_RE.fullmatch(raw) is None:
        raise ValueError(f"Invalid YIU workspace name: {raw!r}.")
    return raw


def yiu_workspace_path(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    parent = default_cruncher_workspaces_root() if root is None else Path(root).expanduser().resolve()
    return parent / workspace_name


def _validate_starter_sequence(sequence: str) -> str:
    normalized = str(sequence).strip().upper()
    if not normalized:
        return DEFAULT_DEMO_SEQUENCE
    if _EXACT_DNA_SEQUENCE_RE.fullmatch(normalized) is None:
        raise ValueError("YIU starter sequence must use exact A/C/G/T bases only.")
    return normalized


def _validate_starter_junction_mode(junction_mode: str) -> str:
    normalized = str(junction_mode).strip()
    if normalized not in STARTER_JUNCTION_MODES:
        allowed = ", ".join(STARTER_JUNCTION_MODES)
        raise ValueError(f"Unsupported YIU starter junction mode {normalized!r}. Expected one of: {allowed}.")
    return normalized


def init_yiu_workspace(
    workspace_root: Path,
    *,
    force_overwrite: bool = False,
    sequence: str = DEFAULT_DEMO_SEQUENCE,
    junction_mode: str = "center_locked",
) -> YiuWorkspaceScaffoldResult:
    resolved_root = Path(workspace_root).expanduser().resolve()
    workspace_name = resolved_root.name
    starter_sequence = _validate_starter_sequence(sequence)
    starter_junction_mode = _validate_starter_junction_mode(junction_mode)
    if resolved_root.exists():
        if not force_overwrite:
            raise ValueError(f"YIU workspace already exists: {resolved_root}")
        shutil.rmtree(resolved_root)
    (resolved_root / "configs" / "yiu").mkdir(parents=True, exist_ok=True)
    (resolved_root / "outputs").mkdir(parents=True, exist_ok=True)
    (resolved_root / "motifs").mkdir(parents=True, exist_ok=True)

    spec_path = resolved_root / "configs" / "yiu" / DEMO_SPEC_FILENAME
    spec_path.write_text(
        canonical_spec_text(sequence=starter_sequence, junction_mode=starter_junction_mode),
        encoding="utf-8",
    )

    advanced_spec_path = resolved_root / "configs" / "yiu" / ADVANCED_SPEC_FILENAME
    advanced_spec_path.write_text(advanced_pwm_spec_text(), encoding="utf-8")

    pwm_context_path = resolved_root / "motifs" / PWM_CONTEXT_FILENAME
    pwm_context_path.write_text(example_pwm_context_text(), encoding="utf-8")

    gitignore_path = resolved_root / ".gitignore"
    gitignore_path.write_text(_workspace_gitignore_text(), encoding="utf-8")

    runbook_path = resolved_root / "configs" / "runbook.yaml"
    runbook_path.write_text(yaml.safe_dump(runbook_payload(workspace_name), sort_keys=False), encoding="utf-8")

    runbook_doc_path = resolved_root / "runbook.md"
    workspace_display_path = resolved_root.relative_to(_repo_root_from(resolved_root) or resolved_root.parent)
    runbook_doc_path.write_text(
        runbook_markdown(workspace_name=workspace_name, workspace_display_path=workspace_display_path),
        encoding="utf-8",
    )

    return YiuWorkspaceScaffoldResult(
        workspace_root=resolved_root,
        runbook_path=runbook_path,
        runbook_doc_path=runbook_doc_path,
        spec_path=spec_path,
    )
