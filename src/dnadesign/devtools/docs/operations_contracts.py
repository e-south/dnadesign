"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/operations_contracts.py

Operations contracts for documentation validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from dnadesign.devtools.ci.changes import discover_repo_tools
from dnadesign.devtools.docs.check_contracts import (
    DISALLOWED_REPO_ROOT_OUTPUT_DIR_NAMES,
    DISALLOWED_SHARED_UTILS_PATHS,
    OPERATIONAL_RUNBOOK_SCAN_PRUNE_DIRS,
    OPS_DEPRECATED_SEMANTICS_DOC_PATHS,
    OPS_DEPRECATED_SEMANTICS_TERMS,
    OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES,
    OPS_OPERATIONAL_RUNBOOK_FALLBACK_SCAN_ROOTS,
    OPS_OPERATIONAL_WORKFLOW_IDS,
    OVERLAY_GUARD_DOC_PATHS,
    RUNBOOK_DEMO_CONTROL_PREFIXES,
    RUNBOOK_DEMO_HEREDOC_PATTERN,
    RUNBOOK_DEMO_SHELL_LANGS,
    RUNBOOK_DEMO_YAML_LANGS,
    RUNBOOK_DEMO_YAML_VALUE_PATTERN,
    STALE_OVERLAY_GUARD_TERMS,
    TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_files_from_relative_paths,
)


def _is_runbook_demo_doc(*, path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if "/archived/" in rel or "/prototypes/" in rel:
        return False
    if rel.endswith("/runbook.md"):
        return True
    if "/docs/demos/" in rel:
        return True
    if "/docs/tutorials/" in rel:
        return True
    if "/docs/workflows/" in rel:
        return True
    if "/docs/howto/" in rel:
        return True
    if "/docs/operations/" in rel:
        return True
    if "/campaigns/demo_" in rel and rel.endswith("/README.md"):
        return True
    if rel.endswith("/workspaces/README.md"):
        return True
    if rel.startswith("src/dnadesign/densegen/workspaces/") and rel.endswith("/README.md"):
        return True
    return False


def _collect_runbook_demo_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    files: list[Path] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        tool_root = src_root / tool_name
        if not tool_root.exists():
            continue
        for path in sorted(tool_root.rglob("*.md")):
            if _is_runbook_demo_doc(path=path, repo_root=repo_root):
                files.append(path)
    return files


def _is_shell_control_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped in {"{", "}", ";;", "in", "PY"}:
        return True
    if stripped.endswith(" then"):
        return True
    if stripped.endswith("{"):
        return True
    if any(stripped.startswith(prefix) for prefix in RUNBOOK_DEMO_CONTROL_PREFIXES):
        return True
    if stripped.startswith(("cruncher() {", "dense() {")):
        return True
    return False


def _find_runbook_demo_snippet_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []

    for path in _collect_runbook_demo_markdown_files(repo_root):
        lines = path.read_text(encoding="utf-8").splitlines()
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx]
            if not line.startswith("```"):
                line_idx += 1
                continue

            lang = line[3:].strip().lower()
            if lang not in RUNBOOK_DEMO_SHELL_LANGS and lang not in RUNBOOK_DEMO_YAML_LANGS:
                line_idx += 1
                continue

            block_start = line_idx + 1  # 1-based line number of the first block line.
            block_lines: list[str] = []
            line_idx += 1
            while line_idx < len(lines) and not lines[line_idx].startswith("```"):
                block_lines.append(lines[line_idx])
                line_idx += 1

            if lang in RUNBOOK_DEMO_SHELL_LANGS:
                heredoc_end: str | None = None
                for idx, raw in enumerate(block_lines):
                    stripped = raw.strip()

                    if heredoc_end is not None:
                        if stripped == heredoc_end:
                            heredoc_end = None
                        continue

                    if _is_shell_control_line(raw):
                        continue

                    prev_non_empty: str | None = None
                    for prev in range(idx - 1, -1, -1):
                        previous = block_lines[prev].strip()
                        if previous:
                            prev_non_empty = block_lines[prev]
                            break

                    if prev_non_empty is not None and prev_non_empty.rstrip().endswith("\\"):
                        continue

                    has_inline_comment = " #" in raw
                    prev_is_comment = prev_non_empty is not None and prev_non_empty.strip().startswith("#")
                    if not has_inline_comment and not prev_is_comment:
                        line_no = block_start + idx
                        issues.append(f"{path}:{line_no}: command in shell block needs an explanatory comment.")

                    heredoc_match = RUNBOOK_DEMO_HEREDOC_PATTERN.search(stripped)
                    if heredoc_match is not None:
                        heredoc_end = heredoc_match.group(1)

            if lang in RUNBOOK_DEMO_YAML_LANGS:
                for idx, raw in enumerate(block_lines):
                    stripped = raw.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if ":" not in raw:
                        continue
                    if not RUNBOOK_DEMO_YAML_VALUE_PATTERN.match(raw):
                        continue
                    _, value = raw.split(":", 1)
                    value_text = value.strip()
                    if not value_text or value_text in {"|", ">"}:
                        continue
                    if "#" in value:
                        continue
                    line_no = block_start + idx
                    issues.append(
                        f"{path}:{line_no}: yaml key/value in runbook/demo snippets needs a right-side inline comment."
                    )

    return issues


def _is_ops_operational_runbook_contract(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: operational runbook yaml is invalid: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{path}: operational runbook yaml is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        return False
    runbook = payload.get("runbook")
    if not isinstance(runbook, dict):
        return False
    workflow_id = runbook.get("workflow_id")
    if not isinstance(workflow_id, str):
        return False
    return workflow_id in OPS_OPERATIONAL_WORKFLOW_IDS


def _is_allowed_operational_runbook_path(*, relative_path: Path) -> bool:
    for prefix in OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES:
        if relative_path == prefix or prefix in relative_path.parents:
            return True
    parts = relative_path.parts
    if "outputs" in parts and "logs" in parts and "ops" in parts and "runbooks" in parts:
        return True
    return False


def _find_operational_runbook_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path in _iter_operational_runbook_candidate_yaml_files(repo_root):
        if not _is_ops_operational_runbook_contract(path):
            continue
        relative_path = path.relative_to(repo_root)
        if _is_allowed_operational_runbook_path(relative_path=relative_path):
            continue
        issues.append(
            f"{path}: operational runbook path is outside allowed locations; "
            "use workspace outputs/logs/ops/runbooks/ or src/dnadesign/ops/runbooks/presets/."
        )
    return issues


def _iter_operational_runbook_candidate_yaml_files(repo_root: Path):
    tracked_paths = _list_git_tracked_yaml_files(repo_root)
    if tracked_paths is not None:
        yield from tracked_paths
        return
    yield from _iter_bounded_operational_runbook_yaml_files(repo_root)


def _list_git_tracked_yaml_files(repo_root: Path) -> tuple[Path, ...] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", "*.yaml", "*.yml"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "").strip().lower()
        if "not a git repository" in stderr:
            return None
        raise ValueError(f"git ls-files failed while collecting tracked yaml candidates: {stderr or exc}") from exc

    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_line in completed.stdout.splitlines():
        relative = Path(str(raw_line).strip())
        if not relative.parts:
            continue
        candidate = repo_root / relative
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".yaml", ".yml"}:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(candidate)
    return tuple(candidates)


def _iter_bounded_operational_runbook_yaml_files(repo_root: Path):
    seen: set[Path] = set()
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted(repo_root.glob(suffix)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
    for relative_root in OPS_OPERATIONAL_RUNBOOK_FALLBACK_SCAN_ROOTS:
        target = repo_root / relative_root
        if not target.exists():
            continue
        for suffix in ("*.yaml", "*.yml"):
            for path in sorted(target.rglob(suffix)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def _should_descend_operational_runbook_dir(relative_parts: tuple[str, ...]) -> bool:
    if not relative_parts:
        return True
    name = relative_parts[-1]
    if name in OPERATIONAL_RUNBOOK_SCAN_PRUNE_DIRS:
        return False
    if "outputs" in relative_parts:
        outputs_idx = relative_parts.index("outputs")
        tail = relative_parts[outputs_idx + 1 :]
        if not tail:
            return True
        if tail[0] != "logs":
            return False
        if len(tail) == 1:
            return True
        if tail[1] != "ops":
            return False
        if len(tail) == 2:
            return True
        if tail[2] != "runbooks":
            return False
    return True


def _find_transient_operational_artifact_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for dir_name in DISALLOWED_REPO_ROOT_OUTPUT_DIR_NAMES:
        candidate = repo_root / dir_name
        if not candidate.exists():
            continue
        if not candidate.is_dir():
            continue
        if not any(candidate.iterdir()):
            continue
        issues.append(
            f"{candidate}: generated artifact directory is not allowed at repository root; "
            "use a tool or study workspace outputs/ root instead."
        )
    for dir_name in TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES:
        candidate = repo_root / dir_name
        if not candidate.exists():
            continue
        if not candidate.is_dir():
            continue
        if not any(candidate.iterdir()):
            continue
        issues.append(
            f"{candidate}: transient operational artifact directory is not allowed at repo root; "
            "use workspace-scoped outputs/logs/ops paths or /scratch for disposable working state."
        )
    return issues


def _find_shared_utils_path_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for relative_path in DISALLOWED_SHARED_UTILS_PATHS:
        candidate = repo_root / relative_path
        if not candidate.exists():
            continue
        issues.append(f"{candidate}: shared utils package is not allowed; keep utilities under src/dnadesign/<tool>/.")
    return issues


def _find_stale_overlay_guard_term_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=OVERLAY_GUARD_DOC_PATHS)
    for path in target_files:
        content = path.read_text(encoding="utf-8")
        for term in STALE_OVERLAY_GUARD_TERMS:
            if term in content:
                issues.append(
                    f"{path}: stale overlay guard term '{term}' is not allowed; "
                    "use usr-overlay-guard and overlay_namespace."
                )
    return issues


def _find_ops_deprecated_semantics_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    target_files = _collect_markdown_files_from_relative_paths(
        repo_root,
        relative_paths=OPS_DEPRECATED_SEMANTICS_DOC_PATHS,
    )
    for path in target_files:
        content = path.read_text(encoding="utf-8")
        for term in OPS_DEPRECATED_SEMANTICS_TERMS:
            if term in content:
                issues.append(
                    f"{path}: deprecated ops semantics term '{term}' is not allowed; "
                    "use transport-neutral workflow ids and the presets surface only."
                )
    return issues
