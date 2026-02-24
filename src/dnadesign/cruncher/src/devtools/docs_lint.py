"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/devtools/docs_lint.py

Lint Cruncher docs for catalog completeness, structure, links, schema contract
consistency, and workspace runbook step coupling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.cruncher.devtools.docs_ia import catalog_path, docs_root, load_docs_catalog, workspaces_root

_LAST_UPDATED_PATTERN = re.compile(
    r"^\s*\*\*Last updated by:\*\*\s+cruncher-maintainers on \d{4}-\d{2}-\d{2}\s*$",
    re.MULTILINE,
)
_TOC_OPTOUT = "<!-- docs:toc:off -->"
_CONTENTS_PATTERN = re.compile(r"^#{2,4}\s+Contents\b", re.MULTILINE)
_NEXT_STEPS_PATTERN = re.compile(r"^#{2,4}\s+Next steps\b", re.MULTILINE)
_YAML_FENCE_PATTERN = re.compile(r"```(?:yaml|yml)\n(.*?)```", re.DOTALL | re.IGNORECASE)
_SCHEMA_PATTERN = re.compile(r"^\s*schema_version\s*:\s*(\d+)\s*$", re.MULTILINE)
_WORKSPACE_PATH_PATTERN = re.compile(r"workspaces/([A-Za-z0-9_.-]+)/")
_STEP_PATTERN = re.compile(r"--step\s+([A-Za-z0-9_.-]+)")
_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class LintIssue:
    path: Path
    rule_id: str
    message: str
    fix: str

    def render(self, repo_root: Path) -> str:
        rel = self.path.relative_to(repo_root)
        return f"{rel}: {self.rule_id}: {self.message} | fix: {self.fix}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _docs_files() -> list[Path]:
    root = docs_root()
    files = sorted(root.rglob("*.md"))
    return [path for path in files if "meta/templates" not in path.as_posix()]


def _catalog_paths() -> set[Path]:
    root = docs_root()
    catalog = load_docs_catalog()
    paths: set[Path] = set()
    for page in catalog.pages:
        rel = Path(page.path)
        if rel.is_absolute():
            raise ValueError(f"Catalog path must be relative: {page.path}")
        paths.add((root / rel).resolve())
    return paths


def _lint_catalog() -> list[LintIssue]:
    issues: list[LintIssue] = []
    repo_root = _repo_root()
    files = _docs_files()
    catalog_paths = _catalog_paths()

    for path in files:
        resolved = path.resolve()
        if resolved not in catalog_paths:
            issues.append(
                LintIssue(
                    path=path,
                    rule_id="CAT001",
                    message="Doc file missing from docs catalog.",
                    fix=f"Add `{path.relative_to(docs_root())}` to `{catalog_path().relative_to(repo_root)}`.",
                )
            )
    for path in sorted(catalog_paths):
        if not path.exists():
            issues.append(
                LintIssue(
                    path=catalog_path(),
                    rule_id="CAT002",
                    message=f"Catalog entry points to missing file: {path.relative_to(docs_root())}",
                    fix="Remove stale entry or create the missing doc file.",
                )
            )
    return issues


def _lint_structure() -> list[LintIssue]:
    issues: list[LintIssue] = []
    for path in _docs_files():
        text = path.read_text(encoding="utf-8")
        head = "\n".join(text.splitlines()[:40])
        if not _LAST_UPDATED_PATTERN.search(head):
            issues.append(
                LintIssue(
                    path=path,
                    rule_id="DOC001",
                    message="Missing or invalid top status line.",
                    fix="Add `**Last updated by:** cruncher-maintainers on YYYY-MM-DD` near the top.",
                )
            )
        if not _CONTENTS_PATTERN.search(text) and _TOC_OPTOUT not in text:
            issues.append(
                LintIssue(
                    path=path,
                    rule_id="DOC002",
                    message="Missing `Contents` section without explicit toc opt-out.",
                    fix=f"Add a `Contents` heading or include `{_TOC_OPTOUT}`.",
                )
            )
        if _NEXT_STEPS_PATTERN.search(text):
            issues.append(
                LintIssue(
                    path=path,
                    rule_id="DOC003",
                    message="Contrived `Next steps` section is not allowed.",
                    fix="Remove the `Next steps` heading and keep action links in flow-specific sections.",
                )
            )
    return issues


def _slugify_heading(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[`*_]", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-")


def _heading_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                slugs.add(_slugify_heading(heading))
    return slugs


def _lint_links() -> list[LintIssue]:
    issues: list[LintIssue] = []
    for path in _docs_files():
        text = path.read_text(encoding="utf-8")
        for match in _LINK_PATTERN.finditer(text):
            target = match.group(1).strip()
            if not target:
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("/"):
                issues.append(
                    LintIssue(
                        path=path,
                        rule_id="LNK001",
                        message=f"Absolute link target is not allowed: {target}",
                        fix="Use relative markdown links within Cruncher docs.",
                    )
                )
                continue
            target_path, _, anchor = target.partition("#")
            if not target_path:
                target_file = path
            else:
                target_file = (path.parent / target_path).resolve()
            if not target_file.exists():
                issues.append(
                    LintIssue(
                        path=path,
                        rule_id="LNK002",
                        message=f"Broken relative link target: {target}",
                        fix=f"Update link to an existing path relative to `{path.parent}`.",
                    )
                )
                continue
            if anchor:
                slug = _slugify_heading(anchor)
                if slug not in _heading_slugs(target_file):
                    issues.append(
                        LintIssue(
                            path=path,
                            rule_id="LNK003",
                            message=f"Anchor not found in target doc: {target}",
                            fix="Update the anchor to match an existing heading slug.",
                        )
                    )
    return issues


def _lint_schema_mentions() -> list[LintIssue]:
    issues: list[LintIssue] = []
    for path in _docs_files():
        text = path.read_text(encoding="utf-8")
        for block in _YAML_FENCE_PATTERN.findall(text):
            found_versions = _SCHEMA_PATTERN.findall(block)
            if not found_versions:
                continue
            for found in found_versions:
                if found != "3":
                    issues.append(
                        LintIssue(
                            path=path,
                            rule_id="SCH001",
                            message=f"Found non-v3 schema_version mention: {found}",
                            fix="Update docs examples to schema_version: 3.",
                        )
                    )
            if not re.search(r"^\s*(cruncher|study|portfolio)\s*:", block, re.MULTILINE):
                issues.append(
                    LintIssue(
                        path=path,
                        rule_id="SCH002",
                        message="YAML config example mentions schema_version but no recognized root key.",
                        fix="Wrap config examples with one of: `cruncher:`, `study:`, or `portfolio:`.",
                    )
                )
    return issues


def _runbook_step_ids(workspace_name: str) -> set[str]:
    runbook_path = workspaces_root() / workspace_name / "configs" / "runbook.yaml"
    if not runbook_path.exists():
        raise FileNotFoundError(f"Workspace runbook not found: {runbook_path}")
    payload = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runbook payload: {runbook_path}")
    runbook = payload.get("runbook")
    if not isinstance(runbook, dict):
        raise ValueError(f"Runbook missing root key `runbook`: {runbook_path}")
    steps = runbook.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Runbook steps must be a list: {runbook_path}")
    ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Runbook step must be mapping: {runbook_path}")
        step_id = str(step.get("id", "")).strip()
        if step_id:
            ids.add(step_id)
    return ids


def _lint_runbook_coupling() -> list[LintIssue]:
    issues: list[LintIssue] = []
    for path in _docs_files():
        text = path.read_text(encoding="utf-8")
        workspaces = sorted(set(_WORKSPACE_PATH_PATTERN.findall(text)))
        if len(workspaces) != 1:
            continue
        workspace = workspaces[0]
        runbook_path = workspaces_root() / workspace / "configs" / "runbook.yaml"
        if "configs/runbook.yaml" in text and not runbook_path.exists():
            issues.append(
                LintIssue(
                    path=path,
                    rule_id="RBK001",
                    message=f"Doc references runbook but workspace runbook is missing: {workspace}",
                    fix=f"Create `{runbook_path.relative_to(_repo_root())}` or fix workspace path in docs.",
                )
            )
            continue
        step_ids = _runbook_step_ids(workspace)
        for step in _STEP_PATTERN.findall(text):
            token = step.strip()
            if not token or "<" in token or ">" in token:
                continue
            if token not in step_ids:
                issues.append(
                    LintIssue(
                        path=path,
                        rule_id="RBK002",
                        message=f"Unknown runbook step id `{token}` for workspace `{workspace}`.",
                        fix=f"Use one of the step IDs defined in `{runbook_path.relative_to(_repo_root())}`.",
                    )
                )
    return issues


def run_lint() -> list[LintIssue]:
    issues: list[LintIssue] = []
    issues.extend(_lint_catalog())
    issues.extend(_lint_structure())
    issues.extend(_lint_links())
    issues.extend(_lint_schema_mentions())
    issues.extend(_lint_runbook_coupling())
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint Cruncher docs IA and coupling contracts.")
    parser.parse_args(argv)
    issues = run_lint()
    if issues:
        repo_root = _repo_root()
        print("Cruncher docs lint failed:")
        for issue in issues:
            print(f"- {issue.render(repo_root)}")
        return 1
    print("Cruncher docs lint passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
