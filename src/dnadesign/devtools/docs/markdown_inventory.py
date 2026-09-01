"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/markdown_inventory.py

Markdown inventory for documentation validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dnadesign.devtools.ci.changes import discover_repo_tools
from dnadesign.devtools.docs.check_contracts import (
    LINK_PATTERN,
    MARKDOWN_HEADING_PATTERN,
    ROOT_MARKDOWN_FILES,
)


def _collect_markdown_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        raise FileNotFoundError("docs/ directory is missing")

    docs_md_files = _collect_visible_markdown_files(repo_root, docs_root)
    tool_docs_md_files = _collect_tool_docs_markdown_files(repo_root)
    tool_readme_md_files = _collect_tool_readme_markdown_files(repo_root)
    all_md_files = list(docs_md_files)
    all_md_files.extend(tool_docs_md_files)
    all_md_files.extend(tool_readme_md_files)
    for name in ROOT_MARKDOWN_FILES:
        path = repo_root / name
        if path.exists():
            all_md_files.append(path)
    deduped = sorted(set(all_md_files))
    return docs_md_files, deduped


def _collect_tool_docs_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    tool_docs: set[Path] = set()
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        docs_root = src_root / tool_name / "docs"
        if not docs_root.exists():
            continue
        for path in _collect_visible_markdown_files(repo_root, docs_root):
            tool_docs.add(path)
    return sorted(tool_docs)


def _collect_visible_markdown_files(repo_root: Path, root: Path) -> list[Path]:
    """Return tracked and nonignored-new Markdown below ``root``.

    Git is the public-tree authority when available. A non-repository fixture
    falls back to filesystem discovery so the checker remains usable in unit
    tests and extracted documentation trees.
    """

    try:
        relative_root = root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return []

    try:
        top_level_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        top_level_result = None

    if (
        top_level_result is None
        or top_level_result.returncode != 0
        or Path(top_level_result.stdout.strip()).resolve() != repo_root.resolve()
    ):
        return sorted(root.rglob("*.md"))

    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                relative_root.as_posix(),
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=False,
        )
    except OSError:
        raise RuntimeError("git ls-files failed while inventorying documentation") from None

    if result.returncode != 0:
        detail = os.fsdecode(result.stderr).strip()
        message = "git ls-files failed while inventorying documentation"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)

    files: set[Path] = set()
    for relative_path_bytes in result.stdout.split(b"\0"):
        if not relative_path_bytes:
            continue
        relative_path = os.fsdecode(relative_path_bytes)
        path = repo_root / relative_path
        if path.suffix == ".md" and path.is_file():
            files.add(path)
    return sorted(files)


def _collect_tool_readme_markdown_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    files: list[Path] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if readme_path.exists():
            files.append(readme_path)
    return files


def _collect_markdown_files_from_relative_paths(repo_root: Path, *, relative_paths: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for rel in relative_paths:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file() and target.suffix == ".md":
            files.add(target)
            continue
        if target.is_dir():
            for path in target.rglob("*.md"):
                files.add(path)
    return sorted(files)


def _find_bad_doc_names(docs_md_files: list[Path]) -> list[Path]:
    return [path for path in docs_md_files if "_" in path.name]


def _find_broken_links(md_files: list[Path], *, repo_root: Path | None = None) -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    anchor_cache: dict[Path, set[str]] = {}
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else None
    for src in md_files:
        text = _markdown_text_without_fenced_code(src.read_text(encoding="utf-8"))
        for raw in LINK_PATTERN.findall(text):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target_rel, anchor = (link.split("#", 1) + [""])[:2]
            if not target_rel:
                target = src.resolve()
            else:
                target = (src.parent / target_rel).resolve()
            if resolved_repo_root is not None:
                try:
                    target.relative_to(resolved_repo_root)
                except ValueError:
                    broken.append((src, f"{link} (local link escapes repository)"))
                    continue
            if not target.exists():
                broken.append((src, link))
                continue
            if anchor and target.suffix == ".md":
                if target not in anchor_cache:
                    anchor_cache[target] = _collect_markdown_anchors(target)
                if anchor not in anchor_cache[target]:
                    broken.append((src, f"{link} (missing anchor '{anchor}')"))
    return broken


def _markdown_text_without_fenced_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence_marker: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        marker = None
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"

        if marker is not None:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            lines.append("")
            continue

        lines.append("" if in_fence else raw_line)
    return "\n".join(lines)


def _collect_markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    for _, _, heading_text in _collect_markdown_headings_outside_fences(path):
        slug = _slugify_markdown_heading(heading_text)
        if not slug:
            continue
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        if count == 0:
            anchors.add(slug)
        else:
            anchors.add(f"{slug}-{count}")
    return anchors


def _slugify_markdown_heading(value: str) -> str:
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum() or char in {" ", "-", "_"}:
            chars.append(char)
    slug = "".join(chars).replace(" ", "-")
    return slug.strip("-")


def _extract_level2_section_lines(text: str, heading: str) -> list[str]:
    section_lines: list[str] = []
    in_section = False
    target = f"## {heading}"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == target:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section:
            section_lines.append(line)
    return section_lines


def _readme_tool_table_rows(text: str) -> list[list[str]]:
    section_lines = _extract_level2_section_lines(text, "Available tools")
    if not section_lines:
        return []
    rows: list[list[str]] = []
    for line in section_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if len(cells) >= 2 and cells[0].lower() == "tool" and cells[1].lower() == "description":
            continue
        if set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        rows.append(cells)
    return rows


def _normalize_relative_markdown_path(value: str) -> str:
    return str(Path(value).as_posix().lstrip("./"))


def _collect_markdown_headings_outside_fences(path: Path) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = MARKDOWN_HEADING_PATTERN.match(raw_line)
        if match is None:
            continue
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        headings.append((line_no, level, heading_text))
    return headings
