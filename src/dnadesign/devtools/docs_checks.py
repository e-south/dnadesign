"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs_checks.py

Validates docs markdown naming, local links, and public interface doc contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from .ci_changes import discover_repo_tools

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
README_TOOL_LINK_PATTERN = re.compile(r"\[\*\*(?P<tool>[a-z0-9_-]+)\*\*\]\((?P<link>[^)]+)\)")
README_COVERAGE_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?P<link>[^)]+)\)")
TOOL_README_BANNER_PATTERN = re.compile(r"!\[[^\]]*banner[^\]]*\]\((?P<link>[^)]+)\)", flags=re.IGNORECASE)
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOOL_README_TOP_LINK_SCAN_LINES = 80
RUNBOOK_DEMO_SHELL_LANGS = {"bash", "sh", "zsh"}
RUNBOOK_DEMO_YAML_LANGS = {"yaml", "yml"}
RUNBOOK_DEMO_HEREDOC_PATTERN = re.compile(r"""<<[-~]?\s*['"]?([A-Za-z_][A-Za-z0-9_]*)['"]?""")
RUNBOOK_DEMO_YAML_VALUE_PATTERN = re.compile(r"^\s*(?:-\s*)?[A-Za-z0-9_\-]+\s*:\s*.+$")
RUNBOOK_DEMO_CONTROL_PREFIXES = (
    "if ",
    "then",
    "else",
    "elif ",
    "fi",
    "for ",
    "while ",
    "do",
    "done",
    "case ",
    "esac",
    "function ",
    "return ",
)
ROOT_MARKDOWN_FILES = (
    "README.md",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "DESIGN.md",
    "SECURITY.md",
    "RELIABILITY.md",
    "PLANS.md",
    "QUALITY_SCORE.md",
)
SOR_MARKDOWN_FILES = (
    "ARCHITECTURE.md",
    "DESIGN.md",
    "SECURITY.md",
    "RELIABILITY.md",
    "PLANS.md",
    "QUALITY_SCORE.md",
)
INDEX_MARKDOWN_FILES = (
    "docs/README.md",
    "docs/architecture/README.md",
    "docs/architecture/decisions/README.md",
    "docs/security/README.md",
    "docs/reliability/README.md",
    "docs/quality/README.md",
    "docs/exec-plans/README.md",
    "docs/templates/README.md",
    "docs/dev/README.md",
    "docs/bu-scc/README.md",
    "docs/notify/README.md",
)
RUNBOOK_MARKDOWN_FILES = (
    "docs/installation.md",
    "docs/dependencies.md",
    "docs/notebooks.md",
    "docs/marimo-reference.md",
    "docs/bu-scc/quickstart.md",
    "docs/bu-scc/install.md",
    "docs/bu-scc/batch-notify.md",
    "docs/notify/usr-events.md",
)
OWNER_PATTERN = re.compile(r"^\*\*Owner:\*\*\s*(.+?)\s*$", re.MULTILINE)
LAST_VERIFIED_PATTERN = re.compile(r"^\*\*Last verified:\*\*\s*(.+?)\s*$", re.MULTILINE)
TYPE_PATTERN = re.compile(r"^\*\*Type:\*\*\s*(.+?)\s*$", re.MULTILINE)
STATUS_PATTERN = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)
CREATED_PATTERN = re.compile(r"^\*\*Created:\*\*\s*(.+?)\s*$", re.MULTILINE)
SECTION_HEADING_PATTERN = re.compile(r"^#{2,6}\s+(.+?)\s*$", re.MULTILINE)
CHECKLIST_ITEM_PATTERN = re.compile(r"^\s*-\s*\[[ xX]\]\s+", re.MULTILINE)
PROGRESS_ITEM_TIMESTAMP_PATTERN = re.compile(r"^\s*-\s*\[[ xX]\]\s+\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}Z\)\s+.+$")
_EXEC_PLAN_STATUSES = {"active", "paused", "completed"}
_EXEC_PLAN_REQUIRED_SECTIONS = (
    "Purpose / Big Picture",
    "Progress",
    "Surprises & Discoveries",
    "Decision Log",
    "Outcomes & Retrospective",
    "Context and Orientation",
    "Plan of Work",
    "Concrete Steps",
    "Validation and Acceptance",
)
PUBLIC_INTERFACE_DOC_PATHS = (
    "src/dnadesign/cruncher/docs/demos",
    "src/dnadesign/cruncher/docs/reference/cli.md",
    "src/dnadesign/cruncher/workspaces",
    "src/dnadesign/densegen/README.md",
    "src/dnadesign/densegen/docs/howto",
    "src/dnadesign/densegen/docs/tutorials",
    "src/dnadesign/densegen/workspaces/README.md",
)
ABSOLUTE_DOC_PATH_TOKENS = ("/Users/", "/private/", "/tmp/", "/home/", "/var/", "C:\\")
INTERNAL_SOURCE_INREACH_PATTERN = re.compile(r"(?:dnadesign\.[a-z0-9_]+\.src\.|src/dnadesign/[a-z0-9_-]+/src/)")
ENTRYPOINT_MARKDOWN_FILES = ("README.md", "docs/README.md")
ENTRYPOINT_LOCAL_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<path>(?:\.\./|\.\/)?(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+)(?![A-Za-z0-9_./-])"
)
DENSEGEN_DOC_LANGUAGE_PATHS = (
    "src/dnadesign/densegen/README.md",
    "src/dnadesign/densegen/AGENTS.md",
    "src/dnadesign/densegen/docs",
    "src/dnadesign/densegen/workspaces",
)
DENSEGEN_DISALLOWED_TERM_PATTERN = re.compile(r"\bcanonical\b", flags=re.IGNORECASE)


def _collect_markdown_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        raise FileNotFoundError("docs/ directory is missing")

    docs_md_files = sorted(docs_root.rglob("*.md"))
    tool_docs_md_files = _collect_tool_docs_markdown_files(repo_root)
    all_md_files = list(docs_md_files)
    all_md_files.extend(tool_docs_md_files)
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
        for path in docs_root.rglob("*.md"):
            tool_docs.add(path)
    return sorted(tool_docs)


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


def _find_broken_links(md_files: list[Path]) -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    anchor_cache: dict[Path, set[str]] = {}
    for src in md_files:
        text = src.read_text(encoding="utf-8")
        for raw in LINK_PATTERN.findall(text):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target_rel, anchor = (link.split("#", 1) + [""])[:2]
            if not target_rel:
                target = src.resolve()
            else:
                target = (src.parent / target_rel).resolve()
            if not target.exists():
                broken.append((src, link))
                continue
            if anchor and target.suffix == ".md":
                if target not in anchor_cache:
                    anchor_cache[target] = _collect_markdown_anchors(target)
                if anchor not in anchor_cache[target]:
                    broken.append((src, f"{link} (missing anchor '{anchor}')"))
    return broken


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


def _is_valid_codecov_component_link(*, tool_name: str, link: str) -> bool:
    parsed = urlparse(link)
    if parsed.scheme != "https":
        return False
    if parsed.netloc not in {"codecov.io", "www.codecov.io", "app.codecov.io"}:
        return False
    if not parsed.path.startswith("/gh/"):
        return False
    component_values = parse_qs(parsed.query).get("component")
    if component_values is None:
        return False
    return any(value.strip() == tool_name for value in component_values)


def _find_readme_tool_catalog_issues(repo_root: Path) -> list[str]:
    readme_path = repo_root / "README.md"
    src_root = repo_root / "src" / "dnadesign"
    if not readme_path.exists() or not src_root.exists():
        return []

    repo_tools = discover_repo_tools(repo_root=repo_root)
    if not repo_tools:
        return []

    readme_text = readme_path.read_text(encoding="utf-8")
    rows = _readme_tool_table_rows(readme_text)
    if not rows:
        return [f"{readme_path}: section '## Available tools' must include a markdown tool table."]

    issues: list[str] = []
    declared_tools: set[str] = set()
    for row in rows:
        if len(row) < 3:
            issues.append(f"{readme_path}: tool table rows must include Tool, Description, and Coverage columns.")
            continue

        tool_cell = row[0]
        coverage_cell = row[2]
        match = README_TOOL_LINK_PATTERN.search(tool_cell)
        if match is None:
            issues.append(f"{readme_path}: tool cell must use [**tool**](src/dnadesign/tool) format ({tool_cell}).")
            continue

        tool_name = match.group("tool")
        tool_link = match.group("link")
        if tool_name in declared_tools:
            issues.append(f"{readme_path}: duplicate tool row for '{tool_name}'.")
            continue
        declared_tools.add(tool_name)

        expected_rel = Path("src") / "dnadesign" / tool_name / "README.md"
        if _normalize_relative_markdown_path(tool_link) != expected_rel.as_posix():
            issues.append(
                f"{readme_path}: tool '{tool_name}' must link to '{expected_rel.as_posix()}' (found '{tool_link}')."
            )

        tool_readme = (repo_root / tool_link).resolve()
        if not tool_readme.exists() or not tool_readme.is_file():
            issues.append(
                f"{readme_path}: tool '{tool_name}' link target does not exist as a markdown file: {tool_link}."
            )

        coverage_match = README_COVERAGE_LINK_PATTERN.search(coverage_cell)
        if coverage_match is None:
            issues.append(
                f"{readme_path}: coverage cell for '{tool_name}' must include a markdown link "
                "to a Codecov component URL."
            )
            continue
        coverage_link = coverage_match.group("link")
        if not _is_valid_codecov_component_link(tool_name=tool_name, link=coverage_link):
            issues.append(
                f"{readme_path}: coverage link for '{tool_name}' must target Codecov with query "
                f"'component={tool_name}' (found '{coverage_link}')."
            )

    missing_tools = sorted(repo_tools - declared_tools)
    extra_tools = sorted(declared_tools - repo_tools)
    if missing_tools:
        issues.append(f"{readme_path}: missing tool rows for: {', '.join(missing_tools)}.")
    if extra_tools:
        issues.append(f"{readme_path}: unknown tool rows not found in src/dnadesign: {', '.join(extra_tools)}.")
    return issues


def _find_root_docs_entrypoint_issues(repo_root: Path) -> list[str]:
    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        return []

    text = readme_path.read_text(encoding="utf-8")
    if "dnadesign banner" not in text.lower():
        return []

    linked_targets: set[str] = set()
    for raw in LINK_PATTERN.findall(text):
        link = raw.strip().split()[0]
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            continue
        linked_targets.add(_normalize_relative_markdown_path(target_rel))

    issues: list[str] = []
    if "docs/README.md" not in linked_targets:
        issues.append(f"{readme_path}: bannered root README must include a markdown link to docs/README.md.")
    return issues


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


def _find_docs_root_heading_style_issues(repo_root: Path) -> list[str]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        return []

    issues: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        headings = _collect_markdown_headings_outside_fences(path)
        if not headings:
            continue

        first_line_no, first_level, _ = headings[0]
        if first_level != 2:
            issues.append(f"{path}:{first_line_no}: docs root markdown must start with '## ' heading style.")

        level2_count = sum(1 for _, level, _ in headings if level == 2)
        if level2_count > 1:
            issues.append(
                f"{path}: docs root markdown should use a single level-2 heading; use level-3+ for subsections."
            )
    return issues


def _find_deprecated_docs_entrypoint_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    deprecated_start_here = repo_root / "docs" / "start-here.md"
    if deprecated_start_here.exists():
        issues.append(f"{deprecated_start_here}: deprecated docs shim must not exist; consolidate into docs/README.md.")

    check_paths = (repo_root / "README.md", repo_root / "docs" / "README.md")
    disallowed_targets = {"docs/start-here.md", "start-here.md"}
    for path in check_paths:
        if not path.exists():
            continue
        for raw in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
            link = raw.strip().split()[0]
            target_rel = link.split("#", 1)[0].strip()
            if target_rel in disallowed_targets:
                issues.append(f"{path}: must not link to docs/start-here.md; use docs/README.md.")
    return issues


def _find_entrypoint_local_path_literal_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for relative_path in ENTRYPOINT_MARKDOWN_FILES:
        path = repo_root / relative_path
        if not path.exists():
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        for line_no, line in enumerate(lines, start=1):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            line_without_links = LINK_PATTERN.sub("", line)
            for match in ENTRYPOINT_LOCAL_PATH_PATTERN.finditer(line_without_links):
                token = match.group("path").strip("()[]{}<>.,:;!?")
                if not token:
                    continue
                if token.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                repo_target = (repo_root / token).resolve()
                relative_target = (path.parent / token).resolve()
                if not repo_target.exists() and not relative_target.exists():
                    continue

                issues.append(
                    f"{path}:{line_no}: local path literal '{token}' should be a markdown hyperlink for navigation."
                )
    return issues


def _find_densegen_disallowed_term_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    targets = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=DENSEGEN_DOC_LANGUAGE_PATHS)
    for path in targets:
        content = path.read_text(encoding="utf-8")
        match = DENSEGEN_DISALLOWED_TERM_PATTERN.search(content)
        if match is None:
            continue
        line_no = content[: match.start()].count("\n") + 1
        issues.append(f"{path}:{line_no}: term '{match.group(0)}' is not allowed in DenseGen docs.")
    return issues


def _find_codecov_component_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []
    repo_tools = discover_repo_tools(repo_root=repo_root)
    if not repo_tools:
        return []

    codecov_path = repo_root / "codecov.yml"
    if not codecov_path.exists():
        return [f"{codecov_path}: missing Codecov configuration file."]

    try:
        config = yaml.safe_load(codecov_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{codecov_path}: unable to parse YAML ({exc})."]

    if not isinstance(config, dict):
        return [f"{codecov_path}: expected a top-level YAML mapping."]

    component_management = config.get("component_management")
    if not isinstance(component_management, dict):
        return [f"{codecov_path}: missing 'component_management' mapping."]

    default_rules = component_management.get("default_rules")
    if not isinstance(default_rules, dict):
        return [f"{codecov_path}: missing component_management.default_rules mapping."]
    statuses = default_rules.get("statuses")
    if not isinstance(statuses, list):
        return [f"{codecov_path}: missing component_management.default_rules.statuses list."]

    has_required_status = False
    for status in statuses:
        if not isinstance(status, dict):
            continue
        if (
            status.get("type") == "project"
            and status.get("target") == "auto"
            and status.get("if_ci_failed") == "error"
            and status.get("if_not_found") == "failure"
        ):
            has_required_status = True
            break
    if not has_required_status:
        return [
            f"{codecov_path}: component_management.default_rules.statuses must include a "
            "project status with target=auto, if_ci_failed=error, if_not_found=failure."
        ]

    individual_components = component_management.get("individual_components")
    if not isinstance(individual_components, list):
        return [f"{codecov_path}: missing component_management.individual_components list."]

    component_ids: set[str] = set()
    issues: list[str] = []

    for component in individual_components:
        if not isinstance(component, dict):
            issues.append(f"{codecov_path}: each individual component must be a mapping.")
            continue

        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            issues.append(f"{codecov_path}: each component must define a non-empty component_id.")
            continue
        if component_id in component_ids:
            issues.append(f"{codecov_path}: duplicate component_id '{component_id}'.")
            continue
        component_ids.add(component_id)

        paths = component.get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            issues.append(f"{codecov_path}: component '{component_id}' must define 'paths' as a list of strings.")
            continue
        expected_path = f"src/dnadesign/{component_id}/**"
        if expected_path not in paths:
            issues.append(f"{codecov_path}: component '{component_id}' must include path '{expected_path}'.")

    missing_components = sorted(repo_tools - component_ids)
    extra_components = sorted(component_ids - repo_tools)
    if missing_components:
        issues.append(f"{codecov_path}: missing component_id entries for: {', '.join(missing_components)}.")
    if extra_components:
        issues.append(
            f"{codecov_path}: unknown component_id entries not found in src/dnadesign: {', '.join(extra_components)}."
        )

    return issues


def _extract_metadata_field(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_iso_date(value: str, *, field_name: str, path: Path) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {field_name} must use YYYY-MM-DD.") from exc


def _find_sor_metadata_issues(repo_root: Path, *, max_age_days: int) -> list[str]:
    today = dt.date.today()
    issues: list[str] = []

    for name in SOR_MARKDOWN_FILES:
        path = repo_root / name
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        doc_type = _extract_metadata_field(text, TYPE_PATTERN)
        if doc_type is None:
            issues.append(f"{path}: missing '**Type:**' metadata field.")
            continue
        if doc_type != "system-of-record":
            issues.append(f"{path}: '**Type:**' must be exactly 'system-of-record'.")
            continue

        owner = _extract_metadata_field(text, OWNER_PATTERN)
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            continue
        if not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            continue
        if not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        age_days = (today - last_verified).days
        if age_days > max_age_days:
            issues.append(f"{path}: Last verified date is stale by {age_days} days (max allowed {max_age_days}).")

    return issues


def _find_index_metadata_issues(repo_root: Path, *, max_age_days: int) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=INDEX_MARKDOWN_FILES,
        max_age_days=max_age_days,
    )


def _find_runbook_metadata_issues(repo_root: Path, *, max_age_days: int) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=RUNBOOK_MARKDOWN_FILES,
        max_age_days=max_age_days,
    )


def _find_tool_docs_metadata_issues(repo_root: Path, *, max_age_days: int) -> list[str]:
    tool_docs = _collect_tool_docs_markdown_files(repo_root)
    return _find_owner_last_verified_metadata_issues_for_files(
        paths=tool_docs,
        max_age_days=max_age_days,
    )


def _find_owner_last_verified_metadata_issues(
    repo_root: Path,
    *,
    relative_paths: tuple[str, ...],
    max_age_days: int,
) -> list[str]:
    files = [repo_root / relative_path for relative_path in relative_paths]
    return _find_owner_last_verified_metadata_issues_for_files(
        paths=files,
        max_age_days=max_age_days,
    )


def _find_owner_last_verified_metadata_issues_for_files(
    *,
    paths: list[Path],
    max_age_days: int,
) -> list[str]:
    today = dt.date.today()
    issues: list[str] = []

    for path in paths:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        owner = _extract_metadata_field(text, OWNER_PATTERN)
        owner_valid = True
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            owner_valid = False
        elif not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        last_verified_valid = True
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            last_verified_valid = False
        elif not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            last_verified_valid = False

        if not owner_valid or not last_verified_valid:
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        age_days = (today - last_verified).days
        if age_days > max_age_days:
            issues.append(f"{path}: Last verified date is stale by {age_days} days (max allowed {max_age_days}).")

    return issues


def _find_exec_plan_metadata_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    exec_root = repo_root / "docs" / "exec-plans"
    if not exec_root.exists():
        return issues

    for lane_name in ("active", "completed"):
        lane_root = exec_root / lane_name
        if not lane_root.exists():
            continue
        for plan_path in sorted(lane_root.rglob("*.md")):
            if plan_path.name == "README.md":
                continue

            text = plan_path.read_text(encoding="utf-8")
            present_sections = {heading.strip() for heading in SECTION_HEADING_PATTERN.findall(text)}
            section_bodies = _extract_section_bodies(text)

            status = _extract_metadata_field(text, STATUS_PATTERN)
            if status is None:
                issues.append(f"{plan_path}: missing '**Status:**' metadata field.")
                continue
            if status not in _EXEC_PLAN_STATUSES:
                allowed_statuses = ", ".join(sorted(_EXEC_PLAN_STATUSES))
                issues.append(f"{plan_path}: invalid status '{status}' (expected one of: {allowed_statuses}).")
                continue
            if lane_name == "completed" and status != "completed":
                issues.append(f"{plan_path}: plans under completed/ must set status to 'completed'.")
            if lane_name == "active" and status == "completed":
                issues.append(f"{plan_path}: plans under active/ cannot set status to 'completed'.")

            owner = _extract_metadata_field(text, OWNER_PATTERN)
            if owner is None:
                issues.append(f"{plan_path}: missing '**Owner:**' metadata field.")
            elif not owner:
                issues.append(f"{plan_path}: '**Owner:**' must not be empty.")

            created_raw = _extract_metadata_field(text, CREATED_PATTERN)
            if created_raw is None:
                issues.append(f"{plan_path}: missing '**Created:**' metadata field.")
            elif not created_raw:
                issues.append(f"{plan_path}: '**Created:**' must not be empty.")
            else:
                try:
                    _parse_iso_date(created_raw, field_name="Created", path=plan_path)
                except ValueError as exc:
                    issues.append(str(exc))

            if not LINK_PATTERN.search(text):
                issues.append(f"{plan_path}: execution plans must include at least one markdown link for traceability.")

            missing_sections = [name for name in _EXEC_PLAN_REQUIRED_SECTIONS if name not in present_sections]
            if missing_sections:
                missing_csv = ", ".join(missing_sections)
                issues.append(f"{plan_path}: missing required execution-plan sections: {missing_csv}.")
                continue

            progress_body = section_bodies.get("Progress", "")
            if not CHECKLIST_ITEM_PATTERN.search(progress_body):
                issues.append(f"{plan_path}: '## Progress' must include checklist items (e.g., '- [ ] ...').")
            progress_items = [line for line in progress_body.splitlines() if CHECKLIST_ITEM_PATTERN.match(line)]
            for progress_item in progress_items:
                if PROGRESS_ITEM_TIMESTAMP_PATTERN.match(progress_item):
                    continue
                issues.append(
                    f"{plan_path}: progress checklist items must include a UTC timestamp "
                    f"in '(YYYY-MM-DD HH:MMZ)' format."
                )
                break

            for section_name, body in section_bodies.items():
                if section_name == "Progress":
                    continue
                if CHECKLIST_ITEM_PATTERN.search(body):
                    issues.append(
                        f"{plan_path}: checklist items are only allowed under "
                        f"'## Progress' (found in '## {section_name}')."
                    )

    return issues


def _extract_section_bodies(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = SECTION_HEADING_PATTERN.match(line)
        if match is not None:
            current = match.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _collect_public_interface_markdown_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()
    for rel in PUBLIC_INTERFACE_DOC_PATHS:
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


def _find_public_interface_doc_contract_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path in _collect_public_interface_markdown_files(repo_root):
        text = path.read_text(encoding="utf-8")
        for token in ABSOLUTE_DOC_PATH_TOKENS:
            if token in text:
                issues.append(
                    f"{path}: absolute filesystem path token '{token}' is not allowed; "
                    "use workspace-relative commands/paths."
                )
                break
        if INTERNAL_SOURCE_INREACH_PATTERN.search(text):
            issues.append(
                f"{path}: internal source inreach detected ('dnadesign.<tool>.src.*' or "
                "'src/dnadesign/<tool>/src/'); use public CLI/artifact contracts."
            )
    return issues


def _find_tool_readme_banner_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    issues: list[str] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if not readme_path.exists():
            continue

        text = readme_path.read_text(encoding="utf-8")
        top_block = "\n".join(text.splitlines()[:25])
        banner_match = TOOL_README_BANNER_PATTERN.search(top_block)
        if banner_match is None:
            issues.append(f"{readme_path}: missing top banner image markdown line with '* banner' alt text.")
            continue

        link = banner_match.group("link").strip().split()[0]
        parsed = urlparse(link)
        if parsed.scheme or link.startswith("mailto:") or not link.lower().endswith(".svg"):
            issues.append(f"{readme_path}: banner link must target a local .svg asset.")
            continue

        target_rel = link.split("#", 1)[0].strip()
        if not target_rel:
            issues.append(f"{readme_path}: banner link must include a relative asset path.")
            continue

        target_path = (readme_path.parent / target_rel).resolve()
        if not target_path.exists():
            issues.append(f"{readme_path}: banner asset target does not exist: {target_rel}.")

        if "placeholder" in top_block.lower():
            issues.append(f"{readme_path}: banner copy must not use placeholder wording.")

    return issues


def _find_tool_readme_structure_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    issues: list[str] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if not readme_path.exists():
            continue

        text = readme_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if not non_empty_indices:
            issues.append(f"{readme_path}: README is empty.")
            continue

        first_index = non_empty_indices[0]
        first_line = lines[first_index].strip()
        if TOOL_README_BANNER_PATTERN.search(first_line) is None:
            issues.append(f"{readme_path}: first non-empty line must be the banner image line.")
            continue

        next_index = next((idx for idx in non_empty_indices if idx > first_index), None)
        if next_index is not None and lines[next_index].lstrip().startswith("#"):
            issues.append(
                f"{readme_path}: line after the banner must be narrative text; avoid repeating a top title heading."
            )

        top_block = "\n".join(lines[:TOOL_README_TOP_LINK_SCAN_LINES])
        has_local_markdown_link = False
        for raw in LINK_PATTERN.findall(top_block):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_rel = link.split("#", 1)[0].strip()
            if target_rel.lower().endswith(".md"):
                has_local_markdown_link = True
                break
        if not has_local_markdown_link:
            issues.append(f"{readme_path}: top section must include a local markdown link to deeper documentation.")

    return issues


def _is_runbook_demo_doc(*, path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if "/archived/" in rel or "/prototypes/" in rel:
        return False
    if rel == "src/dnadesign/notify/docs/usr-events.md":
        return True
    if rel == "src/dnadesign/densegen/workspaces/catalog.md":
        return True
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check docs markdown naming and local links.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--max-sor-age-days",
        type=int,
        default=90,
        help="Maximum allowed age in days for root system-of-record Last verified dates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root

    try:
        docs_md_files, all_md_files = _collect_markdown_files(repo_root)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    bad_names = _find_bad_doc_names(docs_md_files)
    if bad_names:
        print("Docs naming check failed: use kebab-case markdown filenames.")
        for path in bad_names:
            print(f" - {path}")
        return 1

    sor_metadata_issues = _find_sor_metadata_issues(repo_root, max_age_days=args.max_sor_age_days)
    if sor_metadata_issues:
        print("Root system-of-record metadata check failed:")
        for issue in sor_metadata_issues:
            print(f" - {issue}")
        return 1

    index_metadata_issues = _find_index_metadata_issues(repo_root, max_age_days=args.max_sor_age_days)
    if index_metadata_issues:
        print("Docs index metadata check failed:")
        for issue in index_metadata_issues:
            print(f" - {issue}")
        return 1

    runbook_metadata_issues = _find_runbook_metadata_issues(repo_root, max_age_days=args.max_sor_age_days)
    if runbook_metadata_issues:
        print("Docs runbook metadata check failed:")
        for issue in runbook_metadata_issues:
            print(f" - {issue}")
        return 1

    tool_docs_metadata_issues = _find_tool_docs_metadata_issues(repo_root, max_age_days=args.max_sor_age_days)
    if tool_docs_metadata_issues:
        print("Tool docs metadata check failed:")
        for issue in tool_docs_metadata_issues:
            print(f" - {issue}")
        return 1

    interface_doc_issues = _find_public_interface_doc_contract_issues(repo_root)
    if interface_doc_issues:
        print("Public interface docs contract check failed:")
        for issue in interface_doc_issues:
            print(f" - {issue}")
        return 1

    densegen_disallowed_term_issues = _find_densegen_disallowed_term_issues(repo_root)
    if densegen_disallowed_term_issues:
        print("DenseGen docs language check failed:")
        for issue in densegen_disallowed_term_issues:
            print(f" - {issue}")
        return 1

    runbook_demo_snippet_issues = _find_runbook_demo_snippet_issues(repo_root)
    if runbook_demo_snippet_issues:
        print("Runbook/demo snippet annotation check failed:")
        for issue in runbook_demo_snippet_issues:
            print(f" - {issue}")
        return 1

    tool_readme_banner_issues = _find_tool_readme_banner_issues(repo_root)
    if tool_readme_banner_issues:
        print("Tool README banner contract check failed:")
        for issue in tool_readme_banner_issues:
            print(f" - {issue}")
        return 1

    tool_readme_structure_issues = _find_tool_readme_structure_issues(repo_root)
    if tool_readme_structure_issues:
        print("Tool README structure contract check failed:")
        for issue in tool_readme_structure_issues:
            print(f" - {issue}")
        return 1

    readme_tool_catalog_issues = _find_readme_tool_catalog_issues(repo_root)
    if readme_tool_catalog_issues:
        print("README tool catalog check failed:")
        for issue in readme_tool_catalog_issues:
            print(f" - {issue}")
        return 1

    root_docs_entrypoint_issues = _find_root_docs_entrypoint_issues(repo_root)
    if root_docs_entrypoint_issues:
        print("Root docs entrypoint check failed:")
        for issue in root_docs_entrypoint_issues:
            print(f" - {issue}")
        return 1

    deprecated_docs_entrypoint_issues = _find_deprecated_docs_entrypoint_issues(repo_root)
    if deprecated_docs_entrypoint_issues:
        print("Deprecated docs entrypoint check failed:")
        for issue in deprecated_docs_entrypoint_issues:
            print(f" - {issue}")
        return 1

    docs_root_heading_style_issues = _find_docs_root_heading_style_issues(repo_root)
    if docs_root_heading_style_issues:
        print("Docs root heading style check failed:")
        for issue in docs_root_heading_style_issues:
            print(f" - {issue}")
        return 1

    entrypoint_local_path_issues = _find_entrypoint_local_path_literal_issues(repo_root)
    if entrypoint_local_path_issues:
        print("Entrypoint local path hyperlink check failed:")
        for issue in entrypoint_local_path_issues:
            print(f" - {issue}")
        return 1

    codecov_component_issues = _find_codecov_component_issues(repo_root)
    if codecov_component_issues:
        print("Codecov component contract check failed:")
        for issue in codecov_component_issues:
            print(f" - {issue}")
        return 1

    exec_plan_issues = _find_exec_plan_metadata_issues(repo_root)
    if exec_plan_issues:
        print("Execution plan metadata check failed:")
        for issue in exec_plan_issues:
            print(f" - {issue}")
        return 1

    broken = _find_broken_links(all_md_files)
    if broken:
        print("Docs link check failed:")
        for src, link in broken:
            print(f" - {src}: {link}")
        return 1

    print(f"Docs checks passed ({len(all_md_files)} markdown files, including root system-of-record docs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
