"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/public_surface_contracts.py

Public surface contracts for documentation validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.devtools.ci.changes import discover_repo_tools
from dnadesign.devtools.docs.banner_contracts import (
    _top_rendered_readme_banner,
)
from dnadesign.devtools.docs.check_contracts import (
    ABSOLUTE_DOC_PATH_TOKENS,
    AGENTS_CODE_SPAN_PATTERN,
    AGENTS_NEGATIVE_LINE_MARKERS,
    AGENTS_PATH_LITERAL_EXCEPTIONS,
    AGENTS_REPO_RELATIVE_PREFIXES,
    AGENTS_ROOT_FILENAMES,
    CONSTRUCT_LEGACY_OPERATOR_PATTERNS,
    CONSTRUCT_OPERATOR_DOC_PATHS,
    DENSEGEN_DISALLOWED_TERM_PATTERN,
    DENSEGEN_DOC_LANGUAGE_PATHS,
    ENTRYPOINT_LOCAL_PATH_PATTERN,
    ENTRYPOINT_MARKDOWN_FILES,
    INTERNAL_SOURCE_INREACH_PATTERN,
    LEGACY_CONTRACT_SURFACE_DOC_PATTERNS,
    LINK_PATTERN,
    PUBLIC_INTERFACE_DOC_PATHS,
    README_TOOL_CATALOG_EXCLUDED_TOOLS,
    README_TOOL_COMPONENT_COVERAGE_PATTERN,
    README_TOOL_LINK_PATTERN,
    TOOL_README_MAX_LINES,
    TOOL_README_SELF_REFERENTIAL_INTRO_PATTERN,
    TOOL_README_TOP_LINK_SCAN_LINES,
)
from dnadesign.devtools.docs.document_metadata import (
    _markdown_body_without_frontmatter,
)
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_markdown_files,
    _collect_markdown_files_from_relative_paths,
    _collect_markdown_headings_outside_fences,
    _collect_tool_readme_markdown_files,
    _extract_level2_section_lines,
    _normalize_relative_markdown_path,
    _readme_tool_table_rows,
)


def _find_readme_tool_catalog_issues(repo_root: Path) -> list[str]:
    readme_path = repo_root / "README.md"
    src_root = repo_root / "src" / "dnadesign"
    if not readme_path.exists() or not src_root.exists():
        return []

    repo_tools = discover_repo_tools(repo_root=repo_root) - README_TOOL_CATALOG_EXCLUDED_TOOLS
    if not repo_tools:
        return []

    readme_text = readme_path.read_text(encoding="utf-8")
    rows = _readme_tool_table_rows(readme_text)
    if not rows:
        return [f"{readme_path}: section '## Available tools' must include a markdown tool table."]

    issues: list[str] = []
    available_tools_text = "\n".join(_extract_level2_section_lines(readme_text, "Available tools"))
    if README_TOOL_COMPONENT_COVERAGE_PATTERN.search(available_tools_text):
        issues.append(f"{readme_path}: tool catalog must not repeat per-tool Codecov badges or component links.")
    declared_tools: set[str] = set()
    for row in rows:
        if len(row) != 2:
            issues.append(f"{readme_path}: tool table rows must include exactly Tool and Description columns.")
            continue

        tool_cell = row[0]
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
        issues.append(f"{readme_path}: root README must include a markdown link to docs/README.md.")
    return issues


def _find_docs_root_heading_style_issues(repo_root: Path) -> list[str]:
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        return []

    issues: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        if "outputs" in path.relative_to(repo_root).parts:
            continue
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


def _find_agents_path_reference_issues(repo_root: Path) -> list[str]:
    resolved_repo_root = repo_root.expanduser().resolve()
    issues: list[str] = []
    skipped_dir_names = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".worktrees",
        "__pycache__",
    }
    for agents_path in sorted(resolved_repo_root.rglob("AGENTS.md")):
        relative_agents_path = agents_path.relative_to(resolved_repo_root)
        if any(part in skipped_dir_names for part in relative_agents_path.parts):
            continue
        for line_no, line in enumerate(agents_path.read_text(encoding="utf-8").splitlines(), start=1):
            line_lower = line.lower()
            if any(marker in line_lower for marker in AGENTS_NEGATIVE_LINE_MARKERS):
                continue
            for raw_value in AGENTS_CODE_SPAN_PATTERN.findall(line):
                target_literal = raw_value.strip().strip("()[]{}<>,:;!?")
                if not target_literal or target_literal in AGENTS_PATH_LITERAL_EXCEPTIONS:
                    continue
                if _should_skip_agents_path_literal(target_literal):
                    continue

                target_path = _resolve_agents_path_literal(
                    target_literal,
                    agents_path=agents_path,
                    repo_root=resolved_repo_root,
                )
                if target_path is None:
                    continue
                if not target_path.exists():
                    issues.append(f"{agents_path}:{line_no}: referenced path does not exist: `{target_literal}`")
    return issues


def _should_skip_agents_path_literal(value: str) -> bool:
    if value.startswith(("http://", "https://", "mailto:", "#", "/Users/", "/private/", "/tmp/", "/home/", "/var/")):
        return True
    if any(char.isspace() for char in value):
        return True
    if any(char in value for char in "*?[]{}<>$|;&="):
        return True
    if "://" in value or ":" in value:
        return True

    path_part = value.split("#", 1)[0]
    if path_part.startswith(AGENTS_REPO_RELATIVE_PREFIXES):
        return False
    if path_part in AGENTS_ROOT_FILENAMES:
        return False
    if path_part.startswith("./") and path_part[2:] in AGENTS_ROOT_FILENAMES:
        return False
    return True


def _resolve_agents_path_literal(value: str, *, agents_path: Path, repo_root: Path) -> Path | None:
    path_part = value.split("#", 1)[0]
    if path_part.startswith("./") and path_part[2:] in AGENTS_ROOT_FILENAMES:
        path_part = path_part[2:]
    if path_part.startswith(AGENTS_REPO_RELATIVE_PREFIXES):
        repo_target = repo_root / path_part
        relative_target = agents_path.parent / path_part
        target = relative_target if relative_target.exists() else repo_target
    elif path_part in AGENTS_ROOT_FILENAMES:
        target = repo_root / path_part
    else:
        target = agents_path.parent / path_part
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(repo_root)
    except ValueError:
        return None
    return resolved_target


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


def _find_construct_legacy_operator_doc_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    targets = _collect_markdown_files_from_relative_paths(repo_root, relative_paths=CONSTRUCT_OPERATOR_DOC_PATHS)
    for path in targets:
        content = path.read_text(encoding="utf-8")
        for pattern in CONSTRUCT_LEGACY_OPERATOR_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            line_no = content[: match.start()].count("\n") + 1
            issues.append(
                f"{path}:{line_no}: deprecated construct flat-key contract "
                f"'{match.group(0)}' is not allowed in operator docs."
            )
    return issues


def _find_legacy_contract_surface_doc_issues(repo_root: Path) -> list[str]:
    try:
        _docs_md_files, all_md_files = _collect_markdown_files(repo_root)
    except FileNotFoundError:
        return []

    issues: list[str] = []
    for path in all_md_files:
        content = path.read_text(encoding="utf-8")
        for pattern in LEGACY_CONTRACT_SURFACE_DOC_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            line_no = content[: match.start()].count("\n") + 1
            issues.append(
                f"{path}:{line_no}: legacy repo-root contract surface reference "
                f"'{match.group(0)}' is not allowed in docs."
            )
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


def _collect_public_interface_markdown_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set(_collect_tool_readme_markdown_files(repo_root))
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


def _find_tool_readme_structure_issues(repo_root: Path) -> list[str]:
    src_root = repo_root / "src" / "dnadesign"
    if not src_root.exists():
        return []

    issues: list[str] = []
    for tool_name in sorted(discover_repo_tools(repo_root=repo_root)):
        readme_path = src_root / tool_name / "README.md"
        if not readme_path.exists():
            continue

        text = _markdown_body_without_frontmatter(readme_path.read_text(encoding="utf-8"))
        lines = text.splitlines()
        if len(lines) > TOOL_README_MAX_LINES:
            issues.append(
                f"{readme_path}: top-level tool README has {len(lines)} lines; "
                f"keep it at or below {TOOL_README_MAX_LINES} lines and route detail into docs/."
            )

        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if not non_empty_indices:
            issues.append(f"{readme_path}: README is empty.")
            continue

        first_index = non_empty_indices[0]
        banner = _top_rendered_readme_banner(text)
        if banner is None or banner[0] != first_index + 1:
            issues.append(f"{readme_path}: first non-empty line must be the banner image line.")
            continue

        next_index = next((idx for idx in non_empty_indices if idx > first_index), None)
        if next_index is not None and lines[next_index].lstrip().startswith("#"):
            issues.append(
                f"{readme_path}: line after the banner must be narrative text; avoid repeating a top title heading."
            )

        first_heading_index = next(
            (idx for idx in range(first_index + 1, len(lines)) if lines[idx].lstrip().startswith("#")),
            len(lines),
        )
        intro_lines = lines[first_index + 1 : first_heading_index]
        while intro_lines and not intro_lines[0].strip():
            intro_lines.pop(0)
        while intro_lines and not intro_lines[-1].strip():
            intro_lines.pop()

        if not intro_lines:
            issues.append(f"{readme_path}: banner must be followed by one narrative paragraph before docs links.")
        else:
            if any(not line.strip() for line in intro_lines):
                issues.append(
                    f"{readme_path}: intro after the banner must be one paragraph; route extra setup into docs/."
                )
            intro_text = " ".join(line.strip() for line in intro_lines)
            if TOOL_README_SELF_REFERENTIAL_INTRO_PATTERN.search(intro_text):
                issues.append(
                    f"{readme_path}: intro must describe what the tool does; avoid self-referential "
                    "package/layer-in-dnadesign wording."
                )

        if first_heading_index == len(lines):
            issues.append(f"{readme_path}: top-level tool README must include a '## Documentation' section.")
        elif lines[first_heading_index].strip() != "## Documentation":
            issues.append(f"{readme_path}: first heading after the intro must be '## Documentation' for the link map.")

        top_block = "\n".join(lines[:TOOL_README_TOP_LINK_SCAN_LINES])
        first_local_markdown_link: str | None = None
        for raw in LINK_PATTERN.findall(top_block):
            link = raw.strip().split()[0]
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_rel = link.split("#", 1)[0].strip()
            if target_rel.lower().endswith(".md"):
                first_local_markdown_link = target_rel
                break
        if first_local_markdown_link is None:
            issues.append(f"{readme_path}: top section must include a local markdown link to deeper documentation.")
            continue

        docs_index = None
        for candidate in (readme_path.parent / "docs" / "README.md", readme_path.parent / "docs" / "index.md"):
            if candidate.exists():
                docs_index = candidate.resolve()
                break
        if docs_index is None:
            continue

        first_target = (readme_path.parent / first_local_markdown_link).resolve()
        if first_target != docs_index:
            expected_rel = docs_index.relative_to(readme_path.parent.resolve()).as_posix()
            issues.append(
                f"{readme_path}: first local markdown link must point to the tool docs index "
                f"'{expected_rel}', not '{first_local_markdown_link}'."
            )

    return issues
