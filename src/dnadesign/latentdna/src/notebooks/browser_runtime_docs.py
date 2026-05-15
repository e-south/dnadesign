"""Study-document parsing helpers for the LatentDNA browser runtime."""

from __future__ import annotations

from ..labels import humanize_plot_title


def _humanize_plot_id(plot_id: str) -> str:
    return humanize_plot_title(plot_id)


def _markdown_heading_level(line: str) -> int | None:
    hash_count = len(line) - len(line.lstrip("#"))
    if hash_count == 0 or hash_count > 6:
        return None
    if len(line) <= hash_count or line[hash_count] != " ":
        return None
    return hash_count


def _next_heading_at_or_above(lines: list[str], start: int, level: int) -> int:
    for index in range(start + 1, len(lines)):
        heading_level = _markdown_heading_level(lines[index])
        if heading_level is not None and heading_level <= level:
            return index
    return len(lines)


def _parse_deliverable_markdown(markdown: str) -> dict[str, object]:
    lines = markdown.splitlines()
    summary_lines: list[str] = []
    plot_sections: dict[str, dict[str, str]] = {}

    first_h1 = next((index for index, line in enumerate(lines) if line.startswith("# ")), None)
    if first_h1 is not None:
        index = first_h1 + 1
        while index < len(lines):
            line = lines[index]
            heading_level = _markdown_heading_level(line)
            if heading_level is not None and heading_level <= 2:
                break
            summary_lines.append(line)
            index += 1

    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 3]
    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 3)
        heading = lines[start][4:].strip()
        if "|" not in heading:
            continue
        plot_id_text, title_text = (part.strip() for part in heading.split("|", 1))
        plot_sections[plot_id_text] = {
            "title": title_text,
            "markdown": "\n".join(lines[start + 1 : end]).strip(),
        }

    return {
        "summary_markdown": "\n".join(summary_lines).strip(),
        "plot_sections": plot_sections,
    }


def _extract_plot_details(markdown: str) -> str:
    lines = markdown.splitlines()
    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 4]
    if not heading_indices:
        return ""

    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 4)
        title = lines[start][5:].strip()
        if title.casefold() != "plot details":
            continue
        return "\n".join(lines[start + 1 : end]).strip()
    return ""


def _strip_plot_details(markdown: str) -> str:
    lines = markdown.splitlines()
    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 4]
    if not heading_indices:
        return markdown.strip()

    kept_blocks: list[str] = []
    cursor = 0
    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 4)
        if cursor < start:
            kept_blocks.append("\n".join(lines[cursor:start]).strip())
        title = lines[start][5:].strip()
        if title.casefold() != "plot details":
            kept_blocks.append("\n".join(lines[start:end]).strip())
        cursor = end
    if cursor < len(lines):
        kept_blocks.append("\n".join(lines[cursor:]).strip())
    return "\n\n".join(block for block in kept_blocks if block).strip()


def resolve_plot_doc_block(
    *,
    plot_id: str,
    deliverable_summary: str,
    parsed_markdown: dict[str, object] | None,
) -> dict[str, object]:
    plot_sections = parsed_markdown.get("plot_sections", {}) if isinstance(parsed_markdown, dict) else {}
    summary_markdown = (
        str(parsed_markdown.get("summary_markdown") or "").strip() if isinstance(parsed_markdown, dict) else ""
    )
    plot_entry = plot_sections.get(plot_id) if isinstance(plot_sections, dict) else None
    if isinstance(plot_entry, dict):
        markdown = str(plot_entry.get("markdown") or "").strip()
        plot_details_md = _extract_plot_details(markdown)
        return {
            "title": str(plot_entry.get("title") or _humanize_plot_id(plot_id)),
            "markdown": _strip_plot_details(markdown),
            "plot_details_md": plot_details_md,
            "warning": None,
        }
    fallback_markdown = summary_markdown or deliverable_summary.strip()
    return {
        "title": _humanize_plot_id(plot_id),
        "markdown": fallback_markdown,
        "plot_details_md": "",
        "warning": f"Missing plot-specific study-doc subsection for `{plot_id}`.",
    }
