"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/badges.py

Enforces the restrained Markdown badge surface used by repository docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

BADGE_SOURCE_PATTERN = re.compile(
    r"(?:shields\.io|codecov\.io|(?:^|[/_.-])badge(?:[./?_-]|$))",
    flags=re.IGNORECASE,
)
BADGE_LABEL_PATTERN = re.compile(r"\s*(?:ci|coverage|codecov|license)\s*", flags=re.IGNORECASE)
MARKDOWN = MarkdownIt("commonmark")
ROOT_README_ALLOWED_BADGES = frozenset(
    {
        "[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)]"
        "(https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)",
        "[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)]"
        "(https://codecov.io/gh/e-south/dnadesign)",
        "[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](LICENSE)",
    }
)


@dataclass(frozen=True, slots=True)
class _ImageSpec:
    label: str
    source: str
    linked: bool


@dataclass(frozen=True, slots=True)
class _ImageOccurrence:
    line_no: int
    spec: _ImageSpec


@dataclass(frozen=True, slots=True)
class _RelativeImageOccurrence:
    line_offset: int
    spec: _ImageSpec


class _HTMLImageParser(HTMLParser):
    def __init__(self, *, anchor_depth: int, markdown_link_depth: int) -> None:
        super().__init__(convert_charrefs=True)
        self.anchor_depth = anchor_depth
        self.markdown_link_depth = markdown_link_depth
        self.images: list[tuple[int, _ImageSpec]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "a":
            self.anchor_depth += 1
            return
        if normalized_tag not in {"img", "source"}:
            return
        attributes = {name.casefold(): value or "" for name, value in attrs}
        sources = " ".join(value for name in ("src", "srcset") if (value := attributes.get(name, "")))
        self.images.append(
            (
                self.getpos()[0] - 1,
                _ImageSpec(
                    label=attributes.get("alt", ""),
                    source=sources,
                    linked=self.markdown_link_depth > 0 or self.anchor_depth > 0,
                ),
            )
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"a", "img", "source"}:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a":
            self.anchor_depth = max(0, self.anchor_depth - 1)


def _looks_like_badge(*, label: str, source: str, linked: bool) -> bool:
    return BADGE_SOURCE_PATTERN.search(source) is not None or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _html_fragment_image_specs(
    fragment: str,
    *,
    markdown_link_depth: int,
    anchor_depth: int,
) -> tuple[tuple[tuple[int, _ImageSpec], ...], int]:
    parser = _HTMLImageParser(anchor_depth=anchor_depth, markdown_link_depth=markdown_link_depth)
    parser.feed(fragment)
    parser.close()
    return tuple(parser.images), parser.anchor_depth


def _inline_image_specs(
    children: Iterable[Token],
    *,
    anchor_depth: int,
) -> tuple[tuple[_RelativeImageOccurrence, ...], int]:
    images: list[_RelativeImageOccurrence] = []
    markdown_link_depth = 0
    line_offset = 0
    for token in children:
        if token.type == "link_open":
            markdown_link_depth += 1
            continue
        if token.type == "link_close":
            markdown_link_depth = max(0, markdown_link_depth - 1)
            continue
        if token.type == "image":
            images.append(
                _RelativeImageOccurrence(
                    line_offset=line_offset,
                    spec=_ImageSpec(
                        label=token.content,
                        source=token.attrGet("src") or "",
                        linked=markdown_link_depth > 0 or anchor_depth > 0,
                    ),
                )
            )
        elif token.type == "html_inline":
            html_images, anchor_depth = _html_fragment_image_specs(
                token.content,
                markdown_link_depth=markdown_link_depth,
                anchor_depth=anchor_depth,
            )
            images.extend(
                _RelativeImageOccurrence(line_offset=line_offset + relative_line, spec=spec)
                for relative_line, spec in html_images
            )
        if token.type in {"softbreak", "hardbreak"}:
            line_offset += 1
        else:
            line_offset += token.content.count("\n")
    return tuple(images), anchor_depth


def _match_image_lines(
    relative_occurrences: tuple[_RelativeImageOccurrence, ...],
    *,
    line_candidates: list[tuple[int, _ImageSpec]],
    start_line_no: int,
) -> tuple[_ImageOccurrence, ...]:
    occurrences: list[_ImageOccurrence] = []
    candidate_cursor = 0
    for relative_occurrence in relative_occurrences:
        spec = relative_occurrence.spec
        line_no = start_line_no + relative_occurrence.line_offset
        for candidate_index in range(candidate_cursor, len(line_candidates)):
            candidate_line_no, candidate_spec = line_candidates[candidate_index]
            if candidate_spec == spec:
                line_no = candidate_line_no
                candidate_cursor = candidate_index + 1
                break
        occurrences.append(_ImageOccurrence(line_no=line_no, spec=spec))
    return tuple(occurrences)


def _rendered_image_occurrences(content: str) -> tuple[_ImageOccurrence, ...]:
    lines = content.splitlines()
    environment: dict[str, object] = {}
    tokens = MARKDOWN.parse(content, environment)
    occurrences: list[_ImageOccurrence] = []
    anchor_depth = 0
    for token in tokens:
        if token.type == "inline" and token.map is not None:
            initial_anchor_depth = anchor_depth
            relative_occurrences, anchor_depth = _inline_image_specs(
                token.children or (),
                anchor_depth=anchor_depth,
            )
            if not relative_occurrences:
                continue
            start_line, end_line = token.map
            line_candidates: list[tuple[int, _ImageSpec]] = []
            candidate_anchor_depth = initial_anchor_depth
            for line_index in range(start_line, min(end_line, len(lines))):
                line_tokens = MARKDOWN.parseInline(lines[line_index], environment)
                for line_token in line_tokens:
                    line_occurrences, candidate_anchor_depth = _inline_image_specs(
                        line_token.children or (),
                        anchor_depth=candidate_anchor_depth,
                    )
                    line_candidates.extend(
                        (line_index + 1 + occurrence.line_offset, occurrence.spec) for occurrence in line_occurrences
                    )
            occurrences.extend(
                _match_image_lines(
                    relative_occurrences,
                    line_candidates=line_candidates,
                    start_line_no=start_line + 1,
                )
            )
            continue
        if token.type == "html_block" and token.map is not None:
            html_images, anchor_depth = _html_fragment_image_specs(
                token.content,
                markdown_link_depth=0,
                anchor_depth=anchor_depth,
            )
            for relative_line, spec in html_images:
                occurrences.append(
                    _ImageOccurrence(
                        line_no=token.map[0] + relative_line + 1,
                        spec=spec,
                    )
                )
    return tuple(occurrences)


def _rendered_badge_occurrences(content: str) -> tuple[_ImageOccurrence, ...]:
    return tuple(
        occurrence
        for occurrence in _rendered_image_occurrences(content)
        if _looks_like_badge(
            label=occurrence.spec.label,
            source=occurrence.spec.source,
            linked=occurrence.spec.linked,
        )
    )


def rendered_markdown_badge_lines(content: str) -> tuple[str, ...]:
    """Return rendered badge lines after excluding literal Markdown examples."""
    lines = content.splitlines()
    return tuple(lines[occurrence.line_no - 1].strip() for occurrence in _rendered_badge_occurrences(content))


def find_markdown_badge_policy_issues(repo_root: Path, markdown_files: Iterable[Path]) -> list[str]:
    """Return badge-policy violations without changing documentation."""
    root_readme = (repo_root / "README.md").resolve()
    issues: list[str] = []
    reported_locations: set[tuple[Path, int]] = set()
    root_badge_counts: dict[str, int] = {}
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        is_root_readme = path.resolve() == root_readme
        for occurrence in _rendered_badge_occurrences(content):
            line_no = occurrence.line_no
            location = (path, line_no)
            if location in reported_locations:
                continue
            reported_locations.add(location)
            line = lines[line_no - 1].strip()
            if not is_root_readme:
                issues.append(
                    f"{path}:{line_no}: badges belong only in the root README; use a plain text link instead."
                )
                continue
            if line not in ROOT_README_ALLOWED_BADGES:
                issues.append(
                    f"{path}:{line_no}: root README badge is not in the restrained CI, coverage, and license set."
                )
                continue
            root_badge_counts[line] = root_badge_counts.get(line, 0) + 1
            if root_badge_counts[line] > 1:
                issues.append(f"{path}:{line_no}: root README must include each approved badge at most once.")
    return issues
