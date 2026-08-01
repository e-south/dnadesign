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
INERT_HTML_CONTAINERS = frozenset(
    {
        "iframe",
        "noembed",
        "noframes",
        "noscript",
        "plaintext",
        "script",
        "style",
        "template",
        "textarea",
        "title",
        "xmp",
    }
)
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


@dataclass(slots=True)
class _HTMLContext:
    anchor_depth: int = 0
    inert_depth: int = 0
    picture_depth: int = 0

    def clone(self) -> _HTMLContext:
        return _HTMLContext(
            anchor_depth=self.anchor_depth,
            inert_depth=self.inert_depth,
            picture_depth=self.picture_depth,
        )


class _HTMLImageParser(HTMLParser):
    def __init__(self, *, context: _HTMLContext, markdown_link_depth: int) -> None:
        super().__init__(convert_charrefs=True)
        self.context = context
        self.markdown_link_depth = markdown_link_depth
        self.images: list[tuple[int, _ImageSpec]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in INERT_HTML_CONTAINERS:
            self.context.inert_depth += 1
            return
        if self.context.inert_depth > 0:
            return
        if normalized_tag == "a":
            self.context.anchor_depth = 1
            self.markdown_link_depth = 0
            return
        if normalized_tag == "picture":
            self.context.picture_depth += 1
            return
        if normalized_tag not in {"img", "source"}:
            return
        attributes = {name.casefold(): value or "" for name, value in attrs}
        source_names = ("src", "srcset") if normalized_tag == "img" else ("srcset",)
        sources = " ".join(value for name in source_names if (value := attributes.get(name, "")))
        if normalized_tag == "source" and self.context.picture_depth == 0:
            return
        self.images.append(
            (
                self.getpos()[0] - 1,
                _ImageSpec(
                    label=attributes.get("alt", ""),
                    source=sources,
                    linked=self.markdown_link_depth > 0 or self.context.anchor_depth > 0,
                ),
            )
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in INERT_HTML_CONTAINERS:
            self.context.inert_depth = max(0, self.context.inert_depth - 1)
            return
        if self.context.inert_depth > 0:
            return
        if normalized_tag == "a":
            self.context.anchor_depth = max(0, self.context.anchor_depth - 1)
        elif normalized_tag == "picture":
            self.context.picture_depth = max(0, self.context.picture_depth - 1)


def _looks_like_badge(*, label: str, source: str, linked: bool) -> bool:
    return BADGE_SOURCE_PATTERN.search(source) is not None or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _html_fragment_image_specs(
    fragment: str,
    *,
    markdown_link_depth: int,
    context: _HTMLContext,
) -> tuple[tuple[tuple[int, _ImageSpec], ...], int]:
    parser = _HTMLImageParser(context=context, markdown_link_depth=markdown_link_depth)
    parser.feed(fragment)
    parser.close()
    return tuple(parser.images), parser.markdown_link_depth


def _inline_image_specs(
    children: Iterable[Token],
    *,
    context: _HTMLContext,
    source: str,
) -> tuple[_RelativeImageOccurrence, ...]:
    images: list[_RelativeImageOccurrence] = []
    markdown_link_depth = 0
    line_offset = 0
    source_cursor = 0
    for token in children:
        if token.type == "link_open":
            if context.inert_depth == 0:
                context.anchor_depth = 0
            markdown_link_depth += 1
            continue
        if token.type == "link_close":
            markdown_link_depth = max(0, markdown_link_depth - 1)
            continue
        if token.type == "image":
            if context.inert_depth == 0:
                images.append(
                    _RelativeImageOccurrence(
                        line_offset=line_offset,
                        spec=_ImageSpec(
                            label=token.content,
                            source=token.attrGet("src") or "",
                            linked=markdown_link_depth > 0 or context.anchor_depth > 0,
                        ),
                    )
                )
        elif token.type == "html_inline":
            fragment_start = source.find(token.content, source_cursor)
            if fragment_start >= 0:
                line_offset = source[:fragment_start].count("\n")
                source_cursor = fragment_start + len(token.content)
            html_images, markdown_link_depth = _html_fragment_image_specs(
                token.content,
                markdown_link_depth=markdown_link_depth,
                context=context,
            )
            images.extend(
                _RelativeImageOccurrence(line_offset=line_offset + relative_line, spec=spec)
                for relative_line, spec in html_images
            )
        elif token.type == "code_inline" and token.markup:
            opening = source.find(token.markup, source_cursor)
            closing = source.find(token.markup, opening + len(token.markup)) if opening >= 0 else -1
            if closing >= 0:
                source_cursor = closing + len(token.markup)
                line_offset = source[:source_cursor].count("\n")
        elif token.type == "text" and token.content:
            text_start = source.find(token.content, source_cursor)
            if text_start >= 0:
                source_cursor = text_start + len(token.content)
        if token.type in {"softbreak", "hardbreak"}:
            newline = source.find("\n", source_cursor)
            if newline >= 0:
                source_cursor = newline + 1
            line_offset = source[:source_cursor].count("\n")
    return tuple(images)


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
    context = _HTMLContext()
    for token in tokens:
        if token.type == "inline" and token.map is not None:
            initial_context = context.clone()
            relative_occurrences = _inline_image_specs(
                token.children or (),
                context=context,
                source=token.content,
            )
            if not relative_occurrences:
                continue
            start_line, end_line = token.map
            line_candidates: list[tuple[int, _ImageSpec]] = []
            candidate_context = initial_context
            for line_index in range(start_line, min(end_line, len(lines))):
                line_tokens = MARKDOWN.parseInline(lines[line_index], environment)
                for line_token in line_tokens:
                    line_occurrences = _inline_image_specs(
                        line_token.children or (),
                        context=candidate_context,
                        source=line_token.content,
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
            html_images, _markdown_link_depth = _html_fragment_image_specs(
                token.content,
                markdown_link_depth=0,
                context=context,
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
