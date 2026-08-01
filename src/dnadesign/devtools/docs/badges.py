"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/badges.py

Enforces the restrained Markdown badge surface used by repository docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
import secrets
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from pathlib import Path
from xml.etree.ElementTree import Element

import html5lib
from markdown_it import MarkdownIt
from markdown_it.rules_inline import html_inline as markdown_html_inline_rule
from markdown_it.rules_inline import image as markdown_image_rule
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token

BADGE_SOURCE_PATTERN = re.compile(
    r"(?:shields\.io|codecov\.io|(?:^|[/_.-])badge(?:[./?_-]|$))",
    flags=re.IGNORECASE,
)
BADGE_LABEL_PATTERN = re.compile(r"\s*(?:ci|coverage|codecov|license)\s*", flags=re.IGNORECASE)
FLOATING_POINT_PATTERN = re.compile(r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
NON_NEGATIVE_INTEGER_PATTERN = re.compile(r"[0-9]+")
ASCII_WHITESPACE = frozenset("\t\n\f\r ")
RAW_IMAGE_TAGS = frozenset({"image", "img", "source"})
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF_ATTRIBUTE = "{http://www.w3.org/1999/xlink}href"
NON_RENDERING_CONTAINERS = frozenset(
    {
        (XHTML_NAMESPACE, "template"),
        (SVG_NAMESPACE, "desc"),
        (SVG_NAMESPACE, "title"),
    }
)
INLINE_SOURCE_SPAN_META = "dnadesign_source_span"
MAX_IMAGE_DESCRIPTOR_INTEGER = "2147483647"
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
    ordinal: int
    spec: _ImageSpec


@dataclass(frozen=True, slots=True)
class _SrcsetCandidate:
    url: str
    width: str | None = None
    density: float | None = None
    future_height: str | None = None


class _StartTagState(Enum):
    BEFORE_ATTRIBUTE_NAME = auto()
    ATTRIBUTE_NAME = auto()
    AFTER_ATTRIBUTE_NAME = auto()
    BEFORE_ATTRIBUTE_VALUE = auto()
    ATTRIBUTE_VALUE_DOUBLE_QUOTED = auto()
    ATTRIBUTE_VALUE_SINGLE_QUOTED = auto()
    ATTRIBUTE_VALUE_UNQUOTED = auto()
    AFTER_ATTRIBUTE_VALUE_QUOTED = auto()
    SELF_CLOSING_START_TAG = auto()


def _record_inline_source_span(
    state: StateInline,
    silent: bool,
    rule: Callable[[StateInline, bool], bool],
    *,
    expected_type: str,
) -> bool:
    start = state.pos
    token_count = len(state.tokens)
    matched = rule(state, silent)
    if not matched or silent:
        return matched
    produced_tokens = state.tokens[token_count:]
    if not produced_tokens or produced_tokens[-1].type != expected_type:
        raise RuntimeError(f"source-mapped inline rule did not emit its expected {expected_type} token")
    produced_tokens[-1].meta[INLINE_SOURCE_SPAN_META] = (start, state.pos)
    return True


def _source_mapped_image_rule(state: StateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_image_rule, expected_type="image")


def _source_mapped_html_inline_rule(state: StateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_html_inline_rule, expected_type="html_inline")


MARKDOWN = MarkdownIt("commonmark")
MARKDOWN.inline.ruler.at("image", _source_mapped_image_rule)
MARKDOWN.inline.ruler.at("html_inline", _source_mapped_html_inline_rule)


def _looks_like_badge(*, label: str, source: str, linked: bool) -> bool:
    return BADGE_SOURCE_PATTERN.search(source) is not None or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _start_tag_boundary(source: str, start: int) -> tuple[int, int] | None:
    state = _StartTagState.BEFORE_ATTRIBUTE_NAME
    self_closing_slash: int | None = None
    index = start
    while index < len(source):
        character = source[index]
        if state is _StartTagState.BEFORE_ATTRIBUTE_NAME:
            if character in ASCII_WHITESPACE:
                pass
            elif character == "/":
                self_closing_slash = index
                state = _StartTagState.SELF_CLOSING_START_TAG
            elif character == ">":
                return index, index
            else:
                state = _StartTagState.ATTRIBUTE_NAME
        elif state is _StartTagState.ATTRIBUTE_NAME:
            if character in ASCII_WHITESPACE:
                state = _StartTagState.AFTER_ATTRIBUTE_NAME
            elif character == "/":
                self_closing_slash = index
                state = _StartTagState.SELF_CLOSING_START_TAG
            elif character == "=":
                state = _StartTagState.BEFORE_ATTRIBUTE_VALUE
            elif character == ">":
                return index, index
        elif state is _StartTagState.AFTER_ATTRIBUTE_NAME:
            if character in ASCII_WHITESPACE:
                pass
            elif character == "/":
                self_closing_slash = index
                state = _StartTagState.SELF_CLOSING_START_TAG
            elif character == "=":
                state = _StartTagState.BEFORE_ATTRIBUTE_VALUE
            elif character == ">":
                return index, index
            else:
                state = _StartTagState.ATTRIBUTE_NAME
        elif state is _StartTagState.BEFORE_ATTRIBUTE_VALUE:
            if character in ASCII_WHITESPACE:
                pass
            elif character == '"':
                state = _StartTagState.ATTRIBUTE_VALUE_DOUBLE_QUOTED
            elif character == "'":
                state = _StartTagState.ATTRIBUTE_VALUE_SINGLE_QUOTED
            elif character == ">":
                return index, index
            else:
                state = _StartTagState.ATTRIBUTE_VALUE_UNQUOTED
        elif state is _StartTagState.ATTRIBUTE_VALUE_DOUBLE_QUOTED:
            if character == '"':
                state = _StartTagState.AFTER_ATTRIBUTE_VALUE_QUOTED
        elif state is _StartTagState.ATTRIBUTE_VALUE_SINGLE_QUOTED:
            if character == "'":
                state = _StartTagState.AFTER_ATTRIBUTE_VALUE_QUOTED
        elif state is _StartTagState.ATTRIBUTE_VALUE_UNQUOTED:
            if character in ASCII_WHITESPACE:
                state = _StartTagState.BEFORE_ATTRIBUTE_NAME
            elif character == ">":
                return index, index
        elif state is _StartTagState.AFTER_ATTRIBUTE_VALUE_QUOTED:
            if character in ASCII_WHITESPACE:
                state = _StartTagState.BEFORE_ATTRIBUTE_NAME
            elif character == "/":
                self_closing_slash = index
                state = _StartTagState.SELF_CLOSING_START_TAG
            elif character == ">":
                return index, index
            else:
                state = _StartTagState.BEFORE_ATTRIBUTE_NAME
                continue
        elif state is _StartTagState.SELF_CLOSING_START_TAG:
            if character == ">":
                assert self_closing_slash is not None
                return index, self_closing_slash
            self_closing_slash = None
            state = _StartTagState.BEFORE_ATTRIBUTE_NAME
            continue
        index += 1
    return None


def _marker_value(*, line_no: int, ordinal: int) -> str:
    return f"{line_no}:{ordinal}"


def _annotate_raw_html(
    source: str,
    *,
    start_line_no: int,
    marker_attribute: str,
    ordinals: Iterator[int],
) -> str:
    replacements: list[tuple[int, str]] = []
    cursor = 0
    line_cursor = 0
    line_offset = 0
    while cursor < len(source):
        tag_start = source.find("<", cursor)
        if tag_start < 0:
            break
        line_offset += source[line_cursor:tag_start].count("\n")
        line_cursor = tag_start
        name_start = tag_start + 1
        if name_start >= len(source) or not source[name_start].isascii() or not source[name_start].isalpha():
            cursor = tag_start + 1
            continue
        name_end = name_start + 1
        while (
            name_end < len(source) and source[name_end] not in ASCII_WHITESPACE and source[name_end] not in {"/", ">"}
        ):
            name_end += 1
        boundary = _start_tag_boundary(source, name_end)
        if boundary is None:
            break
        tag_end, insertion = boundary
        tag = source[name_start:name_end].casefold()
        if tag in RAW_IMAGE_TAGS:
            ordinal = next(ordinals)
            line_no = start_line_no + line_offset
            replacements.append(
                (
                    insertion,
                    f' {marker_attribute}="{_marker_value(line_no=line_no, ordinal=ordinal)}"',
                )
            )
        cursor = tag_end + 1

    chunks: list[str] = []
    cursor = 0
    for insertion, marker in replacements:
        chunks.extend((source[cursor:insertion], marker))
        cursor = insertion
    chunks.append(source[cursor:])
    return "".join(chunks)


def _annotate_inline_candidates(
    children: Sequence[Token],
    *,
    source: str,
    start_line_no: int,
    marker_attribute: str,
    ordinals: Iterator[int],
) -> None:
    line_cursor = 0
    line_offset = 0
    for token in children:
        if token.type not in {"html_inline", "image"}:
            continue
        span = token.meta.get(INLINE_SOURCE_SPAN_META)
        if not isinstance(span, tuple) or len(span) != 2 or not all(isinstance(position, int) for position in span):
            raise RuntimeError(f"{token.type} token is missing its source-span contract")
        span_start, span_end = span
        if span_start < line_cursor or span_end < span_start or span_end > len(source):
            raise RuntimeError(f"{token.type} token has an invalid source span")
        line_offset += source[line_cursor:span_start].count("\n")
        line_cursor = span_start
        token_line_no = start_line_no + line_offset
        if token.type == "image":
            ordinal = next(ordinals)
            token.attrSet(
                marker_attribute,
                _marker_value(line_no=token_line_no, ordinal=ordinal),
            )
        else:
            if token.content != source[span_start:span_end]:
                raise RuntimeError("html_inline token content differs from its declared source span")
            token.content = _annotate_raw_html(
                token.content,
                start_line_no=token_line_no,
                marker_attribute=marker_attribute,
                ordinals=ordinals,
            )


def _render_markdown_with_markers(content: str) -> tuple[str, str]:
    environment: dict[str, object] = {}
    tokens = MARKDOWN.parse(content, environment)
    marker_attribute = f"data-dnadesign-image-{secrets.token_hex(12)}"
    ordinals = count()
    for token in tokens:
        if token.type == "inline" and token.map is not None:
            _annotate_inline_candidates(
                token.children or (),
                source=token.content,
                start_line_no=token.map[0] + 1,
                marker_attribute=marker_attribute,
                ordinals=ordinals,
            )
        elif token.type == "html_block" and token.map is not None:
            token.content = _annotate_raw_html(
                token.content,
                start_line_no=token.map[0] + 1,
                marker_attribute=marker_attribute,
                ordinals=ordinals,
            )
    return MARKDOWN.renderer.render(tokens, MARKDOWN.options, environment), marker_attribute


def _positive_integer_descriptor(value: str) -> str | None:
    if NON_NEGATIVE_INTEGER_PATTERN.fullmatch(value) is None:
        return None
    normalized = value.lstrip("0")
    if not normalized:
        return None
    if len(normalized) > len(MAX_IMAGE_DESCRIPTOR_INTEGER) or (
        len(normalized) == len(MAX_IMAGE_DESCRIPTOR_INTEGER) and normalized > MAX_IMAGE_DESCRIPTOR_INTEGER
    ):
        return None
    return normalized


def _parse_descriptors(descriptors: Sequence[str]) -> tuple[str | None, float | None, str | None] | None:
    width: str | None = None
    density: float | None = None
    future_height: str | None = None
    for descriptor in descriptors:
        if descriptor.endswith("w") and (value := _positive_integer_descriptor(descriptor[:-1])) is not None:
            if width is not None or density is not None:
                return None
            width = value
            continue
        if descriptor.endswith("x") and FLOATING_POINT_PATTERN.fullmatch(descriptor[:-1]):
            value = float(descriptor[:-1])
            if (
                not math.isfinite(value)
                or value < 0
                or width is not None
                or density is not None
                or future_height is not None
            ):
                return None
            density = value
            continue
        if descriptor.endswith("h") and (value := _positive_integer_descriptor(descriptor[:-1])) is not None:
            if future_height is not None or density is not None:
                return None
            future_height = value
            continue
        return None
    if future_height is not None and width is None:
        return None
    return width, density, future_height


def _parse_srcset(value: str) -> tuple[_SrcsetCandidate, ...]:
    candidates: list[_SrcsetCandidate] = []
    position = 0
    while position < len(value):
        while position < len(value) and (value[position] in ASCII_WHITESPACE or value[position] == ","):
            position += 1
        if position >= len(value):
            break

        url_start = position
        while position < len(value) and value[position] not in ASCII_WHITESPACE:
            position += 1
        url = value[url_start:position]
        descriptors: list[str] = []
        if url.endswith(","):
            url = url.rstrip(",")
        else:
            while position < len(value) and value[position] in ASCII_WHITESPACE:
                position += 1
            descriptor: list[str] = []
            in_parentheses = False
            while position < len(value):
                character = value[position]
                if character == "," and not in_parentheses:
                    if descriptor:
                        descriptors.append("".join(descriptor))
                        descriptor = []
                    position += 1
                    break
                if character in ASCII_WHITESPACE and not in_parentheses:
                    if descriptor:
                        descriptors.append("".join(descriptor))
                        descriptor = []
                else:
                    if character == "(" and not in_parentheses:
                        in_parentheses = True
                    elif character == ")" and in_parentheses:
                        in_parentheses = False
                    descriptor.append(character)
                position += 1
            if descriptor:
                descriptors.append("".join(descriptor))

        parsed = _parse_descriptors(descriptors)
        if url and parsed is not None:
            width, density, future_height = parsed
            candidates.append(
                _SrcsetCandidate(
                    url=url,
                    width=width,
                    density=density,
                    future_height=future_height,
                )
            )
    return tuple(candidates)


def _selectable_srcset_candidates(value: str) -> tuple[_SrcsetCandidate, ...]:
    selected: list[_SrcsetCandidate] = []
    density_values: set[float] = set()
    width_values: set[str] = set()
    for candidate in _parse_srcset(value):
        if candidate.width is not None:
            if candidate.width in width_values:
                continue
            width_values.add(candidate.width)
        else:
            density = candidate.density if candidate.density is not None else 1.0
            if density in density_values:
                continue
            density_values.add(density)
        selected.append(candidate)
    if any(candidate.width is not None or candidate.density is None or candidate.density > 0 for candidate in selected):
        selected = [candidate for candidate in selected if candidate.density != 0]
    return tuple(selected)


def _html_image_sources(element: Element) -> tuple[str, ...]:
    candidates = _selectable_srcset_candidates(element.attrib.get("srcset", ""))
    has_width = any(candidate.width is not None for candidate in candidates)
    has_density_one = any(
        candidate.width is None and (candidate.density is None or candidate.density == 1) for candidate in candidates
    )
    default_source = element.attrib.get("src", "")
    if default_source and not has_width and not has_density_one:
        candidates = tuple(candidate for candidate in candidates if candidate.density != 0)
    sources = [candidate.url for candidate in candidates]
    if default_source and not has_width and not has_density_one:
        sources.append(default_source)
    return tuple(sources)


def _element_name(element: Element) -> tuple[str, str] | None:
    if not isinstance(element.tag, str) or not element.tag.startswith("{"):
        return None
    namespace, separator, local_name = element.tag[1:].partition("}")
    if not separator:
        return None
    return namespace, local_name.casefold()


def _element_is_link(element: Element) -> bool:
    name = _element_name(element)
    return name in {(XHTML_NAMESPACE, "a"), (SVG_NAMESPACE, "a")} and (
        "href" in element.attrib or XLINK_HREF_ATTRIBUTE in element.attrib
    )


def _occurrence_for_element(
    element: Element,
    *,
    marker_attribute: str,
    source: str,
    linked: bool,
) -> _ImageOccurrence | None:
    marker = element.attrib.get(marker_attribute)
    if marker is None:
        return None
    line_text, separator, ordinal_text = marker.partition(":")
    if not separator or not line_text.isdigit() or not ordinal_text.isdigit():
        return None
    return _ImageOccurrence(
        line_no=int(line_text),
        ordinal=int(ordinal_text),
        spec=_ImageSpec(
            label=element.attrib.get("alt", ""),
            source=source,
            linked=linked,
        ),
    )


def _dom_image_occurrences(rendered_html: str, *, marker_attribute: str) -> tuple[_ImageOccurrence, ...]:
    root = html5lib.parseFragment(
        rendered_html,
        treebuilder="etree",
        namespaceHTMLElements=True,
        scripting=True,
    )
    occurrences: list[_ImageOccurrence] = []
    seen_markers: set[str] = set()

    def record(element: Element, *, linked: bool, sources: Sequence[str]) -> None:
        marker = element.attrib.get(marker_attribute)
        if marker is None or marker in seen_markers:
            return
        occurrence = _occurrence_for_element(
            element,
            marker_attribute=marker_attribute,
            source=" ".join(source for source in sources if source),
            linked=linked,
        )
        if occurrence is not None:
            seen_markers.add(marker)
            occurrences.append(occurrence)

    stack: list[tuple[Element, bool]] = [(root, False)]
    while stack:
        element, ancestor_linked = stack.pop()
        name = _element_name(element)
        if name in NON_RENDERING_CONTAINERS:
            continue
        linked = ancestor_linked or _element_is_link(element)
        children = list(element)

        if name == (XHTML_NAMESPACE, "picture"):
            previous_sources: list[Element] = []
            for child in children:
                child_name = _element_name(child)
                if child_name == (XHTML_NAMESPACE, "source"):
                    previous_sources.append(child)
                    continue
                if child_name == (XHTML_NAMESPACE, "img"):
                    for source_element in previous_sources:
                        record(
                            source_element,
                            linked=linked,
                            sources=tuple(
                                candidate.url
                                for candidate in _selectable_srcset_candidates(source_element.attrib.get("srcset", ""))
                            ),
                        )
        elif name == (XHTML_NAMESPACE, "img"):
            record(
                element,
                linked=ancestor_linked,
                sources=_html_image_sources(element),
            )
        elif name == (SVG_NAMESPACE, "image"):
            record(
                element,
                linked=ancestor_linked,
                sources=(element.attrib.get("href", ""), element.attrib.get(XLINK_HREF_ATTRIBUTE, "")),
            )

        stack.extend((child, linked) for child in reversed(children))
    return tuple(sorted(occurrences, key=lambda occurrence: occurrence.ordinal))


def _rendered_image_occurrences(content: str) -> tuple[_ImageOccurrence, ...]:
    rendered_html, marker_attribute = _render_markdown_with_markers(content)
    return _dom_image_occurrences(rendered_html, marker_attribute=marker_attribute)


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
    root_badge_counts: dict[str, int] = {}
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        is_root_readme = path.resolve() == root_readme
        for occurrence in _rendered_badge_occurrences(content):
            line_no = occurrence.line_no
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
