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
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from xml.etree.ElementTree import Element

import html5lib
from html5lib._tokenizer import HTMLTokenizer, tokenTypes
from html5lib.html5parser import HTMLParser, _ReparseException
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
CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
MIME_SUBTYPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*")
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


@dataclass(frozen=True, slots=True)
class _RawHTMLFragment:
    content: str
    source_start_line: int


@dataclass(frozen=True, slots=True)
class _RawHTMLSourceSpan:
    rendered_start: int
    rendered_end: int
    rendered_start_line: int
    source_start_line: int


@dataclass(frozen=True, slots=True)
class _RawHTMLSourceMap:
    spans: tuple[_RawHTMLSourceSpan, ...]
    rendered_starts: tuple[int, ...]

    @classmethod
    def from_fragments(
        cls,
        rendered_html: str,
        fragments: Sequence[_RawHTMLFragment],
    ) -> _RawHTMLSourceMap:
        line_starts = _line_start_offsets(rendered_html)
        spans: list[_RawHTMLSourceSpan] = []
        search_start = 0
        for fragment in fragments:
            if not fragment.content or fragment.source_start_line < 1:
                raise RuntimeError("raw HTML source fragment has an invalid source contract")
            rendered_start = rendered_html.find(fragment.content, search_start)
            if rendered_start < 0:
                raise RuntimeError("rendered Markdown omitted a declared raw HTML source fragment")
            rendered_end = rendered_start + len(fragment.content)
            spans.append(
                _RawHTMLSourceSpan(
                    rendered_start=rendered_start,
                    rendered_end=rendered_end,
                    rendered_start_line=bisect_right(line_starts, rendered_start),
                    source_start_line=fragment.source_start_line,
                )
            )
            search_start = rendered_end
        return cls(
            spans=tuple(spans),
            rendered_starts=tuple(span.rendered_start for span in spans),
        )

    def source_line_for(self, *, rendered_offset: int, rendered_line: int) -> int | None:
        span_index = bisect_right(self.rendered_starts, rendered_offset) - 1
        if span_index < 0:
            return None
        span = self.spans[span_index]
        if rendered_offset >= span.rendered_end:
            return None
        line_offset = rendered_line - span.rendered_start_line
        if line_offset < 0:
            raise RuntimeError("HTML tokenizer resolved a tag before its raw source fragment")
        return span.source_start_line + line_offset


def _line_start_offsets(content: str) -> tuple[int, ...]:
    return (0, *(index + 1 for index, character in enumerate(content) if character == "\n"))


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


def _annotate_inline_candidates(
    children: Sequence[Token],
    *,
    source: str,
    start_line_no: int,
    marker_attribute: str,
) -> tuple[_RawHTMLFragment, ...]:
    raw_fragments: list[_RawHTMLFragment] = []
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
            token.attrSet(marker_attribute, str(token_line_no))
        else:
            if token.content != source[span_start:span_end]:
                raise RuntimeError("html_inline token content differs from its declared source span")
            raw_fragments.append(
                _RawHTMLFragment(
                    content=token.content,
                    source_start_line=token_line_no,
                )
            )
    return tuple(raw_fragments)


def _render_markdown_with_markers(content: str) -> tuple[str, str, _RawHTMLSourceMap]:
    environment: dict[str, object] = {}
    tokens = MARKDOWN.parse(content, environment)
    marker_attribute = f"data-dnadesign-image-{secrets.token_hex(12)}"
    raw_fragments: list[_RawHTMLFragment] = []
    for token in tokens:
        if token.type == "inline" and token.map is not None:
            raw_fragments.extend(
                _annotate_inline_candidates(
                    token.children or (),
                    source=token.content,
                    start_line_no=token.map[0] + 1,
                    marker_attribute=marker_attribute,
                )
            )
        elif token.type == "html_block" and token.map is not None:
            raw_fragments.append(
                _RawHTMLFragment(
                    content=token.content,
                    source_start_line=token.map[0] + 1,
                )
            )
    rendered_html = MARKDOWN.renderer.render(tokens, MARKDOWN.options, environment)
    return (
        rendered_html,
        marker_attribute,
        _RawHTMLSourceMap.from_fragments(rendered_html, raw_fragments),
    )


class _SourceMappedHTMLTokenizer(HTMLTokenizer):
    def __init__(
        self,
        stream: str,
        *,
        parser: HTMLParser,
        marker_attribute: str,
        raw_source_map: _RawHTMLSourceMap,
        **kwargs: object,
    ) -> None:
        if not isinstance(stream, str):
            raise TypeError("source-mapped HTML parsing requires rendered text")
        self._line_starts = _line_start_offsets(stream)
        super().__init__(stream, parser=parser, **kwargs)
        self._marker_attribute = marker_attribute
        self._raw_source_map = raw_source_map
        self._ordinals = count()

    def tagOpenState(self) -> bool:
        rendered_line, rendered_column = self.stream.position()
        rendered_offset = self._line_starts[rendered_line - 1] + rendered_column - 1
        previous_token = getattr(self, "currentToken", None)
        result = super().tagOpenState()
        token = getattr(self, "currentToken", None)
        if token is not previous_token and token is not None and token.get("type") == tokenTypes["StartTag"]:
            token["dnadesign_rendered_line"] = rendered_line
            token["dnadesign_rendered_offset"] = rendered_offset
        return result

    def emitCurrentToken(self) -> None:
        token = self.currentToken
        rendered_line = token.pop("dnadesign_rendered_line", None)
        rendered_offset = token.pop("dnadesign_rendered_offset", None)
        super().emitCurrentToken()
        if token.get("type") != tokenTypes["StartTag"] or token.get("name") not in RAW_IMAGE_TAGS:
            return
        attributes = token.get("data")
        if not isinstance(attributes, dict):
            raise RuntimeError("HTML tokenizer emitted a start tag without normalized attributes")
        source_line = None
        if isinstance(rendered_line, int) and isinstance(rendered_offset, int):
            source_line = self._raw_source_map.source_line_for(
                rendered_offset=rendered_offset,
                rendered_line=rendered_line,
            )
        if source_line is None:
            declared_line = attributes.get(self._marker_attribute)
            if isinstance(declared_line, str) and declared_line.isdigit():
                source_line = int(declared_line)
        if source_line is None:
            return
        if source_line < 1:
            raise RuntimeError("HTML tokenizer resolved an invalid source line")
        attributes[self._marker_attribute] = f"{source_line}:{next(self._ordinals)}"


class _SourceMappedHTMLParser(HTMLParser):
    def __init__(self, *, marker_attribute: str, raw_source_map: _RawHTMLSourceMap) -> None:
        self._marker_attribute = marker_attribute
        self._raw_source_map = raw_source_map
        super().__init__(tree=html5lib.getTreeBuilder("etree"), namespaceHTMLElements=True)

    def _parse(
        self,
        stream: str,
        innerHTML: bool = False,
        container: str = "div",
        scripting: bool = False,
        **kwargs: object,
    ) -> None:
        self.innerHTMLMode = innerHTML
        self.container = container
        self.scripting = scripting
        self.tokenizer = _SourceMappedHTMLTokenizer(
            stream,
            parser=self,
            marker_attribute=self._marker_attribute,
            raw_source_map=self._raw_source_map,
            **kwargs,
        )
        self.reset()
        try:
            self.mainLoop()
        except _ReparseException:
            self.reset()
            self.mainLoop()


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


def _picture_source_is_potentially_eligible(element: Element) -> bool:
    media = element.attrib.get("media")
    if media is not None:
        normalized_media = " ".join(CSS_COMMENT_PATTERN.sub(" ", media).casefold().split())
        if normalized_media == "not all":
            return False
        # Unknown or invalid queries remain inspectable: without a viewport and
        # full CSS evaluator, treating them as ineligible could hide a badge.
    declared_type = element.attrib.get("type")
    if declared_type is None:
        return True
    media_type = declared_type.partition(";")[0].strip().casefold()
    if not media_type:
        return True
    prefix, separator, subtype = media_type.partition("/")
    return separator == "/" and prefix == "image" and MIME_SUBTYPE_PATTERN.fullmatch(subtype) is not None


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


def _dom_image_occurrences(
    rendered_html: str,
    *,
    marker_attribute: str,
    raw_source_map: _RawHTMLSourceMap,
) -> tuple[_ImageOccurrence, ...]:
    parser = _SourceMappedHTMLParser(
        marker_attribute=marker_attribute,
        raw_source_map=raw_source_map,
    )
    root = parser.parseFragment(rendered_html, container="div", scripting=True)
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
                    if _picture_source_is_potentially_eligible(child):
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
    rendered_html, marker_attribute, raw_source_map = _render_markdown_with_markers(content)
    return _dom_image_occurrences(
        rendered_html,
        marker_attribute=marker_attribute,
        raw_source_map=raw_source_map,
    )


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
