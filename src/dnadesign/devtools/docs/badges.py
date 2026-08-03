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
from pathlib import Path
from urllib.parse import unquote
from xml.etree.ElementTree import Element

from markdown_it import MarkdownIt
from upa_url import URL

from dnadesign.devtools.docs.parser_compat import (
    MarkdownStateInline,
    MarkdownToken,
    SourceMappedHTMLParser,
    markdown_html_inline_rule,
    markdown_image_rule,
)

BADGE_PATH_PATTERN = re.compile(r"(?:^|[/_.-])badges?(?:[./?_-]|$)", flags=re.IGNORECASE)
BADGE_PROVIDER_HOSTS = frozenset({"shields.io", "codecov.io"})
BADGE_LABEL_PATTERN = re.compile(r"\s*(?:ci|coverage|codecov|license)\s*", flags=re.IGNORECASE)
FLOATING_POINT_PATTERN = re.compile(r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
NON_NEGATIVE_INTEGER_PATTERN = re.compile(r"[0-9]+")
ASCII_WHITESPACE = frozenset("\t\n\f\r ")
C0_CONTROL_OR_SPACE = "".join(chr(codepoint) for codepoint in range(0x21))
CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
MIME_SUBTYPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*")
MARKDOWN_RENDER_BASE_URL = "https://example.test/docs/"
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF_ATTRIBUTE = "{http://www.w3.org/1999/xlink}href"
NON_RENDERING_CONTAINERS = frozenset(
    {
        (XHTML_NAMESPACE, "template"),
        (SVG_NAMESPACE, "clippath"),
        (SVG_NAMESPACE, "desc"),
        (SVG_NAMESPACE, "defs"),
        (SVG_NAMESPACE, "marker"),
        (SVG_NAMESPACE, "mask"),
        (SVG_NAMESPACE, "metadata"),
        (SVG_NAMESPACE, "pattern"),
        (SVG_NAMESPACE, "script"),
        (SVG_NAMESPACE, "style"),
        (SVG_NAMESPACE, "symbol"),
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
    sources: tuple[str, ...]
    linked: bool


@dataclass(frozen=True, slots=True)
class _ImageOccurrence:
    line_no: int
    ordinal: int
    spec: _ImageSpec


@dataclass(frozen=True, slots=True)
class RenderedMarkdownImage:
    """A rendered image and its source location in Markdown."""

    line_no: int
    label: str
    sources: tuple[str, ...]
    linked: bool


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
    state: MarkdownStateInline,
    silent: bool,
    rule: Callable[[MarkdownStateInline, bool], bool],
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


def _source_mapped_image_rule(state: MarkdownStateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_image_rule, expected_type="image")


def _source_mapped_html_inline_rule(state: MarkdownStateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_html_inline_rule, expected_type="html_inline")


MARKDOWN = MarkdownIt("commonmark")
MARKDOWN.inline.ruler.at("image", _source_mapped_image_rule)
MARKDOWN.inline.ruler.at("html_inline", _source_mapped_html_inline_rule)


def _source_has_badge_hint(source: str) -> bool:
    if any("\ud800" <= character <= "\udfff" for character in source):
        return False
    parsed_source = URL.parse(source, MARKDOWN_RENDER_BASE_URL)
    if parsed_source is None:
        return False
    if _is_badge_provider_hostname(parsed_source.hostname):
        return True
    path = unquote(parsed_source.pathname)
    return BADGE_PATH_PATTERN.search(path) is not None


def _is_badge_provider_hostname(hostname: str) -> bool:
    normalized_hostname = hostname[:-1] if hostname.endswith(".") else hostname
    return any(
        normalized_hostname == provider or normalized_hostname.endswith(f".{provider}")
        for provider in BADGE_PROVIDER_HOSTS
    )


def _looks_like_badge(*, label: str, sources: Sequence[str], linked: bool) -> bool:
    return any(_source_has_badge_hint(source) for source in sources) or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _annotate_inline_candidates(
    children: Sequence[MarkdownToken],
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


def _element_hides_descendants(element: Element) -> bool:
    name = _element_name(element)
    if name is None:
        return False
    if "hidden" in element.attrib:
        return True
    return name == (XHTML_NAMESPACE, "dialog") and "open" not in element.attrib


def _normalized_media_query(media: str) -> str:
    return " ".join(CSS_COMMENT_PATTERN.sub(" ", media).casefold().split())


def _picture_source_is_potentially_eligible(element: Element) -> bool:
    media = element.attrib.get("media")
    if media is not None:
        if _normalized_media_query(media) == "not all":
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


def _picture_source_is_terminal(element: Element, *, sources: Sequence[str]) -> bool:
    """Return whether this source ends selection in every rendering environment."""
    media = element.attrib.get("media")
    media_is_unconditional = media is None or _normalized_media_query(media) in {"", "all"}
    return bool(sources) and media_is_unconditional and "type" not in element.attrib


def _picture_sources_before_image(
    children: Sequence[Element],
    *,
    image_index: int,
) -> tuple[tuple[tuple[Element, tuple[str, ...]], ...], bool]:
    """Return reachable sources and whether the image fallback URL is unreachable."""
    candidates: list[tuple[Element, tuple[str, ...]]] = []
    for element in children[:image_index]:
        if _element_name(element) != (XHTML_NAMESPACE, "source") or _element_hides_descendants(element):
            continue
        if not _picture_source_is_potentially_eligible(element):
            continue
        sources = tuple(candidate.url for candidate in _selectable_srcset_candidates(element.attrib.get("srcset", "")))
        if not sources:
            continue
        candidates.append((element, sources))
        if _picture_source_is_terminal(element, sources=sources):
            return tuple(candidates), True
    return tuple(candidates), False


def _occurrence_for_element(
    element: Element,
    *,
    marker_attribute: str,
    sources: Sequence[str],
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
            sources=tuple(source for source in sources if source),
            linked=linked,
        ),
    )


def _dom_image_occurrences(
    rendered_html: str,
    *,
    marker_attribute: str,
    raw_source_map: _RawHTMLSourceMap,
) -> tuple[_ImageOccurrence, ...]:
    parser = SourceMappedHTMLParser(
        marker_attribute=marker_attribute,
        raw_source_map=raw_source_map,
    )
    root = parser.parseFragment(rendered_html, container="div", scripting=True)
    occurrences: list[_ImageOccurrence] = []
    seen_markers: set[str] = set()
    terminal_picture_images: set[int] = set()

    def record(element: Element, *, linked: bool, sources: Sequence[str]) -> None:
        marker = element.attrib.get(marker_attribute)
        if marker is None or marker in seen_markers:
            return
        occurrence = _occurrence_for_element(
            element,
            marker_attribute=marker_attribute,
            sources=sources,
            linked=linked,
        )
        if occurrence is not None:
            seen_markers.add(marker)
            occurrences.append(occurrence)

    stack: list[tuple[Element, bool]] = [(root, False)]
    while stack:
        element, ancestor_linked = stack.pop()
        name = _element_name(element)
        if name in NON_RENDERING_CONTAINERS or _element_hides_descendants(element):
            continue
        linked = ancestor_linked or _element_is_link(element)
        children = list(element)

        if name == (XHTML_NAMESPACE, "picture"):
            for image_index, child in enumerate(children):
                if _element_name(child) == (XHTML_NAMESPACE, "img") and not _element_hides_descendants(child):
                    previous_sources, is_terminal = _picture_sources_before_image(
                        children,
                        image_index=image_index,
                    )
                    for source_element, sources in previous_sources:
                        record(
                            source_element,
                            linked=linked,
                            sources=sources,
                        )
                    if is_terminal:
                        terminal_picture_images.add(id(child))
        elif name == (XHTML_NAMESPACE, "img"):
            record(
                element,
                linked=ancestor_linked,
                sources=() if id(element) in terminal_picture_images else _html_image_sources(element),
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


def rendered_markdown_images(content: str) -> tuple[RenderedMarkdownImage, ...]:
    """Return images that survive Markdown rendering and HTML visibility rules."""

    return tuple(
        RenderedMarkdownImage(
            line_no=occurrence.line_no,
            label=occurrence.spec.label,
            sources=occurrence.spec.sources,
            linked=occurrence.spec.linked,
        )
        for occurrence in _rendered_image_occurrences(content)
    )


def _rendered_badge_occurrences(content: str) -> tuple[_ImageOccurrence, ...]:
    return tuple(
        occurrence
        for occurrence in _rendered_image_occurrences(content)
        if _looks_like_badge(
            label=occurrence.spec.label,
            sources=occurrence.spec.sources,
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
